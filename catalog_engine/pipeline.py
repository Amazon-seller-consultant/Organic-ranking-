"""Pipeline orchestration: parse -> validate input -> generate -> verify ->
output, with per-seller scoping, incremental reprocessing, and progress.

Scope policy (first pass, per product owner):
- content is generated for Active rows only, including parent rows;
- Inactive/Removed rows are parsed and reported but not sent to the LLM;
- bullets 4/5 and other untouched attributes are preserved via partial update.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from . import category_rules, generator, output, validator
from .exceptions import GenerationError
from .models import (
    Issue,
    Parentage,
    ParseResult,
    ProductRecord,
    RunSummary,
    Severity,
    SkuOutcome,
)
from .store import Store


def fingerprint(record: ProductRecord) -> str:
    """Hash of the content-relevant source data; unchanged hash => skip SKU."""
    snapshot = generator.build_source_snapshot(record)
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def in_scope(record: ProductRecord, include_statuses=("active",)) -> tuple[bool, str]:
    status = record.listing_status.strip().lower()
    if status in include_statuses:
        return True, ""
    return False, (
        f"listing_status is '{record.listing_status or 'blank'}' — not in this "
        f"seller's include_statuses {list(include_statuses)}"
    )


def select_pilot(records: list[ProductRecord]) -> list[ProductRecord]:
    """One representative per product type: prefer a standalone; else the
    first child plus its parent (so variation copy is exercised too)."""
    by_sku = {r.sku: r for r in records}
    chosen: list[ProductRecord] = []
    seen_types: set[str] = set()
    for rec in records:
        if rec.product_type in seen_types:
            continue
        if rec.parentage is Parentage.STANDALONE:
            chosen.append(rec)
            seen_types.add(rec.product_type)
    for rec in records:
        if rec.product_type in seen_types:
            continue
        if rec.parentage is Parentage.CHILD:
            chosen.append(rec)
            if rec.parent_sku and rec.parent_sku in by_sku:
                chosen.append(by_sku[rec.parent_sku])
            seen_types.add(rec.product_type)
    for rec in records:  # product types with only parents
        if rec.product_type not in seen_types:
            chosen.append(rec)
            seen_types.add(rec.product_type)
    return chosen


def run_pipeline(
    seller_id: str,
    file_path: str | Path,
    store: Store,
    pilot: bool = False,
    limit: Optional[int] = None,
    force: bool = False,
    refresh_rules: bool = False,
    workers: int = 1,
    progress: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> RunSummary:
    from .parser import parse_flat_file  # local import keeps module load light

    config = store.get_seller(seller_id)
    run_id = store.start_run(seller_id, str(file_path))
    out_dir = store.seller_dir(seller_id) / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. PARSE ---------------------------------------------------------
    progress(f"[{run_id}] parsing {Path(file_path).name} ...")
    parse = parse_flat_file(file_path, seller_id=seller_id)
    progress(
        f"  {len(parse.records)} records, {len(parse.families)} variation "
        f"families, {len(parse.warnings)} parse warnings"
    )

    # ---- 2. VALIDATE INPUT (category rules from the file's own sheets) ----
    progress("  extracting/loading per-product-type attribute rules ...")
    rules = category_rules.get_rules(
        file_path, parse.settings, store, force_refresh=refresh_rules
    )
    input_issues = category_rules.validate_input(parse.records, rules)
    issues_by_sku: dict[str, list[Issue]] = {}
    for iss in input_issues:
        issues_by_sku.setdefault(iss.sku, []).append(iss)
    progress(f"  input validation: {len(input_issues)} findings across "
             f"{len(issues_by_sku)} SKUs (missing mandatory attrs are flagged, never invented)")

    # ---- select work set ---------------------------------------------------
    fingerprints = store.get_fingerprints(seller_id)
    outcomes: list[SkuOutcome] = []
    work: list[ProductRecord] = []
    for rec in parse.records:
        ok, why = in_scope(rec, tuple(
            s.strip().lower() for s in (config.include_statuses or ["active"])))
        if not ok:
            outcomes.append(SkuOutcome(record=rec, generated=None,
                                       issues=issues_by_sku.get(rec.sku, []),
                                       status="skipped", skip_reason=why))
            continue
        if not force and fingerprints.get(rec.sku) == fingerprint(rec):
            outcomes.append(SkuOutcome(record=rec, generated=None,
                                       issues=issues_by_sku.get(rec.sku, []),
                                       status="skipped_unchanged",
                                       skip_reason="source data unchanged since last run"))
            continue
        work.append(rec)

    if pilot:
        pilot_set = {r.sku for r in select_pilot(work)}
        for rec in work:
            if rec.sku not in pilot_set:
                outcomes.append(SkuOutcome(record=rec, generated=None,
                                           issues=issues_by_sku.get(rec.sku, []),
                                           status="skipped",
                                           skip_reason="not in pilot subset"))
        work = [r for r in work if r.sku in pilot_set]
    if limit is not None:
        for rec in work[limit:]:
            outcomes.append(SkuOutcome(record=rec, generated=None,
                                       issues=issues_by_sku.get(rec.sku, []),
                                       status="skipped",
                                       skip_reason=f"beyond --limit {limit}"))
        work = work[:limit]

    progress(f"  generating content for {len(work)} SKUs "
             f"({'pilot subset' if pilot else 'full in-scope set'}) "
             f"with {config.generation_model} ...")

    # ---- 3. GENERATE + 4. VERIFY ------------------------------------------
    # Workers only make API calls and run the (pure) validator; every SQLite
    # write happens on this thread as futures complete — the sqlite3
    # connection is not shared across threads.
    client = generator.make_client()
    t0 = time.time()
    workers = max(1, min(workers, 16))

    def process(rec: ProductRecord):
        gen = generator.generate_for_record(client, rec, config, run_id, store=None)
        return validator.verify_generated(rec, gen, config, run_id)

    def finish(rec: ProductRecord, i: int, result=None, exc=None) -> None:
        base_issues = issues_by_sku.get(rec.sku, [])
        if exc is not None:
            progress(f"    [{i}/{len(work)}] {rec.sku} ({rec.product_type}) FAILED: {exc}")
            outcomes.append(SkuOutcome(
                record=rec, generated=None, issues=base_issues + [Issue(
                    sku=rec.sku, field_name="generation", rule="generation_failed",
                    severity=Severity.ERROR, message=str(exc))],
                status="failed", skip_reason=str(exc)))
            return
        verified, ver_issues, log_entries = result
        store.record_usage(
            seller_id, run_id, rec.sku, config.generation_model,
            verified.input_tokens, verified.output_tokens)
        for entry in log_entries:
            store.log_compliance(entry)
        has_error = any(i.severity is Severity.ERROR for i in ver_issues)
        outcomes.append(SkuOutcome(
            record=rec, generated=verified,
            issues=base_issues + ver_issues, compliance_log=log_entries,
            status="needs_review" if has_error else "ok"))
        if not has_error:
            store.set_fingerprint(seller_id, rec.sku, fingerprint(rec))
        progress(f"    [{i}/{len(work)}] {rec.sku} ({rec.product_type}) "
                 f"{'needs_review' if has_error else 'ok'}")

    if workers == 1:
        for i, rec in enumerate(work, 1):
            try:
                finish(rec, i, result=process(rec))
            except GenerationError as exc:
                finish(rec, i, exc=exc)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, rec): rec for rec in work}
            for fut in as_completed(futures):
                rec = futures[fut]
                done += 1
                try:
                    finish(rec, done, result=fut.result())
                except GenerationError as exc:
                    finish(rec, done, exc=exc)

    elapsed = time.time() - t0
    n_ok = sum(1 for o in outcomes if o.status == "ok")
    n_review = sum(1 for o in outcomes if o.status == "needs_review")
    n_failed = sum(1 for o in outcomes if o.status == "failed")
    n_unchanged = sum(1 for o in outcomes if o.status == "skipped_unchanged")
    progress(f"  generation+verification done in {elapsed:.0f}s: "
             f"{n_ok} ok, {n_review} needs review, {n_failed} failed")

    # ---- 5. OUTPUT ----------------------------------------------------------
    progress("  writing upload file, diff report, and audit artifacts ...")
    artifacts = output.write_outputs(parse, outcomes, run_id, out_dir)

    usage = store.usage_totals(seller_id)
    summary = RunSummary(
        run_id=run_id, seller_id=seller_id, source_file=str(file_path),
        total_rows=len(parse.records), in_scope=len(work) + n_unchanged,
        generated=n_ok + n_review, skipped_unchanged=n_unchanged,
        needs_review=n_review, failed=n_failed,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        outputs=artifacts,
    )
    store.finish_run(run_id, {
        "total_rows": summary.total_rows, "generated": summary.generated,
        "needs_review": summary.needs_review, "failed": summary.failed,
        "skipped_unchanged": summary.skipped_unchanged, "outputs": artifacts,
        "parse_warnings": parse.warnings,
    })
    return summary
