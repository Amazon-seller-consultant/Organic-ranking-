"""Web UI router for the catalog engine.

Mounts under /catalog in the existing FastAPI app. Runs execute in a
background thread with progress streamed to an in-memory buffer the page
polls. Every endpoint is seller-scoped; artifact downloads are restricted to
files inside the seller's own output directory. Nothing here auto-publishes
to Amazon — downloads only.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .exceptions import CatalogEngineError, SellerNotFoundError
from .models import SellerConfig
from .store import Store

router = APIRouter(prefix="/catalog")

DATA_DIR = "data"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# run_id -> {"lines": [...], "done": bool, "error": str|None, "summary": dict|None}
_PROGRESS: dict[str, dict[str, Any]] = {}
_PROGRESS_LOCK = threading.Lock()


def _store() -> Store:
    return Store(DATA_DIR)


def _check_id(value: str, kind: str) -> str:
    if not _SAFE_ID.match(value or ""):
        raise HTTPException(400, f"invalid {kind}")
    return value


def _seller_or_404(store: Store, seller_id: str) -> SellerConfig:
    try:
        return store.get_seller(seller_id)
    except SellerNotFoundError:
        raise HTTPException(404, f"seller '{seller_id}' not found")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def page() -> HTMLResponse:
    html_path = Path("static/catalog.html")
    if not html_path.exists():
        raise HTTPException(500, "static/catalog.html is missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sellers
# ---------------------------------------------------------------------------
@router.get("/api/sellers")
def list_sellers() -> list[dict[str, Any]]:
    store = _store()
    try:
        rows = store._conn.execute(
            "SELECT config_json FROM sellers ORDER BY seller_id"
        ).fetchall()
        return [json.loads(r[0]) for r in rows]
    finally:
        store.close()


@router.post("/api/sellers")
def upsert_seller(body: dict) -> dict[str, Any]:
    seller_id = _check_id(body.get("seller_id", ""), "seller_id")
    store = _store()
    try:
        try:
            cfg = store.get_seller(seller_id)
        except SellerNotFoundError:
            cfg = SellerConfig(seller_id=seller_id)
        for field_name in ("display_name", "marketplace", "on_limit_violation",
                           "brand_voice", "generation_model"):
            if field_name in body and body[field_name] is not None:
                setattr(cfg, field_name, body[field_name])
        store.upsert_seller(cfg)
        return {"ok": True, "seller": json.loads(json.dumps(cfg.__dict__))}
    finally:
        store.close()


@router.post("/api/sellers/{seller_id}/attest")
def attest(seller_id: str, body: dict) -> dict[str, Any]:
    _check_id(seller_id, "seller_id")
    term = (body.get("term") or "").strip().lower()
    if not term:
        raise HTTPException(400, "term is required")
    store = _store()
    try:
        cfg = _seller_or_404(store, seller_id)
        if body.get("remove"):
            cfg.attested_terms.pop(term, None)
        else:
            note = (body.get("note") or "").strip()
            if not note:
                raise HTTPException(400, "note is required — the attestation "
                                          "text is the audit record")
            cfg.attested_terms[term] = note
        store.upsert_seller(cfg)
        return {"ok": True, "attested_terms": cfg.attested_terms}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Upload + runs
# ---------------------------------------------------------------------------
@router.post("/api/sellers/{seller_id}/upload")
async def upload_file(seller_id: str, file: UploadFile = File(...)) -> dict:
    _check_id(seller_id, "seller_id")
    name = Path(file.filename or "").name
    if not name.lower().endswith((".xlsm", ".xlsx")):
        raise HTTPException(400, f"'{name}' is not an .xlsm/.xlsx file — upload "
                                 "the Category Listing Report exactly as "
                                 "downloaded from Seller Central")
    store = _store()
    try:
        _seller_or_404(store, seller_id)
        dest = store.seller_dir(seller_id) / "uploads" / name
        content = await file.read()
        dest.write_bytes(content)
        return {"ok": True, "filename": name, "bytes": len(content)}
    finally:
        store.close()


@router.get("/api/sellers/{seller_id}/uploads")
def list_uploads(seller_id: str) -> list[dict[str, Any]]:
    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        _seller_or_404(store, seller_id)
        updir = store.seller_dir(seller_id) / "uploads"
        return sorted(
            ({"filename": p.name, "bytes": p.stat().st_size,
              "mtime": p.stat().st_mtime}
             for p in updir.glob("*.xls[mx]")),
            key=lambda d: -d["mtime"],
        )
    finally:
        store.close()


def _run_worker(seller_id: str, file_path: Path, opts: dict, run_box: dict) -> None:
    from .pipeline import run_pipeline

    store = Store(DATA_DIR)  # thread-local connection

    def progress(msg: str) -> None:
        with _PROGRESS_LOCK:
            run_box["lines"].append(msg)
            # first progress line carries the run id: "[<run_id>] parsing ..."
            if run_box.get("run_id") is None and msg.startswith("["):
                run_box["run_id"] = msg[1:].split("]", 1)[0]

    try:
        summary = run_pipeline(
            seller_id=seller_id, file_path=file_path, store=store,
            pilot=opts.get("pilot", False), limit=opts.get("limit"),
            force=opts.get("force", False), workers=opts.get("workers", 8),
            progress=progress,
        )
        with _PROGRESS_LOCK:
            run_box["summary"] = {
                "run_id": summary.run_id, "total_rows": summary.total_rows,
                "generated": summary.generated,
                "needs_review": summary.needs_review, "failed": summary.failed,
                "skipped_unchanged": summary.skipped_unchanged,
                "outputs": summary.outputs,
            }
            run_box["run_id"] = summary.run_id
    except Exception as exc:  # surfaced to the page, never swallowed
        with _PROGRESS_LOCK:
            run_box["error"] = str(exc)
    finally:
        with _PROGRESS_LOCK:
            run_box["done"] = True
        store.close()


@router.post("/api/sellers/{seller_id}/runs")
def start_run(seller_id: str, body: dict) -> dict[str, Any]:
    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        _seller_or_404(store, seller_id)
        filename = Path(body.get("filename", "")).name
        file_path = store.seller_dir(seller_id) / "uploads" / filename
        if not filename or not file_path.exists():
            raise HTTPException(400, f"uploaded file '{filename}' not found — "
                                     "upload the report first")
    finally:
        store.close()

    ticket = f"t{int(time.time()*1000)}"
    run_box: dict[str, Any] = {"lines": [], "done": False, "error": None,
                               "summary": None, "run_id": None,
                               "seller_id": seller_id}
    with _PROGRESS_LOCK:
        _PROGRESS[ticket] = run_box
    opts = {
        "pilot": bool(body.get("pilot")),
        "force": bool(body.get("force")),
        "workers": int(body.get("workers") or 8),
        "limit": int(body["limit"]) if body.get("limit") else None,
    }
    threading.Thread(
        target=_run_worker, args=(seller_id, file_path, opts, run_box),
        daemon=True,
    ).start()
    return {"ok": True, "ticket": ticket}


@router.get("/api/progress/{ticket}")
def progress(ticket: str) -> dict[str, Any]:
    with _PROGRESS_LOCK:
        box = _PROGRESS.get(ticket)
        if box is None:
            raise HTTPException(404, "unknown run ticket")
        return {k: box[k] for k in ("lines", "done", "error", "summary", "run_id")}


@router.get("/api/sellers/{seller_id}/runs")
def list_runs(seller_id: str) -> list[dict[str, Any]]:
    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        _seller_or_404(store, seller_id)
        return store.list_runs(seller_id)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
_ARTIFACT_NAMES = {"upload.xlsx", "results.json", "results.csv", "results.xlsx",
                   "report.html", "input_issues.json", "approved_upload.xlsx",
                   "merged_upload.xlsx"}


def _run_dir(store: Store, seller_id: str, run_id: str) -> Path:
    _check_id(run_id, "run_id")
    d = store.seller_dir(seller_id) / "outputs" / run_id
    if not d.is_dir():
        raise HTTPException(404, f"run '{run_id}' has no artifacts")
    return d


@router.get("/api/sellers/{seller_id}/runs/{run_id}/artifacts")
def list_artifacts(seller_id: str, run_id: str) -> list[str]:
    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        d = _run_dir(store, seller_id, run_id)
        return sorted(p.name for p in d.iterdir() if p.name in _ARTIFACT_NAMES)
    finally:
        store.close()


@router.get("/api/sellers/{seller_id}/runs/{run_id}/file/{name}")
def download(seller_id: str, run_id: str, name: str):
    _check_id(seller_id, "seller_id")
    if name not in _ARTIFACT_NAMES:
        raise HTTPException(404, "unknown artifact")
    store = _store()
    try:
        path = _run_dir(store, seller_id, run_id) / name
        if not path.exists():
            raise HTTPException(404, f"{name} not found for run {run_id}")
        media = "text/html" if name.endswith(".html") else None
        disposition = "inline" if name.endswith((".html", ".json")) else "attachment"
        return FileResponse(
            path, media_type=media, filename=None if disposition == "inline" else name,
            content_disposition_type=disposition,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Dashboard rows (before/after/corrections) + export rebuild
# ---------------------------------------------------------------------------
_PARSE_CACHE: dict[str, Any] = {}  # source_file -> ParseResult


def _cached_parse(source_file: str, seller_id: str):
    from .parser import parse_flat_file

    parse = _PARSE_CACHE.get(source_file)
    if parse is None:
        if not Path(source_file).exists():
            raise HTTPException(
                410, "the original report file for this run is no longer on "
                     "disk, so originals cannot be shown — re-upload it")
        parse = parse_flat_file(source_file, seller_id=seller_id)
        _PARSE_CACHE.clear()  # keep at most one big workbook in memory
        _PARSE_CACHE[source_file] = parse
    return parse


@router.get("/api/sellers/{seller_id}/runs/{run_id}/rows")
def dashboard_rows(seller_id: str, run_id: str) -> dict[str, Any]:
    """Before/after/corrections per SKU for the in-page dashboard."""
    from .output import _BULK_HEADER, build_bulk_rows_from_results

    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        results = _load_results(store, seller_id, run_id)
        parse = _cached_parse(results["source_file"], seller_id)
        return {"header": _BULK_HEADER,
                "rows": build_bulk_rows_from_results(parse, results)}
    finally:
        store.close()


@router.post("/api/sellers/{seller_id}/runs/{run_id}/rebuild-exports")
def rebuild_exports(seller_id: str, run_id: str) -> dict[str, Any]:
    """Regenerate results.csv/results.xlsx for a run with the current export
    format (before/after columns). Safe to call repeatedly."""
    from .output import (_write_results_csv_rows, _write_results_xlsx_rows,
                         build_bulk_rows_from_results)

    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        results = _load_results(store, seller_id, run_id)
        parse = _cached_parse(results["source_file"], seller_id)
        rows = build_bulk_rows_from_results(parse, results)
        d = _run_dir(store, seller_id, run_id)
        _write_results_csv_rows(rows, d / "results.csv")
        _write_results_xlsx_rows(rows, d / "results.xlsx")
        return {"ok": True, "rows": len(rows)}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Review: flagged SKUs, approve, merge
# ---------------------------------------------------------------------------
def _load_results(store: Store, seller_id: str, run_id: str) -> dict:
    path = _run_dir(store, seller_id, run_id) / "results.json"
    if not path.exists():
        raise HTTPException(404, "results.json missing for this run")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/api/sellers/{seller_id}/runs/{run_id}/flagged")
def flagged(seller_id: str, run_id: str) -> list[dict[str, Any]]:
    _check_id(seller_id, "seller_id")
    store = _store()
    try:
        results = _load_results(store, seller_id, run_id)
        out = []
        for s in results["skus"]:
            if s["status"] != "needs_review":
                continue
            out.append({
                "sku": s["sku"], "product_type": s["product_type"],
                "issues": [i["message"] for i in s["issues"]
                           if i["severity"] == "error"],
                "generated": s["generated"],
            })
        return out
    finally:
        store.close()


@router.post("/api/sellers/{seller_id}/runs/{run_id}/approve")
def approve(seller_id: str, run_id: str, body: dict) -> dict[str, Any]:
    """Human approves flagged SKUs after review: writes approved_upload.xlsx
    containing exactly those SKUs and records the approval in the audit log."""
    from .models import ComplianceLogEntry
    from .output import write_upload_subset
    from .parser import parse_flat_file
    from .pipeline import fingerprint

    _check_id(seller_id, "seller_id")
    skus = [s for s in (body.get("skus") or []) if isinstance(s, str)]
    if not skus:
        raise HTTPException(400, "no SKUs given")
    store = _store()
    try:
        results = _load_results(store, seller_id, run_id)
        by_sku = {s["sku"]: s for s in results["skus"]}
        chosen: list[tuple[str, dict]] = []
        for sku in skus:
            entry = by_sku.get(sku)
            if entry is None or entry["status"] != "needs_review":
                raise HTTPException(
                    400, f"SKU '{sku}' is not a flagged SKU of run {run_id}")
            chosen.append((sku, entry["generated"]))

        parse = parse_flat_file(results["source_file"], seller_id=seller_id)
        out_path = _run_dir(store, seller_id, run_id) / "approved_upload.xlsx"
        n = write_upload_subset(parse, chosen, out_path)
        recs = parse.by_sku()
        for sku, _ in chosen:
            store.log_compliance(ComplianceLogEntry(
                seller_id=seller_id, run_id=run_id, sku=sku,
                field_name="all", category="approved", removed_text="",
                reason="flagged content approved by seller after human review "
                       "(web UI)",
            ))
            if sku in recs:
                store.set_fingerprint(seller_id, sku, fingerprint(recs[sku]))
        return {"ok": True, "written": n, "artifact": "approved_upload.xlsx"}
    except CatalogEngineError as exc:
        raise HTTPException(400, str(exc))
    finally:
        store.close()


@router.post("/api/sellers/{seller_id}/merged-upload")
def merged_upload(seller_id: str, body: dict) -> dict[str, Any]:
    """Merge the upload-ready SKUs of several runs (same source catalog) into
    one flat file, written into the newest run's directory."""
    from .output import write_upload_subset
    from .parser import parse_flat_file

    _check_id(seller_id, "seller_id")
    run_ids = [r for r in (body.get("run_ids") or []) if isinstance(r, str)]
    if not run_ids:
        raise HTTPException(400, "no run_ids given")
    store = _store()
    try:
        merged: dict[str, dict] = {}  # sku -> generated (later runs win)
        source_file: Optional[str] = None
        for run_id in sorted(run_ids):
            results = _load_results(store, seller_id, run_id)
            source_file = results["source_file"]
            for s in results["skus"]:
                if s["status"] == "ok" and s.get("generated"):
                    merged[s["sku"]] = s["generated"]
        if not merged:
            raise HTTPException(400, "no upload-ready SKUs in the given runs")
        parse = parse_flat_file(source_file, seller_id=seller_id)
        target = _run_dir(store, seller_id, sorted(run_ids)[-1]) / "merged_upload.xlsx"
        n = write_upload_subset(parse, sorted(merged.items()), target)
        return {"ok": True, "written": n, "run_id": sorted(run_ids)[-1],
                "artifact": "merged_upload.xlsx"}
    except CatalogEngineError as exc:
        raise HTTPException(400, str(exc))
    finally:
        store.close()
