"""Run artifacts: Amazon-uploadable flat file + JSON/HTML review outputs.

`upload.xlsx` is the file a seller uploads to Seller Central, so correctness
beats elegance here:
- header rows (1 .. dataRow-1) are copied VERBATIM from the original source
  workbook — the settings string in Template!A1 must survive unchanged;
- data cells are located through `parse.columns` (base / instance / leaf from
  the template's own attribute row), never by hardcoded index, because column
  positions shift between exports;
- only SKUs with status "ok" are written, as partial_update rows that touch
  ONLY the regenerated fields (blank cells are ignored by Amazon on partial
  update, so the seller's existing bullets 4-5 and every other attribute
  survive untouched).

Nothing in this module publishes anywhere; it only writes files to out_dir.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import openpyxl

from .exceptions import CatalogEngineError, FlatFileError
from .models import (
    BULLET_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    HIGHLIGHT_MAX_CHARS,
    SEARCH_TERMS_MAX_BYTES,
    TITLE_MAX_CHARS,
    ColumnInfo,
    ParseResult,
    Severity,
    SkuOutcome,
)

TEMPLATE_SHEET = "Template"
RECORD_ACTION_VALUE = "partial_update"

# Display order for per-SKU cards in report.html.
_STATUS_ORDER = {"needs_review": 0, "ok": 1}


# ---------------------------------------------------------------------------
# Column location (never by hardcoded index)
# ---------------------------------------------------------------------------
def _find_column(
    columns: list[ColumnInfo], base: str, instance: int = 1
) -> Optional[ColumnInfo]:
    """Find the data column for `base` #`instance` (the '.value' / bare leaf)."""
    for col in columns:
        if col.base == base and col.instance == instance and col.leaf in ("value", ""):
            return col
    return None


def _require_column(
    columns: list[ColumnInfo], base: str, source_file: str, instance: int = 1
) -> ColumnInfo:
    col = _find_column(columns, base, instance)
    if col is None:
        raise FlatFileError(
            f"cannot build upload.xlsx: column '{base}'"
            + (f" (instance #{instance})" if instance != 1 else "")
            + f" was not found on the attribute row of '{Path(source_file).name}'.",
            "The template is missing a column this tool needs to write. "
            "Re-download the Category Listings Report from Seller Central; if the "
            "column is genuinely absent, this template version is not supported.",
        )
    return col


# ---------------------------------------------------------------------------
# Counting helpers (display only — enforcement lives in validator.py)
# ---------------------------------------------------------------------------
def _chars(s: str) -> int:
    return len(s or "")


def _bytes_utf8(s: str) -> int:
    return len((s or "").encode("utf-8"))


# ---------------------------------------------------------------------------
# upload.xlsx
# ---------------------------------------------------------------------------
def _write_upload_xlsx(
    parse: ParseResult, ok_outcomes: list[SkuOutcome], path: Path
) -> None:
    settings = parse.settings
    cols = parse.columns
    src = parse.source_file

    sku_col = _require_column(cols, "contribution_sku", src)
    pt_col = _require_column(cols, "product_type", src)
    action_col = _require_column(cols, "::record_action", src)
    title_col = _require_column(cols, "item_name", src)
    bullet_cols = [
        _require_column(cols, "bullet_point", src, instance=i) for i in (1, 2, 3)
    ]
    desc_col = _require_column(cols, "product_description", src)
    keyword_col = _require_column(cols, "generic_keyword", src)

    src_path = Path(parse.source_file)
    if not src_path.exists():
        raise FlatFileError(
            f"cannot build upload.xlsx: the original source file '{src_path}' is no "
            "longer available, so the template header rows cannot be copied.",
            "Re-upload the Category Listings Report and re-run.",
        )
    src_wb = openpyxl.load_workbook(src_path, read_only=True, data_only=True)
    try:
        if TEMPLATE_SHEET not in src_wb.sheetnames:
            raise FlatFileError(
                f"cannot build upload.xlsx: '{src_path.name}' has no "
                f"'{TEMPLATE_SHEET}' sheet."
            )
        src_ws = src_wb[TEMPLATE_SHEET]
        header_rows = list(
            src_ws.iter_rows(
                min_row=1, max_row=settings.data_row - 1, values_only=True
            )
        )
    finally:
        src_wb.close()

    out_wb = openpyxl.Workbook()
    ws = out_wb.active
    ws.title = TEMPLATE_SHEET

    # Rows 1 .. data_row-1 verbatim (every column, plain values). This keeps
    # the A1 settings string byte-identical and preserves the blank row(s)
    # between the attribute row and the data row.
    for r, row in enumerate(header_rows, 1):
        for c, value in enumerate(row, 1):
            if value is not None:
                ws.cell(row=r, column=c, value=value)

    row_no = settings.data_row
    for outcome in ok_outcomes:
        gen = outcome.generated
        if gen is None:
            raise CatalogEngineError(
                f"SKU {outcome.record.sku}: status is 'ok' but no generated content "
                "was provided — refusing to write an empty upload row."
            )

        def put(col: ColumnInfo, value: str) -> None:
            if value:
                ws.cell(row=row_no, column=col.index + 1, value=value)

        put(sku_col, outcome.record.sku)
        put(pt_col, outcome.record.product_type)
        put(action_col, RECORD_ACTION_VALUE)
        put(title_col, gen.title)
        # Bullets 1-3 only. #4/#5 stay EMPTY on purpose: partial update ignores
        # blank cells, so the seller's existing bullets 4-5 are preserved.
        for col, bullet in zip(bullet_cols, gen.bullets[:3]):
            put(col, bullet)
        put(desc_col, gen.description)
        # Full space-separated search-terms string in generic_keyword #1 only.
        put(keyword_col, gen.search_terms)
        row_no += 1

    out_wb.save(path)


# ---------------------------------------------------------------------------
# results.json / input_issues.json
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _dump_json(data: Any, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _generated_dict(outcome: SkuOutcome) -> Optional[dict[str, Any]]:
    gen = outcome.generated
    if gen is None:
        return None
    return {
        "title": gen.title,
        "item_highlights": list(gen.item_highlights),
        "bullets": list(gen.bullets),
        "description": gen.description,
        "search_terms": gen.search_terms,
        "counts": {
            "title_chars": _chars(gen.title),
            "bullet_chars": [_chars(b) for b in gen.bullets],
            "description_chars": _chars(gen.description),
            "search_terms_bytes": _bytes_utf8(gen.search_terms),
        },
    }


def _write_results_json(
    parse: ParseResult, outcomes: list[SkuOutcome], run_id: str, path: Path
) -> None:
    data = {
        "run_id": run_id,
        "source_file": parse.source_file,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skus": [
            {
                "sku": o.record.sku,
                "status": o.status,
                "skip_reason": o.skip_reason,
                "product_type": o.record.product_type,
                "generated": _generated_dict(o),
                "issues": [asdict(i) for i in o.issues],
                "compliance_log": [asdict(e) for e in o.compliance_log],
            }
            for o in outcomes
        ],
    }
    _dump_json(data, path)


def _write_input_issues_json(outcomes: list[SkuOutcome], path: Path) -> None:
    issues = [
        asdict(issue)
        for outcome in outcomes
        for issue in outcome.issues
        if issue.field_name.startswith("input:")
    ]
    _dump_json(issues, path)


# ---------------------------------------------------------------------------
# report.html
# ---------------------------------------------------------------------------
_CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       margin: 24px; color: #1f2937; background: #f9fafb; }
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { color: #6b7280; font-size: 13px; margin-bottom: 8px; }
.counts span { display: inline-block; margin-right: 10px; padding: 2px 10px;
               border-radius: 10px; font-size: 12px; background: #e5e7eb; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 16px; margin: 16px 0; page-break-inside: avoid; }
.card h2 { font-size: 16px; margin: 0 0 2px; }
.pt { color: #6b7280; font-size: 12px; margin-bottom: 8px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 9px;
         font-size: 11px; font-weight: 600; color: #fff; vertical-align: middle; }
.badge.ok { background: #16a34a; }
.badge.needs_review { background: #d97706; }
.badge.other { background: #6b7280; }
.badge.error { background: #dc2626; }
.badge.warning { background: #d97706; }
.badge.info { background: #6b7280; }
table.ba { border-collapse: collapse; width: 100%; margin: 8px 0; }
table.ba th, table.ba td { border: 1px solid #e5e7eb; padding: 6px 8px;
                           font-size: 13px; text-align: left;
                           vertical-align: top; word-break: break-word; }
table.ba th { background: #f3f4f6; }
table.ba td.field { white-space: nowrap; font-weight: 600; width: 110px; }
.count { color: #6b7280; font-size: 11px; white-space: nowrap; }
.count.over { color: #dc2626; font-weight: 600; }
.hl-note { color: #b45309; font-size: 12px; margin: 2px 0 4px; }
ul.issues, ul.hl, ul.clog { margin: 4px 0 0; padding-left: 18px; }
ul.issues li, ul.clog li { font-size: 13px; margin: 3px 0; }
ul.hl li { font-size: 13px; margin: 2px 0; }
.rule { font-family: ui-monospace, Menlo, monospace; font-size: 12px;
        color: #374151; }
h3 { font-size: 13px; margin: 12px 0 2px; }
.empty { color: #9ca3af; font-style: italic; }
"""


def _esc(value: Optional[str]) -> str:
    if not value:
        return ""
    return html.escape(str(value))


def _cell(value: Optional[str]) -> str:
    return _esc(value) if value else '<span class="empty">—</span>'


def _count_span(n: int, limit: int, unit: str) -> str:
    cls = "count over" if n > limit else "count"
    return f'<span class="{cls}">{n}/{limit} {unit}</span>'


def _status_badge(status: str) -> str:
    cls = status if status in ("ok", "needs_review") else "other"
    return f'<span class="badge {cls}">{_esc(status)}</span>'


def _ba_row(field: str, before: Optional[str], after: Optional[str],
            count_html: str) -> str:
    return (
        f'<tr><td class="field">{_esc(field)}</td>'
        f"<td>{_cell(before)}</td>"
        f"<td>{_cell(after)} {count_html}</td></tr>"
    )


def _card_html(outcome: SkuOutcome) -> str:
    rec = outcome.record
    gen = outcome.generated
    parts: list[str] = ['<div class="card">']
    parts.append(
        f"<h2>{_esc(rec.sku)} {_status_badge(outcome.status)}</h2>"
        f'<div class="pt">{_esc(rec.product_type)}'
        + (f" — {_esc(outcome.skip_reason)}" if outcome.skip_reason else "")
        + "</div>"
    )

    # Before/After table
    parts.append(
        '<table class="ba"><tr><th>Field</th><th>Before (source)</th>'
        "<th>After (generated)</th></tr>"
    )
    title_after = gen.title if gen else ""
    parts.append(
        _ba_row("Title", rec.title, title_after,
                _count_span(_chars(title_after), TITLE_MAX_CHARS, "chars"))
    )
    before_bullets = rec.bullets
    after_bullets = gen.bullets if gen else []
    for i in range(3):
        before = before_bullets[i] if i < len(before_bullets) else ""
        after = after_bullets[i] if i < len(after_bullets) else ""
        parts.append(
            _ba_row(f"Bullet {i + 1}", before, after,
                    _count_span(_chars(after), BULLET_MAX_CHARS, "chars"))
        )
    desc_after = gen.description if gen else ""
    parts.append(
        _ba_row("Description", rec.description, desc_after,
                _count_span(_chars(desc_after), DESCRIPTION_MAX_CHARS, "chars"))
    )
    terms_after = gen.search_terms if gen else ""
    parts.append(
        _ba_row("Search terms", " ".join(rec.search_terms), terms_after,
                _count_span(_bytes_utf8(terms_after), SEARCH_TERMS_MAX_BYTES,
                            "bytes"))
    )
    parts.append("</table>")

    # Item highlights (after only — no template column exists for these)
    if gen and gen.item_highlights:
        parts.append("<h3>Item Highlights</h3>")
        parts.append(
            '<div class="hl-note">not uploaded — paste into Seller Central '
            "manually</div>"
        )
        parts.append('<ul class="hl">')
        for hl in gen.item_highlights:
            parts.append(
                f"<li>{_esc(hl)} "
                f"{_count_span(_chars(hl), HIGHLIGHT_MAX_CHARS, 'chars')}</li>"
            )
        parts.append("</ul>")

    # Issues
    if outcome.issues:
        parts.append("<h3>Issues</h3>")
        parts.append('<ul class="issues">')
        for issue in outcome.issues:
            sev = issue.severity.value if isinstance(issue.severity, Severity) \
                else str(issue.severity)
            parts.append(
                f'<li><span class="badge {_esc(sev)}">{_esc(sev)}</span> '
                f'<span class="rule">{_esc(issue.rule)}</span> '
                f"({_esc(issue.field_name)}): {_esc(issue.message)}</li>"
            )
        parts.append("</ul>")

    # Compliance log
    if outcome.compliance_log:
        parts.append("<h3>Compliance log</h3>")
        parts.append('<ul class="clog">')
        for entry in outcome.compliance_log:
            parts.append(
                f'<li><span class="rule">{_esc(entry.field_name)}</span> '
                f"[{_esc(entry.category)}]: {_esc(entry.reason)}</li>"
            )
        parts.append("</ul>")

    parts.append("</div>")
    return "".join(parts)


def _write_report_html(
    parse: ParseResult, outcomes: list[SkuOutcome], run_id: str, path: Path
) -> None:
    counts = Counter(o.status for o in outcomes)
    counts_html = "".join(
        f"<span>{_esc(status)}: {n}</span>" for status, n in sorted(counts.items())
    )
    ordered = sorted(
        outcomes, key=lambda o: (_STATUS_ORDER.get(o.status, 2), o.record.sku)
    )
    cards = "".join(_card_html(o) for o in ordered)
    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>Catalog run {_esc(run_id)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>Catalog optimization report</h1>"
        f'<div class="meta">Run {_esc(run_id)} — source '
        f"{_esc(Path(parse.source_file).name)} — generated "
        f"{_esc(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))}</div>"
        f'<div class="counts">{counts_html}</div>'
        f"{cards}"
        "</body></html>"
    )
    path.write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def write_outputs(
    parse: ParseResult,
    outcomes: list[SkuOutcome],
    run_id: str,
    out_dir: Path,
) -> dict[str, str]:
    """Write all run artifacts into out_dir; return artifact name -> path.

    Artifacts: upload.xlsx (Amazon partial-update flat file, ok SKUs only),
    results.json (full per-SKU results incl. item highlights and counts),
    report.html (self-contained human review report), input_issues.json
    (input-data issues, field_name starting 'input:').
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_outcomes = [o for o in outcomes if o.status == "ok"]

    upload_path = out_dir / "upload.xlsx"
    results_path = out_dir / "results.json"
    report_path = out_dir / "report.html"
    input_issues_path = out_dir / "input_issues.json"

    _write_upload_xlsx(parse, ok_outcomes, upload_path)
    _write_results_json(parse, outcomes, run_id, results_path)
    _write_report_html(parse, outcomes, run_id, report_path)
    _write_input_issues_json(outcomes, input_issues_path)

    return {
        "upload.xlsx": str(upload_path),
        "results.json": str(results_path),
        "report.html": str(report_path),
        "input_issues.json": str(input_issues_path),
    }
