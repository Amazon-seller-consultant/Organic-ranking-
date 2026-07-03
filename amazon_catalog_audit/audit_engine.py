"""
Amazon Catalog Intelligence Platform — MVP Audit Engine
=========================================================

Processes three inputs:
  1. Category Listing Report (CLR) — .txt/.csv/.xlsx, supports both a clean
     canonical CSV (item_name, parent_sku, bullet_point1...) and Amazon's
     real multi-row flat-file export (hundreds of columns, machine-name
     header row + human-label row, e.g. `item_name[marketplace_id=...]#1.value`).
  2. Helium10 keyword export — .txt/.csv, supports a clean "Keyword" column
     OR Helium10's raw Cerebro/reverse-ASIN dump (match-type labels like
     "close-match" interleaved with `asin="B0..."` markers and keyword rows).
  3. Business Report — .csv, optional. Used to compute revenue-at-risk.

Runs 7 weighted audit engines and emits a single JSON payload shaped for
direct consumption by frontend chart/table components (Tremor/Shadcn).

No external LLM calls are made anywhere in this module (Zero LLM Call rule).
Advice text is produced by `build_consultant_advice()` from static templates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENGINE_WEIGHTS = {
    "variation": 0.20,
    "duplicate": 0.15,
    "browse_node": 0.10,
    "indexing": 0.15,
    "content": 0.15,
    "aplus": 0.10,
    "suppression": 0.15,
}

TITLE_MAX_LEN = 200
DUPLICATE_TITLE_THRESHOLD = 0.90

BANNED_KEYWORDS_REGEX = re.compile(
    r"\b(best|cheap(?:est)?|free\s*shipping|100%\s*guarantee(?:d)?|guarantee(?:d)?|"
    r"warrant(?:y|ies)|#1|number\s*one|sale|discount)\b",
    re.IGNORECASE,
)

CRITICAL_ISSUE_TYPES = {
    "Suppression Risk",
    "Orphan Child",
    "Incorrect Browse Node",
    "Invalid Variation Theme Structure",
    "Virtual Bundle Duplicate",
}

# Canonical field -> list of regex fragments matched against the lowercased
# machine attribute name *and* the lowercased human label of each column.
# First match wins. Multi-valued fields (bullet points, generic keywords,
# materials) collect every matching column into a list, in column order.
CANONICAL_FIELD_PATTERNS: dict[str, list[str]] = {
    "sku": [r"^sku$", r"contribution_sku", r"seller.?sku", r"^merchant.?sku$"],
    "asin": [r"^asin$", r"^product_id$"],
    "product_id_type": [r"product_id_type"],
    "product_id_value": [r"product_id_value", r"product_id$"],
    "item_name": [r"^item_name", r"^title$"],
    "parent_child": [r"parentage_level", r"^parent.?child$", r"relationship.?type"],
    "parent_sku": [r"parent_sku", r"^parent.?sku$"],
    "variation_theme": [r"variation_theme", r"^variation.?theme$"],
    "color_name": [r"^color(?!_(?:rendering|temperature|map))", r"^colour", r"color_name"],
    "size_name": [r"^size(?!_unit)", r"size_name"],
    "item_type_keyword": [r"item_type_keyword", r"^item.?type.?keyword$"],
    "browse_node": [r"recommended_browse_node", r"browse_node", r"browse_classification"],
    "bullet_point": [r"^bullet_point", r"^bullet.?point\d?$"],
    "generic_keyword": [r"^generic_keyword", r"^backend.?(search.?term|keyword)s?$", r"^search_terms?$"],
    "material_type": [r"^material(?!.*\.material)", r"material_type"],
    "flavor": [r"^flavor", r"^flavour"],
    "ingredients": [r"^ingredients?$"],
    "product_description": [r"product_description"],
    "aplus_page_status": [r"aplus.?page.?status", r"a\+.?content"],
    "brand_story_status": [r"brand.?story"],
    "main_image_url": [r"main_image", r"^main.?image.?url$", r"main_offer_image"],
    "feed_product_type": [r"^product_type", r"feed_product_type"],
}


def _match_canonical(name: str, patterns: list[str]) -> bool:
    name = name.lower()
    return any(re.search(p, name) for p in patterns)


# ---------------------------------------------------------------------------
# CLR Loading — handles both clean CSV and real Amazon flat-file exports
# ---------------------------------------------------------------------------

def _read_raw_table(path: str) -> pd.DataFrame:
    """Read the file as a fully untyped, headerless grid of strings."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, header=None, dtype=str)
    # .txt / .csv — Amazon CLR exports are tab-delimited, plain CSVs use comma.
    # An explicit separator is required: pandas' delimiter sniffing (sep=None)
    # misfires on single-column or sparse-delimiter files, silently truncating
    # values at whatever character it mistakes for a separator.
    sep = "\t" if suffix == ".txt" else ","
    return pd.read_csv(path, header=None, dtype=str, sep=sep, keep_default_na=False)


def _looks_like_header_row(row: pd.Series) -> bool:
    """
    A machine attribute-name row (Amazon flat-file's 2nd header row) has a
    distinctive signature: cells like 'item_name[marketplace_id=...]#1.value'
    or 'contribution_sku#1.value' — underscore plus '[' or '#' or '.'. A real
    data row essentially never matches this combination, which keeps clean
    canonical CSVs (whose first data row is just plain values) from being
    mistaken for a second header row.
    """
    values = [str(v) for v in row if v not in (None, "", "nan")]
    if not values:
        return False
    signature_hits = sum(
        1 for v in values if "_" in v and any(ch in v for ch in ("[", "#", "."))
    )
    return signature_hits / max(len(values), 1) > 0.3


def _detect_header_block(raw: pd.DataFrame) -> tuple[int, int]:
    """
    Locate the (label_row, data_start_row) for an Amazon flat-file export.
    Amazon CLR/Flat-File layout: settings row(s) -> instructions -> group row
    -> human label row -> machine attribute-name row -> data rows.
    We scan the first 10 rows for the row containing a cell matching 'sku'
    (case-insensitive, exact) — that is the human-label row. The row directly
    below it is the machine attribute-name row, and data starts after that.
    """
    scan_limit = min(10, len(raw))
    for i in range(scan_limit):
        row = raw.iloc[i]
        cells = [str(c).strip().lower() for c in row if str(c).strip()]
        if any(c in ("sku", "seller-sku", "item_sku") for c in cells):
            label_row = i
            attr_row = i + 1 if i + 1 < len(raw) and _looks_like_header_row(raw.iloc[i + 1]) else i
            data_start = attr_row + 1
            return label_row, data_start
    # Fallback: assume row 0 is already a clean canonical header.
    return 0, 1


def _build_canonical_columns(raw: pd.DataFrame, label_row: int, attr_row: int) -> dict[str, list[int]]:
    """Map each canonical field name to the list of raw column indices that feed it."""
    labels = raw.iloc[label_row].astype(str).tolist()
    attrs = raw.iloc[attr_row].astype(str).tolist() if attr_row != label_row else labels

    mapping: dict[str, list[int]] = defaultdict(list)
    claimed: set[int] = set()
    for canon, patterns in CANONICAL_FIELD_PATTERNS.items():
        for idx, (lbl, attr) in enumerate(zip(labels, attrs)):
            if idx in claimed and canon not in ("bullet_point", "generic_keyword", "material_type"):
                continue
            if _match_canonical(attr, patterns) or _match_canonical(lbl, patterns):
                mapping[canon].append(idx)
                claimed.add(idx)
    return mapping


def load_clr(path: str) -> pd.DataFrame:
    """
    Load a Category Listing Report into a clean, memory-light canonical
    DataFrame. Multi-valued source columns (bullet points, backend keywords,
    materials) are collapsed into list columns. Fields absent from the
    source file are left as a column of None — downstream engines treat an
    entirely-missing canonical field as "not evaluable" rather than fabricating
    errors for every row.
    """
    raw = _read_raw_table(path)
    label_row, data_start = _detect_header_block(raw)
    attr_row = data_start - 1
    col_map = _build_canonical_columns(raw, label_row, attr_row)

    data = raw.iloc[data_start:].reset_index(drop=True)
    # Drop fully-blank trailing rows some exporters leave at EOF.
    data = data[data.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)

    out = pd.DataFrame(index=data.index)
    multi_valued = {"bullet_point", "generic_keyword", "material_type"}

    for canon in CANONICAL_FIELD_PATTERNS:
        idxs = col_map.get(canon, [])
        if not idxs:
            out[canon] = None
            continue
        if canon in multi_valued:
            sub = data.iloc[:, idxs].apply(
                lambda r: [str(v).strip() for v in r if str(v).strip() and str(v).strip().lower() != "nan"],
                axis=1,
            )
            out[canon] = sub
        else:
            out[canon] = data.iloc[:, idxs[0]].apply(
                lambda v: None if str(v).strip() in ("", "nan") else str(v).strip()
            )

    # Expand bullet_point list into bullet_1..bullet_5 for index-friendly access.
    for i in range(1, 6):
        out[f"bullet_{i}"] = out["bullet_point"].apply(
            lambda lst, i=i: lst[i - 1] if isinstance(lst, list) and len(lst) >= i else None
        )

    out["generic_keywords_text"] = out["generic_keyword"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else ""
    )

    # Amazon's flat-file splits the product identifier into a type column
    # (ASIN/UPC/EAN/...) and a value column. Only promote the value to the
    # canonical `asin` field when its type is actually ASIN — otherwise a
    # UPC/EAN value would be mislabeled as an ASIN.
    if out["asin"].isna().all() and out["product_id_value"].notna().any():
        out["asin"] = [
            val if (str(typ).strip().upper() == "ASIN") else None
            for typ, val in zip(out["product_id_type"], out["product_id_value"])
        ]

    # Stable row id used for cross-engine joins even if SKU/ASIN are blank.
    out["row_id"] = out.index
    if out["sku"].isna().all() and out["asin"].notna().any():
        out["sku"] = out["asin"]
    return out


# ---------------------------------------------------------------------------
# Helium10 keyword list loading
# ---------------------------------------------------------------------------

_H10_MATCH_TYPE_LABELS = {"close-match", "loose-match", "substitutes", "complements", "irrelevant"}
_H10_ASIN_MARKER = re.compile(r'^asin(?:-expanded)?\s*=\s*"?([A-Z0-9]{10})"?$', re.IGNORECASE)


def load_helium10(path: str) -> dict[str, Any]:
    """
    Returns:
      {
        "global": set of all keywords seen (lowercase),
        "by_asin": {asin: set(keywords)}  — only populated if the export
                   carries asin="..." markers; empty dict otherwise.
      }
    Supports a clean single-column 'Keyword' CSV as well as Helium10's raw
    reverse-ASIN dump where keyword rows are interleaved with match-type
    labels (close-match/loose-match/...) and asin="B0..." section markers.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        sep = "\t" if suffix == ".txt" else ","
        df = pd.read_csv(path, dtype=str, sep=sep, keep_default_na=False)

    # Clean format: a column literally named "keyword".
    kw_col = next((c for c in df.columns if str(c).strip().lower() == "keyword"), None)
    if kw_col is not None:
        kws = {str(v).strip().lower() for v in df[kw_col] if str(v).strip()}
        return {"global": kws, "by_asin": {}}

    col = df.columns[0]
    global_kws: set[str] = set()
    by_asin: dict[str, set[str]] = defaultdict(set)
    current_asin: str | None = None

    for raw_val in df[col]:
        val = str(raw_val).strip()
        if not val or val.lower() == "nan":
            continue
        low = val.lower()
        if low in _H10_MATCH_TYPE_LABELS:
            continue
        m = _H10_ASIN_MARKER.match(val)
        if m:
            current_asin = m.group(1).upper()
            continue
        global_kws.add(low)
        if current_asin:
            by_asin[current_asin].add(low)

    return {"global": global_kws, "by_asin": dict(by_asin)}


# ---------------------------------------------------------------------------
# Business Report loading (optional — drives revenue-at-risk)
# ---------------------------------------------------------------------------

_BR_ASIN_PATTERNS = [r"\(child\)\s*asin", r"\(parent\)\s*asin", r"^asin$"]
_BR_SKU_PATTERNS = [r"^sku$", r"merchant.?sku"]
_BR_SESSIONS_PATTERNS = [r"sessions.?-?\s*total", r"^sessions$"]
_BR_REVENUE_PATTERNS = [r"ordered product sales", r"^sales$", r"revenue"]
_BR_UNITS_PATTERNS = [r"units\s*ordered(?!.*b2b)", r"^units\s*ordered$"]


def _find_column(columns: list[str], patterns: list[str]) -> str | None:
    """Patterns are tried in priority order; the first pattern with any match wins.

    Iterating patterns-before-columns (rather than columns-before-patterns) matters
    for real Business Report exports, which list "(Parent) ASIN" before "(Child) ASIN" —
    the catalog audit rows are keyed by the sellable child ASIN, so child must win.
    """
    lowered = [str(col).strip().lower() for col in columns]
    for pattern in patterns:
        for col, low in zip(columns, lowered):
            if re.search(pattern, low):
                return col
    return None


def load_business_report(path: str) -> dict[str, dict[str, float]]:
    """Returns {sku_or_asin: {"sessions": float, "revenue": float, "units": float}}."""
    suffix = Path(path).suffix.lower()
    sep = "\t" if suffix == ".txt" else ","
    chunks = pd.read_csv(path, dtype=str, sep=sep, keep_default_na=False, chunksize=2000)

    result: dict[str, dict[str, float]] = {}
    for chunk in chunks:
        cols = list(chunk.columns)
        key_col = _find_column(cols, _BR_SKU_PATTERNS) or _find_column(cols, _BR_ASIN_PATTERNS)
        sessions_col = _find_column(cols, _BR_SESSIONS_PATTERNS)
        revenue_col = _find_column(cols, _BR_REVENUE_PATTERNS)
        units_col = _find_column(cols, _BR_UNITS_PATTERNS)
        if key_col is None:
            continue
        for _, row in chunk.iterrows():
            key = str(row[key_col]).strip()
            if not key:
                continue
            sessions = _to_float(row.get(sessions_col)) if sessions_col else 0.0
            revenue = _to_float(row.get(revenue_col)) if revenue_col else 0.0
            units = _to_float(row.get(units_col)) if units_col else 0.0
            result[key] = {"sessions": sessions, "revenue": revenue, "units": units}
    return result


def _to_float(val: Any) -> float:
    if val is None:
        return 0.0
    s = re.sub(r"[^\d.\-]", "", str(val))
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Per-row issue accumulator
# ---------------------------------------------------------------------------

@dataclass
class RowIssues:
    issues: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # chart bucket per issue
    severity: str = "Low"

    def add(self, message: str, category: str, severity: str = "Medium") -> None:
        self.issues.append(message)
        self.categories.append(category)
        if _severity_rank(severity) > _severity_rank(self.severity):
            self.severity = severity


def _severity_rank(sev: str) -> int:
    return {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}.get(sev, 0)


# ---------------------------------------------------------------------------
# Engine 1 — Variation Audit
# ---------------------------------------------------------------------------

def run_variation_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int]:
    errors = 0
    n = len(df)
    has_parent_sku_col = df["parent_sku"].notna().any()
    has_parentage_col = df["parent_child"].notna().any()

    child_counts: dict[str, int] = defaultdict(int)
    if has_parent_sku_col:
        for v in df["parent_sku"]:
            if v:
                child_counts[v] += 1

    for i, row in df.iterrows():
        parentage = (row["parent_child"] or "").strip().lower() if has_parentage_col else None
        parent_sku = row["parent_sku"]

        if has_parentage_col and parentage == "child" and not parent_sku:
            row_issues[i].add("Orphan Child (missing parent_sku)", "Orphan Child", "Critical")
            errors += 1

        if has_parent_sku_col and parent_sku and child_counts.get(parent_sku, 0) == 1 and parentage == "child":
            row_issues[i].add(
                f"Single Child Parent: only one child under {parent_sku}", "Single Child Parent", "Medium"
            )
            errors += 1

        theme = (row["variation_theme"] or "").lower() if row["variation_theme"] else ""
        if theme:
            if "color" in theme and not row["color_name"]:
                row_issues[i].add(
                    "Invalid Variation Theme Structure: theme is Color but color_name is empty",
                    "Invalid Variation Theme Structure",
                    "High",
                )
                errors += 1
            if "size" in theme and not row["size_name"]:
                row_issues[i].add(
                    "Invalid Variation Theme Structure: theme is Size but size_name is empty",
                    "Invalid Variation Theme Structure",
                    "High",
                )
                errors += 1

    score = 100 if n == 0 else max(0, round(100 - (errors / n) * 100))
    return score, errors


# ---------------------------------------------------------------------------
# Engine 2 — Duplicate Detection
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(title: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(title.lower()) if len(t) >= 3)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def run_duplicate_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int]:
    """
    Full O(n^2) SequenceMatcher comparison does not scale to thousands of
    ASINs. We bucket by first significant word + length, then apply a cheap
    token-Jaccard pre-filter (set intersection, O(tokens) per pair) and only
    run the expensive SequenceMatcher ratio on pairs that clear a generous
    Jaccard floor — any pair with >=90% character similarity necessarily has
    high token overlap, so this filter cannot drop genuine duplicates.
    """
    titles = [(i, (row["item_name"] or "")) for i, row in df.iterrows() if row["item_name"]]
    buckets: dict[tuple[str, int], list[tuple[int, str, frozenset[str]]]] = defaultdict(list)
    for i, title in titles:
        first_word = title.strip().split()[0].lower() if title.strip() else ""
        buckets[(first_word, len(title) // 20)].append((i, title, _tokenize(title)))

    seen_pairs: set[tuple[int, int]] = set()
    # Collect every match per row first, then emit one consolidated issue per row —
    # a row that resembles 7 other listings should produce one summary line, not 7.
    title_matches: dict[int, list[tuple[str, int]]] = defaultdict(list)
    JACCARD_PREFILTER = 0.5
    for bucket in buckets.values():
        for a_idx in range(len(bucket)):
            for b_idx in range(a_idx + 1, len(bucket)):
                i, title_a, tokens_a = bucket[a_idx]
                j, title_b, tokens_b = bucket[b_idx]
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                if _jaccard(tokens_a, tokens_b) < JACCARD_PREFILTER:
                    continue
                ratio = _title_similarity(title_a.lower(), title_b.lower())
                if ratio >= DUPLICATE_TITLE_THRESHOLD:
                    label_a = df.at[i, "asin"] or df.at[i, "sku"] or f"row{i}"
                    label_b = df.at[j, "asin"] or df.at[j, "sku"] or f"row{j}"
                    pct = round(ratio * 100)
                    title_matches[i].append((label_b, pct))
                    title_matches[j].append((label_a, pct))

    affected_rows: set[int] = set()
    for i, matches in title_matches.items():
        matches.sort(key=lambda m: -m[1])
        preview = ", ".join(f"{label} ({pct}%)" for label, pct in matches[:5])
        more = f", +{len(matches) - 5} more" if len(matches) > 5 else ""
        row_issues[i].add(
            f"Possible Duplicate Listing: matches {len(matches)} other listing(s) — {preview}{more}",
            "Possible Duplicate Listing",
            "High",
        )
        affected_rows.add(i)

    # Virtual bundle detection: SKUs like "L2168-N_L2392-2" and "L2168-N_L2408-2"
    # share a base/core SKU before the underscore — they're the same core product
    # bundled with different add-ons, which is a real cannibalizing-duplicate risk
    # even when title-similarity matching alone wouldn't (or shouldn't) be relied on.
    bundle_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in df.iterrows():
        sku = (row["sku"] or "").strip()
        if sku and "_" in sku:
            base = sku.split("_", 1)[0].strip()
            if base:
                bundle_groups[base].append(i)

    for base, idxs in bundle_groups.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            siblings = [df.at[j, "sku"] or df.at[j, "asin"] or f"row{j}" for j in idxs if j != i]
            preview = ", ".join(siblings[:5])
            more = f" (+{len(siblings) - 5} more)" if len(siblings) > 5 else ""
            row_issues[i].add(
                f"Virtual Bundle Duplicate: shares core product '{base}' with {len(siblings)} other bundle listing(s) — {preview}{more}",
                "Virtual Bundle Duplicate",
                "High",
            )
            affected_rows.add(i)

    # Report and score off the number of affected listings, not raw pair count —
    # a tight cluster of k duplicate titles produces k*(k-1)/2 pairs, which would
    # otherwise blow past the catalog size and make the score/error count unreadable.
    n = len(df)
    duplicate_count = len(affected_rows)
    score = 100 if n == 0 else max(0, round(100 - (duplicate_count / n) * 100))
    return score, duplicate_count


# ---------------------------------------------------------------------------
# Engine 3 — Browse Node Audit
# ---------------------------------------------------------------------------

def run_browse_node_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int]:
    source_col = "browse_node" if df["browse_node"].notna().any() else "item_type_keyword"
    values = df[source_col]
    if values.isna().all():
        return 100, 0  # not evaluable — source data absent

    counts = values.value_counts()
    if counts.empty:
        return 100, 0
    dominant = counts.idxmax()
    dominant_share = counts.max() / counts.sum()

    errors = 0
    # Only flag outliers when there is a clear dominant category (>=60% share);
    # otherwise the catalog is genuinely multi-category and outlier-flagging
    # would just be noise.
    if dominant_share >= 0.6:
        for i, val in values.items():
            if val and val != dominant:
                row_issues[i].add(
                    f"Incorrect Browse Node: '{val}' (store's dominant category is '{dominant}')",
                    "Incorrect Browse Node",
                    "High",
                )
                errors += 1

    n = len(df)
    score = 100 if n == 0 else max(0, round(100 - (errors / n) * 100))
    return score, errors


# ---------------------------------------------------------------------------
# Engine 4 — Indexing Audit
# ---------------------------------------------------------------------------

def run_indexing_engine(
    df: pd.DataFrame, row_issues: dict[int, RowIssues], helium10: dict[str, Any]
) -> tuple[int, int]:
    global_kws: set[str] = helium10.get("global", set())
    by_asin: dict[str, set[str]] = helium10.get("by_asin", {})
    if not global_kws:
        return 100, 0  # no keyword list supplied — not evaluable

    errors = 0
    total_checks = 0
    per_asin_mode = bool(by_asin)

    for i, row in df.iterrows():
        asin = row["asin"]
        if per_asin_mode:
            # Helium10 export is grouped per ASIN. If this product's ASIN has
            # no reverse-ASIN keyword section, there is nothing to check it
            # against — falling back to another product's keyword pool would
            # only produce noise, so we skip rather than fabricate errors.
            keywords = by_asin.get(asin)
        else:
            keywords = global_kws
        if not keywords:
            continue

        haystack_parts = [row["item_name"] or ""]
        for b in range(1, 6):
            haystack_parts.append(row.get(f"bullet_{b}") or "")
        haystack_parts.append(row["generic_keywords_text"] or "")
        haystack = " ".join(haystack_parts).lower()

        missing = [kw for kw in keywords if kw and kw not in haystack]
        total_checks += len(keywords)
        if missing:
            errors += len(missing)
            preview = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            row_issues[i].add(
                f"Unindexed / Missing Keyword: {preview}{more}", "Unindexed Keyword", "Medium"
            )

    score = 100 if total_checks == 0 else max(0, round(100 - (errors / total_checks) * 100))
    return score, errors


# ---------------------------------------------------------------------------
# Engine 5 — Content & Attribute Completeness
# ---------------------------------------------------------------------------

CRITICAL_ATTRIBUTES = ["material_type", "flavor", "size_name", "ingredients", "product_description"]


def run_content_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int, float]:
    errors = 0
    checks = 0
    n = len(df)

    evaluable_attrs = [a for a in CRITICAL_ATTRIBUTES if df[a].notna().any()]

    for i, row in df.iterrows():
        missing_bullets = [b for b in range(1, 6) if not row.get(f"bullet_{b}")]
        if missing_bullets:
            label = ", ".join(f"#{b}" for b in missing_bullets)
            row_issues[i].add(f"Missing Bullet Point(s): {label}", "Missing Bullet Point", "Medium")
            errors += 1
        checks += 1

        for attr in evaluable_attrs:
            checks += 1
            if not row[attr]:
                row_issues[i].add(f"Missing Attribute: {attr}", "Missing Attribute", "Medium")
                errors += 1

    completeness_pct = round(100 - (errors / checks) * 100, 1) if checks else 100.0
    score = 100 if checks == 0 else max(0, round(100 - (errors / checks) * 100))
    return score, errors, completeness_pct


# ---------------------------------------------------------------------------
# Engine 6 — A+ Content & Brand Story Audit
# ---------------------------------------------------------------------------

def run_aplus_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int]:
    has_aplus_col = df["aplus_page_status"].notna().any()
    has_brand_story_col = df["brand_story_status"].notna().any()
    if not has_aplus_col and not has_brand_story_col:
        return 100, 0  # not evaluable — source report carries no A+/brand-story field

    errors = 0
    n = len(df)
    for i, row in df.iterrows():
        if has_aplus_col and not row["aplus_page_status"]:
            row_issues[i].add("A+ Content Missing", "A+ Content Missing", "Medium")
            errors += 1
        if has_brand_story_col and not row["brand_story_status"]:
            row_issues[i].add("Brand Story Missing", "Brand Story Missing", "Low")
            errors += 1

    score = 100 if n == 0 else max(0, round(100 - (errors / n) * 100))
    return score, errors


# ---------------------------------------------------------------------------
# Engine 7 — Suppression Risk
# ---------------------------------------------------------------------------

def run_suppression_engine(df: pd.DataFrame, row_issues: dict[int, RowIssues]) -> tuple[int, int]:
    errors = 0
    n = len(df)
    has_image_col = df["main_image_url"].notna().any()

    for i, row in df.iterrows():
        title = row["item_name"] or ""
        if len(title) > TITLE_MAX_LEN:
            row_issues[i].add(
                f"Suppression Risk: Title is {len(title)} characters (limit {TITLE_MAX_LEN})",
                "Suppression Risk",
                "Critical",
            )
            errors += 1

        text_blob = " ".join(
            [title] + [row.get(f"bullet_{b}") or "" for b in range(1, 6)]
        )
        banned_hits = sorted(set(m.group(0).lower() for m in BANNED_KEYWORDS_REGEX.finditer(text_blob)))
        if banned_hits:
            row_issues[i].add(
                f"Banned Keyword Usage: {', '.join(banned_hits)}", "Banned Keyword Usage", "High"
            )
            errors += 1

        if has_image_col and not row["main_image_url"]:
            row_issues[i].add("Suppression Risk: Main image missing", "Suppression Risk", "Critical")
            errors += 1

    score = 100 if n == 0 else max(0, round(100 - (errors / n) * 100))
    return score, errors


# ---------------------------------------------------------------------------
# Revenue at Risk
# ---------------------------------------------------------------------------

def compute_revenue_at_risk(
    df: pd.DataFrame, row_issues: dict[int, RowIssues], business_report: dict[str, dict[str, float]]
) -> tuple[float, dict[int, float]]:
    revenue_at_risk = 0.0
    revenue_by_row: dict[int, float] = {}
    if not business_report:
        return 0.0, revenue_by_row

    for i, row in df.iterrows():
        # The Business Report may be keyed by either ASIN or SKU depending on
        # which column it exposes — try both rather than picking one based on
        # which happens to be non-empty on this row.
        metrics = business_report.get(row["asin"]) or business_report.get(row["sku"])
        revenue = metrics["revenue"] if metrics else 0.0
        revenue_by_row[i] = revenue
        categories = set(row_issues[i].categories) if i in row_issues else set()
        if categories & CRITICAL_ISSUE_TYPES:
            revenue_at_risk += revenue

    return round(revenue_at_risk, 2), revenue_by_row


# ---------------------------------------------------------------------------
# Zero-Sales Cleanup — dead-inventory candidates from the Business Report
# ---------------------------------------------------------------------------

def find_zero_sales_candidates(
    df: pd.DataFrame, business_report: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Listings that the Business Report confirms had zero units and zero
    revenue over the report's period. These aren't a content/compliance
    problem to fix — they're dead inventory worth deleting outright.

    Only rows that actually appear in the Business Report are considered;
    a row absent from the report has no sales data to judge it by, so it
    is left alone rather than assumed dead.
    """
    if not business_report:
        return []

    candidates: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        key = row["asin"] or row["sku"]
        metrics = business_report.get(row["asin"]) or business_report.get(row["sku"])
        if metrics is None:
            continue
        if metrics["units"] == 0 and metrics["revenue"] == 0:
            candidates.append(
                {
                    "asin": row["asin"] or "",
                    "sku": row["sku"] or "",
                    "title": row["item_name"] or "",
                    "sessions": metrics["sessions"],
                    "units_ordered": metrics["units"],
                    "revenue": metrics["revenue"],
                    "recommendation": (
                        "Zero sales and zero units ordered over the reporting period — recommend deleting "
                        "this listing. No content fixes are worth making here."
                        if metrics["sessions"] == 0
                        else "Gets traffic but has never converted — recommend deleting unless you plan to "
                        "investigate pricing/positioning. No content fixes are worth making here."
                    ),
                }
            )
    return candidates


# ---------------------------------------------------------------------------
# Zero-LLM consultant advice templates
# ---------------------------------------------------------------------------

def build_consultant_advice(
    issues: list[str], categories: list[str], revenue: float, total_revenue: float, title_len: int
) -> str:
    if not issues:
        return ""

    cat_set = set(categories)
    revenue_share = round((revenue / total_revenue) * 100, 1) if total_revenue > 0 else 0.0
    parts: list[str] = []

    if revenue_share > 0:
        parts.append(f"This ASIN drives {revenue_share}% of store revenue.")

    if "Suppression Risk" in cat_set and title_len > TITLE_MAX_LEN:
        parts.append(
            f"Title is {title_len} characters, over the {TITLE_MAX_LEN}-character limit. "
            f"Shorten it to {TITLE_MAX_LEN} characters before Amazon suppresses this listing."
        )
    if "Banned Keyword Usage" in cat_set:
        parts.append("Banned marketing keywords were found in this listing; remove them to reduce suppression risk.")
    if "Possible Duplicate Listing" in cat_set:
        dup = next((s for s in issues if s.startswith("Possible Duplicate")), "")
        parts.append(f"{dup}. Differentiate the title to eliminate duplicate-listing risk.")
    if "Virtual Bundle Duplicate" in cat_set:
        bundle = next((s for s in issues if s.startswith("Virtual Bundle Duplicate")), "")
        parts.append(f"{bundle}. These bundle variants are cannibalizing each other's traffic and reviews — consolidate into fewer, more distinct bundle offers.")
    if "Orphan Child" in cat_set:
        parts.append("This variation isn't linked to a parent SKU; it appears disconnected from its variation family in search results.")
    if "Incorrect Browse Node" in cat_set:
        parts.append("This product is listed outside the store's dominant category; move it to the correct browse node to improve discoverability.")
    if "Unindexed Keyword" in cat_set:
        parts.append("High-volume keywords are missing from the title, bullets, or backend search terms; add them to improve search indexing.")
    if "Missing Attribute" in cat_set or "Missing Bullet Point" in cat_set:
        parts.append("Missing content fields hurt conversion rate and search ranking; fill these in.")
    if "A+ Content Missing" in cat_set or "Brand Story Missing" in cat_set:
        parts.append("Missing A+ Content / Brand Story lowers brand perception and conversion rate.")

    if not parts:
        parts.append("Resolve the issues found above to improve catalog health.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_audit(
    clr_path: str, helium10_path: str | None = None, business_report_path: str | None = None
) -> dict[str, Any]:
    df = load_clr(clr_path)
    helium10 = load_helium10(helium10_path) if helium10_path else {"global": set(), "by_asin": {}}
    business_report = load_business_report(business_report_path) if business_report_path else {}

    row_issues: dict[int, RowIssues] = defaultdict(RowIssues)

    variation_score, variation_errors = run_variation_engine(df, row_issues)
    duplicate_score, duplicate_errors = run_duplicate_engine(df, row_issues)
    browse_node_score, browse_node_errors = run_browse_node_engine(df, row_issues)
    indexing_score, indexing_errors = run_indexing_engine(df, row_issues, helium10)
    content_score, content_errors, completeness_pct = run_content_engine(df, row_issues)
    aplus_score, aplus_errors = run_aplus_engine(df, row_issues)
    suppression_score, suppression_errors = run_suppression_engine(df, row_issues)

    revenue_at_risk, revenue_by_row = compute_revenue_at_risk(df, row_issues, business_report)
    total_revenue = sum(revenue_by_row.values()) or 1.0
    zero_sales_candidates = find_zero_sales_candidates(df, business_report)
    zero_sales_keys = {c["asin"] for c in zero_sales_candidates if c["asin"]} | {
        c["sku"] for c in zero_sales_candidates if c["sku"]
    }

    overall_health_score = round(
        variation_score * ENGINE_WEIGHTS["variation"]
        + duplicate_score * ENGINE_WEIGHTS["duplicate"]
        + browse_node_score * ENGINE_WEIGHTS["browse_node"]
        + indexing_score * ENGINE_WEIGHTS["indexing"]
        + content_score * ENGINE_WEIGHTS["content"]
        + aplus_score * ENGINE_WEIGHTS["aplus"]
        + suppression_score * ENGINE_WEIGHTS["suppression"]
    )

    category_counts: dict[str, int] = defaultdict(int)
    for ri in row_issues.values():
        for cat in ri.categories:
            category_counts[cat] += 1

    asin_matrix_table = []
    for i, row in df.iterrows():
        ri = row_issues.get(i)
        if ri is None or not ri.issues:
            continue
        # Don't recommend content fixes on listings already flagged as zero-sales
        # deletion candidates — fixing content on something headed for deletion
        # is wasted effort, and would clutter the action list.
        if (row["asin"] and row["asin"] in zero_sales_keys) or (row["sku"] and row["sku"] in zero_sales_keys):
            continue
        revenue = revenue_by_row.get(i, 0.0)
        title_len = len(row["item_name"] or "")
        advice = build_consultant_advice(ri.issues, ri.categories, revenue, total_revenue, title_len)
        asin_matrix_table.append(
            {
                "asin": row["asin"] or "",
                "sku": row["sku"] or "",
                "title": row["item_name"] or "",
                "revenue": round(revenue, 2),
                "issues": ri.issues,
                "categories": ri.categories,
                "severity": ri.severity,
                "feed_product_type": row["feed_product_type"] or "",
                "ai_consultant_advice": advice,
            }
        )

    asin_matrix_table.sort(key=lambda r: (_severity_rank(r["severity"]), r["revenue"]), reverse=True)

    return {
        "overall_health_score": overall_health_score,
        "revenue_at_risk": revenue_at_risk,
        "whitelabel_ready": True,
        "modules": {
            "variation_audit": {"score": variation_score, "errors_count": variation_errors},
            "duplicate_detection": {"score": duplicate_score, "errors_count": duplicate_errors},
            "browse_node_audit": {"score": browse_node_score, "errors_count": browse_node_errors},
            "index_coverage": {"score": indexing_score, "errors_count": indexing_errors},
            "content_completeness": {
                "score": content_score,
                "errors_count": content_errors,
                "completeness_pct": completeness_pct,
            },
            "aplus_brand_story": {"score": aplus_score, "errors_count": aplus_errors},
            "suppression_risk": {"score": suppression_score, "errors_count": suppression_errors},
        },
        "error_distribution_for_charts": [
            {"category": cat, "count": count} for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1])
        ],
        "asin_matrix_table": asin_matrix_table,
        "zero_sales_candidates": zero_sales_candidates,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python audit_engine.py <clr_path> [helium10_path] [business_report_path]")
        sys.exit(1)

    clr_path = sys.argv[1]
    helium10_path = sys.argv[2] if len(sys.argv) > 2 else None
    business_report_path = sys.argv[3] if len(sys.argv) > 3 else None

    result = run_audit(clr_path, helium10_path, business_report_path)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
