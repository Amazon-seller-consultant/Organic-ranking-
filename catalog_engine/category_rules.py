"""Per-product-type attribute rules extracted from the workbook's own sheets.

Rules encoded here (do not relax for convenience):
- Everything is read from the uploaded workbook itself ("Data Definitions",
  "AttributePTDMAP", "Dropdown Lists"). Nothing is hardcoded per category and
  nothing is ever invented: an attribute only becomes a rule if the workbook
  says so.
- An attribute applies to a product type iff it appears in AttributePTDMAP
  with a "1" mark under that product type's column. Matching is done on the
  BASE attribute name (brackets/[quals]/#N stripped via
  parser.parse_column_header + parser.attribute_key), e.g. "bullet_point",
  "merchant_shipping_group", "fulfillment_availability.quantity".
- Requirement level comes from the Data Definitions "Required?" column
  (Required / Conditionally Required / Recommended / Optional). Attributes
  present in the PTD map but absent from Data Definitions default to
  "optional" — defaulting DOWN is safe; we never upgrade a requirement.
- Valid variation themes per product type come from the "Dropdown Lists"
  sheet: each column carries the product type code on row 2, the raw
  attribute name (e.g. "variation_theme#1.name") on row 3, and the accepted
  values from row 4 down. This is exactly where the workbook's own defined
  names ("<PT>variation_theme1.name") point.
  NOTE on "Conditions List": its CONDITION_LIST_N columns are anonymous
  named ranges with no product-type mapping anywhere in the workbook (the
  Template sheet's data validations never reference them — verified against
  the raw sheet XML), so they cannot be attributed to a product type without
  guessing. We therefore do NOT read them; the Dropdown Lists columns above
  provide the unambiguous per-PT theme lists instead.
- get_rules() serves from the Store rules cache (keyed by
  settings.template_identifier + product_type) when fresh, else re-extracts
  from the workbook and caches plain-dict serializations (JSON-safe).
- validate_input() never fabricates a value. Missing required attributes are
  WARNING for partial_update rows (sparse rows are normal there) and ERROR
  only for full_update rows. System fields ("::...") and purely operational
  fields (fulfillment_availability, merchant_shipping_group) are never
  flagged as missing content attributes.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .exceptions import FlatFileError
from .models import (
    Issue,
    ProductRecord,
    RecordAction,
    Severity,
    TemplateSettings,
)
from .parser import _load_workbook, attribute_key, parse_column_header
from .store import Store

DATA_DEFINITIONS_SHEET = "Data Definitions"
PTD_MAP_SHEET = "AttributePTDMAP"
DROPDOWN_LISTS_SHEET = "Dropdown Lists"

#: Allowed values for AttributeRule.requirement.
REQUIREMENT_LEVELS = ("required", "conditionally_required", "recommended", "optional")

#: "Required?" column labels -> canonical requirement values. Unrecognized
#: labels fall back to "optional" (defaulting down, never up).
_REQUIREMENT_MAP = {
    "required": "required",
    "conditionally required": "conditionally_required",
    "recommended": "recommended",
    "optional": "optional",
}

#: Strictness order used only to resolve duplicate Data Definitions entries
#: that collapse to the same base name (e.g. bullet_point #1..#5, or the
#: audience=ALL / audience=B2B purchasable_offer variants). Compliance first:
#: when duplicates disagree, keep the strictest.
_STRICTNESS = {"required": 0, "conditionally_required": 1, "recommended": 2, "optional": 3}

#: Base names that are operational plumbing, not product content — never
#: flagged as missing by validate_input().
OPERATIONAL_BASES = {"fulfillment_availability", "merchant_shipping_group"}

#: Sentinel cache row (per template_identifier) remembering which product
#: types the template covers, so cache hits need not reopen the workbook.
_PT_INDEX_KEY = "::product_types"


@dataclass
class AttributeRule:
    attribute: str  # base name per parser.parse_column_header().base
    raw_name: str  # raw Field Name from Data Definitions
    label: str
    requirement: str  # "required" | "conditionally_required" | "recommended" | "optional"


@dataclass
class ProductTypeRules:
    product_type: str
    template_identifier: str
    rules: dict[str, AttributeRule]  # keyed by base attribute name
    valid_variation_themes: list[str] = field(default_factory=list)


def _base_key(raw_name: str) -> str:
    """Normalize a raw attribute header to its base key: [quals] and #N
    stripped; non-'value' leaves kept as 'base.leaf' (matches the keys used
    in ProductRecord.attributes)."""
    return attribute_key(parse_column_header(0, raw_name))


def _find_header_row(rows: list[tuple]) -> Optional[dict[str, int]]:
    """Locate the Data Definitions header row ('Field Name' ... 'Required?')
    within the first few rows and return column indices by header text."""
    for row in rows[:5]:
        cells = {str(v).strip().lower(): i for i, v in enumerate(row) if v is not None}
        if "field name" in cells and "required?" in cells:
            return {
                "field": cells["field name"],
                "label": cells.get("local label name", -1),
                "required": cells["required?"],
            }
    return None


def _read_data_definitions(ws, source: str) -> dict[str, AttributeRule]:
    """Data Definitions -> {base attribute name: AttributeRule}. Group header
    rows (only column A populated) are skipped; duplicate base names keep the
    strictest requirement (see _STRICTNESS)."""
    rows = list(ws.iter_rows(values_only=True))
    header = _find_header_row(rows)
    if header is None:
        raise FlatFileError(
            f"'{source}': the '{DATA_DEFINITIONS_SHEET}' sheet has no "
            "'Field Name' / 'Required?' header row.",
            "This file may be from an unsupported template version.",
        )

    def cell(row: tuple, idx: int) -> str:
        if idx < 0 or idx >= len(row) or row[idx] is None:
            return ""
        return str(row[idx]).strip()

    defs: dict[str, AttributeRule] = {}
    for row in rows:
        raw_name = cell(row, header["field"])
        if not raw_name or raw_name.lower() == "field name":
            continue  # group header row or the header row itself
        requirement = _REQUIREMENT_MAP.get(
            cell(row, header["required"]).lower(), "optional"
        )
        key = _base_key(raw_name)
        existing = defs.get(key)
        if existing is not None and _STRICTNESS[existing.requirement] <= _STRICTNESS[requirement]:
            continue  # first-seen wins unless a stricter duplicate shows up
        defs[key] = AttributeRule(
            attribute=key,
            raw_name=raw_name,
            label=cell(row, header["label"]),
            requirement=requirement,
        )
    return defs


def _read_ptd_map(ws, source: str) -> "tuple[list[str], dict[str, set[str]], dict[str, str]]":
    """AttributePTDMAP -> (product types, {pt: applicable base names},
    {base name: raw attribute name}). Row 1 holds product type codes from
    column B on; a data row applies to a PT when its cell holds a 1."""
    rows = ws.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        header = ()
    product_types = [
        str(v).strip() for v in header[1:] if v is not None and str(v).strip()
    ]
    if not product_types:
        raise FlatFileError(
            f"'{source}': the '{PTD_MAP_SHEET}' sheet has no product type codes "
            "on its first row.",
            "This file may be from an unsupported template version.",
        )

    def marked(v: Any) -> bool:
        if v is None:
            return False
        s = str(v).strip()
        if s == "1":
            return True
        try:
            return float(s) == 1.0
        except ValueError:
            return False

    applicable: dict[str, set[str]] = {pt: set() for pt in product_types}
    raw_names: dict[str, str] = {}
    for row in rows:
        raw = row[0] if row else None
        if raw is None or not str(raw).strip():
            continue  # stray repeated header rows have an empty name cell
        key = _base_key(str(raw))
        raw_names.setdefault(key, str(raw).strip())
        for i, pt in enumerate(product_types):
            if i + 1 < len(row) and marked(row[i + 1]):
                applicable[pt].add(key)
    return product_types, applicable, raw_names


def _read_variation_themes(ws, product_types: list[str]) -> dict[str, list[str]]:
    """Dropdown Lists -> {pt: ordered de-duplicated variation themes}.

    Layout (verified against the workbook's own defined names
    '<PT>variation_theme1.name', which point at exactly these ranges):
    row 2 = product type code, row 3 = raw attribute name, rows 4+ = values.
    """
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 4:
        return {}
    pt_row, attr_row = rows[1], rows[2]
    known = set(product_types)
    themes: dict[str, list[str]] = {}
    width = max(len(pt_row), len(attr_row))
    for ci in range(width):
        pt = str(pt_row[ci]).strip() if ci < len(pt_row) and pt_row[ci] is not None else ""
        raw = str(attr_row[ci]).strip() if ci < len(attr_row) and attr_row[ci] is not None else ""
        if pt not in known or not raw:
            continue
        if parse_column_header(ci, raw).base != "variation_theme":
            continue
        seen: set[str] = set()
        values: list[str] = []
        for row in rows[3:]:
            if ci >= len(row) or row[ci] is None:
                continue
            v = str(row[ci]).strip()
            if v and v not in seen:
                seen.add(v)
                values.append(v)
        themes[pt] = values
    return themes


def extract_rules(
    workbook_path: Union[str, Path], settings: TemplateSettings
) -> dict[str, ProductTypeRules]:
    """Extract per-product-type attribute rules from the workbook itself.

    A rule exists for (product type, attribute) iff the attribute appears in
    AttributePTDMAP with a mark under that product type. Requirement levels
    come from Data Definitions; PTD-mapped attributes missing from Data
    Definitions default to "optional".

    Observed in the real Category Listing Reports: the PTD map only lists the
    offer/fulfillment columns (fulfillment_availability.*,
    merchant_shipping_group, purchasable_offer.*), each marked applicable to
    every product type, so content attributes (item_name, bullet_point, ...)
    do not receive per-PT rules from this sheet. We extract exactly what the
    workbook states rather than inventing a broader mapping.
    """
    path = Path(workbook_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        wb = _load_workbook(path)
    try:
        missing = [
            s for s in (DATA_DEFINITIONS_SHEET, PTD_MAP_SHEET) if s not in wb.sheetnames
        ]
        if missing:
            raise FlatFileError(
                f"'{path.name}' is missing required sheet(s): {', '.join(missing)} "
                f"(found: {', '.join(wb.sheetnames) or 'none'}).",
                "Re-download the Category Listing Report from Seller Central and "
                "upload it unmodified.",
            )
        definitions = _read_data_definitions(wb[DATA_DEFINITIONS_SHEET], path.name)
        product_types, applicable, ptd_raw = _read_ptd_map(wb[PTD_MAP_SHEET], path.name)
        if DROPDOWN_LISTS_SHEET in wb.sheetnames:
            themes = _read_variation_themes(wb[DROPDOWN_LISTS_SHEET], product_types)
        else:
            themes = {}

        result: dict[str, ProductTypeRules] = {}
        for pt in product_types:
            rules: dict[str, AttributeRule] = {}
            for key in sorted(applicable[pt]):
                defined = definitions.get(key)
                if defined is not None:
                    rules[key] = AttributeRule(
                        attribute=key,
                        raw_name=defined.raw_name,
                        label=defined.label,
                        requirement=defined.requirement,
                    )
                else:
                    # In the PTD map but not documented in Data Definitions:
                    # default DOWN to optional, never invent a requirement.
                    rules[key] = AttributeRule(
                        attribute=key,
                        raw_name=ptd_raw.get(key, key),
                        label="",
                        requirement="optional",
                    )
            result[pt] = ProductTypeRules(
                product_type=pt,
                template_identifier=settings.template_identifier,
                rules=rules,
                valid_variation_themes=themes.get(pt, []),
            )
        return result
    finally:
        wb.close()


def _rules_from_dict(data: dict[str, Any]) -> ProductTypeRules:
    """Rebuild a ProductTypeRules from its plain-dict (JSON) serialization."""
    return ProductTypeRules(
        product_type=data["product_type"],
        template_identifier=data["template_identifier"],
        rules={
            key: AttributeRule(**rule) for key, rule in data.get("rules", {}).items()
        },
        valid_variation_themes=list(data.get("valid_variation_themes", [])),
    )


def get_rules(
    workbook_path: Union[str, Path],
    settings: TemplateSettings,
    store: Store,
    force_refresh: bool = False,
) -> dict[str, ProductTypeRules]:
    """Serve rules from the Store cache when fresh; else extract and cache.

    The cache is keyed by (settings.template_identifier, product_type) and
    stores plain dicts. A sentinel row per template remembers the product
    type list so a warm cache never reopens the workbook.
    """
    template_identifier = settings.template_identifier
    if not force_refresh:
        index = store.get_cached_rules(template_identifier, _PT_INDEX_KEY)
        if index and index.get("product_types"):
            cached: dict[str, ProductTypeRules] = {}
            for pt in index["product_types"]:
                data = store.get_cached_rules(template_identifier, pt)
                if data is None:
                    cached = {}
                    break  # partially stale cache -> re-extract everything
                cached[pt] = _rules_from_dict(data)
            if cached:
                return cached

    extracted = extract_rules(workbook_path, settings)
    for pt, ptr in extracted.items():
        store.cache_rules(template_identifier, pt, asdict(ptr))
    store.cache_rules(
        template_identifier, _PT_INDEX_KEY, {"product_types": sorted(extracted)}
    )
    return extracted


def _has_value(record: ProductRecord, base: str) -> bool:
    return any(str(v).strip() for v in record.attributes.get(base, []))


def validate_input(
    records: list[ProductRecord], rules: dict[str, ProductTypeRules]
) -> list[Issue]:
    """Flag missing required attributes on input records. Never fabricates.

    - Missing requirement=="required" attribute: WARNING for partial_update
      records (sparse rows are normal in partial updates, not data-quality
      errors), ERROR for full_update records. Delete records are skipped —
      they carry no content by design. Records with an unrecognized action
      are treated conservatively as partial (WARNING).
    - System fields ("::...") and operational fields
      (fulfillment_availability, merchant_shipping_group) are never flagged.
    - A product type with no rules yields one INFO issue per record
      (rule="unknown_product_type") so the gap is visible, not guessed at.
    """
    issues: list[Issue] = []
    for record in records:
        pt_rules = rules.get(record.product_type)
        if pt_rules is None:
            issues.append(
                Issue(
                    sku=record.sku,
                    field_name="input:product_type",
                    rule="unknown_product_type",
                    severity=Severity.INFO,
                    message=(
                        f"row {record.row_number}: no rules found for product type "
                        f"'{record.product_type}' in this template — required-"
                        "attribute checks were skipped for this SKU"
                    ),
                    action="none",
                    before=record.product_type,
                )
            )
            continue
        if record.record_action is RecordAction.DELETE:
            continue
        severity = (
            Severity.ERROR
            if record.record_action is RecordAction.FULL_UPDATE
            else Severity.WARNING
        )
        for key in sorted(pt_rules.rules):
            rule = pt_rules.rules[key]
            if rule.requirement != "required":
                continue
            if key.startswith("::") or key.split(".", 1)[0] in OPERATIONAL_BASES:
                continue
            if _has_value(record, key):
                continue
            issues.append(
                Issue(
                    sku=record.sku,
                    field_name=f"input:{key}",
                    rule="missing_mandatory_attribute",
                    severity=severity,
                    message=(
                        f"row {record.row_number}: required attribute "
                        f"'{rule.label or key}' ({key}) has no value for product "
                        f"type {record.product_type} "
                        f"({record.record_action.value})"
                    ),
                    action="flagged",
                )
            )
    return issues
