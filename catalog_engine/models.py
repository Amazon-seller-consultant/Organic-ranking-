"""Shared data model. Every module in the pipeline codes against these types.

Rules encoded here (do not relax for convenience):
- Limits are enforced by deterministic code in validator.py, never by an LLM.
- ProductRecord.attributes only ever contains values read from the source
  file. Nothing downstream may add to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Content limits (chars unless noted). TITLE_MAX is the July 27, 2026 rule:
# 75 chars including brand.
# ---------------------------------------------------------------------------
TITLE_MAX_CHARS = 75
HIGHLIGHT_MAX_CHARS = 125
BULLET_MAX_CHARS = 125
BULLET_COUNT = 3
DESCRIPTION_MAX_CHARS = 2000
SEARCH_TERMS_MAX_BYTES = 249  # UTF-8 bytes, space-separated


class RecordAction(str, Enum):
    FULL_UPDATE = "full_update"
    PARTIAL_UPDATE = "partial_update"
    DELETE = "delete"
    UNKNOWN = "unknown"

    @classmethod
    def from_cell(cls, raw: Optional[str]) -> "RecordAction":
        """Normalize both machine values and display labels like
        'Edit (Partial Update)'. Unknown non-empty values map to UNKNOWN so
        the pipeline can flag them instead of guessing."""
        if raw is None or not str(raw).strip():
            return cls.PARTIAL_UPDATE  # Amazon treats blank as partial edit
        s = str(raw).strip().lower()
        if "partial" in s:
            return cls.PARTIAL_UPDATE
        if "full" in s:
            return cls.FULL_UPDATE
        if "delete" in s:
            return cls.DELETE
        return cls.UNKNOWN


class Parentage(str, Enum):
    PARENT = "parent"
    CHILD = "child"
    STANDALONE = "standalone"


@dataclass
class TemplateSettings:
    """Decoded from cell A1 of the Template sheet (never hardcoded)."""

    label_row: int
    attribute_row: int
    data_row: int
    marketplace_id: str
    language_tag: str
    template_identifier: str
    feed_type: str = ""
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class ColumnInfo:
    """One Template-sheet column, parsed from its attributeRow header."""

    index: int  # 0-based column index in the sheet
    raw_name: str  # e.g. bullet_point[marketplace_id=..][language_tag=..]#2.value
    base: str  # e.g. "bullet_point" or "::record_action"
    instance: int  # the #N instance, 1 if absent
    leaf: str  # trailing field after the last '.', e.g. "value", "parent_sku"
    label: str  # human label from labelRow ("" if blank)


@dataclass
class ProductRecord:
    seller_id: str
    sku: str
    row_number: int  # 1-based row in the Template sheet, for error reporting
    product_type: str
    record_action: RecordAction
    listing_status: str
    parentage: Parentage
    parent_sku: Optional[str]
    variation_theme: Optional[str]
    # base attribute name -> ordered values (instance #1, #2, ...).
    # Leaf fields other than "value" are keyed as "base.leaf".
    # SOURCE DATA ONLY — never written to after parsing.
    attributes: dict[str, list[str]] = field(default_factory=dict)
    # raw column header -> cell value, for exact round-tripping in output
    raw_cells: dict[str, str] = field(default_factory=dict)

    def first(self, base: str) -> Optional[str]:
        vals = self.attributes.get(base)
        return vals[0] if vals else None

    @property
    def title(self) -> Optional[str]:
        return self.first("item_name")

    @property
    def brand(self) -> Optional[str]:
        return self.first("brand")

    @property
    def bullets(self) -> list[str]:
        return self.attributes.get("bullet_point", [])

    @property
    def description(self) -> Optional[str]:
        return self.first("product_description")

    @property
    def search_terms(self) -> list[str]:
        return self.attributes.get("generic_keyword", [])


@dataclass
class VariationFamily:
    parent_sku: str
    parent: Optional[ProductRecord]  # None => orphan children (flagged)
    children: list[ProductRecord] = field(default_factory=list)


@dataclass
class ParseResult:
    seller_id: str
    source_file: str
    settings: TemplateSettings
    columns: list[ColumnInfo]
    records: list[ProductRecord]
    families: list[VariationFamily]
    warnings: list[str] = field(default_factory=list)

    def by_sku(self) -> dict[str, ProductRecord]:
        return {r.sku: r for r in self.records}


# ---------------------------------------------------------------------------
# Seller configuration (multi-tenant)
# ---------------------------------------------------------------------------
@dataclass
class SellerConfig:
    seller_id: str
    display_name: str = ""
    marketplace: str = "US"  # US / UK / EU / CA
    on_limit_violation: str = "auto_trim"  # or "flag"
    brand_voice: str = ""  # free-text tone guidance for generation
    generation_model: str = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# Generation + verification
# ---------------------------------------------------------------------------
@dataclass
class GeneratedContent:
    sku: str
    title: str = ""
    item_highlights: list[str] = field(default_factory=list)  # <=125 chars each
    bullets: list[str] = field(default_factory=list)  # exactly 3, <=125 chars
    description: str = ""  # plain text, <=2000 chars
    search_terms: str = ""  # single space-separated string, <=249 bytes UTF-8
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class Severity(str, Enum):
    ERROR = "error"  # blocks output for this SKU unless auto-fixed
    WARNING = "warning"  # shipped but surfaced in report
    INFO = "info"


@dataclass
class Issue:
    """A validation/compliance finding, both for input and output checks."""

    sku: str
    field_name: str  # "title", "bullet_2", "search_terms", "input:material"...
    rule: str  # machine id, e.g. "title_max_chars", "pesticide_claim"
    severity: Severity
    message: str  # seller-facing explanation
    action: str = "flagged"  # "trimmed" | "removed" | "flagged" | "none"
    before: str = ""
    after: str = ""


@dataclass
class ComplianceLogEntry:
    """Structured audit record a seller can point to later.

    e.g. field='bullet_3', category='pesticide',
    removed_text='kills germs',
    reason="removed 'kills germs' — pesticide claim requires EPA registration"
    """

    seller_id: str
    run_id: str
    sku: str
    field_name: str
    category: str  # medical | pesticide | safety | promotional | limit | hallucination
    removed_text: str
    reason: str


@dataclass
class SkuOutcome:
    """Everything the output stage needs for one SKU."""

    record: ProductRecord
    generated: Optional[GeneratedContent]
    issues: list[Issue] = field(default_factory=list)
    compliance_log: list[ComplianceLogEntry] = field(default_factory=list)
    status: str = "ok"  # ok | needs_review | skipped_unchanged | skipped | failed
    skip_reason: str = ""


@dataclass
class RunSummary:
    run_id: str
    seller_id: str
    source_file: str
    total_rows: int
    in_scope: int
    generated: int
    skipped_unchanged: int
    needs_review: int
    failed: int
    input_tokens: int
    output_tokens: int
    outputs: dict[str, str] = field(default_factory=dict)  # artifact name -> path
