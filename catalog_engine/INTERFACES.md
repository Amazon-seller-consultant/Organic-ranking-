# catalog_engine module contracts

Shared types live in `catalog_engine/models.py`. Parsing is done (`parser.py`),
persistence is done (`store.py`), errors in `exceptions.py`. Read those first.
Python 3.9 compatible (`from __future__ import annotations` where needed; no
`match`). Dependencies available: openpyxl, pandas, anthropic, python-dotenv.

## Non-negotiable rules (apply to every module)
1. Never invent/upgrade/assume a product attribute not present in source data.
2. Char/byte limits enforced by deterministic code only.
3. Priority when rules conflict: Amazon compliance > attribute accuracy >
   char/byte limits > keyword retention > writing quality.
4. All seller data scoped by `seller_id`; artifacts go under
   `Store.seller_dir(seller_id)/outputs/<run_id>/`.
5. Fail loudly with specific seller-facing messages (`FlatFileError` pattern),
   never silently guess.

## Sample fixtures (real Amazon files, read-only)
- /Users/denizolmez/Downloads/Category+Listings+Report_06-19-2026.xlsm  (376 records, 812 cols)
- /Users/denizolmez/Downloads/Category+Listings+Report_06-25-2026.xlsm  (482 records, 925 cols)
- /Users/denizolmez/Downloads/Category+Listings+Report_07-02-2026.xlsm  (518 records, 658 cols)
All: labelRow=4, attributeRow=5, dataRow=7 (from Template!A1 settings), US, en_US.
openpyxl emits a harmless "Data Validation extension" UserWarning on load.

## catalog_engine/category_rules.py
Extract per-product-type attribute rules from the workbook's own sheets:
- `Data Definitions` (header row 2: Group Name / Field Name / Local Label Name /
  Accepted Values / Example / Required?). Required? values: Required,
  Conditionally Required, Recommended, Optional. Field Name matches the
  Template attributeRow raw names.
- `AttributePTDMAP`: row 1 = product type codes (col B..), col A = raw attribute
  names; cell "1" = attribute applies to that product type.
- `Conditions List`: per-product-type valid variation themes (inspect the sheet
  yourself; columns CONDITION_LIST_N; treat best-effort, warn if unmappable).

API:
```python
@dataclass
class AttributeRule:
    attribute: str          # base name per parser.parse_column_header().base
    raw_name: str           # raw Field Name from Data Definitions
    label: str
    requirement: str        # "required" | "conditionally_required" | "recommended" | "optional"

@dataclass
class ProductTypeRules:
    product_type: str
    template_identifier: str
    rules: dict[str, AttributeRule]        # keyed by base attribute name
    valid_variation_themes: list[str]

def extract_rules(workbook_path, settings: TemplateSettings) -> dict[str, ProductTypeRules]
def get_rules(workbook_path, settings, store: Store, force_refresh: bool = False) -> dict[str, ProductTypeRules]
    # get_rules: serve from store.get_cached_rules(template_identifier, product_type)
    # when fresh; else extract_rules() and store.cache_rules(). Cache stores
    # dataclasses as plain dicts (JSON).
def validate_input(records: list[ProductRecord], rules: dict[str, ProductTypeRules]) -> list[Issue]
```
validate_input flags missing `required` attributes as Issue(severity=WARNING,
rule="missing_mandatory_attribute", field_name=f"input:{attr}") — WARNING not
ERROR because every sample row is partial_update and sparse rows are NOT
data-quality errors. For records with record_action == FULL_UPDATE, missing
required attributes are ERROR. Unknown product_type (no rules found) -> one
INFO issue per record, rule="unknown_product_type". Never fabricate a value.
Use base attribute names (strip [quals] and #N — reuse
`parser.parse_column_header`).

## catalog_engine/validator.py  (+ compliance_terms.py)
Deterministic verification of GeneratedContent. API:
```python
def char_len(s: str) -> int              # len() over NFC-normalized str
def byte_len_utf8(s: str) -> int
def trim_to_chars(s: str, limit: int) -> str    # trim at word boundary, no trailing punct/space
def trim_to_bytes(s: str, limit: int) -> str    # drop whole space-separated tokens from the end

def verify_generated(
    record: ProductRecord,
    gen: GeneratedContent,
    config: SellerConfig,          # on_limit_violation: "auto_trim" | "flag"
    run_id: str,
) -> tuple[GeneratedContent, list[Issue], list[ComplianceLogEntry]]
```
Checks, in priority order:
1. **Compliance-language scan** on title/highlights/bullets/description/search
   terms using regex term lists in `compliance_terms.py`, grouped by category:
   - medical: cure, treat(s/ment), heal, antibacterial, anti-bacterial,
     antimicrobial, antifungal, kills germs/bacteria/viruses, sanitize(s/r),
     disinfect, FDA approved, therapeutic, relieves pain, anti-inflammatory...
   - pesticide: pesticide, insecticide, repels insects/mosquito, kills
     insects/pests/mold/mildew, fungicide, antimicrobial (also pesticide per
     EPA)...
   - safety: fireproof, flameproof, non-toxic (unless present in source
     attributes), child-safe, BPA free (unless in source), hypoallergenic...
   - promotional: best seller, #1, top rated, hot item, free shipping,
     guarantee(d), warranty (in title/bullets only), sale, discount, cheapest,
     limited time, money back...
   Word-boundary, case-insensitive. Violations are REMOVED from the text
   (excise the offending phrase, tidy whitespace/punctuation), logged as
   ComplianceLogEntry(category=..., reason="removed '<text>' from <field> — <category> claim")
   + Issue(action="removed", severity=WARNING). If removal guts a field
   (<30% of original length left), instead keep original text and mark
   Issue(severity=ERROR, action="flagged") -> caller sets needs_review.
2. **Attribute-diff (hallucination check)**: extract factual claims from
   generated text: numbers with units (e.g. 11x17, 8.5 inch, 24 pack, 500 lb),
   color words, material words (wood, oak, steel, aluminum, acrylic, plastic,
   leather, bamboo, glass, metal, canvas, fabric, vinyl, ceramic, marble...).
   Each claim must appear in some source attribute value (case-insensitive
   substring across `record.attributes` values + sku). Number claims: compare
   numeric value, tolerate formatting (11 x 17 vs 11x17, 8.5" vs 8.5 inch).
   Unsupported claim -> ERROR Issue rule="unsupported_claim",
   action="flagged" (do NOT auto-edit facts) + ComplianceLogEntry
   category="hallucination".
3. **Limits** (after removals): title<=TITLE_MAX_CHARS, each highlight &
   bullet<=125 chars, exactly BULLET_COUNT bullets (more -> keep first 3,
   log; fewer -> ERROR flag), description<=2000 chars, search_terms<=249
   UTF-8 bytes. Bullets must not end with '.' (strip it, log INFO). Over
   limit: if config.on_limit_violation=="auto_trim" -> trim_* + Issue
   action="trimmed" (WARNING) + log entry category="limit"; else Issue ERROR
   action="flagged".
4. Search terms extra: lowercase, dedupe tokens preserving order, remove
   tokens equal to brand words (Amazon ignores brand in search terms),
   then byte-trim.
Return the possibly-modified GeneratedContent copy; never mutate inputs.
Include unit tests in `tests/test_validator.py` (multibyte byte-counting,
trim boundaries, each compliance category, hallucination detection pos/neg).

## catalog_engine/output.py
```python
def write_outputs(
    parse: ParseResult,
    outcomes: list[SkuOutcome],       # only SKUs processed this run
    run_id: str,
    out_dir: Path,                    # Store.seller_dir()/outputs/<run_id>
) -> dict[str, str]                   # artifact name -> path
```
Artifacts:
1. `upload.xlsx` — Amazon-upload-compatible partial-update file. Copy Template
   sheet rows 1..settings.data_row-1 VERBATIM from the source workbook (all
   columns, including A1 settings string; preserve blank row(s) between
   attribute row and data row). Then one row per outcome with status "ok":
   values ONLY in: contribution_sku, product_type, ::record_action (write
   "partial_update"), item_name (new title), bullet_point #1-#3 (leave #4/#5
   cells EMPTY — partial update ignores blanks, existing bullets 4-5 survive),
   product_description, generic_keyword#1 (full space-separated search-terms
   string; leave #2+ empty). Locate columns via parse.columns
   (raw_name/base/instance/leaf) — never by hardcoded index. Item Highlights
   are NOT written to the flat file (no column exists) — JSON/report only.
   Sheet name "Template". Do not copy other sheets.
2. `results.json` — per-SKU: sku, status, generated fields incl.
   item_highlights, char/byte counts, issues, compliance log entries.
3. `report.html` — self-contained human-readable review report: run summary
   header (counts by status), then per-SKU cards: before/after table per field
   (before = source record values, after = generated), char/byte counts vs
   limits, issues with severity badges, compliance-log reasons. Escape HTML.
   No external assets (inline CSS). Include SKUs with status needs_review
   prominently (sort: needs_review, then ok, then skipped/failed).
4. `input_issues.json` — issues passed in via outcomes with field_name
   starting "input:".
Never auto-publish anywhere. Include tests in `tests/test_output.py` (use a
small synthetic ParseResult or parse a sample file; verify upload.xlsx A1
settings survives, header rows identical, only expected cells populated,
bullets 4/5 empty).

## tests/test_parser.py (regression suite)
pytest, runnable via `python3 -m pytest tests/ -q`. Cover, for EACH of the 3
sample files (skip gracefully with pytest.skip if a file is missing):
- settings decoded: label/attribute/data rows == 4/5/7, marketplace US
- record counts == {06-19: 376, 06-25: 482, 07-02: 518}
- every record has sku + product_type; record_action == partial_update for all
- variation family integrity: every CHILD is in exactly one family; family
  counts == {06-19: 60, 06-25: 27, 07-02: 107}; orphan counts == {58, 0, 2}
- column mapping: bullet_point columns exist with instances 1..5;
  item_name, product_description, generic_keyword present
- malformed uploads raise FlatFileError with helpful text: missing file,
  .txt file, corrupt bytes as .xlsm, workbook without Template sheet,
  Template sheet whose A1 lacks 'settings=' (build tiny xlsx fixtures with
  openpyxl in tmp_path)
- RecordAction.from_cell mapping table incl. display labels
