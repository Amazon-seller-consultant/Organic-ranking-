# Amazon Catalog Optimization & Compliance Engine

Multi-tenant pipeline that ingests Amazon Category Listing Report flat files
(.xlsm) and produces compliant, optimized listing content with a full audit
trail. Never auto-publishes: output is an upload-ready flat file plus a
human-readable review report.

## Pipeline

```
parse -> validate input -> generate (LLM) -> verify output (deterministic) -> output
parser.py   category_rules.py   generator.py       validator.py               output.py
```

The LLM (step 3) is the **only** stage that touches a model. Parsing,
counting, limit enforcement, compliance scanning, and attribute-diffing are
deterministic code.

## Usage

```sh
# one-time per seller
python3 -m catalog_engine.cli seller create --seller acme \
    --name "Acme Displays" --marketplace US --on-limit auto_trim \
    --brand-voice "confident, practical, no fluff"

# process a flat file (pilot = one representative SKU per product type)
python3 -m catalog_engine.cli run --seller acme --file report.xlsm --pilot
python3 -m catalog_engine.cli run --seller acme --file report.xlsm          # full catalog
python3 -m catalog_engine.cli run --seller acme --file report.xlsm --force  # ignore fingerprints

python3 -m catalog_engine.cli usage --seller acme
```

Requires `ANTHROPIC_API_KEY` (loaded from the project `.env`).

Artifacts land in `data/sellers/<seller_id>/outputs/<run_id>/`:

| Artifact | Purpose |
|---|---|
| `upload.xlsx` | Partial-update flat file for Seller Central. Header rows copied verbatim from the source (incl. the A1 settings string). Only touched cells populated. |
| `report.html` | Per-SKU before/after review report with char/byte counts, severity badges, and compliance reasons. Review before uploading. |
| `results.json` | Structured per-SKU results incl. Item Highlights. |
| `input_issues.json` | Missing-mandatory-attribute findings from input validation. |

## Content rules enforced (validator.py, deterministic)

- Title ≤ 75 chars including brand (July 27, 2026 rule)
- Item Highlights: 3 × ≤ 125 chars — **not written to the flat file** (no
  column exists in the template); delivered via report/JSON for manual entry
  in Seller Central
- Bullets: exactly 3 × ≤ 125 chars, fragments, no terminal periods.
  Bullet columns 4–5 are left empty in the upload file so a partial update
  preserves the seller's existing bullets 4–5
- Description ≤ 2000 chars plain text
- Backend search terms ≤ 249 UTF-8 bytes, lowercase, deduped, brand words
  removed
- Compliance-language scan (medical / pesticide / safety / promotional) —
  violations are excised and logged with reasons; if excision guts a field,
  the SKU is flagged `needs_review` instead
- Attribute-diff: numeric/color/material claims in generated copy must
  appear in the source record; unsupported claims flag the SKU (facts are
  never auto-edited)

Per-seller `on_limit_violation`: `auto_trim` (word-boundary trim + audit log)
or `flag` (needs_review).

## Multi-tenancy

Everything is scoped by `seller_id`: SQLite rows (sellers, runs,
sku_fingerprints, usage, compliance_log) and artifact directories
(`data/sellers/<id>/`). The only shared table is `rules_cache`, keyed by
Amazon's `templateIdentifier` — rules come from Amazon's template, not seller
data. Incremental reprocessing hashes each SKU's content-relevant source
attributes; unchanged SKUs are skipped unless `--force`.

## Known findings from the real sample files (June–July 2026)

1. **Header rows are dynamic but currently 4/5/7** (labelRow/attributeRow/
   dataRow from the `settings=` string in Template!A1). Column positions
   shift between exports (812/925/658 columns in the three samples) — the
   parser maps by attribute name, never index.
2. **AttributePTDMAP does not map content attributes.** In all three sample
   files it contains only offer/fulfillment attributes, marked applicable to
   every product type. Per-product-type *content* mandatory rules therefore
   cannot be derived from the workbook itself; the engine extracts what is
   unambiguous (requirement levels from Data Definitions ∩ PTD map, valid
   variation themes from the Dropdown Lists named ranges) and flags rather
   than guesses. If Amazon publishes richer per-PT requirements, extend
   `category_rules.py`.
3. **Conditions List is unmappable** — anonymous named ranges with no
   references from Template data validations; deliberately unused.
4. All rows in the samples are `Edit (Partial Update)`; sparse rows are not
   data-quality errors and input validation treats missing mandatory
   attributes on partial updates as WARNING, not ERROR.

## Tests

```sh
python3 -m pytest tests/ -q   # 108 tests; regression suite runs against the
                              # 3 real sample files in ~/Downloads when present
```
