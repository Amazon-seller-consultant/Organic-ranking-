"""Content generation — the ONLY pipeline stage that touches an LLM.

The LLM writes copy from source attributes it is given; it never parses,
counts, or validates. Structured outputs (output_config.format with a strict
JSON schema) guarantee a parseable response; deterministic verification
happens downstream in validator.py regardless of what the model claims.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import anthropic

from .exceptions import GenerationError
from .models import (
    BULLET_COUNT,
    BULLET_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    HIGHLIGHT_MAX_CHARS,
    SEARCH_TERMS_MAX_BYTES,
    TITLE_MAX_CHARS,
    GeneratedContent,
    ProductRecord,
    SellerConfig,
)
from .store import Store

# Attributes that carry listing content or facts the copy may draw on.
# Everything the model sees comes from this snapshot — nothing else.
CONTENT_ATTRIBUTE_ORDER = [
    "item_name", "brand", "manufacturer", "product_description", "bullet_point",
    "generic_keyword", "color", "size", "material", "style", "item_shape",
    "number_of_items", "item_package_quantity", "unit_count", "unit_count.type",
    "included_components", "special_feature", "product_site_launch_date",
    "model_number", "model_name", "part_number", "item_type_keyword",
    "target_audience_keyword", "specific_uses_for_product", "room_type",
    "mounting_type", "orientation", "frame_type", "display_type", "finish_type",
    "paper_size", "item_dimensions", "item_display_dimensions",
]

# Excluded entirely: operational/system fields with no copy value.
EXCLUDED_PREFIXES = (
    "::", "fulfillment", "merchant_", "purchasable_offer", "list_price",
    "condition_", "batteries_", "supplier_", "gpsr_", "dsa_",
    "child_parent_sku_relationship", "parentage_level", "variation_theme",
    "main_image", "other_image", "swatch_image",
)

SYSTEM_PROMPT = """You are an Amazon listing copywriter working inside a \
compliance pipeline. You write optimized listing content from SOURCE DATA \
only. Deterministic code later verifies every claim you make against the \
source data and enforces all limits; anything you invent will be caught and \
the listing flagged, so accuracy beats persuasiveness.

HARD RULES
1. Use ONLY facts present in the source data. Never invent, upgrade, or
   assume an attribute (no materials, dimensions, quantities, colors,
   certifications, or capabilities that are not in the data). If the data is
   thin, write thinner copy — do not fill gaps.
2. No medical claims (cure, treat, antibacterial, kills germs, sanitize...),
   no pesticide claims, no safety certifications not in the data, no
   promotional language (best seller, #1, free shipping, guarantee, sale,
   discount, top rated).
3. Priority order: Amazon compliance > attribute accuracy > length limits >
   keyword retention > writing quality.

FIELD RULES
- title: <= {title_max} characters INCLUDING the brand name, which must
  appear at the start. Format: Brand + product type + 1-3 highest-value
  differentiators (size/color/quantity if present). Title case. No promo
  words, no ALL CAPS words except the brand's own styling, no special
  characters like ~!*$?_{{}}#<>|.
- item_highlights: exactly 3 strings, each <= {highlight_max} characters.
  Short scannable phrases of the product's strongest verifiable selling
  points. Sentence fragments, no terminal periods.
- bullets: exactly {bullet_count} strings, each <= {bullet_max} characters.
  Sentence fragments (NOT full sentences), no terminal periods. Start each
  with a capitalized benefit phrase, then supporting fact from the data.
  Do not repeat the same fact across bullets.
- description: plain text, <= {description_max} characters. No HTML tags.
  2-4 short paragraphs separated by a blank line. Natural, factual,
  benefit-oriented. Do not restate the bullets verbatim.
- search_terms: single string of lowercase space-separated words,
  <= {search_terms_max} BYTES of UTF-8. No brand names (yours or
  competitors'), no ASINs, no repetition of any word, no commas, no words
  already doing work in the title, no misspellings, no subjective claims.
  Choose generic discovery words a shopper would actually type.
{brand_voice}"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "item_highlights": {
            "type": "array", "items": {"type": "string"},
        },
        "bullets": {
            "type": "array", "items": {"type": "string"},
        },
        "description": {"type": "string"},
        "search_terms": {"type": "string"},
    },
    "required": ["title", "item_highlights", "bullets", "description", "search_terms"],
    "additionalProperties": False,
}


def build_source_snapshot(record: ProductRecord) -> dict[str, Any]:
    """The exact facts the model is allowed to use — nothing else.

    Ordered dict: known content attributes first, then any remaining
    non-operational attributes so obscure-but-real facts stay usable.
    """
    snapshot: dict[str, Any] = {
        "sku": record.sku,
        "product_type": record.product_type,
    }
    if record.parentage.value != "standalone":
        snapshot["parentage"] = record.parentage.value
        if record.variation_theme:
            snapshot["variation_theme"] = record.variation_theme

    def add(key: str) -> None:
        vals = record.attributes.get(key)
        if vals:
            snapshot[key] = vals[0] if len(vals) == 1 else vals

    for key in CONTENT_ATTRIBUTE_ORDER:
        add(key)
    for key in sorted(record.attributes):
        if key in snapshot or key in CONTENT_ATTRIBUTE_ORDER:
            continue
        if any(key.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        add(key)
    return snapshot


def _system_prompt(config: SellerConfig) -> str:
    brand_voice = ""
    if config.brand_voice:
        brand_voice = (
            "\nBRAND VOICE (style only — never a source of facts): "
            + config.brand_voice
        )
    return SYSTEM_PROMPT.format(
        title_max=TITLE_MAX_CHARS,
        highlight_max=HIGHLIGHT_MAX_CHARS,
        bullet_count=BULLET_COUNT,
        bullet_max=BULLET_MAX_CHARS,
        description_max=DESCRIPTION_MAX_CHARS,
        search_terms_max=SEARCH_TERMS_MAX_BYTES,
        brand_voice=brand_voice,
    )


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=4)


def generate_for_record(
    client: anthropic.Anthropic,
    record: ProductRecord,
    config: SellerConfig,
    run_id: str,
    store: Optional[Store] = None,
) -> GeneratedContent:
    snapshot = build_source_snapshot(record)
    user_msg = (
        "SOURCE DATA (the only permissible facts):\n"
        + json.dumps(snapshot, ensure_ascii=False, indent=1, sort_keys=False)
        + "\n\nWrite the optimized listing content for this product."
    )

    try:
        response = client.messages.create(
            model=config.generation_model,
            max_tokens=3000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # stable prefix cached across the batch; per-SKU data goes last
            system=[{
                "type": "text",
                "text": _system_prompt(config),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIStatusError as exc:
        raise GenerationError(record.sku, f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise GenerationError(record.sku, f"connection error: {exc}") from exc

    if store is not None:
        store.record_usage(
            seller_id=config.seller_id, run_id=run_id, sku=record.sku,
            model=config.generation_model,
            input_tokens=response.usage.input_tokens
            + (response.usage.cache_creation_input_tokens or 0)
            + (response.usage.cache_read_input_tokens or 0),
            output_tokens=response.usage.output_tokens,
        )

    if response.stop_reason == "refusal":
        raise GenerationError(record.sku, "model declined the request (refusal)")
    if response.stop_reason == "max_tokens":
        raise GenerationError(record.sku, "response truncated at max_tokens")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(record.sku, f"model returned invalid JSON: {exc}") from exc

    return GeneratedContent(
        sku=record.sku,
        title=str(data.get("title", "")).strip(),
        item_highlights=[str(h).strip() for h in data.get("item_highlights", [])],
        bullets=[str(b).strip() for b in data.get("bullets", [])],
        description=str(data.get("description", "")).strip(),
        search_terms=str(data.get("search_terms", "")).strip(),
        model=config.generation_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
