"""Tests for catalog_engine.category_rules against the real 07-02 sample file.

The AttributePTDMAP sheet in the real Category Listing Reports only lists the
offer/fulfillment columns (fulfillment_availability.*, merchant_shipping_group,
purchasable_offer.*), each marked applicable to every product type — so every
product type receives the same 18 base-keyed rules and none of them is
"required". The validate_input severity tests therefore inject one synthetic
required rule on top of the real rules to exercise the WARNING/ERROR paths
with real parsed records.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from catalog_engine import category_rules
from catalog_engine.category_rules import (
    REQUIREMENT_LEVELS,
    AttributeRule,
    ProductTypeRules,
    extract_rules,
    get_rules,
    validate_input,
)
from catalog_engine.models import RecordAction, Severity
from catalog_engine.parser import parse_flat_file
from catalog_engine.store import Store

SAMPLE = Path("/Users/denizolmez/Downloads/Category+Listings+Report_07-02-2026.xlsm")

EXPECTED_PRODUCT_TYPES = {
    "SIGNAGE", "LAB_SUPPLY", "ITEM_STAND", "STORAGE_RACK", "MARKING_PEN",
    "WALL_ART", "LIGHT_BOX", "CLOTHES_RACK", "PUMP_DISPENSER", "FOOTREST",
    "ART_EASEL", "OFFICE_PRODUCTS", "HARDWARE_CLAMP_VISE",
    "PORTABLE_ELECTRONIC_DEVICE_STAND", "PINBOARD", "ART_SUPPLIES",
    "ROOM_DIVIDER", "ITEM_CONTAINER", "WRITING_BOARD", "BOOK_DOCUMENT_STAND",
    "PICTURE_FRAME",
}

pytestmark = pytest.mark.skipif(
    not SAMPLE.exists(), reason=f"sample file not present: {SAMPLE}"
)


@pytest.fixture(scope="module")
def parse():
    return parse_flat_file(SAMPLE, seller_id="test-seller")


@pytest.fixture(scope="module")
def rules(parse):
    return extract_rules(SAMPLE, parse.settings)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def test_extracts_all_21_product_types(rules):
    assert set(rules) == EXPECTED_PRODUCT_TYPES
    assert len(rules) == 21


def test_every_product_type_has_nonempty_rules(rules, parse):
    for pt, ptr in rules.items():
        assert isinstance(ptr, ProductTypeRules)
        assert ptr.product_type == pt
        assert ptr.template_identifier == parse.settings.template_identifier
        assert ptr.rules, f"{pt} has no rules"
        for key, rule in ptr.rules.items():
            assert isinstance(rule, AttributeRule)
            assert rule.attribute == key
            # base names: no qualifiers, no instance markers
            assert "[" not in key and "#" not in key
            assert rule.raw_name


def test_requirement_values_only_from_allowed_set(rules):
    seen = set()
    for ptr in rules.values():
        for rule in ptr.rules.values():
            assert rule.requirement in REQUIREMENT_LEVELS, (
                f"{ptr.product_type}/{rule.attribute}: {rule.requirement!r}"
            )
            seen.add(rule.requirement)
    assert seen  # at least one requirement level observed


def test_ptd_mapped_attributes_present(rules):
    # The PTD map marks the offer/fulfillment attributes for every PT.
    for ptr in rules.values():
        assert "merchant_shipping_group" in ptr.rules
        assert "fulfillment_availability.quantity" in ptr.rules
        assert ptr.rules["merchant_shipping_group"].requirement == "conditionally_required"
        assert ptr.rules["fulfillment_availability.quantity"].requirement == "conditionally_required"


def test_variation_themes_extracted(rules):
    # Every PT column in Dropdown Lists carries a variation_theme list.
    for pt, ptr in rules.items():
        assert ptr.valid_variation_themes, f"{pt} has no variation themes"
    assert "COLOR" in rules["SIGNAGE"].valid_variation_themes
    assert len(rules["SIGNAGE"].valid_variation_themes) == 313
    assert len(rules["PICTURE_FRAME"].valid_variation_themes) == 260
    # themes are de-duplicated
    for ptr in rules.values():
        themes = ptr.valid_variation_themes
        assert len(themes) == len(set(themes))


# ---------------------------------------------------------------------------
# caching through the Store
# ---------------------------------------------------------------------------
def test_get_rules_caching_roundtrip(parse, tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    try:
        calls = {"n": 0}
        real_extract = category_rules.extract_rules

        def counting_extract(path, settings):
            calls["n"] += 1
            return real_extract(path, settings)

        monkeypatch.setattr(category_rules, "extract_rules", counting_extract)

        first = get_rules(SAMPLE, parse.settings, store)
        assert calls["n"] == 1
        assert set(first) == EXPECTED_PRODUCT_TYPES

        # second call must be served from the cache, no workbook re-read
        second = get_rules(SAMPLE, parse.settings, store)
        assert calls["n"] == 1
        assert second == first  # dataclasses compare by value

        # force_refresh must re-extract
        third = get_rules(SAMPLE, parse.settings, store, force_refresh=True)
        assert calls["n"] == 2
        assert third == first
    finally:
        store.close()


def test_cached_rules_rebuild_dataclasses(parse, tmp_path):
    store = Store(tmp_path / "data")
    try:
        get_rules(SAMPLE, parse.settings, store)
        cached = get_rules(SAMPLE, parse.settings, store)
        ptr = cached["PICTURE_FRAME"]
        assert isinstance(ptr, ProductTypeRules)
        assert all(isinstance(r, AttributeRule) for r in ptr.rules.values())
        assert isinstance(ptr.valid_variation_themes, list)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------
def _with_required_item_name(rules):
    """Copy of the real rules with one synthetic required content rule so the
    missing-mandatory paths can be exercised (the real PTD map yields no
    requirement=='required' attributes)."""
    out = {}
    for pt, ptr in rules.items():
        merged = dict(ptr.rules)
        merged["item_name"] = AttributeRule(
            attribute="item_name",
            raw_name="item_name[marketplace_id=X][language_tag=en_US]#1.value",
            label="Item Name",
            requirement="required",
        )
        out[pt] = dataclasses.replace(ptr, rules=merged)
    return out


def test_real_rules_produce_no_errors_on_sample(parse, rules):
    issues = validate_input(parse.records, rules)
    assert all(i.severity is not Severity.ERROR for i in issues)


def test_missing_required_is_warning_for_partial_update(parse, rules):
    assert all(
        r.record_action is RecordAction.PARTIAL_UPDATE for r in parse.records
    )
    issues = validate_input(parse.records, _with_required_item_name(rules))
    missing = [i for i in issues if i.rule == "missing_mandatory_attribute"]
    assert missing, "expected some records to be missing item_name"
    for issue in missing:
        assert issue.severity is Severity.WARNING  # partial updates: never ERROR
        assert issue.field_name == "input:item_name"
    # records that do carry an item_name are not flagged for it
    flagged = {i.sku for i in missing}
    with_title = {r.sku for r in parse.records if r.first("item_name")}
    assert not flagged & with_title


def test_missing_required_is_error_for_full_update(parse, rules):
    victim = next(r for r in parse.records if not r.first("item_name"))
    full = dataclasses.replace(victim, record_action=RecordAction.FULL_UPDATE)
    issues = validate_input([full], _with_required_item_name(rules))
    missing = [i for i in issues if i.rule == "missing_mandatory_attribute"]
    assert missing
    assert all(i.severity is Severity.ERROR for i in missing)


def test_unknown_product_type_yields_one_info_per_record(parse, rules):
    some = parse.records[:5]
    pruned = {pt: ptr for pt, ptr in rules.items()
              if pt not in {r.product_type for r in some}}
    issues = validate_input(some, pruned)
    unknown = [i for i in issues if i.rule == "unknown_product_type"]
    assert len(unknown) == len(some)
    for issue in unknown:
        assert issue.severity is Severity.INFO
        assert issue.field_name == "input:product_type"


def test_system_and_operational_fields_never_flagged(parse, rules):
    pt = parse.records[0].product_type
    ptr = rules[pt]
    noisy = dict(ptr.rules)
    for key, raw in [
        ("::listing_status", "::listing_status"),
        ("fulfillment_availability.quantity", "fulfillment_availability#1.quantity"),
        ("merchant_shipping_group", "merchant_shipping_group[marketplace_id=X]#1.value"),
    ]:
        noisy[key] = AttributeRule(
            attribute=key, raw_name=raw, label="", requirement="required"
        )
    record = dataclasses.replace(
        parse.records[0], record_action=RecordAction.FULL_UPDATE, attributes={}
    )
    issues = validate_input([record], {pt: dataclasses.replace(ptr, rules=noisy)})
    flagged = {i.field_name for i in issues if i.rule == "missing_mandatory_attribute"}
    assert "input:::listing_status" not in flagged
    assert "input:fulfillment_availability.quantity" not in flagged
    assert "input:merchant_shipping_group" not in flagged
