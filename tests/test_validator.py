"""Unit tests for catalog_engine.validator — pure in-memory, no fixtures."""

from __future__ import annotations

import dataclasses

import pytest

from catalog_engine.models import (
    GeneratedContent,
    Parentage,
    ProductRecord,
    RecordAction,
    SellerConfig,
    Severity,
    SEARCH_TERMS_MAX_BYTES,
    TITLE_MAX_CHARS,
)
from catalog_engine.validator import (
    byte_len_utf8,
    char_len,
    trim_to_bytes,
    trim_to_chars,
    verify_generated,
)

RUN_ID = "run-test"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def make_record(attributes=None, brand=None, sku="SKU-1"):
    attrs = dict(attributes or {})
    if brand:
        attrs.setdefault("brand", [brand])
    return ProductRecord(
        seller_id="seller1",
        sku=sku,
        row_number=7,
        product_type="poster_frame",
        record_action=RecordAction.PARTIAL_UPDATE,
        listing_status="Active",
        parentage=Parentage.STANDALONE,
        parent_sku=None,
        variation_theme=None,
        attributes=attrs,
    )


def make_gen(**overrides):
    base = dict(
        sku="SKU-1",
        title="Poster Holder Wall Mount",
        item_highlights=["Easy to hang", "Slim profile"],
        bullets=["Easy to hang", "Slim profile design", "Fits standard posters"],
        description="A simple poster holder for home and office walls",
        search_terms="poster holder wall mount",
    )
    base.update(overrides)
    return GeneratedContent(**base)


def make_config(on_limit_violation="auto_trim"):
    return SellerConfig(seller_id="seller1", on_limit_violation=on_limit_violation)


def run(record, gen, mode="auto_trim"):
    return verify_generated(record, gen, make_config(mode), RUN_ID)


def issues_for(issues, rule=None, field=None):
    out = issues
    if rule is not None:
        out = [i for i in out if i.rule == rule]
    if field is not None:
        out = [i for i in out if i.field_name == field]
    return out


# ---------------------------------------------------------------------------
# char/byte counting
# ---------------------------------------------------------------------------
class TestCounting:
    def test_multibyte_bytes_vs_chars(self):
        assert char_len("ü") == 1 and byte_len_utf8("ü") == 2
        assert char_len("™") == 1 and byte_len_utf8("™") == 3
        assert char_len("中文") == 2 and byte_len_utf8("中文") == 6
        assert byte_len_utf8("abc") == 3 == char_len("abc")

    def test_char_len_nfc_normalizes(self):
        # 'u' + combining diaeresis composes to a single char under NFC
        assert char_len("ü") == 1


# ---------------------------------------------------------------------------
# trimming primitives
# ---------------------------------------------------------------------------
class TestTrimming:
    def test_trim_to_chars_word_boundary(self):
        assert trim_to_chars("alpha beta gamma", 12) == "alpha beta"

    def test_trim_to_chars_no_trailing_space_or_punct(self):
        out = trim_to_chars("alpha beta, gamma", 12)
        assert out == "alpha beta"
        assert not out.endswith((" ", ",", ".", "-"))

    def test_trim_to_chars_within_limit_untouched(self):
        assert trim_to_chars("short", 75) == "short"

    def test_trim_to_bytes_drops_whole_tokens(self):
        s = "aa üü 中中 bb"  # 2+1+4+1+6+1+2 = 17 bytes
        out = trim_to_bytes(s, 10)
        assert out == "aa üü"
        assert byte_len_utf8(out) <= 10

    def test_trim_to_bytes_result_within_limit(self):
        s = " ".join("tük%d" % i for i in range(80))
        out = trim_to_bytes(s, SEARCH_TERMS_MAX_BYTES)
        assert byte_len_utf8(out) <= SEARCH_TERMS_MAX_BYTES
        assert all(tok in s.split() for tok in out.split())


# ---------------------------------------------------------------------------
# limit enforcement
# ---------------------------------------------------------------------------
class TestLimits:
    def _long_title(self):
        return " ".join(["Poster"] * 14)  # 97 chars, no claims

    def test_title_auto_trimmed(self):
        record = make_record()
        gen = make_gen(title=self._long_title())
        new_gen, issues, log = run(record, gen, "auto_trim")
        assert char_len(new_gen.title) <= TITLE_MAX_CHARS
        trimmed = issues_for(issues, rule="title_max_chars")
        assert len(trimmed) == 1
        assert trimmed[0].action == "trimmed"
        assert trimmed[0].severity == Severity.WARNING
        assert any(e.category == "limit" and e.field_name == "title" for e in log)

    def test_title_flagged_not_trimmed(self):
        record = make_record()
        gen = make_gen(title=self._long_title())
        new_gen, issues, _ = run(record, gen, "flag")
        assert new_gen.title == gen.title  # untouched
        flagged = issues_for(issues, rule="title_max_chars")
        assert len(flagged) == 1
        assert flagged[0].action == "flagged"
        assert flagged[0].severity == Severity.ERROR

    def test_bullet_trailing_period_stripped(self):
        record = make_record()
        gen = make_gen(bullets=["Easy to hang.", "Slim profile design",
                                "Fits standard posters"])
        new_gen, issues, _ = run(record, gen)
        assert new_gen.bullets[0] == "Easy to hang"
        info = issues_for(issues, rule="bullet_trailing_period")
        assert len(info) == 1
        assert info[0].severity == Severity.INFO

    def test_extra_bullets_truncated_to_three(self):
        record = make_record()
        gen = make_gen(bullets=["One good point", "Two good points",
                                "Three good points", "Four good points"])
        new_gen, issues, log = run(record, gen)
        assert len(new_gen.bullets) == 3
        assert new_gen.bullets == gen.bullets[:3]
        count_issues = issues_for(issues, rule="bullet_count")
        assert len(count_issues) == 1
        assert count_issues[0].action == "trimmed"
        assert any(e.category == "limit" and "Four good points" in e.removed_text
                   for e in log)

    def test_too_few_bullets_is_error(self):
        record = make_record()
        gen = make_gen(bullets=["Only one point", "And a second"])
        new_gen, issues, _ = run(record, gen)
        count_issues = issues_for(issues, rule="bullet_count")
        assert len(count_issues) == 1
        assert count_issues[0].severity == Severity.ERROR
        assert count_issues[0].action == "flagged"
        assert new_gen.bullets == gen.bullets  # never fabricates bullets


# ---------------------------------------------------------------------------
# compliance scan — one positive + one negative per category
# ---------------------------------------------------------------------------
class TestCompliance:
    def test_medical_positive_kills_germs_removed_and_logged(self):
        record = make_record()
        gen = make_gen(bullets=["Kills germs on every surface",
                                "Slim profile design", "Fits standard posters"])
        new_gen, issues, log = run(record, gen)
        assert "kills germs" not in new_gen.bullets[0].lower()
        removed = [i for i in issues if i.action == "removed"
                   and i.field_name == "bullet_1"]
        assert removed and removed[0].severity == Severity.WARNING
        entries = [e for e in log if e.field_name == "bullet_1"]
        assert entries
        assert entries[0].removed_text.lower() == "kills germs"
        assert "'kills germs'".lower() in entries[0].reason.lower()
        assert "bullet_1" in entries[0].reason

    def test_medical_negative_stainless_steel_not_flagged(self):
        record = make_record(attributes={"material": ["Stainless Steel"]})
        gen = make_gen(bullets=["Sturdy stainless steel frame",
                                "Slim profile design", "Fits standard posters"])
        _, issues, log = run(record, gen)
        assert not [i for i in issues if i.severity != Severity.INFO]
        assert not log

    def test_pesticide_positive_repels_mosquitoes_removed(self):
        record = make_record()
        gen = make_gen(description="Repels mosquitoes effectively outdoors")
        new_gen, issues, log = run(record, gen)
        assert "repels mosquitoes" not in new_gen.description.lower()
        assert any(e.category == "pesticide" and e.field_name == "description"
                   for e in log)

    def test_pesticide_negative_insect_theme_not_flagged(self):
        record = make_record()
        gen = make_gen(description="An insect themed art print for kids rooms")
        new_gen, issues, log = run(record, gen)
        assert new_gen.description == gen.description
        assert not [e for e in log if e.category == "pesticide"]

    def test_safety_positive_fireproof_removed(self):
        record = make_record()
        gen = make_gen(description="A fireproof document holder for the office")
        new_gen, issues, log = run(record, gen)
        assert "fireproof" not in new_gen.description.lower()
        assert any(e.category == "safety" for e in log)

    def test_safety_bpa_free_kept_when_in_source(self):
        record = make_record(attributes={"special_feature": ["BPA-free construction"]})
        gen = make_gen(description="Made with BPA free materials for peace of mind")
        new_gen, issues, log = run(record, gen)
        assert "bpa free" in new_gen.description.lower()
        assert not [e for e in log if e.category == "safety"]

    def test_safety_bpa_free_removed_when_absent_from_source(self):
        record = make_record()
        gen = make_gen(description="Made with BPA free materials for peace of mind")
        new_gen, issues, log = run(record, gen)
        assert "bpa free" not in new_gen.description.lower()
        assert any(e.category == "safety" and "bpa free" in e.removed_text.lower()
                   for e in log)

    def test_promotional_positive_best_seller_removed(self):
        record = make_record()
        gen = make_gen(bullets=["A best seller in modern offices",
                                "Slim profile design", "Fits standard posters"])
        new_gen, issues, log = run(record, gen)
        assert "best seller" not in new_gen.bullets[0].lower()
        assert any(e.category == "promotional" for e in log)

    def test_promotional_negative_plain_copy_not_flagged(self):
        record = make_record()
        gen = make_gen(bullets=["Top quality finish throughout",
                                "Slim profile design", "Fits standard posters"])
        _, issues, log = run(record, gen)
        assert not [e for e in log if e.category == "promotional"]

    def test_gutted_field_kept_and_flagged(self):
        record = make_record()
        gen = make_gen(bullets=["Kills germs", "Slim profile design",
                                "Fits standard posters"])
        new_gen, issues, log = run(record, gen)
        assert new_gen.bullets[0] == "Kills germs"  # original kept
        flagged = [i for i in issues if i.field_name == "bullet_1"
                   and i.severity == Severity.ERROR and i.action == "flagged"]
        assert flagged
        assert any("gut" in e.reason for e in log if e.field_name == "bullet_1")


# ---------------------------------------------------------------------------
# hallucination / attribute-diff
# ---------------------------------------------------------------------------
class TestHallucination:
    def test_unsupported_pack_count_flagged(self):
        record = make_record(attributes={"item_package_quantity": ["1"]})
        gen = make_gen(title="Poster Holder Value 24 Pack")
        new_gen, issues, log = run(record, gen)
        assert new_gen.title == gen.title  # facts are never auto-edited
        claims = issues_for(issues, rule="unsupported_claim", field="title")
        assert len(claims) == 1
        assert claims[0].severity == Severity.ERROR
        assert claims[0].action == "flagged"
        assert any(e.category == "hallucination" and "24" in e.removed_text
                   for e in log)

    def test_supported_pack_count_passes(self):
        record = make_record(attributes={"item_package_quantity": ["24"]})
        gen = make_gen(title="Poster Holder Value 24 Pack")
        _, issues, log = run(record, gen)
        assert not issues_for(issues, rule="unsupported_claim")
        assert not [e for e in log if e.category == "hallucination"]

    def test_dimension_format_tolerance_11x17(self):
        record = make_record(
            attributes={"item_display_dimensions": ["11 x 17 inches"]})
        gen = make_gen(title="Poster Holder 11x17 Sign")
        _, issues, _ = run(record, gen)
        assert not issues_for(issues, rule="unsupported_claim")

    def test_dimension_format_tolerance_quote_vs_inch(self):
        record = make_record(attributes={"item_width": ["8.5 inch"]})
        gen = make_gen(title='Document Holder 8.5" Width')
        _, issues, _ = run(record, gen)
        assert not issues_for(issues, rule="unsupported_claim")

    def test_unsupported_dimension_pair_flagged(self):
        record = make_record(
            attributes={"item_display_dimensions": ["11 x 17 inches"]})
        gen = make_gen(title="Poster Holder 24x36 Sign")
        _, issues, log = run(record, gen)
        assert issues_for(issues, rule="unsupported_claim", field="title")
        assert any(e.category == "hallucination" for e in log)

    def test_unsupported_color_flagged_supported_color_passes(self):
        record = make_record(attributes={"color": ["Black"]})
        gen = make_gen(title="Black Poster Holder",
                       bullets=["Deep red finish", "Slim profile design",
                                "Fits standard posters"])
        _, issues, _ = run(record, gen)
        assert not issues_for(issues, rule="unsupported_claim", field="title")
        assert issues_for(issues, rule="unsupported_claim", field="bullet_1")

    def test_unsupported_material_flagged(self):
        record = make_record(attributes={"material": ["Acrylic"]})
        gen = make_gen(description="Solid oak construction built to last")
        _, issues, log = run(record, gen)
        flagged = issues_for(issues, rule="unsupported_claim", field="description")
        assert flagged
        assert any(e.removed_text.lower() == "oak" for e in log
                   if e.category == "hallucination")

    def test_sku_numbers_do_not_false_positive(self):
        record = make_record(sku="FRM-8017", attributes={})
        gen = make_gen(description="Model 8017 fits most walls")
        _, issues, _ = run(record, gen)
        assert not issues_for(issues, rule="unsupported_claim")


# ---------------------------------------------------------------------------
# search terms
# ---------------------------------------------------------------------------
class TestSearchTerms:
    def test_lowercase_dedupe_brand_removed(self):
        record = make_record(brand="M&T Displays")
        gen = make_gen(search_terms="Displays M&T Poster poster Holder holder wall")
        new_gen, issues, _ = run(record, gen)
        tokens = new_gen.search_terms.split()
        assert tokens == ["poster", "holder", "wall"]
        assert "m&t" not in tokens and "displays" not in tokens
        assert issues_for(issues, rule="search_terms_brand_removed")
        assert issues_for(issues, rule="search_terms_deduped")

    def test_byte_trim_multibyte(self):
        record = make_record()
        long_terms = " ".join("tük%d" % i for i in range(60))
        gen = make_gen(search_terms=long_terms)
        new_gen, issues, log = run(record, gen, "auto_trim")
        assert byte_len_utf8(new_gen.search_terms) <= SEARCH_TERMS_MAX_BYTES
        trimmed = issues_for(issues, rule="search_terms_max_bytes")
        assert trimmed and trimmed[0].action == "trimmed"
        assert any(e.category == "limit" and e.field_name == "search_terms"
                   for e in log)

    def test_byte_overflow_flagged_when_config_flag(self):
        record = make_record()
        long_terms = " ".join("tük%d" % i for i in range(60))
        gen = make_gen(search_terms=long_terms)
        new_gen, issues, _ = run(record, gen, "flag")
        assert byte_len_utf8(new_gen.search_terms) > SEARCH_TERMS_MAX_BYTES
        flagged = issues_for(issues, rule="search_terms_max_bytes")
        assert flagged and flagged[0].severity == Severity.ERROR


# ---------------------------------------------------------------------------
# purity: inputs never mutated
# ---------------------------------------------------------------------------
class TestNoMutation:
    def test_inputs_not_mutated(self):
        record = make_record(brand="M&T Displays")
        gen = make_gen(
            title=" ".join(["Poster"] * 14),
            bullets=["Kills germs everywhere today.", "Two good points",
                     "Three good points", "Four good points"],
            search_terms="Displays poster poster holder",
        )
        snapshot = dataclasses.replace(
            gen, item_highlights=list(gen.item_highlights),
            bullets=list(gen.bullets))
        new_gen, _, _ = run(record, gen)
        assert gen == snapshot
        assert new_gen is not gen
        assert new_gen.bullets is not gen.bullets


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
