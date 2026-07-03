"""Regression suite for catalog_engine.parser (see INTERFACES.md).

Real-file cases run against the three Category Listing Report samples in
~/Downloads and are skipped per-file when a sample is absent, so the synthetic
cases (header parsing, RecordAction table, malformed uploads) still pass on a
machine without the fixtures.

Each sample workbook is parsed at most ONCE per test session (module-scoped
cache) because parsing a full report takes a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from catalog_engine.exceptions import FlatFileError
from catalog_engine.models import Parentage, RecordAction
from catalog_engine.parser import parse_column_header, parse_flat_file

DOWNLOADS = Path("/Users/denizolmez/Downloads")
SELLER_ID = "test_seller"

# key -> (filename, expected record count, family count, orphan-family count)
SAMPLES = {
    "06-19": ("Category+Listings+Report_06-19-2026.xlsm", 376, 60, 58),
    "06-25": ("Category+Listings+Report_06-25-2026.xlsm", 482, 27, 0),
    "07-02": ("Category+Listings+Report_07-02-2026.xlsm", 518, 107, 2),
}
SAMPLE_KEYS = list(SAMPLES)


def sample_path(key):
    return DOWNLOADS / SAMPLES[key][0]


@pytest.fixture(scope="module")
def parsed():
    """Callable fixture: parsed(key) -> ParseResult, parsing each sample file
    once per module and skipping the calling test if the file is absent."""
    cache = {}

    def get(key):
        path = sample_path(key)
        if not path.exists():
            pytest.skip("sample file not present: {}".format(path))
        if key not in cache:
            cache[key] = parse_flat_file(path, seller_id=SELLER_ID)
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# Settings decoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_settings_rows_and_marketplace(parsed, key):
    settings = parsed(key).settings
    assert settings.label_row == 4
    assert settings.attribute_row == 5
    assert settings.data_row == 7
    assert settings.marketplace_id == "US"
    assert settings.language_tag == "en_US"


# ---------------------------------------------------------------------------
# Record counts and per-record invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_record_count(parsed, key):
    assert len(parsed(key).records) == SAMPLES[key][1]


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_every_record_has_sku_and_product_type(parsed, key):
    for rec in parsed(key).records:
        assert rec.sku, "row {}: empty sku".format(rec.row_number)
        assert rec.product_type, "SKU {}: empty product_type".format(rec.sku)


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_all_record_actions_are_partial_update(parsed, key):
    actions = {rec.record_action for rec in parsed(key).records}
    assert actions == {RecordAction.PARTIAL_UPDATE}


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_skus_are_unique(parsed, key):
    skus = [rec.sku for rec in parsed(key).records]
    assert len(skus) == len(set(skus))


# ---------------------------------------------------------------------------
# Variation family integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_family_and_orphan_counts(parsed, key):
    result = parsed(key)
    _, _, expected_families, expected_orphans = SAMPLES[key]
    assert len(result.families) == expected_families
    orphans = [f for f in result.families if f.parent is None]
    assert len(orphans) == expected_orphans


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_every_child_in_exactly_one_family(parsed, key):
    result = parsed(key)
    child_skus = {
        rec.sku for rec in result.records if rec.parentage is Parentage.CHILD
    }
    placed = [rec.sku for fam in result.families for rec in fam.children]
    # no child appears twice, and every child record was placed in a family
    assert len(placed) == len(set(placed))
    assert set(placed) == child_skus


@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_family_parents_are_consistent(parsed, key):
    for fam in parsed(key).families:
        if fam.parent is not None:
            assert fam.parent.sku == fam.parent_sku


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", SAMPLE_KEYS)
def test_column_mapping(parsed, key):
    columns = parsed(key).columns
    bases = {c.base for c in columns}
    for base in (
        "contribution_sku",
        "product_type",
        "::record_action",
        "item_name",
        "product_description",
        "generic_keyword",
        "bullet_point",
    ):
        assert base in bases, "missing column base: {}".format(base)

    bullet_instances = {c.instance for c in columns if c.base == "bullet_point"}
    assert bullet_instances >= {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# parse_column_header unit tests
# ---------------------------------------------------------------------------
def test_parse_column_header_bullet_point():
    col = parse_column_header(
        3, "bullet_point[marketplace_id=X][language_tag=Y]#2.value"
    )
    assert col.base == "bullet_point"
    assert col.instance == 2
    assert col.leaf == "value"
    assert col.index == 3


def test_parse_column_header_nested_path():
    col = parse_column_header(
        0, "light_source[m=x]#1.special_features[l=y]#2.value"
    )
    assert col.base == "light_source.special_features"
    assert col.instance == 2
    assert col.leaf == "value"


def test_parse_column_header_record_action():
    col = parse_column_header(0, "::record_action")
    assert col.base == "::record_action"
    assert col.leaf == ""
    assert col.instance == 1


def test_parse_column_header_parent_sku():
    col = parse_column_header(
        0, "child_parent_sku_relationship[m=x]#1.parent_sku"
    )
    assert col.base == "child_parent_sku_relationship"
    assert col.leaf == "parent_sku"
    assert col.instance == 1


# ---------------------------------------------------------------------------
# RecordAction.from_cell mapping table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        # display labels as shown in Seller Central exports
        ("Edit (Partial Update)", RecordAction.PARTIAL_UPDATE),
        ("Update (Full Update)", RecordAction.FULL_UPDATE),
        ("Delete (Delete Product and Offer)", RecordAction.DELETE),
        # machine values
        ("partial_update", RecordAction.PARTIAL_UPDATE),
        ("full_update", RecordAction.FULL_UPDATE),
        ("delete", RecordAction.DELETE),
        # case variants
        ("PARTIAL_UPDATE", RecordAction.PARTIAL_UPDATE),
        ("Full Update", RecordAction.FULL_UPDATE),
        # blank -> Amazon treats as partial edit
        (None, RecordAction.PARTIAL_UPDATE),
        ("", RecordAction.PARTIAL_UPDATE),
        ("   ", RecordAction.PARTIAL_UPDATE),
        # junk -> UNKNOWN, never guessed
        ("banana", RecordAction.UNKNOWN),
        ("upsert", RecordAction.UNKNOWN),
    ],
)
def test_record_action_from_cell(raw, expected):
    assert RecordAction.from_cell(raw) is expected


# ---------------------------------------------------------------------------
# Malformed uploads -> FlatFileError with seller-facing text
# ---------------------------------------------------------------------------
def _save_workbook(tmp_path, filename, sheet_title=None, a1=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    if sheet_title is not None:
        ws.title = sheet_title
    if a1 is not None:
        ws["A1"] = a1
    path = tmp_path / filename
    wb.save(path)
    return path


def test_missing_file(tmp_path):
    with pytest.raises(FlatFileError, match="file not found"):
        parse_flat_file(tmp_path / "does_not_exist.xlsm", SELLER_ID)


def test_txt_file_rejected(tmp_path):
    path = tmp_path / "listing.txt"
    path.write_text("sku\tproduct_type\nABC\tCHAIR\n")
    with pytest.raises(FlatFileError, match="not an Excel workbook"):
        parse_flat_file(path, SELLER_ID)


def test_corrupt_bytes_as_xlsm(tmp_path):
    path = tmp_path / "corrupt.xlsm"
    path.write_bytes(b"this is definitely not a zip archive\x00\x01\x02")
    with pytest.raises(FlatFileError, match="corrupted or truncated"):
        parse_flat_file(path, SELLER_ID)


def test_workbook_without_template_sheet(tmp_path):
    path = _save_workbook(tmp_path, "notemplate.xlsx", sheet_title="Sheet1")
    with pytest.raises(FlatFileError, match="no 'Template' sheet"):
        parse_flat_file(path, SELLER_ID)


def test_template_a1_without_settings_prefix(tmp_path):
    path = _save_workbook(
        tmp_path, "badsettings.xlsx", sheet_title="Template", a1="hello world"
    )
    with pytest.raises(FlatFileError, match="settings="):
        parse_flat_file(path, SELLER_ID)


def test_settings_missing_label_row(tmp_path):
    a1 = (
        "settings=attributeRow=5&dataRow=7"
        "&primaryMarketplaceId=amzn1.mp.o.ATVPDKIKX0DER"
        "&contentLanguageTag=en_US"
    )
    path = _save_workbook(
        tmp_path, "nolabelrow.xlsx", sheet_title="Template", a1=a1
    )
    with pytest.raises(FlatFileError, match="labelRow"):
        parse_flat_file(path, SELLER_ID)
