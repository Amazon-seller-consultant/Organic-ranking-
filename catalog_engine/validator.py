"""Deterministic verification of LLM-generated listing content.

Everything here is plain code — no model calls. Checks run in the priority
order mandated by the module contract (compliance > attribute accuracy >
limits > keyword retention):

1. Compliance-language scan (``compliance_terms.py``): prohibited phrases are
   excised from the text; if removal would gut a field (<30% of the original
   length left) the original text is kept and the field is flagged for review.
2. Attribute-diff hallucination check: factual claims (numbers with units,
   colors, materials) in the generated text must appear in the source
   attributes. Unsupported claims are FLAGGED, never auto-edited.
3. Char/byte limit enforcement, honoring ``SellerConfig.on_limit_violation``
   ("auto_trim" trims deterministically; "flag" raises an ERROR issue).
4. Search-term hygiene: lowercase, dedupe, brand-word removal, byte trim.

``verify_generated`` returns a modified copy — inputs are never mutated.
"""

from __future__ import annotations

import dataclasses
import html as html_mod
import re
import unicodedata
from typing import List, Optional, Tuple

from .compliance_terms import (
    COMPLIANCE_TERMS,
    SOURCE_CONDITIONAL_PATTERNS,
    TITLE_BULLET_ONLY_PATTERNS,
)
from .models import (
    BULLET_COUNT,
    BULLET_MAX_CHARS,
    DESCRIPTION_MAX_CHARS,
    HIGHLIGHT_MAX_CHARS,
    SEARCH_TERMS_MAX_BYTES,
    TITLE_MAX_CHARS,
    ComplianceLogEntry,
    GeneratedContent,
    Issue,
    ProductRecord,
    SellerConfig,
    Severity,
)

# Fraction of the original field that must survive a compliance excision;
# below this we keep the original text and flag for human review instead.
GUT_THRESHOLD = 0.30

# Minimum fill targets: content below these leaves indexable space unused.
# INFO-level findings only — short-but-accurate copy is never blocked.
UTILIZATION_FLOORS = {
    "title": 60,          # of 75 chars
    "highlight": 90,      # of 125 chars
    "bullet": 95,         # of 125 chars
    "description": 1400,  # of 2000 chars
}
SEARCH_TERMS_FLOOR_BYTES = 200  # of 249 bytes

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*/?>|<!--.*?-->", re.DOTALL)
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,7}|#x[0-9a-fA-F]{1,6});")

# ---------------------------------------------------------------------------
# Hallucination-check vocabularies (deterministic, no LLM)
# ---------------------------------------------------------------------------
COLOR_WORDS = (
    "black", "white", "red", "blue", "green", "silver", "gold", "gray",
    "grey", "brown", "natural", "clear", "yellow", "orange", "purple",
    "pink", "beige", "ivory", "tan", "navy", "teal", "bronze", "charcoal",
    "turquoise", "burgundy", "maroon", "cream",
)

MATERIAL_WORDS = (
    "wood", "oak", "steel", "aluminum", "aluminium", "acrylic", "plastic",
    "leather", "bamboo", "glass", "metal", "canvas", "fabric", "vinyl",
    "ceramic", "marble", "brass", "copper", "iron", "pine", "walnut",
    "maple", "mahogany", "mdf", "plywood", "polyester", "cotton", "nylon",
    "silicone", "rubber", "stone", "granite", "wicker", "rattan", "velvet",
    "linen", "foam", "cork",
)

# spelling variants normalized before haystack lookups (both directions)
_WORD_EQUIV = {"grey": "gray", "aluminium": "aluminum", "transparent": "clear"}

_NUM = r"\d+(?:\.\d+)?"
_DIM_PAIR_RE = re.compile(r"\b(%s)\s*[x×]\s*(%s)\b" % (_NUM, _NUM), re.IGNORECASE)
_NUM_UNIT_RE = re.compile(
    r"\b(%s)[\s-]*(inches|inch|in|feet|foot|ft|cm|mm|lbs|lb|pounds|pound|"
    r"ounces|ounce|oz|kg|packs|pack|pieces|piece|pcs|count|ct|sheets|sheet|"
    r"gallons|gallon)\b" % _NUM,
    re.IGNORECASE,
)
_NUM_QUOTE_RE = re.compile(r"\b(%s)\s*[\"”″]" % _NUM)
_PACK_OF_RE = re.compile(r"\b(?:pack|set|box|case)s?\s+of\s+(%s)\b" % _NUM, re.IGNORECASE)
# bare numbers: only 3+ digit values are checked (small bare numbers are
# too ambiguous to flag with confidence)
_BARE_NUM_RE = re.compile(r"(?<![\w.\-])(\d{3,})(?![\w.\-])")

_COLOR_RE = re.compile(r"\b(%s)\b" % "|".join(COLOR_WORDS), re.IGNORECASE)
_MATERIAL_RE = re.compile(r"\b(%s)\b" % "|".join(MATERIAL_WORDS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Length primitives
# ---------------------------------------------------------------------------
def char_len(s: str) -> int:
    """Character count over the NFC-normalized string (what Amazon counts)."""
    return len(unicodedata.normalize("NFC", s))


def byte_len_utf8(s: str) -> int:
    """UTF-8 byte length (search-terms limit is bytes, not chars)."""
    return len(s.encode("utf-8"))


def trim_to_chars(s: str, limit: int) -> str:
    """Trim to at most ``limit`` chars at a word boundary.

    Never leaves a trailing space or dangling punctuation/separator.
    """
    s = unicodedata.normalize("NFC", s)
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if not s[limit].isspace():
        # the cut landed mid-word: drop the partial word
        idx = cut.rfind(" ")
        if idx > 0:
            cut = cut[:idx]
    return cut.rstrip(" \t,;:.!?-–—&/")


def trim_to_bytes(s: str, limit: int) -> str:
    """Trim to at most ``limit`` UTF-8 bytes by dropping whole
    space-separated tokens from the end."""
    if byte_len_utf8(s) <= limit:
        return s
    tokens = s.split(" ")
    while tokens and byte_len_utf8(" ".join(tokens)) > limit:
        tokens.pop()
    result = " ".join(tokens).rstrip()
    if not result and s:
        # single oversized token: hard cut at a valid UTF-8 boundary
        result = s.encode("utf-8")[:limit].decode("utf-8", errors="ignore").rstrip()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_words(s: str) -> str:
    """Lowercase, unify hyphens/underscores to spaces, map spelling variants."""
    s = re.sub(r"[-_]", " ", s.lower())
    s = re.sub(r"\s+", " ", s)
    for variant, canon in _WORD_EQUIV.items():
        s = re.sub(r"\b%s\b" % variant, canon, s)
    return s


def _build_haystack(record: ProductRecord) -> Tuple[str, List[float]]:
    """Lowercased searchable text + numeric values from all source
    attribute values and the SKU."""
    parts: List[str] = []
    for values in record.attributes.values():
        for v in values:
            if v:
                parts.append(str(v))
    parts.append(record.sku)
    raw = " | ".join(parts).lower()
    numbers = [float(m.group(0)) for m in re.finditer(_NUM, raw)]
    return _normalize_words(raw), numbers


def _in_haystack(phrase: str, haystack_norm: str) -> bool:
    needle = _normalize_words(phrase)
    return re.search(r"\b%s\b" % re.escape(needle), haystack_norm) is not None


def _num_supported(value: float, numbers: List[float]) -> bool:
    return any(abs(value - n) < 1e-6 for n in numbers)


def _tidy(s: str) -> str:
    """Collapse the debris left behind after excising a phrase."""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)  # ', ,' -> ','
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"^[\s,;:.!?\-–—&/]+", "", s)
    s = re.sub(r"[\s,;:\-–—&/]+$", "", s)
    return s.strip()


def _overlaps(span: Tuple[int, int], claimed: List[Tuple[int, int]]) -> bool:
    return any(s < span[1] and span[0] < e for s, e in claimed)


# ---------------------------------------------------------------------------
# Check 0: HTML stripping (Amazon rejects/renders-broken HTML in all fields)
# ---------------------------------------------------------------------------
def _strip_html_field(
    record: ProductRecord,
    field_name: str,
    text: str,
    issues: List[Issue],
) -> str:
    """Remove HTML tags and decode entities. Deterministic guarantee that no
    markup ships in any field, whatever the LLM produced."""
    if not text:
        return text
    cleaned = _HTML_TAG_RE.sub(" ", text)
    if _HTML_ENTITY_RE.search(cleaned):
        cleaned = html_mod.unescape(cleaned)
    if cleaned != text:
        cleaned = _tidy(cleaned)
        issues.append(Issue(
            sku=record.sku, field_name=field_name, rule="html_stripped",
            severity=Severity.INFO,
            message="HTML markup/entities removed from %s — Amazon fields "
                    "must be plain text" % field_name,
            action="removed", before=text, after=cleaned,
        ))
    return cleaned


# ---------------------------------------------------------------------------
# Check: Item Highlight must not repeat the item name (Amazon's own spec:
# "Do not repeat information already in the item name.")
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with", "of", "in", "on", "to",
    "by", "your", "you", "is", "are", "this", "that", "from", "at", "as",
}


def _significant_words(s: str) -> set:
    words = _normalize_words(s).split()
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _check_highlight_repeats_title(
    record: ProductRecord, title: str, highlight: str, issues: List[Issue]
) -> None:
    hl_words = _significant_words(highlight)
    if not hl_words:
        return
    title_words = _significant_words(title)
    if hl_words <= title_words:  # every meaningful word already in the title
        issues.append(Issue(
            sku=record.sku, field_name="item_highlight",
            rule="highlight_repeats_title", severity=Severity.WARNING,
            message="Item Highlight repeats words already in the title — "
                    "Amazon's guidance says this adds no new information",
            action="flagged", before=highlight, after=highlight,
        ))


# ---------------------------------------------------------------------------
# Check 5: limit utilization (INFO) — unused chars are unused SEO space
# ---------------------------------------------------------------------------
def _check_utilization(
    record: ProductRecord,
    field_name: str,
    text: str,
    floor: int,
    limit: int,
    issues: List[Issue],
    unit: str = "chars",
) -> None:
    n = byte_len_utf8(text) if unit == "bytes" else char_len(text)
    if text and n < floor:
        issues.append(Issue(
            sku=record.sku, field_name=field_name, rule="underutilized_limit",
            severity=Severity.INFO,
            message="%s uses %d of %d %s — %d %s of indexable space unused"
                    % (field_name, n, limit, unit, limit - n, unit),
            action="none", before=text, after=text,
        ))


# ---------------------------------------------------------------------------
# Check 1: compliance-language scan
# ---------------------------------------------------------------------------
def _scan_compliance_field(
    record: ProductRecord,
    field_name: str,
    text: str,
    haystack_norm: str,
    run_id: str,
    issues: List[Issue],
    log: List[ComplianceLogEntry],
) -> str:
    """Excise prohibited phrases from one field; returns the cleaned text
    (or the original text when removal would gut the field)."""
    if not text:
        return text
    title_or_bullet = field_name == "title" or field_name.startswith("bullet")
    claimed: List[Tuple[int, int]] = []
    matches: List[Tuple[int, int, str, str, str]] = []  # start,end,text,cat,reason
    for category, rx, reason in COMPLIANCE_TERMS:
        if rx.pattern in TITLE_BULLET_ONLY_PATTERNS and not title_or_bullet:
            continue
        for m in rx.finditer(text):
            if rx.pattern in SOURCE_CONDITIONAL_PATTERNS and _in_haystack(
                m.group(0), haystack_norm
            ):
                continue  # claim is backed by source data — allowed
            span = (m.start(), m.end())
            if _overlaps(span, claimed):
                continue  # same text already claimed by an earlier category
            claimed.append(span)
            matches.append((m.start(), m.end(), m.group(0), category, reason))
    if not matches:
        return text

    matches.sort()
    cleaned = text
    for start, end, _, _, _ in sorted(matches, key=lambda t: -t[0]):
        cleaned = cleaned[:start] + " " + cleaned[end:]
    cleaned = _tidy(cleaned)

    if char_len(cleaned) < GUT_THRESHOLD * char_len(text):
        # Removal would gut the field: keep original, flag for review.
        for _, _, phrase, category, reason in matches:
            issues.append(Issue(
                sku=record.sku, field_name=field_name,
                rule="%s_claim" % category, severity=Severity.ERROR,
                message="'%s' is a prohibited %s term but removing it would "
                        "gut %s — rewrite required" % (phrase, category, field_name),
                action="flagged", before=text, after=text,
            ))
            log.append(ComplianceLogEntry(
                seller_id=record.seller_id, run_id=run_id, sku=record.sku,
                field_name=field_name, category=category, removed_text=phrase,
                reason="flagged '%s' in %s — %s (removal would gut field, "
                       "review required)" % (phrase, field_name, reason),
            ))
        return text

    for _, _, phrase, category, reason in matches:
        issues.append(Issue(
            sku=record.sku, field_name=field_name,
            rule="%s_claim" % category, severity=Severity.WARNING,
            message="removed prohibited %s term '%s'" % (category, phrase),
            action="removed", before=text, after=cleaned,
        ))
        log.append(ComplianceLogEntry(
            seller_id=record.seller_id, run_id=run_id, sku=record.sku,
            field_name=field_name, category=category, removed_text=phrase,
            reason="removed '%s' from %s — %s" % (phrase, field_name, reason),
        ))
    return cleaned


# ---------------------------------------------------------------------------
# Check 2: attribute-diff hallucination check (flags, never edits)
# ---------------------------------------------------------------------------
def _check_claims_field(
    record: ProductRecord,
    field_name: str,
    text: str,
    haystack_norm: str,
    numbers: List[float],
    run_id: str,
    issues: List[Issue],
    log: List[ComplianceLogEntry],
    attested: Optional[dict] = None,  # normalized term -> attestation note
) -> None:
    if not text:
        return
    attested = attested or {}
    flagged: set = set()  # normalized claim -> flag once per field
    claimed_spans: List[Tuple[int, int]] = []

    def flag(phrase: str, kind: str) -> None:
        key = phrase.lower().strip()
        if key in flagged:
            return
        flagged.add(key)
        note = attested.get(_normalize_words(phrase))
        if note is not None:
            # Seller has attested this term: allowed, but leave an audit
            # record so the acceptance is traceable to the attestation.
            issues.append(Issue(
                sku=record.sku, field_name=field_name, rule="attested_claim",
                severity=Severity.INFO,
                message="%s claim '%s' accepted per seller attestation: %s"
                        % (kind, phrase, note),
                action="none", before=text, after=text,
            ))
            log.append(ComplianceLogEntry(
                seller_id=record.seller_id, run_id=run_id, sku=record.sku,
                field_name=field_name, category="attested", removed_text="",
                reason="allowed '%s' in %s — seller attestation: %s"
                       % (phrase, field_name, note),
            ))
            return
        issues.append(Issue(
            sku=record.sku, field_name=field_name, rule="unsupported_claim",
            severity=Severity.ERROR,
            message="%s claim '%s' not found in any source attribute — "
                    "verify or remove" % (kind, phrase),
            action="flagged", before=text, after=text,
        ))
        log.append(ComplianceLogEntry(
            seller_id=record.seller_id, run_id=run_id, sku=record.sku,
            field_name=field_name, category="hallucination", removed_text=phrase,
            reason="flagged '%s' in %s — %s claim not supported by source "
                   "attributes" % (phrase, field_name, kind),
        ))

    # (a) dimension pairs like 11x17 / 11 x 17
    for m in _DIM_PAIR_RE.finditer(text):
        claimed_spans.append(m.span())
        a, b = float(m.group(1)), float(m.group(2))
        if not (_num_supported(a, numbers) and _num_supported(b, numbers)):
            flag(m.group(0), "dimension")

    # (b) numbers with units / quote marks / "pack of N"
    for rx, kind in ((_NUM_UNIT_RE, "quantity"), (_NUM_QUOTE_RE, "dimension"),
                     (_PACK_OF_RE, "quantity")):
        for m in rx.finditer(text):
            if _overlaps(m.span(), claimed_spans):
                continue
            claimed_spans.append(m.span())
            if not _num_supported(float(m.group(1)), numbers):
                flag(m.group(0), kind)

    # (c) bare 3+ digit numbers (smaller bare numbers are ignored on purpose)
    for m in _BARE_NUM_RE.finditer(text):
        if _overlaps(m.span(), claimed_spans):
            continue
        claimed_spans.append(m.span())
        if not _num_supported(float(m.group(1)), numbers):
            flag(m.group(1), "quantity")

    # (d) colors and (e) materials
    for rx, kind in ((_COLOR_RE, "color"), (_MATERIAL_RE, "material")):
        for m in rx.finditer(text):
            if not _in_haystack(m.group(0), haystack_norm):
                flag(m.group(0), kind)


# ---------------------------------------------------------------------------
# Check 3 helper: char-limit enforcement per config
# ---------------------------------------------------------------------------
def _enforce_char_limit(
    record: ProductRecord,
    field_name: str,
    text: str,
    limit: int,
    rule: str,
    auto_trim: bool,
    run_id: str,
    issues: List[Issue],
    log: List[ComplianceLogEntry],
) -> str:
    n = char_len(text)
    if n <= limit:
        return text
    if auto_trim:
        trimmed = trim_to_chars(text, limit)
        issues.append(Issue(
            sku=record.sku, field_name=field_name, rule=rule,
            severity=Severity.WARNING,
            message="%s was %d chars (limit %d) — auto-trimmed at word "
                    "boundary" % (field_name, n, limit),
            action="trimmed", before=text, after=trimmed,
        ))
        log.append(ComplianceLogEntry(
            seller_id=record.seller_id, run_id=run_id, sku=record.sku,
            field_name=field_name, category="limit",
            removed_text=text[len(trimmed):].strip(),
            reason="trimmed %s from %d to %d chars — '%s' removed to meet "
                   "the %d-char limit" % (field_name, n, char_len(trimmed),
                                          text[len(trimmed):].strip(), limit),
        ))
        return trimmed
    issues.append(Issue(
        sku=record.sku, field_name=field_name, rule=rule,
        severity=Severity.ERROR,
        message="%s is %d chars, over the %d-char limit" % (field_name, n, limit),
        action="flagged", before=text, after=text,
    ))
    log.append(ComplianceLogEntry(
        seller_id=record.seller_id, run_id=run_id, sku=record.sku,
        field_name=field_name, category="limit", removed_text="",
        reason="flagged %s — %d chars exceeds the %d-char limit "
               "(on_limit_violation=flag)" % (field_name, n, limit),
    ))
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def verify_generated(
    record: ProductRecord,
    gen: GeneratedContent,
    config: SellerConfig,
    run_id: str,
) -> Tuple[GeneratedContent, List[Issue], List[ComplianceLogEntry]]:
    """Run all deterministic checks on ``gen``.

    Returns ``(modified_copy, issues, compliance_log)``. Neither ``record``
    nor ``gen`` is mutated.
    """
    issues: List[Issue] = []
    log: List[ComplianceLogEntry] = []
    haystack_norm, numbers = _build_haystack(record)
    auto_trim = config.on_limit_violation == "auto_trim"

    title = gen.title or ""
    highlight = gen.item_highlight or ""
    bullets = list(gen.bullets)
    description = gen.description or ""
    search_terms = gen.search_terms or ""

    # -- 0. HTML stripping (plain text guarantee) ------------------------
    def strip_html(field_name: str, text: str) -> str:
        return _strip_html_field(record, field_name, text, issues)

    title = strip_html("title", title)
    highlight = strip_html("item_highlight", highlight)
    bullets = [strip_html("bullet_%d" % (i + 1), b)
               for i, b in enumerate(bullets)]
    description = strip_html("description", description)
    search_terms = strip_html("search_terms", search_terms)

    # -- 1. compliance-language scan -----------------------------------
    def scan(field_name: str, text: str) -> str:
        return _scan_compliance_field(
            record, field_name, text, haystack_norm, run_id, issues, log)

    title = scan("title", title)
    highlight = scan("item_highlight", highlight)
    bullets = [scan("bullet_%d" % (i + 1), b) for i, b in enumerate(bullets)]
    description = scan("description", description)
    search_terms = scan("search_terms", search_terms)

    # -- 2. attribute-diff hallucination check (flags only) ------------
    attested = {
        _normalize_words(term): note
        for term, note in (config.attested_terms or {}).items()
    }
    claim_fields = [("title", title), ("item_highlight", highlight)]
    claim_fields += [("bullet_%d" % (i + 1), b) for i, b in enumerate(bullets)]
    claim_fields.append(("description", description))
    for field_name, text in claim_fields:
        _check_claims_field(
            record, field_name, text, haystack_norm, numbers, run_id, issues,
            log, attested=attested)

    # -- 3. limits (after removals) -------------------------------------
    title = _enforce_char_limit(
        record, "title", title, TITLE_MAX_CHARS, "title_max_chars",
        auto_trim, run_id, issues, log)
    highlight = _enforce_char_limit(
        record, "item_highlight", highlight, HIGHLIGHT_MAX_CHARS,
        "highlight_max_chars", auto_trim, run_id, issues, log)
    if highlight:
        _check_highlight_repeats_title(record, title, highlight, issues)

    if len(bullets) > BULLET_COUNT:
        dropped = bullets[BULLET_COUNT:]
        issues.append(Issue(
            sku=record.sku, field_name="bullets", rule="bullet_count",
            severity=Severity.WARNING,
            message="%d bullets generated — kept the first %d"
                    % (len(bullets), BULLET_COUNT),
            action="trimmed", before=" | ".join(bullets),
            after=" | ".join(bullets[:BULLET_COUNT]),
        ))
        log.append(ComplianceLogEntry(
            seller_id=record.seller_id, run_id=run_id, sku=record.sku,
            field_name="bullets", category="limit",
            removed_text=" | ".join(dropped),
            reason="trimmed bullets — dropped %d extra bullet(s) '%s' to "
                   "meet the %d-bullet limit"
                   % (len(dropped), " | ".join(dropped), BULLET_COUNT),
        ))
        bullets = bullets[:BULLET_COUNT]
    elif len(bullets) < BULLET_COUNT:
        issues.append(Issue(
            sku=record.sku, field_name="bullets", rule="bullet_count",
            severity=Severity.ERROR,
            message="only %d bullet(s) generated — exactly %d required"
                    % (len(bullets), BULLET_COUNT),
            action="flagged",
        ))
        log.append(ComplianceLogEntry(
            seller_id=record.seller_id, run_id=run_id, sku=record.sku,
            field_name="bullets", category="limit", removed_text="",
            reason="flagged bullets — only %d generated, %d required"
                   % (len(bullets), BULLET_COUNT),
        ))

    fixed_bullets: List[str] = []
    for i, b in enumerate(bullets):
        field_name = "bullet_%d" % (i + 1)
        b = _enforce_char_limit(
            record, field_name, b, BULLET_MAX_CHARS, "bullet_max_chars",
            auto_trim, run_id, issues, log)
        if b.endswith("."):
            stripped = b.rstrip(".").rstrip()
            issues.append(Issue(
                sku=record.sku, field_name=field_name,
                rule="bullet_trailing_period", severity=Severity.INFO,
                message="bullets must not end with a period — stripped",
                action="removed", before=b, after=stripped,
            ))
            b = stripped
        fixed_bullets.append(b)
    bullets = fixed_bullets

    description = _enforce_char_limit(
        record, "description", description, DESCRIPTION_MAX_CHARS,
        "description_max_chars", auto_trim, run_id, issues, log)

    # -- 4. search-term hygiene ------------------------------------------
    original_terms = search_terms
    brand_words = set((record.brand or "").lower().split())
    seen: set = set()
    kept: List[str] = []
    dropped_brand: List[str] = []
    dropped_dupes: List[str] = []
    for token in search_terms.lower().split():
        if token in brand_words:
            dropped_brand.append(token)
        elif token in seen:
            dropped_dupes.append(token)
        else:
            seen.add(token)
            kept.append(token)
    search_terms = " ".join(kept)
    if dropped_brand:
        issues.append(Issue(
            sku=record.sku, field_name="search_terms",
            rule="search_terms_brand_removed", severity=Severity.INFO,
            message="removed brand word(s) %s — Amazon ignores brand in "
                    "search terms" % ", ".join("'%s'" % t for t in sorted(set(dropped_brand))),
            action="removed", before=original_terms, after=search_terms,
        ))
    if dropped_dupes:
        issues.append(Issue(
            sku=record.sku, field_name="search_terms",
            rule="search_terms_deduped", severity=Severity.INFO,
            message="removed duplicate token(s) %s"
                    % ", ".join("'%s'" % t for t in sorted(set(dropped_dupes))),
            action="removed", before=original_terms, after=search_terms,
        ))

    n_bytes = byte_len_utf8(search_terms)
    if n_bytes > SEARCH_TERMS_MAX_BYTES:
        if auto_trim:
            trimmed = trim_to_bytes(search_terms, SEARCH_TERMS_MAX_BYTES)
            cut = search_terms[len(trimmed):].strip()
            issues.append(Issue(
                sku=record.sku, field_name="search_terms",
                rule="search_terms_max_bytes", severity=Severity.WARNING,
                message="search terms were %d bytes (limit %d) — dropped "
                        "trailing tokens" % (n_bytes, SEARCH_TERMS_MAX_BYTES),
                action="trimmed", before=search_terms, after=trimmed,
            ))
            log.append(ComplianceLogEntry(
                seller_id=record.seller_id, run_id=run_id, sku=record.sku,
                field_name="search_terms", category="limit", removed_text=cut,
                reason="trimmed search_terms from %d to %d bytes — dropped "
                       "'%s' to meet the %d-byte limit"
                       % (n_bytes, byte_len_utf8(trimmed), cut,
                          SEARCH_TERMS_MAX_BYTES),
            ))
            search_terms = trimmed
        else:
            issues.append(Issue(
                sku=record.sku, field_name="search_terms",
                rule="search_terms_max_bytes", severity=Severity.ERROR,
                message="search terms are %d bytes, over the %d-byte limit"
                        % (n_bytes, SEARCH_TERMS_MAX_BYTES),
                action="flagged", before=search_terms, after=search_terms,
            ))
            log.append(ComplianceLogEntry(
                seller_id=record.seller_id, run_id=run_id, sku=record.sku,
                field_name="search_terms", category="limit", removed_text="",
                reason="flagged search_terms — %d bytes exceeds the %d-byte "
                       "limit (on_limit_violation=flag)"
                       % (n_bytes, SEARCH_TERMS_MAX_BYTES),
            ))

    # -- 5. limit utilization (INFO only — unused space = unused SEO) ----
    _check_utilization(record, "title", title,
                       UTILIZATION_FLOORS["title"], TITLE_MAX_CHARS, issues)
    _check_utilization(record, "item_highlight", highlight,
                       UTILIZATION_FLOORS["highlight"], HIGHLIGHT_MAX_CHARS,
                       issues)
    for i, b in enumerate(bullets):
        _check_utilization(record, "bullet_%d" % (i + 1), b,
                           UTILIZATION_FLOORS["bullet"], BULLET_MAX_CHARS, issues)
    _check_utilization(record, "description", description,
                       UTILIZATION_FLOORS["description"], DESCRIPTION_MAX_CHARS,
                       issues)
    _check_utilization(record, "search_terms", search_terms,
                       SEARCH_TERMS_FLOOR_BYTES, SEARCH_TERMS_MAX_BYTES,
                       issues, unit="bytes")

    new_gen = dataclasses.replace(
        gen,
        title=title,
        item_highlight=highlight,
        bullets=bullets,
        description=description,
        search_terms=search_terms,
    )
    return new_gen, issues, log
