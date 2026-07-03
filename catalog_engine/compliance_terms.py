"""Prohibited-language term lists used by the validator compliance scan.

Amazon suppresses or legally exposes listings containing unsubstantiated
medical, pesticide, safety, or promotional claims. Each entry below is a
``(category, compiled_regex, reason)`` triple:

- ``category``: one of ``medical`` | ``pesticide`` | ``safety`` | ``promotional``
  (matches ``ComplianceLogEntry.category``);
- ``compiled_regex``: word-boundary, case-insensitive pattern;
- ``reason``: short human explanation suitable for the audit log, appended to
  "removed '<text>' from <field> — ".

The list is plain data on purpose: per-marketplace extensions (e.g. stricter
DE/FR wording) can append entries without touching validator logic.

Two companion sets refine behaviour for specific patterns:
- ``SOURCE_CONDITIONAL_PATTERNS``: terms that are allowed when the same claim
  already exists in the seller's source attributes (e.g. "BPA free" on a
  product whose source data says BPA free);
- ``TITLE_BULLET_ONLY_PATTERNS``: terms only prohibited in title/bullet fields
  (e.g. "warranty" is fine in the description).
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

ComplianceTerm = Tuple[str, "Pattern[str]", str]


def _rx(pattern: str) -> "Pattern[str]":
    return re.compile(pattern, re.IGNORECASE)


# Patterns needing special handling (referenced by their .pattern string).
_NON_TOXIC = r"\bnon[-\s]?toxic\b"
_BPA_FREE = r"\bbpa[-\s]?free\b"
_WARRANTY = r"\bwarrant(?:y|ies|ied)\b"

COMPLIANCE_TERMS: List[ComplianceTerm] = [
    # ------------------------------------------------------------------
    # medical — disease/health claims require FDA substantiation
    # ------------------------------------------------------------------
    ("medical", _rx(r"\bcure[sd]?\b"), "medical claim (cure) requires FDA substantiation"),
    ("medical", _rx(r"\btreat(?:s|ments?|ing|ed)?\b"), "medical claim (treatment) requires FDA substantiation"),
    ("medical", _rx(r"\bheal(?:s|ing|ed)?\b"), "medical claim (healing) requires FDA substantiation"),
    ("medical", _rx(r"\banti[-\s]?bacterial\b"), "medical claim (antibacterial) requires FDA substantiation"),
    ("medical", _rx(r"\banti[-\s]?microbial\b"), "antimicrobial claim requires FDA/EPA substantiation"),
    ("medical", _rx(r"\banti[-\s]?fungal\b"), "medical claim (antifungal) requires FDA substantiation"),
    ("medical", _rx(r"\bkills?\s+(?:germs|bacteria|viruses)\b"), "germ/virus kill claim requires FDA/EPA registration"),
    ("medical", _rx(r"\bsaniti[sz](?:es?|er|ers|ing)\b"), "sanitizing claim requires EPA registration"),
    ("medical", _rx(r"\bdisinfect(?:s|ant|ants|ing|ed)?\b"), "disinfectant claim requires EPA registration"),
    ("medical", _rx(r"\bfda[-\s]approved\b"), "'FDA approved' claim prohibited on listings"),
    ("medical", _rx(r"\btherapeutic\b"), "medical claim (therapeutic) requires FDA substantiation"),
    ("medical", _rx(r"\brelieves?\s+pain\b"), "pain-relief claim requires FDA substantiation"),
    ("medical", _rx(r"\banti[-\s]?inflammatory\b"), "medical claim (anti-inflammatory) requires FDA substantiation"),
    # ------------------------------------------------------------------
    # pesticide — EPA-regulated language (FIFRA)
    # ------------------------------------------------------------------
    ("pesticide", _rx(r"\bpesticides?\b"), "pesticide claim requires EPA registration"),
    ("pesticide", _rx(r"\binsecticides?\b"), "insecticide claim requires EPA registration"),
    ("pesticide", _rx(r"\bfungicides?\b"), "fungicide claim requires EPA registration"),
    ("pesticide", _rx(r"\brepels?\s+(?:insects?|mosquito(?:e?s)?|bugs?|pests?)\b"), "pest-repellent claim requires EPA registration"),
    ("pesticide", _rx(r"\bkills?\s+(?:insects?|pests?|mold|mildew)\b"), "pest/mold kill claim requires EPA registration"),
    ("pesticide", _rx(r"\banti[-\s]?microbial\b"), "antimicrobial is a pesticide claim per EPA"),
    # ------------------------------------------------------------------
    # safety — prohibited or source-conditional safety assurances
    # ------------------------------------------------------------------
    ("safety", _rx(r"\bfire[-\s]?proof\b"), "fireproof claim prohibited without certification"),
    ("safety", _rx(r"\bflame[-\s]?proof\b"), "flameproof claim prohibited without certification"),
    ("safety", _rx(_NON_TOXIC), "non-toxic claim not supported by source data"),
    ("safety", _rx(r"\bchild[-\s]?(?:safe|proof)\b"), "child-safety claim prohibited without certification"),
    ("safety", _rx(_BPA_FREE), "BPA-free claim not supported by source data"),
    ("safety", _rx(r"\bhypo[-\s]?allergenic\b"), "hypoallergenic claim prohibited without substantiation"),
    # ------------------------------------------------------------------
    # promotional — subjective/promotional language Amazon prohibits
    # ------------------------------------------------------------------
    ("promotional", _rx(r"\bbest[-\s]?sellers?\b"), "promotional claim (best seller) prohibited"),
    ("promotional", _rx(r"(?<![\w#])#\s?1\b"), "promotional claim (#1) prohibited"),
    ("promotional", _rx(r"\btop[-\s]?rated\b"), "promotional claim (top rated) prohibited"),
    ("promotional", _rx(r"\bhot\s+item\b"), "promotional claim (hot item) prohibited"),
    ("promotional", _rx(r"\bfree\s+shipping\b"), "shipping promotion prohibited in listing content"),
    ("promotional", _rx(r"\bguaranteed?\b"), "guarantee language prohibited in listing content"),
    ("promotional", _rx(_WARRANTY), "warranty language prohibited in title/bullets"),
    ("promotional", _rx(r"\bsale\b"), "promotional claim (sale) prohibited"),
    ("promotional", _rx(r"\bdiscount(?:s|ed)?\b"), "promotional claim (discount) prohibited"),
    ("promotional", _rx(r"\bcheapest\b"), "promotional claim (cheapest) prohibited"),
    ("promotional", _rx(r"\blimited\s+time\b"), "promotional claim (limited time) prohibited"),
    ("promotional", _rx(r"\bmoney[-\s]?back\b"), "promotional claim (money back) prohibited"),
]

# Allowed when the matched claim already appears in the source attributes.
SOURCE_CONDITIONAL_PATTERNS = frozenset({_NON_TOXIC, _BPA_FREE})

# Only enforced in title / bullet fields (allowed in description etc.).
TITLE_BULLET_ONLY_PATTERNS = frozenset({_WARRANTY})
