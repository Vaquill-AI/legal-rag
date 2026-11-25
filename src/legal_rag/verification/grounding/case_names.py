"""Case-name query parsing.

Detects when a string reads like a case caption ("Smith v. Jones") and splits it into two cleaned party names.
Strings that merely contain "v" or "vs" between document-version labels ("draft vs final") are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VS_PATTERN = re.compile(
    r"^(.+?)\s+(?:vs?\.?|versus|v/s)\s+(.+)$",
    re.IGNORECASE,
)

# A "party" that is really a document-version label, not a litigant.
# Anchored to the WHOLE party so real names survive: "Old Colony Trust" is not "old".
_VERSION_PARTY_RE = re.compile(
    r"^(?:the|my|our|this|just|only)?\s*"
    r"(?:v\.?\s?\d+|ver(?:sion)?\.?\s*\d*|rev(?:ision)?\.?\s*\d*|"
    r"draft|final|latest|newest|previous|prior|original|current|old|new|"
    r"first|second|third|last|copy|upload|attachment|doc|document|file|policy)"
    r"(?:\s*(?:v\.?\s?\d+|\d+|version|draft|copy|upload|doc|document|file))?\s*$",
    re.IGNORECASE,
)

# Pure legal noise only. Words such as "state", "of" and "united" distinguish government parties.
_STRIP_WORDS = frozenset(
    {"and", "et", "al", "others", "another", "etc", "the", "mr", "mrs", "ms", "dr"}
)

_MIN_TOKEN_LEN = 3
_MAX_PARTY_WORDS = 6
_MAX_PARTY_CHARS = 60


@dataclass(frozen=True)
class ParsedCaseName:
    """The two cleaned party names of a case caption."""

    party_a: str
    party_b: str
    raw_query: str


def is_version_party(party: str) -> bool:
    """True when a parsed party is a document-version label, not a litigant."""
    return bool(_VERSION_PARTY_RE.match((party or "").strip()))


def _clean_party_name(name: str) -> str:
    name = re.sub(r"[.,]", "", name.lower().strip())
    tokens = [t for t in name.split() if t not in _STRIP_WORDS and len(t) >= 2]
    return " ".join(tokens)


def parse_case_name_query(query: str) -> ParsedCaseName | None:
    """Parse "X v. Y" into party names, or return ``None`` when it is not a caption.

    >>> parse_case_name_query("Marbury v. Madison").party_b
    'madison'
    """
    query = (query or "").strip()
    if not query:
        return None
    match = _VS_PATTERN.match(query)
    if not match:
        return None
    party_a = _clean_party_name(match.group(1).strip())
    party_b = _clean_party_name(match.group(2).strip())
    if len(party_a) < _MIN_TOKEN_LEN or len(party_b) < _MIN_TOKEN_LEN:
        return None
    # Long strings are questions that happen to contain "vs", not captions.
    if (
        len(party_a.split()) > _MAX_PARTY_WORDS
        or len(party_b.split()) > _MAX_PARTY_WORDS
        or len(party_a) > _MAX_PARTY_CHARS
        or len(party_b) > _MAX_PARTY_CHARS
    ):
        return None
    if is_version_party(party_a) or is_version_party(party_b):
        return None
    return ParsedCaseName(party_a=party_a, party_b=party_b, raw_query=query)
