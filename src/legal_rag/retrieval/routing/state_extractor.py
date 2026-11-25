"""Extract referenced state codes from a US legal question.

Used to filter a state-statute corpus down to the states the question actually
asks about, instead of returning NY/TX/FL chunks for a question that only
asks about CA.

Detection is regex-only (deterministic, ~0ms). Returns the set of two-letter
state codes mentioned in the question. Returns an empty set if no specific
state is referenced - in which case the caller should NOT filter (broad
retrieval is the right move when the user asks about "state law on X"
generally).
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Final


# US states + DC + PR + common territory abbreviations. This map is
# wider than the supported jurisdiction set on purpose: the extractor
# detects state mentions from arbitrary text (chat, query, doc) and must
# still recognize references to jurisdictions a corpus may not cover.
def _load_state_names() -> dict[str, str]:
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/us_states.json")
        .open("r", encoding="utf-8") as fh
    ):
        return {code.upper(): name for code, name in json.load(fh).items()}


_STATE_NAMES: Final[dict[str, str]] = _load_state_names()

# Statute-prefix abbreviations that anchor a state without using the
# state's full name. E.g. "Fla. Stat. § 732" -> FL; "Cal. Civ. Code § 1559"
# -> CA. Patterns must be specific enough to avoid matching common words -
# short abbreviations like "ic" or "orc" would hit inside "biometric" or
# "enforcement" and are intentionally omitted.
_PREFIX_TO_STATE: Final[dict[str, str]] = {
    "fla. stat.": "FL",
    "fla stat ": "FL",
    "n.y. eptl": "NY",
    "ny eptl": "NY",
    "n.y. cpl": "NY",
    "ny cpl": "NY",
    "n.y. lab. law": "NY",
    "ny lab law": "NY",
    "n.y. bus. law": "NY",
    "n.y. gen. bus. law": "NY",
    "ny gen bus law": "NY",
    "cal. civ. proc. code": "CA",
    "cal. civ. code": "CA",
    "cal. lab. code": "CA",
    "cal. bus. & prof. code": "CA",
    "cal. penal code": "CA",
    "tex. bus. & com. code": "TX",
    "tex. civ. prac. & rem. code": "TX",
    "tex. lab. code": "TX",
    "tex. penal code": "TX",
    "tex. citizens participation act": "TX",
    "del. c.": "DE",
    "delaware code": "DE",
    "delaware llc act": "DE",
    "740 ilcs": "IL",
    "740 ill. comp. stat.": "IL",
    "biometric information privacy act": "IL",  # BIPA is IL-specific
    "mass. gen. laws": "MA",
    "n.j. stat.": "NJ",
    "n.j.s.a.": "NJ",
    "ga. code ann.": "GA",
    "ohio rev. code": "OH",
    "ohio revised code": "OH",
    "wash. rev. code": "WA",
    "or. rev. stat.": "OR",
    "oregon revised statutes": "OR",
    "mich. comp. laws": "MI",
    "ariz. rev. stat.": "AZ",
    "arizona revised statutes": "AZ",
    "colo. rev. stat.": "CO",
    "colorado revised statutes": "CO",
    "minn. stat.": "MN",
    "n.c. gen. stat.": "NC",
    "tenn. code ann.": "TN",
    "va. code ann.": "VA",
    "ind. code": "IN",
}

# Pre-compiled word-boundary regex per state full name (case-insensitive).
_NAME_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (code, re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE))
    for code, name in _STATE_NAMES.items()
)

# Two-letter abbreviation regex (must be uppercase and stand alone - avoids
# matching "OR" in "memorandum or note"). Only fires for specific state-
# abbreviation patterns like "(CA)", "Cal." (already in prefix list above).
# We deliberately keep this conservative.
_ABBREV_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (code, re.compile(rf"\b{code}\b")) for code in _STATE_NAMES
)


def extract_state_codes(question: str) -> frozenset[str]:
    """Return the set of two-letter state codes referenced in the question.

    Returns an empty set when no specific state is mentioned, which signals
    the caller to leave the retrieval unfiltered.

    Examples:
        >>> extract_state_codes("Compare California and New York non-compete law")
        frozenset({"CA", "NY"})
        >>> extract_state_codes("Fla. Stat. § 732.502 will execution requirements")
        frozenset({"FL"})
        >>> extract_state_codes("What is the federal Brady rule?")
        frozenset()
    """
    q = (question or "").strip()
    if not q:
        return frozenset()
    q_lower = q.lower()
    found: set[str] = set()

    # Pass 1: full state-name matches (the most reliable signal)
    for code, pat in _NAME_PATTERNS:
        if pat.search(q):
            found.add(code)

    # Pass 2: statute-prefix abbreviations (FL via "Fla. Stat.", DE via "Del. C.")
    for prefix, code in _PREFIX_TO_STATE.items():
        if prefix in q_lower:
            found.add(code)

    # Pass 3: bare two-letter ALL-CAPS state codes (CA, NY, TX, etc.).
    # Conservative: ignore in mid-sentence - only flag when surrounded by
    # ad-hoc separators that suggest a state-tag use (parens, commas,
    # leading "in").
    bare = re.findall(
        r"(?:\bin\s+|\(|,\s+)([A-Z]{2})\b",
        q,
    )
    for code in bare:
        if code in _STATE_NAMES:
            found.add(code)

    return frozenset(found)


__all__ = ["extract_state_codes"]
