"""Extract explicit US Code and CFR references as ``(code, title, section)`` tuples.

Useful when a question names an exact section and a lookup should pin that section instead of relying on semantic retrieval, which tends to return adjacent sections.
``code`` is ``"usc"`` or ``"cfr"``.

    >>> extract_code_refs("Can you give the full text of 7 U.S.C. § 8307?")
    [('usc', 7, '8307')]

    >>> extract_code_refs("Compare 29 CFR 825.110 and 29 USC § 2611")
    [('cfr', 29, '825.110'), ('usc', 29, '2611')]

Section components keep their full form including subsections (``501(c)(3)``, ``825.110(b)(1)``).
The title is always an ``int``.
"""

from __future__ import annotations

import re
from typing import Final

# Case-insensitive, tolerant of punctuation variation: "7 U.S.C. § 8307", "7 USC 8307(a)".
_USC_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2})\s*"
    r"(?:U\.?\s*S\.?\s*C\.?)\s*"
    r"(?:§|sec\.?|section)?\s*"
    r"(\d+[\w()\-.]*)",
    re.IGNORECASE,
)

_CFR_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2})\s*"
    r"(?:C\.?\s*F\.?\s*R\.?)\s*"
    r"(?:§|sec\.?|section)?\s*"
    r"(\d+[\w()\-.]*)",
    re.IGNORECASE,
)


def _clean_section(s: str) -> str:
    """Strip trailing punctuation that sticks when the cite ends a sentence."""
    return s.rstrip(".,;:)")


def extract_code_refs(question: str) -> list[tuple[str, int, str]]:
    """Return all US Code and CFR references in the text, deduplicated, in order."""
    if not question:
        return []
    out: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for code, pattern in (("usc", _USC_RE), ("cfr", _CFR_RE)):
        for m in pattern.finditer(question):
            section = _clean_section(m.group(2))
            if not section:
                continue
            key = (code, int(m.group(1)), section)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def normalise_for_lookup(section: str) -> str:
    """Strip subsection brackets so ``8307(a)`` and ``8307`` both match a stored ``8307``.

    >>> normalise_for_lookup("825.110(b)(1)")
    '825.110'
    """
    return section.split("(", 1)[0] if "(" in section else section


__all__ = ["extract_code_refs", "normalise_for_lookup"]
