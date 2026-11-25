"""US court case-number extraction and normalization.

Case numbers are a reliable signal that a set of documents mixes several distinct
proceedings. Parties do not disambiguate (the same entity often recurs across related
cases), so case numbers are extracted from document text and normalized to one
canonical key, which lets the same case written different ways group together.
Examples that must all map to the SAME key ``CA-2025-2720``:

    2025-CA-2720        (dashed)
    25CA002720AX        (Florida uniform case number)
    42-2025-CA-002720   (county-prefixed, zero-padded)

Coverage spans common formats: Florida uniform, county-prefixed, year-first,
type-first, district-court-of-appeal, federal district (``2:25-cv-01234``), the
California Superior Court current format (``25STUD01234``) and legacy format
(``BC691405``), and the statewide uniform format (``30-2025-01234567-CU-BC-CJC``).

Typical uses: warn when one collection mixes several cases, populate a structured
``case_number`` for case-aware retrieval, and normalize a model-extracted caption
number.
"""

from __future__ import annotations

import re

# Candidate tokens in free text. Broad on purpose (recall); _normalize() is the
# precision gate. Covers year-first, type-first, county-prefixed, the Florida
# uniform format, district-court-of-appeal numbers, federal district numbers,
# the California (LA Superior Court) current format, and the statewide/Orange
# County uniform format.
_CASE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}:\d{2}-[A-Za-z]{2}-\d{3,7}"  # federal district: 2:25-cv-01234
    r"|\d{2,3}-(?:19|20)\d{2}-\d{6,8}-[A-Z]{2}-[A-Z]{2}-[A-Z]{2,3}"  # CA uniform: 30-2025-01234567-CU-BC-CJC
    r"|\d{1,3}-(?:19|20)\d{2}-[A-Z]{2}-\d{2,7}"  # county-prefixed: 42-2025-CA-002720
    r"|(?:19|20)\d{2}[-\s]?[A-Z]{2}[-\s]?\d{2,7}"  # year-first: 2025-CA-2720
    r"|[A-Z]{2,3}[-\s]?(?:19|20)\d{2}[-\s]?\d{3,7}(?:\.\d{1,3})?"  # type-first: JU-2023-000406.01
    r"|\d{2}[A-Za-z]{4}\d{5,7}"  # CA (LA Superior Court) current: 25STUD01234
    r"|\d{2}[A-Z]{2}\d{5,7}[A-Z]{0,4}"  # FL uniform: 25CA002720AX
    r"|\dD(?:19|20)\d{2}[-\s]?\d{3,6}"  # DCA: 5D2026-0361
    r")\b"
)

# Two-letter tokens that look like case-type codes but aren't.
_TYPE_BLOCKLIST = frozenset({"NO", "PM", "AM", "ID", "PO", "US", "PP", "EX"})

# Federal district-court case-type codes (the alpha block in 2:25-cv-01234).
_FEDERAL_TYPES = frozenset({"CV", "CR", "MD", "MC", "BK", "AP", "MJ"})


def _key(ctype: str, year: str, num: str) -> str | None:
    if ctype in _TYPE_BLOCKLIST:
        return None
    return f"{ctype}-{year}-{int(num)}"  # int() strips leading zeros


def normalize_case_number(raw: str | None) -> str | None:
    """Normalize a raw case number (any common US format) to a canonical key.

    Canonical key is ``TYPE-YYYY-N`` (leading zeros stripped). Returns None if it
    doesn't look like a case number.
    """
    if not raw:
        return None
    s = re.sub(r"\s+", "", raw).upper()
    s = re.sub(r"\.\d{1,3}$", "", s)  # drop ".01" sub-case suffix

    # Federal district court: 2:25-cv-01234[-ABC]. District prefix + 2-digit year
    # + case-type code + sequence (judge initials, if any, ignored).
    m = re.match(r"^\d{1,2}:(\d{2})-?([A-Z]{2})-?0*(\d+)", s)
    if m and m.group(2) in _FEDERAL_TYPES:
        yy = int(m.group(1))
        year = 2000 + yy if yy < 70 else 1900 + yy
        return _key(m.group(2), str(year), m.group(3))

    # California / statewide uniform: 30-2025-01234567-CU-BC-CJC. Court code +
    # year + 7-8 digit sequence + category codes. Must run before the leading
    # numeric-prefix strip below, which would otherwise mangle it.
    m = re.match(r"^\d{2,3}-((?:19|20)\d{2})-0*(\d+)-([A-Z]{2})-[A-Z]{2}", s)
    if m:
        return _key(m.group(3), m.group(1), m.group(2))

    # Florida uniform case number: YY + TYPE + zero-padded seq + alpha suffix
    m = re.match(r"^(\d{2})([A-Z]{2})0*(\d+)[A-Z]{0,4}$", s)
    if m:
        yy = int(m.group(1))
        year = 2000 + yy if yy < 70 else 1900 + yy
        return _key(m.group(2), str(year), m.group(3))

    # California (LA Superior Court) current format: YY + 4-letter district+type
    # block + sequence, e.g. 25STUD01234 (Stanley Mosk, Unlawful Detainer). The
    # full 4-letter block is kept in the key so cases that share a type but sit
    # in different districts (25STUD.. vs 25VAUD..) stay distinct.
    m = re.match(r"^(\d{2})([A-Z]{4})0*(\d+)$", s)
    if m:
        yy = int(m.group(1))
        year = 2000 + yy if yy < 70 else 1900 + yy
        return _key(m.group(2), str(year), m.group(3))

    # Legacy LA Superior Court: district letter + category (C civil/criminal or
    # Q family/probate) + 6 digits, no embedded year (e.g. BC691405). Yearless,
    # so the key keeps the verbatim prefix + sequence.
    m = re.match(r"^([A-Z][CQ])(\d{6})$", s)
    if m and m.group(1) not in _TYPE_BLOCKLIST:
        return f"{m.group(1)}-{int(m.group(2))}"

    # District Court of Appeal: 5D2026-0361
    m = re.match(r"^(\d)D-?((?:19|20)\d{2})-?0*(\d+)$", s)
    if m:
        return _key(f"{m.group(1)}DCA", m.group(2), m.group(3))

    # Strip a leading county/district numeric prefix: 42-..., 08-...
    s = re.sub(r"^\d{1,3}-", "", s)

    # Year-first: 2025-CA-2720 / 2024CA001356 / 2024-DR001937
    m = re.match(r"^((?:19|20)\d{2})-?([A-Z]{2,3})-?0*(\d+)$", s)
    if m:
        return _key(m.group(2), m.group(1), m.group(3))

    # Type-first: JU-2023-000406 / CV2023194 / CL-2025-1067
    m = re.match(r"^([A-Z]{2,3})-?((?:19|20)\d{2})-?0*(\d+)$", s)
    if m:
        return _key(m.group(1), m.group(2), m.group(3))

    return None


def extract_case_numbers(text: str | None) -> set[str]:
    """Return the set of normalized case-number keys found in ``text``."""
    if not text:
        return set()
    out: set[str] = set()
    for raw in _CASE_RE.findall(text):
        key = normalize_case_number(raw)
        if key:
            out.add(key)
    return out


def distinct_cases_in_sources(sources: list[dict]) -> set[str]:
    """Normalized case numbers referenced across a list of retrieved sources.

    Looks at each source's ``text`` (the retrieved chunk content). Returns the
    set of distinct case keys, len >= 2 means the answer drew on more than one
    case and is at risk of conflating them.
    """
    found: set[str] = set()
    for s in sources or []:
        found |= extract_case_numbers(s.get("text"))
    return found
