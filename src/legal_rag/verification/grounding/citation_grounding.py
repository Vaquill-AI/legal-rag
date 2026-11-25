"""Flag CASE citations in an answer that are NOT grounded in the retrieved sources.

Free-text case citations a model writes in prose ("Smith v. Jones, 123 F.3d 456", "45 N.Y.2d 12", "[2019] UKSC 10") are a leading hallucination vector in legal answers.
A bracketed ``[N]`` cross-reference validator only checks index refs and their quotes, so prose citations go unchecked.

A case citation that appears in the answer but in NONE of the retrieved source passages is unverifiable against the sources and possibly fabricated.
This module surfaces those so a caller can strip or flag them.

Coverage is deliberately limited to **case reporter and neutral citations**, where absence from the sources is a high-precision fabrication signal:

  * US federal reporters   123 F.3d 456, 467 U.S. 837, 72 F. Supp. 3d 100
  * US state and regional  45 N.Y.2d 12, 88 P.3d 100, 300 So. 3d 1
  * Foreign neutral cites  [2019] UKSC 10, 2020 SCC 10

It deliberately does NOT strip statutes, regulations, public laws or Federal Register cites.
Those are often cited correctly from the model's own knowledge and are often written in a form that differs from the retrieved snippet ("16 C.F.R. Part 910" in the source and "16 C.F.R. 910.1" in the answer, both correct).
Absence from the sources is a weak, high-false-positive signal for them.

Matching is intentionally PERMISSIVE: when in doubt a citation is treated as grounded, because deleting a real citation is worse than leaving an unverifiable one for the reader to confirm.

Pure and deterministic, so it is cheap (no LLM, no network) and unit-testable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from legal_rag.verification.citations.statutes import CASE_REPORTER_PATTERN
from legal_rag.verification.citations.us_patterns import STATE_REPORTER_PATTERN

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Fields a retrieved source may carry its text under (dicts or objects).
_SOURCE_TEXT_FIELDS = ("text", "content", "snippet", "quote", "passage", "excerpt")


# Additional case-reporter families not in the statute extractor

# Foreign neutral and standard case citations that web search can surface.
# Kept tight and anchored to their court and reporter tokens.
_FOREIGN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # UK neutral: [2019] UKSC 10, [2019] EWCA Civ 123, [2020] EWHC 55
    re.compile(
        r"\[\d{4}\]\s?(?:UKSC|UKHL|UKPC|EWCA(?:\s?(?:Civ|Crim))?|EWHC|EWFC|UKUT|UKFTT)\s?\d+",
        re.IGNORECASE,
    ),
    # Canada neutral: 2020 SCC 10, 2019 FCA 42, 2021 ONCA 5
    re.compile(
        r"\b\d{4}\s?(?:SCC|FCA|FC|ONCA|BCCA|ABCA|ONSC|BCSC|QCCA)\s?\d+",
        re.IGNORECASE,
    ),
    # Canada reporter: [1999] 1 S.C.R. 497
    re.compile(r"\[\d{4}\]\s?\d?\s?S\.?C\.?R\.?\s?\d+", re.IGNORECASE),
)


def _normalize(text: str | None) -> str:
    """Collapse whitespace and drop the punctuation that varies between citation
    renderings (periods, commas) so '123 F.3d 456' and '123 F. 3d 456' compare
    against sources robustly."""
    return re.sub(r"[\s.,§]+", "", text or "").lower()


# ── Per-family extraction: (display, grounding-key) ────────────────────────
# ``display`` is the exact substring as it appears in the answer, used by the
# caller to strip it. ``key`` is the normalized core identifier searched inside
# the (normalized) source text -- pinpoints are dropped so a pin cite into a
# genuinely cited case is treated as grounded.


def _reporter_pairs(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in pattern.finditer(text):
        g = m.groupdict()
        vol = (g.get("volume") or "").strip()
        rep = re.sub(r"\s+", " ", (g.get("reporter") or "").strip())
        page = (g.get("page") or "").strip()
        if vol and rep and page:
            out.append((m.group(0).strip(), _normalize(f"{vol}{rep}{page}")))
    return out


def _foreign_pairs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pattern in _FOREIGN_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0).strip()
            out.append((raw, _normalize(raw)))
    return out


def extract_case_citations(text: str | None) -> list[str]:
    """Distinct FEDERAL reporter-style case citations in ``text`` (e.g.
    '123 F.3d 456'), in order of first appearance. Kept for callers that only
    care about federal case reporters."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for display, _key in _reporter_pairs(text, CASE_REPORTER_PATTERN):
        seen.setdefault(display, None)
    return list(seen.keys())


def _extract_case_pairs(text: str | None) -> list[tuple[str, str]]:
    """All case citations in ``text`` as (display, grounding-key), deduplicated
    by display, in order of first appearance -- federal + state/regional
    reporters and foreign neutral citations."""
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    pairs.extend(_reporter_pairs(text, CASE_REPORTER_PATTERN))
    pairs.extend(_reporter_pairs(text, STATE_REPORTER_PATTERN))
    pairs.extend(_foreign_pairs(text))
    seen: dict[str, str] = {}
    for display, key in pairs:
        seen.setdefault(display, key)
    return list(seen.items())


def _source_text(source) -> str:
    if isinstance(source, dict):
        parts = [source.get(k) for k in _SOURCE_TEXT_FIELDS]
        meta = source.get("metadata")
        if isinstance(meta, dict):
            parts.extend(meta.get(k) for k in _SOURCE_TEXT_FIELDS)
    else:
        parts = [getattr(source, k, None) for k in _SOURCE_TEXT_FIELDS]
    return " ".join(str(p) for p in parts if p)


def _ungrounded(
    answer: str | None,
    sources: list | None,
    verified_citations: Iterable[str] | None,
    extractor: Callable[[str | None], list[tuple[str, str]]],
) -> list[str]:
    pairs = extractor(answer)
    if not pairs:
        return []
    haystack = _normalize(" ".join(_source_text(s) for s in (sources or [])))
    verified = {_normalize(c) for c in (verified_citations or [])}
    return [display for display, key in pairs if key not in haystack and key not in verified]


def ungrounded_legal_citations(
    answer: str | None,
    sources: list | None,
    verified_citations: Iterable[str] | None = None,
) -> list[str]:
    """Case citations present in ``answer`` but found in NONE of ``sources`` --
    across federal + state/regional reporters and foreign neutral citations.

    Matching is punctuation-insensitive and permissive: a citation whose core
    identifier appears anywhere in the retrieved passages is treated as grounded,
    so only citations that appear nowhere are returned. ``verified_citations``
    (e.g. resolved via a citation-lookup tool in agentic mode) are treated as
    grounded even when absent from ``sources``.

    Statutes, regulations, public laws, and Federal Register cites are
    intentionally NOT covered -- see the module docstring for why string-matching
    them against sources is a high-false-positive signal.
    """
    return _ungrounded(answer, sources, verified_citations, _extract_case_pairs)


def ungrounded_case_citations(
    answer: str | None,
    sources: list | None,
    verified_citations: Iterable[str] | None = None,
) -> list[str]:
    """FEDERAL case-reporter citations in ``answer`` grounded in none of
    ``sources``. Backward-compatible narrow entry point; new callers should
    prefer :func:`ungrounded_legal_citations`, which also covers state reporters
    and foreign citations."""
    return _ungrounded(
        answer,
        sources,
        verified_citations,
        lambda t: _reporter_pairs(t or "", CASE_REPORTER_PATTERN),
    )
