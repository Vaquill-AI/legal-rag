"""Detect case citations that no retrieved opinion backs.

A model can name a case that was never retrieved as an opinion.
A common route is mining a brief or article that was in context and re-citing cases out of its table of authorities as if the model had found them.

The deterministic signature is a case named in the answer that matches none of the retrieved court-opinion sources.
This module surfaces those cases as telemetry so the problem is measurable and the citations can be marked unverified.

It does NOT strip them.
A model legitimately cites well-known cases from its own training that were never retrieved, and removing those would delete good citations.
Provenance is a signal, not a verdict.
Only a verifier that reads each cited opinion can tell "real but mis-described" from "correct".
This is the cheap first layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# "United States v. Smith", "People v. Jones", "Doe v. Acme Corporation". Requires capitalized party names around a "v." so it does not
# fire on prose. Non-greedy party names, capped so a run-on sentence can't match.
_CASE_CITE_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4})\s+v\.?\s+"
    r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4})"
)

# Party tokens that carry no identifying signal, so matching on them alone would
# call any "United States v. X" backed by any other "United States v. Y".
_GENERIC_PARTY_TOKENS = frozenset(
    {"united", "states", "state", "people", "commonwealth", "in", "re", "ex", "rel", "us", "usa"}
)

# Source-type tags that mark a retrieved court opinion. Override per call for your own tags.
DEFAULT_OPINION_SOURCE_TYPES = frozenset({"case", "us_case", "opinion"})

# Capitalized words that can precede a case name in prose ("See United States v.
# Smith", "In Atkins v. ...") and would otherwise be captured as part of party 1.
_LEADING_CONNECTORS = frozenset(
    {"see", "in", "under", "also", "and", "but", "the", "cf", "accord", "e.g", "compare", "per"}
)


def _clean_party(raw: str, *, is_first: bool) -> str:
    """Trim a captured party name of leading connectors and trailing punctuation."""
    name = raw.strip().strip(".,;:()").strip()
    if is_first:
        words = name.split()
        while len(words) > 1 and words[0].lower().strip(".") in _LEADING_CONNECTORS:
            words = words[1:]
        name = " ".join(words)
    return name


def _tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower()) if len(t) > 1}


def _distinctive(name: str) -> set[str]:
    """Identifying tokens for a party: drop generic ones (United, States, People)."""
    return _tokens(name) - _GENERIC_PARTY_TOKENS


def extract_case_citations(text: str | None) -> list[str]:
    """Every "X v. Y" case name in the text, de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CASE_CITE_RE.finditer(text or ""):
        p1 = _clean_party(m.group(1), is_first=True)
        p2 = _clean_party(m.group(2), is_first=False)
        if not p1 or not p2:
            continue
        name = f"{p1} v. {p2}"
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _opinion_titles(
    sources: list[dict[str, Any]] | None, source_types: frozenset[str]
) -> list[str]:
    titles: list[str] = []
    for s in sources or []:
        st = s.get("source_type") or (s.get("metadata") or {}).get("source_type")
        if st in source_types:
            titles.append(str(s.get("title") or s.get("filename") or ""))
    return titles


def _is_backed(citation: str, opinion_titles: list[str]) -> bool:
    """A citation is backed when a retrieved opinion shares its distinctive tokens.

    Requires at least one distinctive token to overlap, so "United States v.
    Smith" is not matched by "United States v. Jones" (both lose the generic
    "united states" and keep only "smith" vs "jones").
    """
    want = _distinctive(citation)
    if not want:
        # Both parties were generic (e.g. "State v. State"); cannot attribute.
        return True
    for title in opinion_titles:
        have = _distinctive(title)
        if want & have:
            return True
    return False


def unbacked_case_citations(
    answer: str | None,
    sources: list[dict[str, Any]] | None,
    *,
    opinion_source_types: frozenset[str] = DEFAULT_OPINION_SOURCE_TYPES,
) -> list[str]:
    """Case citations in the answer that no retrieved opinion backs.

    Empty when no opinions were retrieved at all: the model is then answering
    from training, which is a different situation than laundering a citation out
    of a brief that WAS retrieved. Only meaningful when case-law retrieval ran.

    The empty result is ambiguous when no opinion was retrieved (including an
    empty ``sources``): nothing could be checked, which is not the same as "no
    unbacked citation". The return type stays ``list[str]``, so that case is
    signalled by one ``WARNING`` log record per call from this module's
    logger. Callers must test for missing opinions themselves.
    """
    titles = _opinion_titles(sources, opinion_source_types)
    if not titles:
        if answer:
            logger.warning(
                "unbacked_case_citations_skipped: no retrieved opinions to check against, "
                "an empty result means nothing was checked"
            )
        return []
    return [c for c in extract_case_citations(answer) if not _is_backed(c, titles)]
