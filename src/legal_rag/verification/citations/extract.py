"""Extract US legal citations from text.

``extract(text)`` returns ``Citation`` objects in document order.
It recognizes US Code, CFR, federal rules, public laws, Federal Register cites, federal and state case reporter cites, and common state statute forms.
State statute forms live in ``data/state_statute_forms.json``.
Only US formats are recognized.

    >>> [c.normalized for c in extract("See 42 U.S.C. 1983 and 625 ILCS 5/11-601.")]
    ['42 U.S.C. § 1983', '625 ILCS 5/11-601']
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any

from legal_rag.verification.citations.models import Citation, CitationKind
from legal_rag.verification.citations.statutes import extract_us_citations
from legal_rag.verification.citations.us_patterns import STATE_REPORTER_PATTERN

_FEDERAL_KINDS: dict[str, CitationKind] = {
    "usc": CitationKind.USC,
    "cfr": CitationKind.CFR,
    "frcp": CitationKind.FEDERAL_RULE,
    "frcrmp": CitationKind.FEDERAL_RULE,
    "fre": CitationKind.FEDERAL_RULE,
    "frap": CitationKind.FEDERAL_RULE,
    "frbp": CitationKind.FEDERAL_RULE,
    "case": CitationKind.CASE,
    "publaw": CitationKind.PUBLIC_LAW,
    "fedreg": CitationKind.FEDERAL_REGISTER,
}

_COMPONENT_KEYS = (
    "title",
    "section",
    "subsection",
    "rule",
    "reporter",
    "volume",
    "page",
    "pinpoint",
    "congress",
    "number",
    "url",
)


@lru_cache(maxsize=1)
def _state_statute_forms() -> tuple[tuple[str, str, re.Pattern[str], str, dict[str, str]], ...]:
    raw = json.loads(
        resources.files("legal_rag.verification.citations")
        .joinpath("data/state_statute_forms.json")
        .read_text(encoding="utf-8")
    )
    return tuple(
        (
            f["id"],
            f["state"],
            re.compile(f["pattern"], re.IGNORECASE),
            f["template"],
            f.get("code_names", {}),
        )
        for f in raw["forms"]
    )


def _federal_candidates(text: str) -> list[Citation]:
    out: list[Citation] = []
    for d in extract_us_citations(text, include_duplicates=True):
        kind = _FEDERAL_KINDS.get(d["type"])
        if kind is None:
            continue
        components: dict[str, Any] = {k: d[k] for k in _COMPONENT_KEYS if k in d}
        if kind is CitationKind.FEDERAL_RULE:
            components["rule_set"] = d["type"]
        out.append(
            Citation(
                raw=d["match_text"],
                kind=kind,
                normalized=d["citation"],
                start=d["match_start"],
                end=d["match_end"],
                components=components,
            )
        )
    return out


def _state_statute_candidates(text: str) -> list[Citation]:
    out: list[Citation] = []
    for form_id, state, pattern, template, code_names in _state_statute_forms():
        for m in pattern.finditer(text):
            groups = {k: re.sub(r"\s+", " ", v).strip() for k, v in m.groupdict().items() if v}
            if "code" in groups and code_names:
                key = re.sub(r"[^a-z]", "", groups["code"].lower())
                groups["code"] = code_names.get(key, groups["code"])
            out.append(
                Citation(
                    raw=m.group(0),
                    kind=CitationKind.STATE_STATUTE,
                    normalized=template.format(**groups),
                    start=m.start(),
                    end=m.end(),
                    components={"state": state, "form": form_id, **groups},
                )
            )
    return out


def _state_reporter_candidates(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in STATE_REPORTER_PATTERN.finditer(text):
        reporter = re.sub(r"\s+", " ", m.group("reporter")).strip()
        out.append(
            Citation(
                raw=m.group(0),
                kind=CitationKind.CASE,
                normalized=f"{m.group('volume')} {reporter} {m.group('page')}",
                start=m.start(),
                end=m.end(),
                components={
                    "volume": int(m.group("volume")),
                    "reporter": reporter,
                    "page": int(m.group("page")),
                },
            )
        )
    return out


def extract(text: str | None) -> list[Citation]:
    """Return every recognized citation in ``text``, in document order.

    When two matches overlap, the one found by the more specific extractor wins
    (federal, then state statute, then state reporter).
    """
    if not text:
        return []
    accepted: list[Citation] = []
    for group in (
        _federal_candidates(text),
        _state_statute_candidates(text),
        _state_reporter_candidates(text),
    ):
        for cand in group:
            if not any(cand.start < a.end and a.start < cand.end for a in accepted):
                accepted.append(cand)
    accepted.sort(key=lambda c: (c.start, c.end))
    return accepted


def extract_unique(text: str | None) -> list[Citation]:
    """Like ``extract`` but keep only the first occurrence of each normalized citation."""
    seen: set[str] = set()
    out: list[Citation] = []
    for c in extract(text):
        if c.normalized not in seen:
            seen.add(c.normalized)
            out.append(c)
    return out
