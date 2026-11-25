"""US legal citation extractor and canonical URL builder.

Parses US legal text (USC, CFR, federal rules of procedure and evidence, public
laws, Federal Register cites and federal reporter case citations) and emits
structured citation dicts.

Each result is a dict such as::

    {
      "citation": "28 U.S.C. § 1331",
      "type": "usc",   # usc | cfr | frcp | fre | frap | frcrmp | frbp | case | publaw | fedreg
      "title": 28,
      "section": "1331",
      "subsection": "a",
      "url": "https://uscode.house.gov/...",
      "match_text": "28 U.S.C. § 1331(a)",
      "match_start": 10,
      "match_end": 30,
    }

Extraction is regex-only and stateless.
A match means the text has the SHAPE of a citation, never that the cited
section exists.
Confirming existence is the job of a resolver you inject elsewhere.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

# ------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------

# 28 U.S.C. § 1331        -> title=28, section="1331"
# 28 U.S.C. § 1331(a)     -> + subsection="a"
# 28 U.S.C. § 1331(a)(1)  -> + subsection="a(1)" (kept inline)
# Tolerates "U.S.C.A.", "USC", " U S C " spaced
USC_PATTERN = re.compile(
    r"""
    (?P<title>\d+)
    \s*
    U\.?\s*S\.?\s*C\.?(?:A\.?|S\.?)?
    \s*
    (?:§§?|sec(?:tion)?s?\.?)?
    \s*
    (?P<section>\d+(?:[a-z]+)?(?:[\-\.]\w+)*)
    (?P<subsection>(?:\([a-zA-Z0-9]+\))*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 17 C.F.R. § 240.10b-5
# 26 C.F.R. § 1.401(k)-1
CFR_PATTERN = re.compile(
    r"""
    (?P<title>\d+)
    \s*
    C\.?\s*F\.?\s*R\.?
    \s*
    (?:§§?|sec(?:tion)?s?\.?)?
    \s*
    (?P<section>\d+(?:\.\w[\w\-]*)*)
    (?P<subsection>(?:\([a-zA-Z0-9]+\))*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Federal Rule X(b)(6): Civil/Evidence/Appellate/Criminal
FED_RULE_PATTERN = re.compile(
    r"""
    \b
    (?:
        (?P<frcp>(?:FRCP|Fed\.?\s*R\.?\s*Civ\.?\s*P\.?))
      | (?P<frcrmp>(?:FRCrP|Fed\.?\s*R\.?\s*Crim\.?\s*P\.?))
      | (?P<fre>(?:FRE|Fed\.?\s*R\.?\s*Evid\.?))
      | (?P<frap>(?:FRAP|Fed\.?\s*R\.?\s*App\.?\s*P\.?))
      | (?P<frbp>(?:FRBP|Fed\.?\s*R\.?\s*Bankr\.?\s*P\.?))
    )
    \s*
    (?P<rule>\d+(?:\.\d+)?)
    (?P<sub>(?:\([a-zA-Z0-9]+\))*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 556 U.S. 662  /  410 U.S. 113  /  354 F.3d 1218  /  72 F. Supp. 3d 100
# Optional pin: "556 U.S. 662, 678".
CASE_REPORTER_PATTERN = re.compile(
    r"""
    (?P<volume>\d+)
    \s+
    (?P<reporter>(?:U\.?\s?S\.?|S\.?\s?Ct\.?|F\.\s*\d?[a-z]*|F\.?\s*Supp\.?(?:\s*\d?[a-z]*)?|F\.\s*App'?x?|L\.?\s?Ed\.?(?:\s*\d?[a-z]*)?|F\.\s?R\.?\s?D\.?|B\.\s?R\.?))
    \s+
    (?P<page>\d+)
    (?:\s*,\s*(?P<pinpoint>\d+))?
    """,
    re.VERBOSE,
)

# Pub. L. No. 117-103   /   PL 117-103
PUBLAW_PATTERN = re.compile(
    r"""
    \b
    (?:Pub\.?\s*L\.?(?:\s*No\.?)?|PL)
    \s+
    (?P<congress>\d+)
    \s*[-\u2013]\s*
    (?P<number>\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 89 Fed. Reg. 12,345
FEDREG_PATTERN = re.compile(
    r"""
    \b
    (?P<volume>\d+)
    \s*
    Fed\.?\s*Reg\.?
    \s*
    (?P<page>\d{1,3}(?:,\d{3})+|\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Federal-rule kind -> official uscourts.gov landing-page slug. The publisher prints each
# set as one page, so the link targets the set and not the individual rule.
_FED_RULE_PAGE_SLUG: dict[str, str] = {
    "frcp": "federal-rules-civil-procedure",
    "frcrmp": "federal-rules-criminal-procedure",
    "fre": "federal-rules-evidence",
    "frap": "federal-rules-appellate-procedure",
    "frbp": "federal-rules-bankruptcy-procedure",
}
_FED_RULES_INDEX = "https://www.uscourts.gov/forms-rules/current-rules-practice-procedure"


def _normalize_reporter(raw: str) -> str:
    """Map noisy reporter strings to the standard abbreviation."""
    n = raw.strip()
    n = n.replace("U. S.", "U.S.").replace("U S", "U.S.")
    n = re.sub(r"\s+", " ", n)
    # Common normalizations
    if re.match(r"U\.?\s*S\.?$", n, re.IGNORECASE):
        return "U.S."
    if re.match(r"S\.?\s*Ct\.?$", n, re.IGNORECASE):
        return "S. Ct."
    if re.match(r"F\.\s*2d$", n, re.IGNORECASE):
        return "F.2d"
    if re.match(r"F\.\s*3d$", n, re.IGNORECASE):
        return "F.3d"
    if re.match(r"F\.\s*4th$", n, re.IGNORECASE):
        return "F.4th"
    if re.match(r"F\.?\s*Supp\.?\s*3d$", n, re.IGNORECASE):
        return "F. Supp. 3d"
    if re.match(r"F\.?\s*Supp\.?\s*2d$", n, re.IGNORECASE):
        return "F. Supp. 2d"
    if re.match(r"F\.?\s*Supp\.?$", n, re.IGNORECASE):
        return "F. Supp."
    if re.match(r"F\.?\s*App'?x?$", n, re.IGNORECASE):
        return "F. App'x"
    if re.match(r"L\.?\s*Ed\.?\s*2d$", n, re.IGNORECASE):
        return "L. Ed. 2d"
    if re.match(r"L\.?\s*Ed\.?$", n, re.IGNORECASE):
        return "L. Ed."
    if re.match(r"F\.?\s?R\.?\s?D\.?$", n, re.IGNORECASE):
        return "F.R.D."
    return n


def _build_usc_url(title: str, section: str, subsection: str) -> str:
    """Official US Code browse URL for a section. ``subsection`` is accepted for symmetry."""
    del subsection
    return "https://uscode.house.gov/view.xhtml?req=" + urllib.parse.quote(
        f"granuleid:USC-prelim-title{title}-section{section}"
    )


def _build_cfr_url(title: str, section: str) -> str:
    """Official eCFR URL for a section."""
    return f"https://www.ecfr.gov/current/title-{title}/section-{section.strip()}"


def _build_rule_url(kind: str, rule: str) -> str:
    """Official landing page for the rule SET. `rule` is deliberately unused.

    Kept in the signature because every call site passes it and because the
    per-rule anchor comes back the moment uscourts.gov or our own browse
    surface exposes one.
    """
    del rule
    slug = _FED_RULE_PAGE_SLUG.get(kind.lower(), _FED_RULE_PAGE_SLUG["frcp"])
    return f"{_FED_RULES_INDEX}/{slug}"


def _build_publaw_url(congress: str, number: str) -> str:
    return f"https://www.congress.gov/{congress}/plaws/publ{number}/PLAW-{congress}publ{number}.pdf"


def _build_fedreg_url(volume: str, page: str) -> str:
    page_clean = page.replace(",", "")
    return f"https://www.federalregister.gov/citation/{volume}-FR-{page_clean}"


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def _make_citation(**kw: Any) -> dict[str, Any]:
    """Drop None values to keep the dict clean."""
    return {k: v for k, v in kw.items() if v is not None}


def extract_us_citations(
    text: str,
    *,
    include_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Extract every recognized US legal citation in the text.

    Returns a list of citation dicts in document order.

    By default the same authority cited twice in the same paragraph
    yields only one dict (handy for the right-rail Citations panel).
    Pass `include_duplicates=True` to get every occurrence, required
    by the DOCX exporter (every span needs its own hyperlink) and by
    the Bluebook formatter (Id. detection works on consecutive cites).
    """
    if not text:
        return []

    seen: set[tuple] = set()
    seen_positions: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []

    def _push(c: dict[str, Any]) -> None:
        # Always drop exact same-position duplicates (one regex
        # accidentally matching what another already grabbed).
        pos_key = (c.get("match_start"), c.get("match_end"))
        if pos_key[0] is not None and pos_key in seen_positions:
            return
        if pos_key[0] is not None:
            seen_positions.add(pos_key)
        if include_duplicates:
            out.append(c)
            return
        key = (
            c.get("type"),
            c.get("title"),
            c.get("section"),
            c.get("rule"),
            c.get("citation"),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(c)

    # USC
    for m in USC_PATTERN.finditer(text):
        title = m.group("title")
        section = m.group("section")
        subsection = m.group("subsection") or ""
        canon = f"{title} U.S.C. § {section}{subsection}"
        _push(
            _make_citation(
                citation=canon,
                type="usc",
                title=int(title),
                section=section,
                subsection=subsection or None,
                url=_build_usc_url(title, section, subsection),
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    # CFR
    for m in CFR_PATTERN.finditer(text):
        title = m.group("title")
        section = m.group("section")
        subsection = m.group("subsection") or ""
        canon = f"{title} C.F.R. § {section}{subsection}"
        _push(
            _make_citation(
                citation=canon,
                type="cfr",
                title=int(title),
                section=section,
                subsection=subsection or None,
                url=_build_cfr_url(title, section),
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    # Federal Rules
    for m in FED_RULE_PATTERN.finditer(text):
        kind = next(
            (k for k in ("frcp", "frcrmp", "fre", "frap", "frbp") if m.group(k)),
            None,
        )
        if not kind:
            continue
        rule = m.group("rule")
        sub = m.group("sub") or ""
        kind_label = {
            "frcp": "Fed. R. Civ. P.",
            "frcrmp": "Fed. R. Crim. P.",
            "fre": "Fed. R. Evid.",
            "frap": "Fed. R. App. P.",
            "frbp": "Fed. R. Bankr. P.",
        }[kind]
        canon = f"{kind_label} {rule}{sub}"
        _push(
            _make_citation(
                citation=canon,
                type=kind,
                rule=rule,
                subsection=sub or None,
                url=_build_rule_url(kind, rule),
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    # Federal cases (skip "X U.S.C." which the USC regex already grabbed)
    for m in CASE_REPORTER_PATTERN.finditer(text):
        rep = m.group("reporter")
        # Avoid accidentally re-matching "X U.S.C." -- if next char after
        # the reporter is "C" or ".C", it is USC, skip.
        end = m.end("reporter")
        tail = text[end : end + 4]
        if re.match(r"\.?\s*C\.?", tail, re.IGNORECASE):
            continue
        volume = m.group("volume")
        page = m.group("page")
        pin = m.group("pinpoint")
        rep_norm = _normalize_reporter(rep)
        canon = f"{volume} {rep_norm} {page}"
        if pin:
            canon += f", {pin}"
        _push(
            _make_citation(
                citation=canon,
                type="case",
                reporter=rep_norm,
                volume=int(volume),
                page=int(page),
                pinpoint=int(pin) if pin else None,
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    # Public Laws
    for m in PUBLAW_PATTERN.finditer(text):
        congress = m.group("congress")
        number = m.group("number")
        canon = f"Pub. L. No. {congress}-{number}"
        _push(
            _make_citation(
                citation=canon,
                type="publaw",
                congress=int(congress),
                number=int(number),
                url=_build_publaw_url(congress, number),
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    # Federal Register
    for m in FEDREG_PATTERN.finditer(text):
        volume = m.group("volume")
        page = m.group("page")
        canon = f"{volume} Fed. Reg. {page}"
        _push(
            _make_citation(
                citation=canon,
                type="fedreg",
                volume=int(volume),
                page=int(page.replace(",", "")),
                url=_build_fedreg_url(volume, page),
                match_text=m.group(0),
                match_start=m.start(),
                match_end=m.end(),
            )
        )

    return out


__all__ = [
    "CASE_REPORTER_PATTERN",
    "CFR_PATTERN",
    "FEDREG_PATTERN",
    "FED_RULE_PATTERN",
    "PUBLAW_PATTERN",
    "USC_PATTERN",
    "extract_us_citations",
]
