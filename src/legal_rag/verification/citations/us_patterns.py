"""Canonical US legal citation patterns, in one place.

Re-deriving these per call site leads to blind spots, for example a query classifier that only knew non-US formats and scored "15 U.S.C. 78j(b)" as plain prose.
Import these patterns rather than writing another copy.

The high-precision set (``US_CITATION_GUARD_PATTERNS``) is for places where a false positive costs something.
The looser set (``US_SIGNAL_PATTERNS``) is for places where a false positive is free and a false negative is expensive.
"""

from __future__ import annotations

import re

# Structural citations: a volume/title + reporter/code + number. Unambiguous:
# this shape does not occur in ordinary prose.
US_CITATION_PATTERN = re.compile(
    r"\b\d+\s+U\.?\s?S\.?\s?C\.?"  # 15 U.S.C. (United States Code)
    r"|\b\d+\s+C\.?\s?F\.?\s?R\.?"  # 29 C.F.R. (Code of Federal Regulations)
    r"|\b\d+\s+U\.?\s?S\.?\s+\d+\b"  # 347 U.S. 483 (US Reports)
    r"|\b\d+\s+F\.?\s?(?:2d|3d|4th|Supp\.?(?:\s?2d|\s?3d)?|App'?x)\.?\s+\d+"  # 500 F.3d 100
    r"|\b\d+\s+S\.?\s?Ct\.?\s+\d+"  # 123 S. Ct. 456
    r"|\b\d+\s+L\.?\s?Ed\.?(?:\s?2d)?\s+\d+",  # 98 L. Ed. 2d 100
    re.IGNORECASE,
)

# Case caption "X v. Y" / "X vs. Y" with capitalized party names (high precision:
# both sides must start uppercase, so it does not fire on "compare A v B" prose).
US_CASE_PATTERN = re.compile(r"\b[A-Z][\w.&'-]+\s+v\.?s?\.?\s+[A-Z][\w.&'-]+")

# ANY reporter citation: "<volume> <Reporter> <page>", e.g. "384 U.S. 436",
# "16 Cal.App.4th 943", "734 So. 2d 1038", "379 So. 3d 1171", "48 Cal. 3d 644".
# The reporter token must contain a period or carry a series marker (2d/3d/4th),
# so "Section 230" style references do NOT false-match.
#
# This generalizes US_CITATION_PATTERN, which enumerates only federal reporters
# (U.S., F.3d, S. Ct., L. Ed.) and therefore misses every STATE reporter such as
# "218 Cal.App.3d 914". Enumerating reporters is the wrong shape (there are hundreds), so
# match the citation's STRUCTURE instead.
US_REPORTER_CITATION_PATTERN = re.compile(
    r"\b\d{1,4}\s+"  # volume
    r"(?:[A-Z][A-Za-z.']*\.\s*)+"  # reporter tokens, at least one with a period
    r"(?:\d?(?:2d|3d|4th|5th)\s+)?"  # optional series marker
    r"\d{1,4}\b",  # page
    re.IGNORECASE,
)

# State official + regional reporters. Series number (2d/3d/4th) is required for
# the ambiguous single/short regional abbreviations (A., P., So., N.E.) so we
# never mistake ordinary prose ("5 A. 10") for a citation; the low-ambiguity
# official reporters (Cal., N.Y.) may omit it. Case-sensitive on purpose:
# reporters are capitalized, which further cuts false positives.
STATE_REPORTER_PATTERN = re.compile(
    r"""
    (?P<volume>\d+)
    \s+
    (?P<reporter>
        (?:Cal\.?(?:\s?App\.?)?(?:\s?Supp\.?)?(?:\s?(?:2d|3d|4th|5th))?)
      | (?:N\.?\s?Y\.?\s?S\.?\s?(?:2d|3d))
      | (?:N\.?\s?Y\.?(?:\s?(?:2d|3d))?)
      | (?:(?:A|P|So|N\.?\s?E|N\.?\s?W|S\.?\s?E|S\.?\s?W)\.?\s?(?:2d|3d|4th))
    )
    \s+
    (?P<page>\d+)
    """,
    re.VERBOSE,
)

# Common US federal statutes by name or acronym. Weaker than the two above: these
# are real words in other contexts ("ADA" is also a name), so they are a good
# in-domain hint but a poor gate. Callers that need precision should skip this.
US_STATUTE_PATTERN = re.compile(
    r"\b(?:FLSA|FMLA|ADA|ADEA|ERISA|HIPAA|COBRA|DMCA|CCPA|CPRA|FCRA|FDCPA|TCPA|"
    r"FOIA|RICO|OSHA|Title\s+VII|Title\s+IX|WARN\s+Act|Sherman\s+Act|Clayton\s+Act|"
    r"Lanham\s+Act|Securities\s+(?:Exchange\s+)?Act|Exchange\s+Act|Sarbanes[- ]?Oxley|"
    r"Dodd[- ]?Frank|FTC\s+Act|Bankruptcy\s+Code|Internal\s+Revenue\s+Code|"
    r"Uniform\s+Commercial\s+Code|UCC)\b",
    re.IGNORECASE,
)

# State-law citations: "Nev. Rev. Stat. § 587.877", "Cal. CIV § 1942",
# "10 Del. C. § 1", "RCW 63.14.136", "5 ILCS 80/4.36". Formats differ per state
# (some use §, some sec./art., some nothing at all), so this recognizes the
# SHAPE rather than modelling 51 grammars. Recognition only: resolving a state
# citation to a section is the job of a resolver.
US_STATE_CITATION_PATTERN = re.compile(
    r"\b(?:\d+\s+)?(?:[A-Z][A-Za-z.'’&]*\s+){1,6}?(?:§+\s*|sec\.\s*|art\.\s*)[\w.\-()/]+"
    r"|\b(?:\d+\s+)?(?:RCW|ILCS|NRS|CGS|USCA|V\.S\.A\.|Del\.\s?C\.)\s+\d[\w.\-/]*",
)

# The high-precision set: both sides of a case caption must be capitalized and a
# structural citation cannot occur in ordinary prose. Use where a false positive
# would cost something.
US_CITATION_GUARD_PATTERNS: tuple[re.Pattern[str], ...] = (
    US_CITATION_PATTERN,
    US_REPORTER_CITATION_PATTERN,
    US_CASE_PATTERN,
)

# Every US signal, including the looser ones. Use where a false positive is FREE
# and a false negative is expensive, for example a guard that decides whether to
# run retrieval at all. Wrongly treating "who is Ada Lovelace" as legal costs one
# unnecessary retrieval; wrongly treating a UCC question as trivia costs the answer.
US_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    US_CITATION_PATTERN,
    US_REPORTER_CITATION_PATTERN,
    US_CASE_PATTERN,
    US_STATUTE_PATTERN,
    US_STATE_CITATION_PATTERN,
)


def looks_like_us_citation(text: str | None) -> bool:
    """True when `text` names a US authority in an unambiguous citation form."""
    if not text:
        return False
    return any(p.search(text) for p in US_CITATION_GUARD_PATTERNS)
