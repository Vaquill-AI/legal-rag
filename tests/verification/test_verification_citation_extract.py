"""Citation extraction: statutes, rules, case reporters and state forms."""

from __future__ import annotations

import pytest

from legal_rag.verification.citations import (
    CitationKind,
    extract,
    extract_code_refs,
    extract_unique,
    looks_like_us_citation,
    normalise_for_lookup,
)
from legal_rag.verification.citations.statutes import extract_us_citations


def test_extract_returns_citation_objects_with_offsets() -> None:
    text = "Liability arises under 42 U.S.C. 1983 for state actors."
    [cite] = extract(text)
    assert cite.raw == "42 U.S.C. 1983"
    assert cite.kind is CitationKind.USC
    assert cite.normalized == "42 U.S.C. § 1983"
    assert text[cite.start : cite.end] == cite.raw
    assert cite.components["title"] == 42
    assert cite.components["section"] == "1983"


def test_extract_state_statute_form() -> None:
    [cite] = extract("Speeding is governed by 625 ILCS 5/11-601 in Illinois.")
    assert cite.kind is CitationKind.STATE_STATUTE
    assert cite.normalized == "625 ILCS 5/11-601"
    assert cite.components["state"] == "il"


def test_extract_mixed_in_document_order() -> None:
    text = (
        "See 29 C.F.R. § 825.110(b)(1); Marbury v. Madison, 5 U.S. 137 (1803); "
        "Fed. R. Civ. P. 12(b)(6); Pub. L. No. 117-103; 89 Fed. Reg. 12,345; RCW 9A.36.011."
    )
    cites = extract(text)
    assert [c.kind for c in cites] == [
        CitationKind.CFR,
        CitationKind.CASE,
        CitationKind.FEDERAL_RULE,
        CitationKind.PUBLIC_LAW,
        CitationKind.FEDERAL_REGISTER,
        CitationKind.STATE_STATUTE,
    ]
    assert [c.start for c in cites] == sorted(c.start for c in cites)
    assert all(text[c.start : c.end] == c.raw for c in cites)
    assert cites[1].normalized == "5 U.S. 137"
    assert cites[4].raw == "89 Fed. Reg. 12,345"


def test_extract_state_reporter_and_code_names() -> None:
    cites = extract("Compare 45 N.Y.2d 12 with Cal. Civ. Code § 1942 and Tex. Penal Code § 1.01.")
    assert [c.normalized for c in cites] == [
        "45 N.Y.2d 12",
        "Cal. Civ. Code § 1942",
        "Tex. Penal Code § 1.01",
    ]
    assert cites[0].kind is CitationKind.CASE


def test_overlapping_matches_keep_the_specific_extractor() -> None:
    cites = extract("42 U.S.C. 1983")
    assert len(cites) == 1
    assert cites[0].kind is CitationKind.USC


def test_extract_unique_drops_repeats() -> None:
    text = "42 U.S.C. 1983, then 42 USC § 1983 again."
    assert len(extract(text)) == 2
    assert len(extract_unique(text)) == 1


@pytest.mark.parametrize("text", ["", None, "No citations here, only prose."])
def test_extract_empty_and_prose(text: str | None) -> None:
    assert extract(text) == []


def test_indian_formats_are_not_in_the_default_set() -> None:
    assert extract("As held in AIR 2024 SC 123 and (2024) 5 SCC 123.") == []


def test_extract_us_citations_dicts_carry_official_links() -> None:
    [usc] = extract_us_citations("28 U.S.C. § 1331(a)")
    assert usc["type"] == "usc"
    assert usc["subsection"] == "(a)"
    assert usc["url"].startswith("https://uscode.house.gov/")
    assert "verified" not in usc
    [cfr] = extract_us_citations("17 C.F.R. § 240.10b-5")
    assert cfr["url"] == "https://www.ecfr.gov/current/title-17/section-240.10b-5"


def test_code_refs_and_lookup_normalisation() -> None:
    refs = extract_code_refs("Compare 29 CFR 825.110 and 29 USC § 2611, then 29 USC § 2611 again.")
    assert refs == [("usc", 29, "2611"), ("cfr", 29, "825.110")]
    assert normalise_for_lookup("825.110(b)(1)") == "825.110"
    assert normalise_for_lookup("1983") == "1983"


def test_looks_like_us_citation() -> None:
    assert looks_like_us_citation("What does 15 U.S.C. 78j(b) cover?")
    assert looks_like_us_citation("Marbury v. Madison")
    assert not looks_like_us_citation("who is ada lovelace")
    assert not looks_like_us_citation(None)
