"""Index validation, provenance and the stdlib fuzzy helpers."""

from __future__ import annotations

import logging

import pytest

from legal_rag.verification import _fuzz
from legal_rag.verification.citations import unbacked_case_citations
from legal_rag.verification.citations.validator import (
    CitationValidationStatus,
    CrossReferenceCitationValidator,
    citation_failure_ratio,
    quick_citation_validation,
    strip_fabricated_citations,
)

SOURCES = [
    {"text": "A person acting under color of state law is liable under 42 U.S.C. 1983."},
    {"text": "The privilege against self-incrimination applies to custodial interrogation."},
]


def test_quick_validation_flags_out_of_range_indices() -> None:
    ok, bad = quick_citation_validation("Claim one [1]. Claim two [2]. Claim three [5].", 2)
    assert not ok
    assert bad == [5]
    assert quick_citation_validation("Claim [1][2].", 2) == (True, [])


def test_strip_fabricated_citations_removes_only_bad_markers() -> None:
    cleaned, removed = strip_fabricated_citations("Liability applies [1] and more [7].", 2)
    assert removed == [7]
    assert "[7]" not in cleaned
    assert "[1]" in cleaned
    assert citation_failure_ratio("Liability applies [1] and more [7].", 2) == (2, 1)


def test_cross_reference_validator_finds_quote_in_source() -> None:
    validator = CrossReferenceCitationValidator()
    answer = 'The source says "a person acting under color of state law is liable" [1].'
    result = validator.validate(answer, SOURCES)
    assert result.total_citations >= 1
    assert result.invalid_citations == 0
    assert all(
        r.status is not CitationValidationStatus.INVALID_INDEX for r in result.citation_results
    )


def test_cross_reference_validator_rejects_missing_index() -> None:
    result = CrossReferenceCitationValidator().validate("A claim [9].", SOURCES)
    assert not result.is_valid
    assert result.invalid_indices == [9]


def test_provenance_flags_case_missing_from_retrieved_opinions() -> None:
    sources = [{"source_type": "case", "title": "Marbury v. Madison"}]
    answer = "See Marbury v. Madison and also Smith v. Acme Holdings."
    assert unbacked_case_citations(answer, sources) == ["Smith v. Acme Holdings"]
    assert unbacked_case_citations(answer, []) == []
    custom = [{"source_type": "opinion_text", "title": "Smith v. Acme Holdings"}]
    assert unbacked_case_citations(
        answer, custom, opinion_source_types=frozenset({"opinion_text"})
    ) == ["Marbury v. Madison"]


def test_fuzz_helpers() -> None:
    assert _fuzz.ratio("equal", "equal") == 100
    assert _fuzz.ratio("abc", "xyz") == 0
    assert (
        _fuzz.partial_ratio("state law", "a person acting under color of state law is liable")
        == 100
    )
    assert _fuzz.partial_ratio("", "x") == 0


def test_unbacked_case_citations_warns_when_there_are_no_opinions_to_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    answer = "See Smith v. Acme Holdings."
    with caplog.at_level(logging.WARNING, logger="legal_rag.verification.citations.provenance"):
        assert unbacked_case_citations(answer, []) == []
        assert unbacked_case_citations(answer, [{"source_type": "statute", "title": "x"}]) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2  # one per call
    assert all("no retrieved opinions" in r.getMessage() for r in warnings)


def test_unbacked_case_citations_is_quiet_when_opinions_were_checked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sources = [{"source_type": "case", "title": "Marbury v. Madison"}]
    with caplog.at_level(logging.WARNING, logger="legal_rag.verification.citations.provenance"):
        assert unbacked_case_citations("See Smith v. Acme Holdings.", sources) == [
            "Smith v. Acme Holdings"
        ]
    assert not caplog.records
