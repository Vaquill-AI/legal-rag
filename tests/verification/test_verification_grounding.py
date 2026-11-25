"""Grounding checks that need no model: fabricated cases, state forms, hygiene and the guard."""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources

import pytest

from legal_rag.providers.fakes import FakeEmbedder, FakeVectorStore
from legal_rag.types import RetrievedChunk
from legal_rag.verification import CaseResolver, ResolvedCase
from legal_rag.verification.grounding import (
    HallucinationDetector,
    SourceHygiene,
    SourceTier,
    classify_source,
    detect_fabricated_cases,
    detect_state_mismatches,
    evaluate_named_case_guard,
    evaluate_named_case_guard_with_resolver,
    filter_sources,
    parse_case_name_query,
    ungrounded_case_citations,
)
from legal_rag.verification.grounding.source_hygiene import sanitize_answer_links

SOURCES = [{"case_name": "Marbury v. Madison", "citation": "5 U.S. 137"}]


class FakeCaseResolver:
    """A resolver backed by the fake embedder and vector store."""

    def __init__(self, known: list[ResolvedCase]) -> None:
        self.embedder = FakeEmbedder()
        self.store = FakeVectorStore(
            [RetrievedChunk(id=c.case_name, text=c.case_name, metadata={"case": c}) for c in known]
        )

    async def resolve(self, query: str) -> ResolvedCase | None:
        vector = await self.embedder.embed_query(query)
        needle = {t for t in query.lower().replace(".", " ").split() if len(t) > 3}
        for chunk in await self.store.search(vector, top_k=10):
            hay = chunk.text.lower()
            if needle and all(t in hay for t in needle):
                return chunk.metadata["case"]
        return None


def test_fake_case_resolver_satisfies_protocol() -> None:
    assert isinstance(FakeCaseResolver([]), CaseResolver)


def test_detect_fabricated_cases_flags_only_unretrieved_names() -> None:
    answer = (
        "See Marbury v. Madison, 5 U.S. 137 (1803); "
        "see also Smith v. Jones, 123 F.4th 456 (11th Cir. 2024)."
    )
    flagged = detect_fabricated_cases(answer, SOURCES)
    assert [f.citation for f in flagged] == ["123 F.4th 456"]
    assert detect_fabricated_cases(answer, []) == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "The court held in Smith v. Jones, 123 F.3d 456 (9th Cir. 1999) that the claim fails.",
            "Smith v. Jones",
        ),
        ("The Court held in Smith v. Jones, 123 F.3d 456 (9th Cir. 1999).", "Smith v. Jones"),
        ("See Smith v. Jones, 123 F.3d 456 (9th Cir. 1999).", "Smith v. Jones"),
        ("In Smith v. Jones, 123 F.3d 456 (9th Cir. 1999), the court held X.", "Smith v. Jones"),
        ("However Smith v. Jones, 123 F.3d 456 (9th Cir. 1999) controls.", "Smith v. Jones"),
        (
            "The court relied on United States v. Nixon, 418 U.S. 683 (1974).",
            "United States v. Nixon",
        ),
        (
            "United States v. Nixon, 418 U.S. 683 (1974) is a leading case.",
            "United States v. Nixon",
        ),
        (
            "Under Matter of Acme v. Beta, 100 N.E.2d 50 (1950) the rule holds.",
            "Matter of Acme v. Beta",
        ),
        (
            "The court held in In re Gault v. State, 387 U.S. 1 (1967) that ...",
            "In re Gault v. State",
        ),
        ("In re Gault v. State, 387 U.S. 1 (1967) is a leading case.", "In re Gault v. State"),
        (
            "As held in Ex parte Young v. Doe, 209 U.S. 123 (1908), suits lie.",
            "Ex parte Young v. Doe",
        ),
        (
            "Doe ex rel. Roe v. Board of Education, 1 F.3d 2 (1990).",
            "Doe ex rel. Roe v. Board of Education",
        ),
    ],
)
def test_detect_fabricated_cases_captures_only_the_case_name(answer: str, expected: str) -> None:
    [flagged] = detect_fabricated_cases(answer, SOURCES)
    assert flagged.case_name == expected


def test_prose_lead_does_not_hide_a_grounded_case_or_a_fabricated_one() -> None:
    answer = "The court held in Smith v. Jones, 123 F.3d 456 (9th Cir. 1999) that the claim fails."
    grounded = [*SOURCES, {"case_name": "Smith v. Jones", "citation": "123 F.3d 456"}]
    # The lead prose used to make the name unmatchable, so a retrieved case was flagged.
    assert detect_fabricated_cases(answer, grounded) == []
    # A case that is not in the sources is still flagged, under its clean name.
    other = "The court held in Doe v. Roe, 1 F.3d 2 (1990) that ..."
    assert [f.case_name for f in detect_fabricated_cases(other, grounded)] == ["Doe v. Roe"]


def test_stripped_lead_word_that_is_part_of_a_real_name_is_not_flagged() -> None:
    sources = [{"case_name": "Under Armour Inc. v. Smith", "citation": "1 F.3d 1"}]
    answer = "Under Armour Inc. v. Smith, 1 F.3d 1 (4th Cir. 2000) is on point."
    assert detect_fabricated_cases(answer, sources) == []


def test_in_re_party_name_survives_normalisation() -> None:
    sources = [{"case_name": "In re Gault v. State", "citation": "387 U.S. 1"}]
    assert detect_fabricated_cases("In re Gault v. State, 387 U.S. 1 (1967).", sources) == []
    assert [
        f.case_name
        for f in detect_fabricated_cases("In re Winship v. State, 397 U.S. 358.", sources)
    ] == ["In re Winship v. State"]


def test_state_forms_load_from_data_file() -> None:
    raw = json.loads(
        resources.files("legal_rag.verification.citations")
        .joinpath("data/state_citation_forms.json")
        .read_text(encoding="utf-8")
    )
    assert raw["state_prefixes"]["cal."] == "ca"
    assert set(raw["regional_reporters"]["so. 2d"]) == {"al", "fl", "la", "ms"}


def test_state_mismatch_detection() -> None:
    answer = "Relief is available under Cal. Civ. Code § 1942 for tenants."
    [mismatch] = detect_state_mismatches(answer, "ny")
    assert mismatch.detected_state == "ca"
    assert detect_state_mismatches(answer, "ca") == []
    assert detect_state_mismatches(answer, "federal") == []


def test_ungrounded_case_citations() -> None:
    sources = [{"text": "The Court decided 410 U.S. 113 in 1973."}]
    answer = "See 410 U.S. 113 and 999 F.3d 1."
    assert ungrounded_case_citations(answer, sources) == ["999 F.3d 1"]


def test_parse_case_name_query_rejects_version_labels() -> None:
    parsed = parse_case_name_query("Marbury v. Madison")
    assert parsed is not None
    assert (parsed.party_a, parsed.party_b) == ("marbury", "madison")
    assert parse_case_name_query("just v2 vs latest") is None
    assert parse_case_name_query("") is None


def test_named_case_guard_declines_unretrieved_case() -> None:
    result = evaluate_named_case_guard("Marbury v. Madison", [{"title": "Unrelated memo"}])
    assert result.should_decline
    assert result.message is not None
    assert "I " not in (result.message or "").split(".")[0].split()[:1]
    matched = evaluate_named_case_guard("Marbury v. Madison", [{"case_name": "Marbury v. Madison"}])
    assert not matched.should_decline
    assert matched.matched_source_index == 1
    assert not evaluate_named_case_guard("What is the statute of limitations?", []).should_decline


async def test_named_case_guard_consults_resolver_before_declining() -> None:
    known = ResolvedCase(case_name="Marbury v. Madison", citation="5 U.S. 137")
    resolver = FakeCaseResolver([known])
    result = await evaluate_named_case_guard_with_resolver("Marbury v. Madison", [], resolver)
    assert not result.should_decline
    other = await evaluate_named_case_guard_with_resolver("Smith v. Acme", [], FakeCaseResolver([]))
    assert other.should_decline


def test_source_hygiene_defaults_and_custom_lists() -> None:
    assert classify_source("https://www.law.cornell.edu/uscode") is SourceTier.AGGREGATOR
    assert classify_source("https://www.uscourts.gov/rules") is SourceTier.OFFICIAL
    assert classify_source("https://www.instagram.com/x") is SourceTier.JUNK
    assert classify_source("https://goto") is SourceTier.JUNK
    hygiene = SourceHygiene(paywalled=["paywall.example.net"], blocked=["rival.example.org"])
    assert hygiene.classify("https://paywall.example.net/a") is SourceTier.PAYWALLED
    assert hygiene.should_drop("https://www.rival.example.org/x")
    kept, dropped = filter_sources([{"url": "https://youtu.be/abc"}, {"title": "own upload"}])
    assert (len(kept), dropped) == (1, 1)
    text, removed = sanitize_answer_links(
        "See [Doc](https://youtu.be/abc) and [Act](https://www.congress.gov/x)."
    )
    assert removed == 1
    assert "[Act](https://www.congress.gov/x)" in text


def test_hallucination_detector_scores_supported_text_higher() -> None:
    detector = HallucinationDetector()
    context = (
        "The statute requires employers to pay overtime at one and one half times the regular rate."
    )
    grounded = detector.detect(
        "Employers must pay overtime at one and one half times the regular rate.", context
    )
    ungrounded = detector.detect(
        "Employers must provide free lunches to all employees every single day.", context
    )
    assert grounded.hallucination_score <= ungrounded.hallucination_score


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "The right is rooted in the Constitution. See Hartwell v. Blackstone Holdings, 1 U.S. 1.",
            "Hartwell v. Blackstone Holdings",
        ),
        (
            "That is settled law. In Hartwell v. Blackstone Holdings, 1 U.S. 1, the court agreed.",
            "Hartwell v. Blackstone Holdings",
        ),
        # An abbreviation inside a party name is not a sentence boundary.
        (
            "See Acme Inc. Holdings v. Beta Corp., 1 U.S. 1.",
            "Acme Inc. Holdings v. Beta Corp.",
        ),
    ],
)
def test_a_sentence_boundary_before_a_signal_word_is_not_part_of_the_name(
    answer: str, expected: str
) -> None:
    [flagged] = detect_fabricated_cases(answer, SOURCES)
    assert flagged.case_name == expected


def test_detect_fabricated_cases_warns_when_there_are_no_sources_to_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.WARNING, logger="legal_rag.verification.grounding.fabricated_case"
    ):
        assert detect_fabricated_cases("See Hartwell v. Blackstone Holdings, 1 U.S. 1.", []) == []
    [record] = caplog.records  # exactly one warning for the call
    assert "no sources to check against" in record.getMessage()


def test_detect_fabricated_cases_is_quiet_when_sources_were_checked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.WARNING, logger="legal_rag.verification.grounding.fabricated_case"
    ):
        assert detect_fabricated_cases("See Marbury v. Madison, 5 U.S. 137.", SOURCES) == []
    assert not caplog.records


class RecordingResolver:
    """Resolves nothing and remembers every probe it was sent."""

    def __init__(self) -> None:
        self.probes: list[str] = []

    async def resolve(self, query: str) -> ResolvedCase | None:
        self.probes.append(query)
        return None


@pytest.mark.parametrize(
    ("query", "name_probe"),
    [
        # Case and spelling are kept, and the words around the name are cut.
        ("Marbury v. Madison", "Marbury v. Madison"),
        (
            "What did the court decide in Marbury v. Madison, 5 U.S. 137 about review?",
            "Marbury v. Madison",
        ),
        (
            "Summarize Hartwell v. Blackstone Holdings. Then compare it.",
            "Hartwell v. Blackstone Holdings",
        ),
        ("Tell me the holding of Smith v. Acme Corp., please", "Smith v. Acme Corp."),
        # Without capitals the edges are found by dropping prose words.
        (
            "what did the court hold in hartwell v. blackstone holdings",
            "hartwell v. blackstone holdings",
        ),
    ],
)
async def test_resolver_probe_is_the_case_name_as_written(query: str, name_probe: str) -> None:
    resolver = RecordingResolver()
    result = await evaluate_named_case_guard_with_resolver(query, [], resolver)
    assert result.should_decline
    assert resolver.probes[-1] == name_probe
    # Reporter citations are still sent verbatim, ahead of the name.
    assert resolver.probes[:-1] == list(result.citations)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The U.S. Supreme Court holds that contracts bind the parties.",
            ["The U.S. Supreme Court holds that contracts bind the parties."],
        ),
        (
            "Hartwell v. Blackstone Holdings, Inc. was decided in 1987 by the court.",
            ["Hartwell v. Blackstone Holdings, Inc. was decided in 1987 by the court."],
        ),
        (
            "Smith Co. v. Jones Corp. was argued before the 9th Cir. and the Dist. Ct. in No. 12.",
            [
                "Smith Co. v. Jones Corp. was argued before the 9th Cir. and the Dist. Ct. in No. 12."
            ],
        ),
        (
            "The N.Y. and Cal. statutes are different from the rule in D.C. today.",
            ["The N.Y. and Cal. statutes are different from the rule in D.C. today."],
        ),
        (
            "Jurisdiction is governed by 28 U.S.C. § 1331 and § 12.5 applies too.",
            ["Jurisdiction is governed by 28 U.S.C. § 1331 and § 12.5 applies too."],
        ),
        # Control: two real sentences still split.
        (
            "The court ruled that the contract is void. The statute requires a written notice.",
            [
                "The court ruled that the contract is void.",
                "The statute requires a written notice.",
            ],
        ),
    ],
)
def test_claim_splitting_keeps_legal_abbreviations_whole(text: str, expected: list[str]) -> None:
    assert HallucinationDetector()._extract_claims(text) == expected


def test_detector_counts_a_partially_grounded_claim_as_supported() -> None:
    """The documented rule: only a claim below ``min_overlap_ratio`` counts against the score.

    A claim with 0.4 to under 0.8 of its key terms in the context is ``partial``.
    ``detect`` and ``check`` score it as supported, so one grounded claim plus one
    partial claim scores 1.0 while ``supported_claims`` counts only the grounded one.
    """
    context = "Courts may strike down statutes that conflict with the Constitution. Employers must pay overtime."
    answer = (
        "Courts may strike down statutes that conflict with the Constitution. "
        "Employers must pay overtime pensions lunches vacation bonuses."
    )
    detector = HallucinationDetector()
    levels = [c.grounding_level.value for c in detector.detect(answer, context).claim_analyses]
    assert levels == ["grounded", "partial"]
    checked = asyncio.run(detector.check(answer, [{"text": context}]))
    assert (checked.total_claims, checked.supported_claims) == (2, 1)
    assert (checked.score, checked.verified, checked.unsupported_claims) == (1.0, True, [])
    # A claim below the overlap floor does lower the score.
    worse = asyncio.run(
        detector.check(answer.replace("overtime pensions", "zebras quantum"), [{"text": context}])
    )
    assert worse.score == 0.5
    assert not worse.verified
