"""Rule and data driven classifiers: query shape, jurisdiction, state, topic and doctrine."""

from __future__ import annotations

import asyncio

from legal_rag.providers.fakes import FakeCache
from legal_rag.retrieval.routing.doctrine_detector import (
    build_canonical_chunks,
    detect_doctrine,
    detect_doctrine_fast,
    enrich_canonical_chunks_with_opinions,
)
from legal_rag.retrieval.routing.jurisdiction_classifier import classify_jurisdiction
from legal_rag.retrieval.routing.query_classification import (
    detect_temporal_recency_gap,
    is_general_knowledge_query,
)
from legal_rag.retrieval.routing.query_classifier import QueryClassifier
from legal_rag.retrieval.routing.state_extractor import extract_state_codes
from legal_rag.retrieval.routing.topic_filters import (
    classify_substantive_area_fast,
    is_civil_procedure_question,
)


def run(coro):
    return asyncio.run(coro)


def test_general_knowledge_query_is_detected():
    assert is_general_knowledge_query("who is the president of france")


def test_temporal_recency_gap_extracts_the_year():
    assert detect_temporal_recency_gap("latest case laws of 2026") == (True, 2026)


def test_temporal_recency_gap_honours_the_cutoff_year_argument():
    query = "give me 2025 opinions"
    assert detect_temporal_recency_gap(query, cutoff_year=2025) == (False, 2025)
    assert detect_temporal_recency_gap(query, cutoff_year=2024) == (True, 2025)
    assert detect_temporal_recency_gap("case law from 2020", cutoff_year=2019) == (True, 2020)


def test_query_classifier_keeps_the_raw_query():
    result = QueryClassifier().classify("What does 42 U.S.C. 1983 provide?")
    assert result.raw_query == "What does 42 U.S.C. 1983 provide?"
    assert 0.0 <= result.confidence <= 1.0


def test_state_extractor_finds_state_codes():
    assert "TX" in extract_state_codes("Is this claim barred under Texas law?")


def test_jurisdiction_is_classified_without_a_model():
    question = "What is the limitations period in California?"
    first = run(classify_jurisdiction(question, cache=FakeCache()))
    second = run(classify_jurisdiction(question, cache=FakeCache()))
    assert first.label == second.label
    assert first.source in ("keyword", "llm", "default")


def test_topic_filters_return_an_area():
    assert isinstance(is_civil_procedure_question("how do I serve a complaint under Rule 4"), bool)
    assert classify_substantive_area_fast("tax deduction for home office").area is not None


def test_doctrine_alias_hit_builds_canonical_chunks():
    match = detect_doctrine_fast("explain qualified immunity")
    assert match is not None
    chunks = build_canonical_chunks(match)
    assert chunks
    assert all(c.id and c.text for c in chunks)


def test_doctrine_without_models_returns_none_for_unrelated_text():
    assert run(detect_doctrine("what time is it")) is None


class _Resolver:
    def __init__(self):
        self.resolved: list[str] = []

    def resolve_case(self, case_name, citation, year):
        self.resolved.append(case_name)

    def fetch_excerpt(self, case_id):
        return ""


def test_case_resolver_is_injected():
    match = detect_doctrine_fast("explain qualified immunity")
    assert match is not None
    chunks = build_canonical_chunks(match)
    resolver = _Resolver()
    count = run(enrich_canonical_chunks_with_opinions(chunks, resolver))
    assert isinstance(count, int)
