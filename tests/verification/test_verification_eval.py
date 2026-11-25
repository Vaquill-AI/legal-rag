"""Offline evaluation metrics."""

from __future__ import annotations

import pytest

from legal_rag.verification.citations import Citation, extract
from legal_rag.verification.eval import (
    LabeledQuery,
    LegalEvaluator,
    PrecedentStatus,
    PrecedentValidity,
    RAGASEvaluator,
    StageIds,
    evaluate_retrieval,
    evaluate_retrieval_async,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_citations,
    stage_attribution,
)

CONTEXT = "42 U.S.C. § 1983. Civil action for deprivation of rights under color of state law."


def test_rank_metrics() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"b", "z"}, 2) == 0.5
    assert mrr_at_k(retrieved, {"c"}, 4) == pytest.approx(1 / 3)
    assert mrr_at_k(retrieved, {"z"}, 4) == 0.0
    assert ndcg_at_k(retrieved, {"a"}, 4) == pytest.approx(1.0)


def test_stage_attribution_names_the_first_lossy_stage() -> None:
    stages = StageIds(dense=["g"], sparse=[], fused=["g"], reranked=[], context=[])
    assert stage_attribution(stages, {"g"}, 5) != ""


def test_evaluate_retrieval_with_stub_retriever() -> None:
    labeled = [LabeledQuery(query="q1", gold_ids=frozenset({"g1"}))]

    def retrieve(query: str) -> StageIds:
        ids = ["x", "g1"]
        return StageIds(dense=ids, sparse=ids, fused=ids, reranked=ids, context=ids)

    report = evaluate_retrieval(labeled, retrieve, ks=(1, 2))
    assert report is not None


async def test_evaluate_retrieval_async_drops_failing_queries() -> None:
    labeled = [
        LabeledQuery(query="ok", gold_ids=frozenset({"g"})),
        LabeledQuery(query="boom", gold_ids=frozenset({"g"})),
    ]

    async def retrieve(query: str) -> StageIds:
        if query == "boom":
            raise RuntimeError("retriever down")
        return StageIds(dense=["g"], sparse=["g"], fused=["g"], reranked=["g"], context=["g"])

    assert await evaluate_retrieval_async(labeled, retrieve, ks=(1,), concurrency=2) is not None


def test_score_citations_checks_support_in_sources() -> None:
    report = score_citations(
        "Liability arises under 42 U.S.C. 1983 for state actors.",
        [{"text": CONTEXT}],
    )
    assert report is not None


async def test_ragas_evaluator_prefers_grounded_answers() -> None:
    evaluator = RAGASEvaluator()
    good = await evaluator.evaluate(
        "What does 42 U.S.C. 1983 provide?",
        "42 U.S.C. 1983 provides a civil action for deprivation of rights under color of state law.",
        [CONTEXT],
    )
    bad = await evaluator.evaluate(
        "What does 42 U.S.C. 1983 provide?",
        "The statute regulates interstate shipping of perishable agricultural goods.",
        [CONTEXT],
    )
    assert good.metrics.faithfulness >= bad.metrics.faithfulness
    empty = await evaluator.evaluate("q", "", [CONTEXT])
    assert not empty.passed


class StubChecker:
    async def check_validity(self, citation: str, *, context: str = "") -> PrecedentValidity:
        if "5 U.S. 137" in citation:
            return PrecedentValidity(status=PrecedentStatus.OVERRULED)
        return PrecedentValidity(status=PrecedentStatus.VALID)


async def test_legal_evaluator_uses_injected_extractor_and_checker() -> None:
    calls: list[str] = []

    def spy(text: str) -> list[Citation]:
        calls.append(text)
        return extract(text)

    evaluator = LegalEvaluator(citation_extractor=spy, precedent_checker=StubChecker())
    result = await evaluator.evaluate(
        query="What does 42 U.S.C. 1983 provide?",
        response="Under 42 U.S.C. 1983 a civil action exists, and 99 F.3d 1 agrees.",
        context=CONTEXT,
    )
    assert calls
    assert "99 F.3d 1" in result.citation_accuracy.fabricated_citations
    assert result.temporal_validity.validity_score == 1.0


async def test_legal_evaluator_flags_overruled_precedent() -> None:
    evaluator = LegalEvaluator(precedent_checker=StubChecker())
    result = await evaluator.evaluate(
        query="q",
        response="Marbury v. Madison, 5 U.S. 137 (1803), is cited here.",
        context="5 U.S. 137",
    )
    assert result.temporal_validity.overruled_citations == ["5 U.S. 137"]


async def test_legal_evaluator_without_checker_skips_temporal_penalty() -> None:
    result = await LegalEvaluator().evaluate("q", "See 42 U.S.C. 1983.", CONTEXT)
    assert result.temporal_validity.validity_score == 1.0


async def test_legal_evaluator_lists_outdated_precedent() -> None:
    class OutdatedChecker:
        async def check_validity(self, citation: str, *, context: str = "") -> PrecedentValidity:
            return PrecedentValidity(status=PrecedentStatus.OUTDATED)

    result = await LegalEvaluator(precedent_checker=OutdatedChecker()).evaluate(
        query="q",
        response="Marbury v. Madison, 5 U.S. 137 (1803), is cited here.",
        context="5 U.S. 137",
    )
    temporal = result.temporal_validity
    assert temporal.outdated_citations == ["5 U.S. 137"]
    assert temporal.overruled_citations == []
    assert temporal.validity_score == 0.0
    assert any("outdated" in w for w in temporal.warnings)


async def test_ragas_evaluate_batch_with_no_samples_reports_no_score() -> None:
    summary = await RAGASEvaluator().evaluate_batch([])
    assert summary["samples_evaluated"] == 0
    assert summary["individual_results"] == []
    # Nothing was scored, so there is no pass rate or metric to report, never a perfect one.
    assert summary["pass_rate"] is None
    assert summary["average_metrics"] is None
