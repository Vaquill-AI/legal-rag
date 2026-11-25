"""Rerank stage: one test per behavior, with synthetic chunks and fake providers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline import ComplexityCaps, QueryContext, UsageLedger, get_tier
from legal_rag.pipeline.stages.rerank import (
    RerankStageConfig,
    apply_case_cap,
    apply_document_cap,
    apply_relevance_gate,
    compute_two_bucket_budgets,
    detect_contradiction_hints,
    dynamic_top_k,
    extract_case_names,
    gate_score,
    rerank_stage,
    resolve_document_cap,
    resolve_gate_threshold,
    resolve_rerank_top_k,
    split_generated_for_grading,
)
from legal_rag.providers.fakes import FakeReranker
from legal_rag.retrieval.rerank import BaseReranker, LostInMiddleReorderer, ScoreBasedReranker
from legal_rag.retrieval.rerank.base import RerankOutcome
from legal_rag.retrieval.strategies.crag import CRAGConfig
from legal_rag.types import Message, RerankResult, RetrievedChunk

STANDARD = get_tier("standard")
VOCAB = [f"w{i}" for i in range(30)]


def chunk(cid: str, score: float = 0.0, doc: str | None = None, **meta: Any) -> RetrievedChunk:
    text = meta.pop("text", cid)
    metadata = dict(meta)
    if doc is not None:
        metadata["document_id"] = doc
    return RetrievedChunk(id=cid, text=text, score=score, metadata=metadata)


def ids(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [c.id for c in chunks]


def make_ctx(chunks: Sequence[RetrievedChunk], **kwargs: Any) -> QueryContext:
    kwargs.setdefault("query", "what does the statute require")
    return QueryContext(retrieved=list(chunks), **kwargs)


def make_cfg(tier: Any = None, **kwargs: Any) -> RerankStageConfig:
    """Config with the optional filters off, so each test switches on only what it checks."""
    base: dict[str, Any] = {"use_crag": False, "use_diversity": False, "use_reorder": False}
    base.update(kwargs)
    return RerankStageConfig(tier=tier or STANDARD, **base)


class ScriptedReranker:
    """A provider-level scorer that returns a fixed score per document text."""

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.queries: list[str] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankResult]:
        self.queries.append(query)
        scored = [RerankResult(i, self.scores.get(d, 0.0)) for i, d in enumerate(documents)]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k] if top_k else scored


class RaisingReranker(BaseReranker):
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None, **kwargs: object
    ) -> RerankOutcome:
        raise RuntimeError("scorer is down")


class RaisingProvider:
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankResult]:
        raise TimeoutError("provider timed out")


class GraderLLM:
    """Grades a chunk 0 when its text holds the marker, else 1."""

    def __init__(self, marker: str = "IRRELEVANT") -> None:
        self.marker = marker
        self.prompts: list[str] = []

    async def generate(self, messages: Sequence[Message], **kwargs: Any) -> str:
        return ""

    async def generate_structured(
        self, messages: Sequence[Message], schema: type, *, max_tokens: int | None = None
    ) -> Any:
        prompt = messages[0].content
        self.prompts.append(prompt)
        return schema(score=0.0 if self.marker in prompt else 1.0)


def scored(prefix: str, scores: Sequence[float], **meta: Any) -> list[RetrievedChunk]:
    return [chunk(f"{prefix}{i}", s, **meta) for i, s in enumerate(scores)]


# ---------------------------------------------------------------------------
# Two-bucket budgets (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("top_k", "pool", "expected"),
    [
        (10, 20, (4, 6)),
        (20, 100, (8, 12)),
        (6, 3, (2, 4)),
        (5, 1, (1, 4)),
        (6, 0, (0, 6)),
        (2, 10, (2, 1)),
    ],
)
def test_two_bucket_budgets_are_pure_and_bounded(top_k, pool, expected):
    assert compute_two_bucket_budgets(top_k, pool) == expected


def test_two_bucket_budgets_never_exceed_the_pool_and_keep_a_reference_slot():
    for top_k in range(1, 45):
        for pool in range(50):
            pinned, reference = compute_two_bucket_budgets(top_k, pool)
            assert pinned <= pool
            assert reference >= 1
            assert pinned <= max(2, top_k // 2 + 1)


# ---------------------------------------------------------------------------
# Dynamic top_k and complexity caps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(15, 18), (2, 5), (100, 40), (None, 10), (0, 10), (-3, 10), (True, 10), ("7", 10)],
)
def test_dynamic_top_k_honors_a_valid_requested_count(requested, expected):
    assert dynamic_top_k(10, requested) == expected


def test_complexity_caps_come_from_tier_data():
    deep = get_tier("deep")
    assert resolve_rerank_top_k(STANDARD, complexity="simple") == 5
    assert resolve_rerank_top_k(STANDARD, complexity="moderate") == 10
    assert resolve_rerank_top_k(deep, complexity="simple") == 8
    assert resolve_rerank_top_k(deep, complexity="moderate") == 15
    assert resolve_rerank_top_k(deep, complexity="complex") == 20
    assert resolve_rerank_top_k(deep, complexity="never-heard-of-it") == 20


def test_case_law_and_single_document_floors_raise_the_cap():
    assert resolve_rerank_top_k(STANDARD, complexity="simple", pool_has_case_law=True) == 10
    got = resolve_rerank_top_k(
        STANDARD, requested_count=15, complexity="simple", pool_has_case_law=True
    )
    assert got == 15
    task = "case_law_research"
    assert resolve_rerank_top_k(STANDARD, complexity="moderate", legal_task_type=task) == 10
    # A simple case-law query is not raised: the floor applies from "moderate" up.
    assert resolve_rerank_top_k(STANDARD, complexity="simple", legal_task_type=task) == 5
    assert resolve_rerank_top_k(STANDARD, complexity="simple", pinned_document_count=1) == 12
    assert resolve_rerank_top_k(STANDARD, complexity="simple", pinned_document_count=2) == 5


async def test_stage_applies_dynamic_top_k_from_the_requested_count():
    query = " ".join(VOCAB)
    chunks = [chunk(f"c{i}", text=" ".join(VOCAB[: 30 - i])) for i in range(30)]
    cfg = make_cfg()

    base = await rerank_stage(make_ctx(chunks, query=query), reranker=FakeReranker(), config=cfg)
    asked = await rerank_stage(
        make_ctx(chunks, query=query, metrics={"requested_count": 6}),
        reranker=FakeReranker(),
        config=cfg,
    )
    assert len(base.reranked) == STANDARD.top_k_rerank == 10
    assert len(asked.reranked) == 9
    assert ids(asked.reranked)[:3] == ["c0", "c1", "c2"]
    assert ids(asked.extra["alternative_chunks"])[:2] == ["c9", "c10"]


async def test_zero_complexity_cap_skips_reranking_and_passes_chunks_through():
    tier = replace(STANDARD, complexity=ComplexityCaps.from_mapping({"rerank": {"trivial": 0}}))
    chunks = scored("c", [0.9, 0.5, 0.4, 0.3, 0.2, 0.1])
    ctx = make_ctx(
        chunks, metrics={"query_analysis": {"complexity": "trivial", "legal_task_type": "x"}}
    )
    out = await rerank_stage(ctx, reranker=RaisingReranker(), config=make_cfg(tier))
    assert ids(out.reranked) == ids(chunks)
    assert out.metrics["rerank"]["reranked"] is False


async def test_tier_without_reranking_passes_chunks_through_and_marks_alternatives():
    tier = replace(STANDARD, features=replace(STANDARD.features, use_reranking=False))
    chunks = scored("c", [0.5] * 12)
    out = await rerank_stage(make_ctx(chunks), reranker=RaisingReranker(), config=make_cfg(tier))
    assert ids(out.reranked) == ids(chunks[:10])
    assert ids(out.extra["alternative_chunks"]) == ["c10", "c11"]
    assert out.extra["alternative_chunks"][0].metadata["selected"] is False


# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------


async def test_relevance_gate_drops_chunks_below_the_threshold():
    chunks = scored("c", [0.0] * 6)
    scorer = ScriptedReranker(dict(zip(ids(chunks), [0.9, 0.8, 0.7, 0.6, 0.1, 0.05], strict=True)))
    out = await rerank_stage(make_ctx(chunks), reranker=scorer, config=make_cfg())
    assert ids(out.reranked) == ["c0", "c1", "c2", "c3"]
    assert out.metrics["rerank"]["gate_dropped"] == 2
    assert out.metrics["rerank"]["gate_skipped"] is None


async def test_relevance_gate_keeps_the_minimum_number_of_chunks():
    chunks = scored("c", [0.0] * 5)
    scorer = ScriptedReranker(dict(zip(ids(chunks), [0.9, 0.2, 0.1, 0.05, 0.04], strict=True)))
    out = await rerank_stage(make_ctx(chunks), reranker=scorer, config=make_cfg())
    assert ids(out.reranked) == ["c0", "c1", "c2"]


def test_gate_threshold_is_disabled_when_the_top_score_is_below_it():
    low = scored("c", [0.02, 0.01])
    assert resolve_gate_threshold(low, 0.25) == (0.0, "top_score_below_threshold")
    high = scored("c", [0.9, 0.01])
    assert resolve_gate_threshold(high, 0.25) == (0.25, None)
    assert resolve_gate_threshold(high, 0.25, scorer_degraded=True)[1] == "fallback_scorer_in_use"
    assert resolve_gate_threshold(high, 0.0) == (0.0, None)


def test_gate_reads_the_raw_score_snapshot_not_the_adjusted_score():
    demoted = chunk("a", 0.09, rerank_score=0.9)
    plain = chunk("b", 0.4)
    assert gate_score(demoted) == 0.9
    assert gate_score(plain) == 0.4
    kept = apply_relevance_gate(
        [chunk("x", 0.8), chunk("y", 0.7), chunk("z", 0.6), demoted, chunk("w", 0.01)],
        threshold=0.25,
        min_chunks=3,
    )
    assert ids(kept) == ["x", "y", "z", "a"]


async def test_gate_survives_an_adjuster_that_demotes_a_relevant_chunk():
    chunks = scored("c", [0.0] * 5)
    scorer = ScriptedReranker(dict(zip(ids(chunks), [0.9, 0.8, 0.7, 0.6, 0.5], strict=True)))

    def demote_first(query, found, filters):
        return [replace(c, score=c.score * 0.1) if c.id == "c0" else c for c in found]

    out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, adjusters=[demote_first], config=make_cfg()
    )
    assert "c0" in ids(out.reranked)
    assert ids(out.reranked)[-1] == "c0"


async def test_gate_is_disabled_when_the_fallback_scorer_is_used():
    # Retrieval-scale scores (about 0.02) are far below a 0-1 threshold. Applying the gate would
    # empty the result down to the minimum, so it must switch off instead.
    chunks = [chunk(f"c{i}", 0.020 - i * 0.001, text=f"zz{i}") for i in range(8)]
    ctx = make_ctx(chunks, query="zzzzzz")

    primary_failed = await rerank_stage(
        ctx,
        reranker=RaisingReranker(),
        fallback_reranker=ScoreBasedReranker(),
        config=make_cfg(),
    )
    assert len(primary_failed.reranked) == 8
    assert primary_failed.metrics["rerank"]["gate_skipped"] == "fallback_scorer_in_use"
    assert "rerank_degraded" in primary_failed.warnings

    provider_failed = await rerank_stage(ctx, reranker=RaisingProvider(), config=make_cfg())
    assert len(provider_failed.reranked) == 8
    assert provider_failed.metrics["rerank"]["degraded"] is True


async def test_gate_runs_on_the_fallback_scale_when_one_is_configured():
    chunks = [chunk(f"c{i}", 0.020 - i * 0.001, text=f"zz{i}") for i in range(8)]
    cfg = make_cfg(fallback_relevance_threshold=0.0165)
    out = await rerank_stage(
        make_ctx(chunks, query="zzzzzz"),
        reranker=RaisingReranker(),
        fallback_reranker=ScoreBasedReranker(),
        config=cfg,
    )
    assert ids(out.reranked) == ["c0", "c1", "c2", "c3"]


async def test_per_scorer_threshold_overrides_the_tier_value():
    chunks = scored("c", [0.0] * 6)
    scorer = ScriptedReranker(dict(zip(ids(chunks), [9.0, 8.0, 7.0, 6.0, 1.0, 0.5], strict=True)))
    out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, config=make_cfg(relevance_threshold=5.0)
    )
    assert ids(out.reranked) == ["c0", "c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# CRAG grade filter
# ---------------------------------------------------------------------------


def graded_chunks() -> list[RetrievedChunk]:
    texts = {"g0": "on point", "g1": "on point", "g2": "on point", "g3": "IRRELEVANT noise"}
    return [chunk(cid, 0.9 - i * 0.1, text=text) for i, (cid, text) in enumerate(texts.items())]


def graded_scorer(chunks: Sequence[RetrievedChunk]) -> ScriptedReranker:
    return ScriptedReranker({c.text: c.score for c in chunks})


async def test_crag_filter_drops_chunks_graded_irrelevant():
    chunks = graded_chunks()
    out = await rerank_stage(
        make_ctx(chunks),
        reranker=graded_scorer(chunks),
        llm=GraderLLM(),
        config=make_cfg(use_crag=None),
    )
    assert ids(out.reranked) == ["g0", "g1", "g2"]
    assert out.metrics["crag_filter"]["quality"] == "partial"
    assert out.metrics["crag_filter"]["before"] == 4


async def test_crag_filter_never_leaves_the_context_empty():
    chunks = [chunk(f"g{i}", 0.9 - i * 0.1, text="IRRELEVANT") for i in range(4)]
    no_preserve = CRAGConfig(preserve_for_low_quality=False)
    out = await rerank_stage(
        make_ctx(chunks),
        reranker=graded_scorer(chunks),
        llm=GraderLLM(),
        config=make_cfg(use_crag=True, crag=no_preserve),
    )
    assert len(out.reranked) == 4
    assert out.metrics["crag_filter"]["kept_all"] == "filter_emptied"

    default = await rerank_stage(
        make_ctx(chunks),
        reranker=graded_scorer(chunks),
        llm=GraderLLM(),
        config=make_cfg(use_crag=True),
    )
    assert len(default.reranked) == 3


async def test_crag_filter_is_skipped_for_scoped_retrieval_and_without_a_model():
    chunks = graded_chunks()
    scorer = graded_scorer(chunks)
    pinned = await rerank_stage(
        make_ctx(chunks, pinned_ids=["g3"], enable_shared_corpus=False),
        reranker=scorer,
        llm=GraderLLM(),
        config=make_cfg(use_crag=True),
    )
    assert len(pinned.reranked) == 4
    assert "crag_filter" not in pinned.metrics

    no_model = await rerank_stage(
        make_ctx(chunks), reranker=scorer, llm=None, config=make_cfg(use_crag=True)
    )
    assert len(no_model.reranked) == 4
    assert "crag_filter" not in no_model.metrics


async def test_crag_filter_restores_flagged_chunks_and_regrades_generated_apart():
    chunks = [
        chunk("a", 0.9, text="on point"),
        chunk("b", 0.8, text="on point"),
        chunk("c", 0.7, text="on point"),
        chunk("lead", 0.1, text="IRRELEVANT", is_pinned=True),
        chunk("d", 0.6, text="IRRELEVANT"),
    ]
    out = await rerank_stage(
        make_ctx(chunks),
        reranker=graded_scorer(chunks),
        llm=GraderLLM(),
        config=make_cfg(use_crag=True),
    )
    assert ids(out.reranked) == ["lead", "a", "b", "c"]

    authority = [chunk("x", 0.9), chunk("y", 0.8)]
    gen = chunk("gen", 0.7, is_generated=True)
    to_grade, reappend = split_generated_for_grading([*authority, gen])
    assert ids(to_grade) == ["x", "y"]
    assert ids(reappend) == ["gen"]
    only_generated = [chunk("g1", is_generated=True), chunk("g2", source_type="generated")]
    assert ids(split_generated_for_grading(only_generated)[0]) == ["g1", "g2"]
    assert split_generated_for_grading(only_generated)[1] == []


# ---------------------------------------------------------------------------
# Diversity caps
# ---------------------------------------------------------------------------


def test_document_cap_backfills_from_documents_not_yet_present():
    selected = [
        chunk(c, s, doc=d)
        for c, s, d in [
            ("a1", 0.95, "A"),
            ("a2", 0.94, "A"),
            ("a3", 0.93, "A"),
            ("a4", 0.92, "A"),
            ("a5", 0.91, "A"),
            ("b1", 0.90, "B"),
        ]
    ]
    alternatives = [
        chunk(c, s, doc=d)
        for c, s, d in [
            ("b2", 0.89, "B"),
            ("c1", 0.88, "C"),
            ("d1", 0.87, "D"),
            ("e1", 0.86, "E"),
        ]
    ]
    kept, rest, removed = apply_document_cap(selected, alternatives, max_per_doc=3)
    assert removed == 2
    assert ids(kept) == ["a1", "a2", "a3", "b1", "c1", "d1"]
    assert ids(rest) == ["b2", "e1"]


def test_document_cap_leaves_a_compliant_set_untouched():
    selected = [chunk("a1", 0.9, doc="A"), chunk("b1", 0.8, doc="B")]
    kept, rest, removed = apply_document_cap(selected, [chunk("c1", 0.1, doc="C")], max_per_doc=3)
    assert (ids(kept), ids(rest), removed) == (["a1", "b1"], ["c1"], 0)


def test_document_cap_resolution_tightens_for_case_law_research():
    tasks = ("case_law_research", "comparison")
    assert resolve_document_cap(3, "case_law_research", "x", case_law_task_types=tasks) == 2
    assert resolve_document_cap(3, "general", "x", case_law_task_types=tasks) == 3
    assert resolve_document_cap(1, "comparison", "x", case_law_task_types=tasks) == 1
    landmark = re.compile(r"basic structure")
    wide = resolve_document_cap(
        6,
        "comparison",
        "the basic structure test",
        case_law_task_types=tasks,
        landmark_pattern=landmark,
    )
    assert wide == 4


async def test_stage_caps_chunks_per_document_with_the_tier_default_of_three():
    rows = [("a1", "A"), ("a2", "A"), ("a3", "A"), ("a4", "A"), ("a5", "A"), ("b1", "B")]
    rows += [("b2", "B"), ("c1", "C"), ("d1", "D"), ("e1", "E"), ("f1", "F"), ("g1", "G")]
    scores = [0.95 - i * 0.01 for i in range(len(rows))]
    chunks = [chunk(cid, 0.0, doc=doc) for cid, doc in rows]
    scorer = ScriptedReranker(dict(zip(ids(chunks), scores, strict=True)))
    tier = replace(STANDARD, top_k_rerank=6)
    assert tier.max_chunks_per_document == 3

    out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, config=make_cfg(tier, use_diversity=True)
    )
    docs = [c.metadata["document_id"] for c in out.reranked]
    assert len(out.reranked) == 6
    assert docs.count("A") == 3
    assert "a4" not in ids(out.reranked)
    assert out.metrics["rerank"]["diversity_dropped"] == 2


# ---------------------------------------------------------------------------
# Case-level cap and the case-name wrapper
# ---------------------------------------------------------------------------


async def test_case_name_wrapper_calls_the_ner_extractor_method():
    names = await extract_case_names("Smith v. Jones was decided last term.")
    assert names == ["Smith v. Jones"]


async def test_case_cap_limits_chunks_that_cite_the_same_case():
    pairs = ["Alpha v. Beta", "Gamma v. Delta", "Epsilon v. Zeta", "Theta v. Iota"]
    mentions = ["Smith v. Jones"] * 5 + pairs
    mentions += ["Kappa v. Lambda", "Omega v. Sigma", "Rho v. Tau"]
    chunks = [
        chunk(f"c{i}", 0.0, doc=f"d{i}", text=f"{name} was decided on appeal.")
        for i, name in enumerate(mentions)
    ]
    scorer = ScriptedReranker({c.text: 0.99 - i * 0.01 for i, c in enumerate(chunks)})
    tier = replace(STANDARD, top_k_rerank=7)
    ctx = make_ctx(
        chunks,
        metrics={"query_analysis": {"complexity": "complex", "legal_task_type": "comparison"}},
    )
    out = await rerank_stage(ctx, reranker=scorer, config=make_cfg(tier, use_diversity=True))
    smith = [c for c in out.reranked if c.text.startswith("Smith")]
    assert len(smith) == 3
    assert len(out.reranked) == 7
    assert out.metrics["rerank"]["diversity_dropped"] == 2


async def test_case_cap_ignores_a_failing_extractor():
    async def broken(text: str) -> list[str]:
        raise RuntimeError("no model")

    selected = [chunk("a", 0.9), chunk("b", 0.8)]
    kept, rest, removed = await apply_case_cap(selected, [], max_per_case=1, extractor=broken)
    assert (ids(kept), rest, removed) == (["a", "b"], [], 0)


# ---------------------------------------------------------------------------
# Explicit scope: pinned chunks are exempt from the gate and diversity
# ---------------------------------------------------------------------------


async def test_pinned_chunks_are_exempt_from_the_relevance_gate():
    chunks = [
        chunk("p1", 0.0, doc="docP"),
        *[chunk(f"r{i}", 0.0, doc=f"ref{i}") for i in range(5)],
    ]
    scores = {"p1": 0.05, "r0": 0.9, "r1": 0.8, "r2": 0.7, "r3": 0.6, "r4": 0.1}
    ctx = make_ctx(chunks, pinned_ids=["docP"], enable_shared_corpus=False)
    out = await rerank_stage(ctx, reranker=ScriptedReranker(scores), config=make_cfg())
    assert "p1" in ids(out.reranked)
    assert "r4" not in ids(out.reranked)

    unpinned = await rerank_stage(
        make_ctx(chunks), reranker=ScriptedReranker(scores), config=make_cfg()
    )
    assert "p1" not in ids(unpinned.reranked)


async def test_pinned_chunks_are_exempt_from_the_document_cap():
    chunks = [chunk(f"p{i}", 0.0, doc="docP") for i in range(5)]
    chunks += [chunk(f"x{i}", 0.0, doc="docX") for i in range(4)]
    scores = {c.id: 0.95 - i * 0.01 for i, c in enumerate(chunks)}
    tier = replace(STANDARD, top_k_rerank=9)
    ctx = make_ctx(chunks, pinned_ids=["docP"], enable_shared_corpus=False)
    out = await rerank_stage(
        ctx, reranker=ScriptedReranker(scores), config=make_cfg(tier, use_diversity=True)
    )
    docs = [c.metadata["document_id"] for c in out.reranked]
    assert docs.count("docP") == 5
    assert docs.count("docX") == 3


async def test_scope_lock_ids_match_case_ids_too():
    chunks = [chunk("l1", 0.0, case_id="case-9"), *scored("r", [0.0] * 5)]
    scores = {"l1": 0.02, "r0": 0.9, "r1": 0.8, "r2": 0.7, "r3": 0.6, "r4": 0.5}
    ctx = make_ctx(chunks, scope_lock_ids=["case-9"])
    out = await rerank_stage(ctx, reranker=ScriptedReranker(scores), config=make_cfg())
    assert "l1" in ids(out.reranked)


async def test_two_bucket_mode_reserves_slots_for_pinned_documents():
    pinned = [chunk(f"p{i}", 0.0, doc="docP") for i in range(3)]
    reference = [chunk(f"r{i}", 0.0, doc=f"ref{i}") for i in range(8)]
    scores = {"p0": 0.3, "p1": 0.2, "p2": 0.1}
    scores.update({f"r{i}": 0.9 - i * 0.05 for i in range(8)})
    tier = replace(STANDARD, top_k_rerank=6)
    ctx = make_ctx([*reference, *pinned], pinned_ids=["docP", "docQ"], enable_shared_corpus=True)
    out = await rerank_stage(
        ctx, reranker=ScriptedReranker(scores), config=make_cfg(tier, use_diversity=True)
    )
    assert ids(out.reranked) == ["p0", "p1", "r0", "r1", "r2", "r3"]
    assert out.metrics["rerank"]["two_bucket"] is True
    assert "diversity_dropped" not in out.metrics["rerank"]


async def test_one_empty_bucket_falls_back_to_a_single_pool():
    chunks = scored("r", [0.0] * 6, document_id="ref")
    scores = {c.id: 0.9 - i * 0.05 for i, c in enumerate(chunks)}
    ctx = make_ctx(chunks, pinned_ids=["docP", "docQ"], enable_shared_corpus=True)
    out = await rerank_stage(ctx, reranker=ScriptedReranker(scores), config=make_cfg())
    assert out.metrics["rerank"]["two_bucket"] is False
    assert len(out.reranked) == 6


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


async def test_rerank_failure_falls_back_to_the_retrieved_chunks():
    chunks = scored("c", [0.3, 0.2, 0.1])
    out = await rerank_stage(make_ctx(chunks), reranker=RaisingReranker(), config=make_cfg())
    assert ids(out.reranked) == ["c0", "c1", "c2"]
    assert out.extra["alternative_chunks"] == []
    assert "rerank_failed" in out.warnings
    assert out.metrics["rerank"]["failed"] is True


async def test_empty_retrieval_returns_an_empty_result_without_calling_the_reranker():
    out = await rerank_stage(make_ctx([]), reranker=RaisingReranker(), config=make_cfg())
    assert out.reranked == []
    assert out.warnings == []


async def test_rerank_does_not_mutate_the_input_chunks():
    chunks = scored("c", [0.5, 0.4, 0.3, 0.2, 0.1])
    ctx = make_ctx(chunks)
    scorer = ScriptedReranker({c.id: 0.9 - i * 0.1 for i, c in enumerate(chunks)})
    await rerank_stage(ctx, reranker=scorer, config=make_cfg())
    assert [c.score for c in ctx.retrieved] == [0.5, 0.4, 0.3, 0.2, 0.1]
    assert all("selected" not in c.metadata for c in ctx.retrieved)
    assert ctx.reranked == []


# ---------------------------------------------------------------------------
# Adjusters
# ---------------------------------------------------------------------------


async def test_adjusters_are_applied_in_the_order_given():
    chunks = scored("c", [0.0] * 3)
    scorer = ScriptedReranker({"c0": 0.9, "c1": 0.8, "c2": 0.3})

    def first(query, found, filters):
        for c in found:
            c.metadata.setdefault("trail", []).append("first")
        return [replace(c, score=0.95) if c.id == "c2" else c for c in found]

    def second(query, found, filters):
        for c in found:
            c.metadata.setdefault("trail", []).append("second")
        return [replace(c, score=c.score * 0.5) if c.id == "c2" else c for c in found]

    out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, adjusters=[first, second], config=make_cfg()
    )
    assert ids(out.reranked) == ["c0", "c1", "c2"]
    assert out.reranked[2].metadata["trail"] == ["first", "second"]

    reversed_out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, adjusters=[second, first], config=make_cfg()
    )
    assert ids(reversed_out.reranked)[0] == "c2"


async def test_a_failing_adjuster_is_skipped_and_the_rest_still_run():
    chunks = scored("c", [0.0] * 3)
    scorer = ScriptedReranker({"c0": 0.9, "c1": 0.8, "c2": 0.7})

    def broken(query, found, filters):
        raise ValueError("bad adjuster")

    def promote(query, found, filters):
        return [replace(c, score=2.0) if c.id == "c2" else c for c in found]

    out = await rerank_stage(
        make_ctx(chunks), reranker=scorer, adjusters=[broken, promote], config=make_cfg()
    )
    assert ids(out.reranked)[0] == "c2"


async def test_default_adjuster_demotes_generated_chunks_below_real_authority():
    chunks = [chunk("gen", 0.0, is_generated=True), *scored("a", [0.0] * 3)]
    scorer = ScriptedReranker({"gen": 0.95, "a0": 0.9, "a1": 0.8, "a2": 0.7})
    out = await rerank_stage(make_ctx(chunks), reranker=scorer, config=make_cfg())
    assert ids(out.reranked) == ["a0", "a1", "a2", "gen"]
    assert out.reranked[-1].metadata["rerank_score"] == 0.95

    off = await rerank_stage(
        make_ctx(chunks), reranker=scorer, config=make_cfg(downweight_generated=False)
    )
    assert ids(off.reranked)[0] == "gen"


async def test_default_adjuster_leaves_a_generated_only_set_alone():
    chunks = [chunk(f"g{i}", 0.0, is_generated=True) for i in range(3)]
    scorer = ScriptedReranker({"g0": 0.9, "g1": 0.8, "g2": 0.7})
    out = await rerank_stage(make_ctx(chunks), reranker=scorer, config=make_cfg())
    assert ids(out.reranked) == ["g0", "g1", "g2"]
    assert [c.score for c in out.reranked] == [0.9, 0.8, 0.7]


class FakePlugin:
    condenser_key = "fake"

    def matches(self, filters: Mapping[str, Any]) -> bool:
        return filters.get("body") == "fake"

    def adjusters(self):
        def promote_last(query, found, filters):
            return [replace(c, score=3.0) if c.id == "c2" else c for c in found]

        return [promote_last]


async def test_jurisdiction_plugin_adjusters_apply_only_when_it_matches():
    chunks = scored("c", [0.0] * 3)
    scorer = ScriptedReranker({"c0": 0.9, "c1": 0.8, "c2": 0.7})
    cfg = make_cfg()
    hit = await rerank_stage(
        make_ctx(chunks, filters={"body": "fake"}),
        reranker=scorer,
        jurisdiction=FakePlugin(),
        config=cfg,
    )
    miss = await rerank_stage(
        make_ctx(chunks, filters={"body": "other"}),
        reranker=scorer,
        jurisdiction=FakePlugin(),
        config=cfg,
    )
    assert ids(hit.reranked)[0] == "c2"
    assert ids(miss.reranked) == ["c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# Reorder, contradiction hints, query selection, events and usage
# ---------------------------------------------------------------------------


async def test_lost_in_the_middle_reorder_is_applied_and_flagged_chunks_lead():
    chunks = [*scored("c", [0.0] * 6), chunk("lead", 0.0, is_pinned=True)]
    scores = {f"c{i}": 0.9 - i * 0.05 for i in range(6)}
    ordered = await rerank_stage(
        make_ctx(chunks), reranker=ScriptedReranker(scores), config=make_cfg(use_reorder=True)
    )
    plain = await rerank_stage(
        make_ctx(chunks), reranker=ScriptedReranker(scores), config=make_cfg(use_reorder=False)
    )
    rest = [c for c in plain.reranked if c.id != "lead"]
    expected = LostInMiddleReorderer().reorder(rest).reordered_chunks
    assert ids(plain.reranked) == ["lead", "c0", "c1", "c2", "c3", "c4", "c5"]
    assert ids(ordered.reranked) == ["lead", *ids(expected)]
    assert ids(ordered.reranked)[1:] != ids(rest)


def test_contradiction_hints_flag_a_section_shared_by_two_documents():
    chunks = [
        chunk("a", doc="d1", section_number="5"),
        chunk("b", doc="d2", section_number="5"),
        chunk("c", doc="d3", section_number="9"),
    ]
    hints = detect_contradiction_hints(chunks)
    assert hints == {
        "conflicting_sections": ["5"],
        "document_ids": ["d1", "d2"],
        "section_count": 1,
    }
    assert detect_contradiction_hints(chunks[:1]) is None
    same_doc = [chunk("a", doc="d1", section_number="5"), chunk("b", doc="d1", section_number="5")]
    assert detect_contradiction_hints(same_doc) is None


async def test_contradiction_hints_are_off_unless_asked_for():
    chunks = [
        chunk("a", 0.0, doc="d1", section_number="5"),
        chunk("b", 0.0, doc="d2", section_number="5"),
    ]
    scorer = ScriptedReranker({"a": 0.9, "b": 0.8})
    off = await rerank_stage(make_ctx(chunks), reranker=scorer, config=make_cfg())
    on = await rerank_stage(
        make_ctx(chunks), reranker=scorer, config=make_cfg(detect_contradictions=True)
    )
    assert "contradictions" not in off.metrics
    assert on.metrics["contradictions"]["section_count"] == 1


async def test_rerank_uses_the_condensed_query_when_one_is_available():
    chunks = scored("c", [0.0] * 3)
    scorer = ScriptedReranker({})
    ctx = make_ctx(
        chunks, query="and what about that?", metrics={"condensed_query": " standalone "}
    )
    await rerank_stage(ctx, reranker=scorer, config=make_cfg())
    await rerank_stage(ctx, reranker=scorer, config=make_cfg(use_condensed_query=False))
    assert scorer.queries == ["standalone", "and what about that?"]


async def test_stage_emits_a_completion_event_and_usage():
    events: list[Any] = []

    async def sink(event: Any) -> None:
        events.append(event)

    ledger = UsageLedger()
    chunks = graded_chunks()
    out = await rerank_stage(
        make_ctx(chunks),
        reranker=graded_scorer(chunks),
        llm=GraderLLM(),
        usage=ledger,
        on_event=sink,
        config=make_cfg(use_crag=True),
    )
    assert events and events[0]["type"] == "rerank_complete"
    assert events[0]["input"] == 4
    assert ledger.counts_by_operation().keys() >= {"reranking", "crag_evaluation"}
    assert out.metrics["rerank"]["selected"] == 3


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

RERANK_SOURCE = Path(__file__).resolve().parents[2] / "src/legal_rag/pipeline/stages/rerank.py"
FORBIDDEN = {
    "vaquill": re.compile(r"vaquill", re.IGNORECASE),
    "supabase": re.compile(r"supabase", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "voyage": re.compile(r"voyage", re.IGNORECASE),
    "cohere": re.compile(r"cohere", re.IGNORECASE),
    "organization_id": re.compile(r"organization_id"),
    "user_id": re.compile(r"user_id"),
    "matter_id": re.compile(r"matter_id"),
    "private app import": re.compile(r"(?<![A-Za-z_])app\."),
    "os.environ": re.compile(r"os\.environ|getenv"),
    "settings": re.compile(r"settings", re.IGNORECASE),
    "gpt-": re.compile(r"gpt-", re.IGNORECASE),
    "ticket tag": re.compile(r"Round \d|bug-?fix #\d|GAP \d|EDGE \d", re.IGNORECASE),
    "em dash": re.compile(f"{chr(0x2014)}|{chr(0x2013)}"),
}


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_rerank_source_has_no_forbidden_text(label: str):
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(RERANK_SOURCE.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found in rerank.py:\n" + "\n".join(hits)
