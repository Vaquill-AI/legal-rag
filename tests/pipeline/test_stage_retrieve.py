"""Retrieval stage: strategy dispatch, depth caps, pinning, scope lock, fail-open and hygiene."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline import QueryContext, UsageLedger, get_tier
from legal_rag.pipeline.stages.retrieve import (
    DEFAULT_STRATEGIES,
    EXTRA_CONDENSED_QUERY,
    EXTRA_FORCE_STRATEGY,
    EXTRA_QUERY_ANALYSIS,
    EXTRA_QUERY_EXPANSION,
    EXTRA_REQUIRES_RETRIEVAL,
    EXTRA_SUB_QUERIES,
    FanOutRetriever,
    RetrievalRun,
    RetrieveConfig,
    ScopeFilterKeys,
    UnknownStrategyError,
    deduplicate_chunks,
    plan_retrieval,
    resolve_strategy,
    retrieve_stage,
)
from legal_rag.pipeline.tiers import KNOWN_STRATEGIES
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.retrieval.strategies.crag import RelevanceScore
from legal_rag.retrieval.strategies.gap_analysis import InformationGapAnalysis
from legal_rag.retrieval.strategies.multi_query import QueryVariations
from legal_rag.retrieval.strategies.self_rag import RetrievalAssessment
from legal_rag.types import RetrievedChunk

QUERY = "what notice does the statute require before a tenant is evicted"
HYPOTHESIS = (
    "A landlord must serve written notice before filing, and the notice period is set by statute."
)
REWRITE_A = "required written notice period before eviction proceedings begin"
REWRITE_B = "statutory notice requirements a landlord must satisfy before eviction"


def chunk(cid: str, score: float = 0.5, text: str | None = None, **meta: Any) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text or f"text of {cid}", score=score, metadata=dict(meta))


def make_ctx(strategy: str = "hybrid", tier: str = "standard", **kwargs: Any) -> QueryContext:
    return QueryContext(query=QUERY, tier=tier, retrieval_strategy=strategy, **kwargs)


class RecordingRetriever:
    """Records every call. ``by_query`` maps a query to its chunks, ``default`` is the fallback."""

    def __init__(
        self,
        default: Sequence[RetrievedChunk] = (),
        by_query: Mapping[str, Sequence[RetrievedChunk]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.default = list(default)
        self.by_query = dict(by_query or {})
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "sub_queries": sub_queries}
        )
        if self.error is not None:
            raise self.error
        return list(self.by_query.get(query, self.default))

    @property
    def queries(self) -> list[str]:
        return [c["query"] for c in self.calls]


class FakeLookup:
    def __init__(self, chunks: Sequence[RetrievedChunk] = (), error: Exception | None = None):
        self.chunks = list(chunks)
        self.error = error
        self.queries: list[str] = []

    async def lookup(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class FakeWeb:
    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return list(self.chunks)


class FakePlugin:
    condenser_key = "test"

    def __init__(self, chunks: Sequence[RetrievedChunk], lookups: Sequence[FakeLookup] = ()):
        self.retriever = RecordingRetriever(chunks)
        self.lookups = list(lookups)
        self.applies = True

    def matches(self, filters: Mapping[str, Any]) -> bool:
        return self.applies

    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        return await self.retriever.retrieve(
            query, top_k=top_k, filters=filters, sub_queries=sub_queries
        )

    def adjusters(self):
        return ()

    def exact_lookups(self):
        return self.lookups

    def label_for(self, chunk: RetrievedChunk) -> str:
        return chunk.id


def recording_table(calls: list[str], *, empty: frozenset[str] = frozenset()):
    def make(name: str):
        async def runner(run: RetrievalRun):
            calls.append(name)
            return ([] if name in empty else [chunk(f"{name}-1", 0.9)]), {"ran": name}

        return runner

    return {name: make(name) for name in KNOWN_STRATEGIES}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_default_table_covers_every_known_strategy() -> None:
    assert set(DEFAULT_STRATEGIES) == KNOWN_STRATEGIES


@pytest.mark.parametrize("name", sorted(KNOWN_STRATEGIES))
async def test_each_strategy_name_dispatches_to_its_runner(name: str) -> None:
    calls: list[str] = []
    result = await retrieve_stage(
        make_ctx(name), retriever=RecordingRetriever(), strategies=recording_table(calls)
    )
    assert calls == [name]
    assert [c.id for c in result.retrieved] == [f"{name}-1"]
    assert result.metrics["retrieval"]["strategy"] == name


async def test_unknown_strategy_raises_instead_of_failing_open() -> None:
    with pytest.raises(UnknownStrategyError):
        await retrieve_stage(make_ctx("telepathy"), retriever=RecordingRetriever())
    with pytest.raises(ValueError):
        resolve_strategy("telepathy")


async def test_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        await retrieve_stage(make_ctx(tier="platinum"), retriever=RecordingRetriever())


async def test_strategy_falls_back_to_tier_default_when_context_has_none() -> None:
    calls: list[str] = []
    ctx = make_ctx("")
    await retrieve_stage(ctx, retriever=RecordingRetriever(), strategies=recording_table(calls))
    assert calls == [get_tier("standard").retrieval_strategy]


def test_adaptive_override_moves_decomposable_query_to_multi_hop_when_tier_allows() -> None:
    config = RetrieveConfig()
    analysis = {EXTRA_QUERY_ANALYSIS: {"should_decompose": True, "complexity": "complex"}}
    standard = plan_retrieval(
        make_ctx("hybrid", extra=dict(analysis)), get_tier("standard"), config
    )
    deep = plan_retrieval(
        make_ctx("hybrid", "deep", extra=dict(analysis)), get_tier("deep"), config
    )
    multi_query = plan_retrieval(
        make_ctx("multi_query", "deep", extra=dict(analysis)), get_tier("deep"), config
    )
    forced = plan_retrieval(
        make_ctx("hybrid", "deep", extra={**analysis, EXTRA_FORCE_STRATEGY: "hybrid"}),
        get_tier("deep"),
        config,
    )
    assert standard.strategy == "hybrid"
    assert deep.strategy == "multi_hop"
    assert multi_query.strategy == "multi_query"
    assert forced.strategy == "hybrid" and forced.forced


# ---------------------------------------------------------------------------
# No-results retry and warnings
# ---------------------------------------------------------------------------


async def test_no_results_retry_falls_back_to_hybrid() -> None:
    calls: list[str] = []
    table = recording_table(calls, empty=frozenset({"crag"}))
    result = await retrieve_stage(
        make_ctx("crag"), retriever=RecordingRetriever(), strategies=table
    )
    summary = result.metrics["retrieval"]
    assert calls == ["crag", "hybrid"]
    assert [c.id for c in result.retrieved] == ["hybrid-1"]
    assert summary["strategy"] == "hybrid" and summary["requested_strategy"] == "crag"
    assert summary["strategy_metadata"]["fallback_from"] == "crag"


async def test_retry_can_be_turned_off() -> None:
    calls: list[str] = []
    table = recording_table(calls, empty=frozenset({"crag"}))
    result = await retrieve_stage(
        make_ctx("crag"),
        retriever=RecordingRetriever(),
        strategies=table,
        config=RetrieveConfig(retry_empty_with_hybrid=False),
    )
    assert calls == ["crag"] and result.retrieved == []


async def test_forced_strategy_beats_the_context_and_is_never_retried() -> None:
    calls: list[str] = []
    table = recording_table(calls, empty=frozenset({"hybrid"}))
    ctx = make_ctx("crag", extra={EXTRA_FORCE_STRATEGY: "hybrid"})
    result = await retrieve_stage(ctx, retriever=RecordingRetriever(), strategies=table)
    assert calls == ["hybrid"]
    assert result.retrieval_warning == "no_results"


@pytest.mark.parametrize(
    ("count", "warning"),
    [(0, "no_results"), (1, "limited_results"), (2, "limited_results"), (3, None)],
)
async def test_retrieval_warning_values(count: int, warning: str | None) -> None:
    retriever = RecordingRetriever([chunk(f"c{i}") for i in range(count)])
    result = await retrieve_stage(make_ctx(), retriever=retriever)
    assert result.retrieval_warning == warning


async def test_self_rag_skip_decision_is_trusted_and_not_retried() -> None:
    llm = FakeLanguageModel(
        structured={"RetrievalAssessment": RetrievalAssessment(decision="skip")}
    )
    retriever = RecordingRetriever([chunk("a")])
    result = await retrieve_stage(make_ctx("self_rag"), retriever=retriever, llm=llm)
    assert retriever.calls == []
    assert result.retrieved == []
    assert result.metrics["retrieval"]["strategy"] == "self_rag"


# ---------------------------------------------------------------------------
# Depth, caps and skipping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "complexity", "task", "pinned", "expected"),
    [
        ("standard", "simple", "general", 0, 20),
        ("standard", "moderate", "general", 0, 50),
        ("standard", "complex", "general", 0, 100),
        ("standard", "moderate", "case_law_research", 0, 75),
        ("standard", "simple", "case_law_research", 0, 20),
        ("standard", "mystery", "general", 0, 100),
        ("deep", "simple", "general", 0, 50),
        ("deep", "moderate", "comparison", 0, 100),
        ("standard", "simple", "general", 5, 30),
        ("standard", "simple", "general", 40, 150),
    ],
)
def test_depth_comes_from_tier_caps(
    tier: str, complexity: str, task: str, pinned: int, expected: int
) -> None:
    analysis = {"complexity": complexity, "legal_task_type": task}
    ctx = make_ctx(
        tier=tier,
        extra={EXTRA_QUERY_ANALYSIS: analysis},
        pinned_ids=[f"doc{i}" for i in range(pinned)],
    )
    assert plan_retrieval(ctx, get_tier(tier), RetrieveConfig()).top_k == expected


async def test_trivial_cap_of_zero_skips_retrieval() -> None:
    retriever = RecordingRetriever([chunk("a")])
    ctx = make_ctx(extra={EXTRA_QUERY_ANALYSIS: {"complexity": "trivial"}})
    result = await retrieve_stage(ctx, retriever=retriever)
    assert retriever.calls == []
    assert result.retrieved == []
    assert result.retrieval_warning is None
    assert result.metrics["retrieval"] == {"skipped": True, "skip_source": "complexity_cap"}


@pytest.mark.parametrize(
    ("extra", "source"),
    [
        ({EXTRA_REQUIRES_RETRIEVAL: False}, "requires_retrieval"),
        ({EXTRA_QUERY_ANALYSIS: {"should_retrieve": False}}, "query_analysis"),
    ],
)
async def test_heuristics_skip_retrieval(extra: dict[str, Any], source: str) -> None:
    retriever = RecordingRetriever([chunk("a")])
    result = await retrieve_stage(make_ctx(extra=extra), retriever=retriever)
    assert retriever.calls == []
    assert result.metrics["retrieval"]["skip_source"] == source


@pytest.mark.parametrize("lock", ["pinned_ids", "scope_lock_ids"])
async def test_explicit_scope_overrides_every_skip_rule(lock: str) -> None:
    retriever = RecordingRetriever([chunk("a")])
    extra = {
        EXTRA_REQUIRES_RETRIEVAL: False,
        EXTRA_QUERY_ANALYSIS: {"should_retrieve": False, "complexity": "trivial"},
    }
    ctx = make_ctx(extra=extra, **{lock: ["doc1"]})
    result = await retrieve_stage(ctx, retriever=retriever)
    assert [c.id for c in result.retrieved] == ["a"]
    assert retriever.calls[0]["top_k"] == get_tier("standard").top_k_retrieval


# ---------------------------------------------------------------------------
# Additive layers and scope lock
# ---------------------------------------------------------------------------


def layer_llm() -> FakeLanguageModel:
    return FakeLanguageModel(
        reply=HYPOTHESIS,
        structured={"QueryVariations": QueryVariations(queries=[REWRITE_A, REWRITE_B])},
    )


def layer_retriever() -> RecordingRetriever:
    return RecordingRetriever(
        [chunk("base", 0.9)],
        by_query={
            HYPOTHESIS: [chunk("from-hyde", 0.4)],
            REWRITE_A: [chunk("from-multi-query", 0.3)],
        },
    )


async def test_additive_layers_widen_recall_on_a_tier_that_enables_them() -> None:
    result = await retrieve_stage(
        make_ctx("hybrid", "deep"), retriever=layer_retriever(), llm=layer_llm()
    )
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert {c.id for c in result.retrieved} == {"base", "from-hyde", "from-multi-query"}
    assert meta["hyde_layer"]["added"] == 1
    assert meta["multi_query_layer"]["added"] == 1


@pytest.mark.parametrize("lock", ["pinned_ids", "scope_lock_ids"])
async def test_scope_lock_skips_additive_layers(lock: str) -> None:
    llm = layer_llm()
    result = await retrieve_stage(
        make_ctx("hybrid", "deep", **{lock: ["doc1"]}), retriever=layer_retriever(), llm=llm
    )
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert [c.id for c in result.retrieved] == ["base"]
    assert "hyde_layer" not in meta and "multi_query_layer" not in meta
    assert llm.calls == []


async def test_a_layer_is_skipped_when_the_base_strategy_already_is_that_technique() -> None:
    result = await retrieve_stage(
        make_ctx("hyde", "deep"), retriever=layer_retriever(), llm=layer_llm()
    )
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert "hyde_layer" not in meta and "multi_query_layer" in meta


async def test_standard_tier_runs_no_layers() -> None:
    result = await retrieve_stage(make_ctx("hybrid"), retriever=layer_retriever(), llm=layer_llm())
    assert [c.id for c in result.retrieved] == ["base"]


# ---------------------------------------------------------------------------
# Exact lookups, boost and filters
# ---------------------------------------------------------------------------


async def test_exact_lookup_results_are_pinned_and_first() -> None:
    lookup = FakeLookup([chunk("cited", 0.01, text="the named section")])
    retriever = RecordingRetriever([chunk("x", 0.9), chunk("cited", 0.8), chunk("y", 0.7)])
    result = await retrieve_stage(make_ctx(), retriever=retriever, exact_lookups=[lookup])
    ids = [c.id for c in result.retrieved]
    assert ids == ["cited", "x", "y"]
    assert result.retrieved[0].metadata["is_pinned"] is True
    assert result.retrieved[0].metadata["pinned_by"]
    assert result.retrieved[0].text == "the named section"
    assert result.metrics["retrieval"]["pinned_count"] == 1
    assert lookup.queries == [QUERY]


async def test_a_failing_exact_lookup_is_skipped() -> None:
    good = FakeLookup([chunk("cited")])
    bad = FakeLookup(error=RuntimeError("down"))
    result = await retrieve_stage(
        make_ctx(), retriever=RecordingRetriever([chunk("x")]), exact_lookups=[bad, good]
    )
    assert [c.id for c in result.retrieved] == ["cited", "x"]


async def test_filters_reach_the_retriever_unchanged() -> None:
    filters = {"tenant": "t-1", "years": [2020, 2021]}
    retriever = RecordingRetriever([chunk("a")])
    ctx = make_ctx(
        filters=filters, pinned_ids=["d1"], scope_lock_ids=["c1"], boost_ids=["b1"], collection="co"
    )
    await retrieve_stage(ctx, retriever=retriever)
    assert retriever.calls[0]["filters"] is ctx.filters
    assert retriever.calls[0]["filters"] == {"tenant": "t-1", "years": [2020, 2021]}


async def test_scope_ids_are_added_only_under_the_keys_the_caller_names() -> None:
    keys = ScopeFilterKeys(
        pinned_ids="doc_ids", scope_lock_ids="case_ids", boost_ids="boost", collection="corpus"
    )
    retriever = RecordingRetriever([chunk("a", 0.9)], by_query={}, error=None)
    ctx = make_ctx(
        filters={"tenant": "t-1"},
        pinned_ids=["d1", "d2"],
        scope_lock_ids=["c1"],
        boost_ids=["b1"],
        collection="co",
    )
    result = await retrieve_stage(ctx, retriever=retriever, config=RetrieveConfig(scope_keys=keys))
    main, boost = retriever.calls[0]["filters"], retriever.calls[1]["filters"]
    assert main == {"tenant": "t-1", "doc_ids": ["d1", "d2"], "case_ids": ["c1"], "corpus": "co"}
    assert boost == {"tenant": "t-1", "corpus": "co", "boost": ["b1"]}
    assert ctx.filters == {"tenant": "t-1"}
    assert result.retrieved[0].metadata["pinned_by"] == "boost"
    assert [c.id for c in result.retrieved].count("a") == 1


# ---------------------------------------------------------------------------
# Fan-out, jurisdiction plug-in, query fan-out
# ---------------------------------------------------------------------------


async def test_named_retrievers_fan_out_and_merge_in_declared_order() -> None:
    docs = RecordingRetriever([chunk("d1", 0.2), chunk("shared", 0.3)])
    corpus = RecordingRetriever([chunk("c1", 0.9), chunk("shared", 0.8)])
    result = await retrieve_stage(make_ctx(), retriever={"documents": docs, "corpus": corpus})
    assert [c.id for c in result.retrieved] == ["d1", "shared", "c1"]
    shared = next(c for c in result.retrieved if c.id == "shared")
    assert shared.score == 0.8
    assert result.retrieved[0].metadata["retriever"] == "documents"
    assert result.retrieved[2].metadata["retriever"] == "corpus"
    # The duplicate keeps its best-scored copy (the corpus one) at the first position.
    assert shared.metadata["retriever"] == "corpus"
    assert result.metrics["retrieval"]["sources"] == {"documents": 1, "corpus": 2}
    assert docs.calls[0]["filters"] is corpus.calls[0]["filters"]


async def test_fan_out_skips_a_failing_source_but_raises_when_all_fail() -> None:
    ok = RecordingRetriever([chunk("a")])
    bad = RecordingRetriever(error=RuntimeError("down"))
    merged = await FanOutRetriever({"bad": bad, "ok": ok}).retrieve("q", top_k=5)
    assert [c.id for c in merged] == ["a"]
    with pytest.raises(RuntimeError):
        await FanOutRetriever({"one": bad, "two": bad}).retrieve("q", top_k=5)
    with pytest.raises(ValueError):
        FanOutRetriever({})


async def test_matching_jurisdiction_plugin_retrieves_and_pins_its_lookups() -> None:
    lookup = FakeLookup([chunk("section-5")])
    plugin = FakePlugin([chunk("law-1", 0.7)], lookups=[lookup])
    base = RecordingRetriever([chunk("generic")])
    ctx = make_ctx(extra={EXTRA_SUB_QUERIES: ["notice period for eviction topic"]})
    result = await retrieve_stage(ctx, retriever=base, jurisdiction=plugin)
    assert base.calls == []
    assert [c.id for c in result.retrieved] == ["section-5", "law-1"]
    assert plugin.retriever.calls[0]["sub_queries"] == ["notice period for eviction topic"]

    plugin.applies = False
    result = await retrieve_stage(make_ctx(), retriever=base, jurisdiction=plugin)
    assert [c.id for c in result.retrieved] == ["generic"]


async def test_condenser_sub_queries_are_searched_in_parallel_and_deduplicated() -> None:
    retriever = RecordingRetriever(
        by_query={
            "condensed main question about eviction": [chunk("m", 0.9), chunk("dup", 0.2)],
            "topic one sub query": [chunk("dup", 0.6), chunk("t1", 0.5)],
            "topic two sub query": [chunk("t2", 0.4)],
        }
    )
    extra = {
        EXTRA_CONDENSED_QUERY: "condensed main question about eviction",
        EXTRA_SUB_QUERIES: ["topic one sub query", "topic two sub query"],
    }
    result = await retrieve_stage(make_ctx(extra=extra), retriever=retriever)
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert sorted(retriever.queries) == sorted(
        ["condensed main question about eviction", "topic one sub query", "topic two sub query"]
    )
    assert [c.id for c in result.retrieved] == ["m", "dup", "t1", "t2"]
    assert result.retrieved[1].score == 0.6
    assert meta["condenser_decomposition"] is True and meta["total_before_dedup"] == 5


async def test_expansion_variations_are_searched_when_there_are_no_sub_queries() -> None:
    retriever = RecordingRetriever([chunk("a")])
    expansion = {"applied": True, "variations": ["variation one text", "variation two text"]}
    result = await retrieve_stage(
        make_ctx(extra={EXTRA_QUERY_EXPANSION: expansion}), retriever=retriever
    )
    assert set(retriever.queries) == {QUERY, "variation one text", "variation two text"}
    assert result.metrics["retrieval"]["strategy_metadata"]["expansion_parallel"] is True


# ---------------------------------------------------------------------------
# Real strategy classes with fakes
# ---------------------------------------------------------------------------


async def test_hyde_searches_with_the_hypothetical_answer() -> None:
    retriever = RecordingRetriever(by_query={HYPOTHESIS: [chunk("h")]})
    result = await retrieve_stage(make_ctx("hyde"), retriever=retriever, llm=layer_llm())
    assert retriever.queries == [HYPOTHESIS]
    assert [c.id for c in result.retrieved] == ["h"]


async def test_multi_query_searches_every_phrasing() -> None:
    retriever = RecordingRetriever(
        by_query={
            QUERY: [chunk("a", 0.9)],
            REWRITE_A: [chunk("b", 0.8)],
            REWRITE_B: [chunk("a", 0.2)],
        }
    )
    result = await retrieve_stage(make_ctx("multi_query"), retriever=retriever, llm=layer_llm())
    assert set(retriever.queries) == {QUERY, REWRITE_A, REWRITE_B}
    assert [c.id for c in result.retrieved] == ["a", "b"]


async def test_self_rag_retrieves_and_runs_a_follow_up_when_the_first_pass_is_thin() -> None:
    gap = InformationGapAnalysis(
        is_sufficient=False,
        missing_aspects=["notice period"],
        suggested_followup="statutory notice period length for eviction",
        confidence=0.3,
    )
    llm = FakeLanguageModel(
        structured={
            "RetrievalAssessment": RetrievalAssessment(decision="retrieve"),
            "InformationGapAnalysis": gap,
        }
    )
    retriever = RecordingRetriever(
        by_query={QUERY: [chunk("first", 0.9)], gap.suggested_followup: [chunk("second", 0.5)]}
    )
    result = await retrieve_stage(make_ctx("self_rag"), retriever=retriever, llm=llm)
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert [c.id for c in result.retrieved] == ["first", "second"]
    assert meta["re_retrieved"] is True and meta["decision"] == "retrieve"


async def test_crag_adds_web_chunks_when_graded_quality_is_low() -> None:
    llm = FakeLanguageModel(structured={"RelevanceScore": RelevanceScore(score=0.1)})
    web = FakeWeb([chunk("web-1", 0.5, url="https://example.org/a")])
    retriever = RecordingRetriever([chunk("a"), chunk("b"), chunk("c"), chunk("d")])
    result = await retrieve_stage(
        make_ctx("crag"),
        retriever=retriever,
        llm=llm,
        web_search=web,
        config=RetrieveConfig(re_retrieval=False),
    )
    meta = result.metrics["retrieval"]["strategy_metadata"]
    web_chunk = next(c for c in result.retrieved if c.id == "web-1")
    assert web_chunk.metadata["source_type"] == "web_search"
    assert meta["quality"] == "low" and meta["web_search_executed"] is True
    assert web.calls == [(QUERY, 5)]


async def test_crag_keeps_good_chunks_and_never_calls_web_search() -> None:
    llm = FakeLanguageModel(structured={"RelevanceScore": RelevanceScore(score=0.9)})
    web = FakeWeb([chunk("web-1")])
    retriever = RecordingRetriever([chunk("a"), chunk("b")])
    result = await retrieve_stage(make_ctx("crag"), retriever=retriever, llm=llm, web_search=web)
    assert [c.id for c in result.retrieved] == ["a", "b"]
    assert web.calls == []


async def test_multi_hop_runs_without_a_model_and_reports_progress() -> None:
    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    retriever = RecordingRetriever([chunk("a", 0.9), chunk("b", 0.8)])
    result = await retrieve_stage(
        make_ctx("multi_hop", "deep"), retriever=retriever, on_event=on_event
    )
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert {c.id for c in result.retrieved} == {"a", "b"}
    assert meta["hops_performed"] >= 1
    assert events and events[0].event_type == "multi_hop_started"


@pytest.mark.parametrize("name", ["self_rag", "crag", "hyde", "multi_query"])
async def test_llm_strategies_degrade_to_hybrid_without_a_model(name: str) -> None:
    retriever = RecordingRetriever([chunk("a")])
    result = await retrieve_stage(make_ctx(name), retriever=retriever)
    meta = result.metrics["retrieval"]["strategy_metadata"]
    assert [c.id for c in result.retrieved] == ["a"]
    assert meta["degraded_from"] == name


# ---------------------------------------------------------------------------
# Fail-open, usage and dedup
# ---------------------------------------------------------------------------


async def test_a_retriever_error_fails_open_with_the_input_context() -> None:
    retriever = RecordingRetriever(error=RuntimeError("index unavailable"))
    ctx = make_ctx()
    result = await retrieve_stage(ctx, retriever=retriever)
    assert result is ctx
    assert result.retrieved == []
    assert "retrieval_failed" in result.warnings


async def test_every_fan_out_source_failing_fails_open() -> None:
    bad = RecordingRetriever(error=RuntimeError("down"))
    ctx = make_ctx()
    result = await retrieve_stage(ctx, retriever={"one": bad, "two": bad})
    assert result is ctx and "retrieval_failed" in result.warnings


async def test_stage_emits_one_usage_event() -> None:
    ledger = UsageLedger()
    await retrieve_stage(make_ctx(), retriever=RecordingRetriever([chunk("a")]), usage=ledger)
    assert ledger.counts_by_operation() == {"retrieve": 1}
    assert ledger.events[0].extra["strategy"] == "hybrid"


def test_deduplicate_keeps_the_best_scored_duplicate_best_first() -> None:
    chunks = [chunk("a", 0.2), chunk("b", 0.9), chunk("a", 0.7), chunk("c", 0.9), chunk("b", 0.1)]
    result = deduplicate_chunks(chunks)
    assert [(c.id, c.score) for c in result] == [("b", 0.9), ("c", 0.9), ("a", 0.7)]
    assert deduplicate_chunks([]) == []


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

RETRIEVE_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "legal_rag"
    / "pipeline"
    / "stages"
    / "retrieve.py"
)
FORBIDDEN = {
    "vaquill": re.compile(r"vaquill", re.IGNORECASE),
    "supabase": re.compile(r"supabase", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "organization_id": re.compile(r"organization_id"),
    "user_id": re.compile(r"user_id"),
    "matter_id": re.compile(r"matter_id"),
    "private app import": re.compile(r"(?<![A-Za-z_])app\."),
    "os.environ": re.compile(r"os\.environ|getenv"),
    "settings": re.compile(r"settings", re.IGNORECASE),
    "gpt-": re.compile(r"gpt-", re.IGNORECASE),
    "courtlistener": re.compile(r"courtlistener", re.IGNORECASE),
    "em dash": re.compile(f"{chr(0x2014)}|{chr(0x2013)}"),
}


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_retrieve_module_has_no_forbidden_text(label: str) -> None:
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(RETRIEVE_FILE.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found:\n" + "\n".join(hits)
