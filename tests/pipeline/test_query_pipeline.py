"""Unit behavior of ``QueryPipeline``: tiers, cache layers, retry, usage, events, runner, contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from legal_rag.ingestion import router as ingestion_router
from legal_rag.pipeline import (
    CACHE_DISABLED,
    CACHE_EXACT_HIT,
    CACHE_MISS,
    CACHE_SEMANTIC_HIT,
    DEFAULT_STRATEGIES,
    PINNED_BY_KEY,
    PINNED_KEY,
    ContextBudget,
    PipelineRequest,
    QueryContext,
    QueryPipeline,
    QueryPipelineConfig,
    RetrieveConfig,
    UnknownStrategyError,
    UsageLedger,
    build_cache_keys,
    run_stages,
    select_within_budget,
    whitespace_token_counter,
)
from legal_rag.pipeline.jurisdictions.us import merge_pinned_chunks
from legal_rag.pipeline.query import (
    WARNING_TIER_RESOLUTION_FAILED,
    exact_cache_key,
    normalize_query,
)
from legal_rag.pipeline.stages.retrieve import StrategyOutcome, run_hybrid
from legal_rag.providers import usage as usage_module
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeReranker
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.retrieval.strategies import _common
from legal_rag.types import RetrievedChunk, UsageEvent

QUERY = "what notice must a landlord give before eviction under state law"
QUERY_PATH = Path(__file__).resolve().parents[2] / "src" / "legal_rag" / "pipeline" / "query.py"


def corpus(n: int = 6) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=f"c{i}",
            text=f"A landlord must give written notice before eviction, rule {i} of the state law.",
            score=0.8,
            metadata={"document_id": f"doc-{i}", "title": f"Title {i}"},
        )
        for i in range(n)
    ]


class Retriever:
    def __init__(self, chunks: Sequence[RetrievedChunk] | None = None) -> None:
        self.chunks = corpus() if chunks is None else list(chunks)
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "filters": filters})
        return list(self.chunks)


def make(retriever: Retriever | None = None, **kwargs: Any) -> QueryPipeline:
    return QueryPipeline(retriever=retriever or Retriever(), reranker=FakeReranker(), **kwargs)


def req(**kwargs: Any) -> PipelineRequest:
    kwargs.setdefault("query", QUERY)
    return PipelineRequest(**kwargs)


# ---------------------------------------------------------------------------
# Tier and strategy resolution
# ---------------------------------------------------------------------------


async def test_the_default_tier_is_standard_and_caches_report_disabled() -> None:
    result = await make().run(req())
    assert result.tier == "standard" and result.cache_status == CACHE_DISABLED


async def test_the_request_tier_beats_the_resolver() -> None:
    result = await make().run(req(tier="deep"), tier_resolver=lambda _r: "standard")
    assert result.tier == "deep"


async def test_an_async_resolver_is_awaited() -> None:
    async def resolver(_request: PipelineRequest) -> str:
        return "deep"

    assert (await make().run(req(), tier_resolver=resolver)).tier == "deep"


async def test_a_failing_resolver_falls_back_to_the_default_tier_with_a_warning() -> None:
    def broken(_request: PipelineRequest) -> str:
        raise RuntimeError("plan lookup down")

    result = await make().run(req(), tier_resolver=broken)
    assert result.tier == "standard"
    assert WARNING_TIER_RESOLUTION_FAILED in result.warnings
    assert result.sources, "the query still ran"


async def test_an_unknown_tier_is_a_setup_error_that_touches_nothing() -> None:
    retriever, store = Retriever(), FakeCache()
    pipeline = make(retriever, cache=store)
    with pytest.raises(ValueError, match="Unknown tier"):
        await pipeline.run(req(tier="platinum"))
    with pytest.raises(ValueError, match="Unknown tier"):
        await pipeline.run(req(), tier_resolver=lambda _r: "legacy")
    assert retriever.calls == [] and store.data == {}


async def test_an_unknown_strategy_raises_before_any_cache_or_provider_is_used() -> None:
    retriever, store = Retriever(), FakeCache()
    with pytest.raises(UnknownStrategyError):
        await make(retriever, cache=store).run(req(retrieval_strategy="telepathy"))
    assert retriever.calls == [] and store.data == {}


async def test_a_blank_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="blank"):
        await make().run(req(query="   "))


async def test_the_request_strategy_beats_the_tier_default() -> None:
    result = await make().run(req(retrieval_strategy="hybrid", tier="deep"))
    assert result.strategy == "hybrid"


# ---------------------------------------------------------------------------
# Cache keys and layers
# ---------------------------------------------------------------------------


def test_the_exact_key_normalizes_whitespace_but_keeps_case() -> None:
    request = req()
    keys = build_cache_keys("ns", request, tier="standard", retrieval_strategy="hybrid")
    assert normalize_query("  a   b\n c ") == "a b c"
    same = exact_cache_key("ns", f" {QUERY}\n", keys.scope)
    assert same == keys.exact_key
    assert exact_cache_key("ns", QUERY.upper(), keys.scope) != keys.exact_key


def test_the_namespace_is_part_of_both_keys() -> None:
    request = req()
    one = build_cache_keys("a", request, tier="standard", retrieval_strategy="hybrid")
    two = build_cache_keys("b", request, tier="standard", retrieval_strategy="hybrid")
    assert one.semantic_scope != two.semantic_scope and one.exact_key != two.exact_key
    assert one.semantic_scope.startswith("a:") and one.exact_key.startswith("a:query:")
    assert re.fullmatch(r"a:query:[0-9a-f]{64}", one.exact_key)


def test_a_blank_namespace_is_rejected() -> None:
    with pytest.raises(ValueError, match="namespace"):
        QueryPipelineConfig(namespace="  ")


async def test_layer_one_is_the_semantic_cache_and_layer_two_the_exact_cache() -> None:
    retriever = Retriever()
    semantic_store, exact_store = FakeCache(), FakeCache()
    pipeline = make(
        retriever, semantic_cache=SemanticCache(FakeEmbedder(), semantic_store), cache=exact_store
    )
    assert (await pipeline.run(req())).cache_status == CACHE_MISS
    assert semantic_store.data and exact_store.data
    assert (await pipeline.run(req())).cache_status == CACHE_SEMANTIC_HIT

    await semantic_store.delete(next(iter(semantic_store.data)))
    assert (await pipeline.run(req())).cache_status == CACHE_EXACT_HIT
    assert len(retriever.calls) == 1, "only the first run reached the retriever"


async def test_the_cached_result_round_trips_chunks_and_sources() -> None:
    pipeline = make(cache=FakeCache())
    first = await pipeline.run(req())
    second = await pipeline.run(req())
    assert second.cache_hit and second.tier == first.tier and second.strategy == first.strategy
    assert [(c.id, c.text, c.score) for c in second.chunks] == [
        (c.id, c.text, c.score) for c in first.chunks
    ]
    assert second.context == first.context and second.sources == first.sources
    assert second.metrics["pipeline"].keys() == {"elapsed_ms"}


async def test_an_unusable_cache_entry_is_a_miss() -> None:
    store = FakeCache()
    pipeline = make(cache=store)
    await pipeline.run(req())
    (key,) = store.data
    for garbage in (b"not json", json.dumps({"context": 5}).encode(), json.dumps([]).encode()):
        store.data[key] = garbage
        assert (await pipeline.run(req())).cache_status == CACHE_MISS


async def test_a_cache_that_raises_never_fails_the_query() -> None:
    class BrokenCache:
        async def get(self, key: str) -> bytes | None:
            raise ConnectionError("down")

        async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
            raise ConnectionError("down")

        async def delete(self, key: str) -> None:
            raise ConnectionError("down")

    result = await make(
        cache=BrokenCache(), semantic_cache=SemanticCache(FakeEmbedder(), BrokenCache())
    ).run(req())
    assert result.sources and result.cache_status == CACHE_MISS


async def test_a_source_enrichment_failure_is_degraded_and_not_cached() -> None:
    def enricher(sources: list[dict[str, Any]]) -> None:
        raise RuntimeError("lookup failed")

    store = FakeCache()
    result = await make(cache=store, source_enricher=enricher).run(req())
    assert "source_enrichment_failed" in result.warnings and result.degraded
    assert store.data == {}


# ---------------------------------------------------------------------------
# No-results retry with force_retrieval_strategy
# ---------------------------------------------------------------------------


async def empty_strategy(run: Any) -> StrategyOutcome:
    return [], {"adaptive": "graded everything out"}


def empty_adaptive_table() -> dict[str, Any]:
    return {**DEFAULT_STRATEGIES, "self_rag": empty_strategy}


NO_STAGE_RETRY = QueryPipelineConfig(retrieve=RetrieveConfig(retry_empty_with_hybrid=False))


async def test_no_results_retries_retrieve_and_the_later_stages_with_hybrid() -> None:
    retriever = Retriever()
    events: list[Any] = []

    async def sink(event: Any) -> None:
        events.append(event)

    pipeline = make(retriever, strategies=empty_adaptive_table(), config=NO_STAGE_RETRY)
    result = await pipeline.run(req(), on_event=sink)
    assert result.sources and result.strategy == "hybrid"
    retry = result.metrics["no_results_retry"]
    assert retry["from_strategy"] == "self_rag"
    assert retry["first_pass"]["retrieval"]["chunk_count"] == 0
    assert len(retriever.calls) == 1, "only the forced hybrid pass searched"
    assert [e["type"] for e in events if isinstance(e, Mapping)].count("retry") == 1
    stages = [
        e["stage"] for e in events if isinstance(e, Mapping) and e["type"] == "stage_completed"
    ]
    assert stages == ["analyze", "retrieve", "rerank", "context", "retrieve", "rerank", "context"]
    assert result.retrieval_warning is None and "retrieval_failed" not in result.warnings


async def test_no_retry_when_it_is_off_when_hybrid_already_ran_or_retrieval_was_skipped() -> None:
    off = QueryPipelineConfig(retrieve=NO_STAGE_RETRY.retrieve, retry_no_results_with_hybrid=False)
    result = await make(strategies=empty_adaptive_table(), config=off).run(req())
    assert result.sources == [] and "no_results_retry" not in result.metrics

    hybrid_empty = make(Retriever([]), config=NO_STAGE_RETRY)
    result = await hybrid_empty.run(req(retrieval_strategy="hybrid"))
    assert result.retrieval_warning == "no_results" and "no_results_retry" not in result.metrics

    skipped = await make(config=NO_STAGE_RETRY).run(req(query="thanks"))
    assert skipped.metrics["retrieval"]["skipped"] and "no_results_retry" not in skipped.metrics


async def test_no_retry_after_a_retrieval_failure() -> None:
    class Failing(Retriever):
        async def retrieve(self, query: str, **kwargs: Any) -> list[RetrievedChunk]:
            self.calls.append({})
            raise RuntimeError("boom")

    retriever = Failing()
    result = await make(retriever, config=NO_STAGE_RETRY).run(req())
    assert "retrieval_failed" in result.warnings and "no_results_retry" not in result.metrics
    assert len(retriever.calls) == 1


# ---------------------------------------------------------------------------
# Usage, events, config errors
# ---------------------------------------------------------------------------


async def test_usage_reaches_the_ledger_and_the_downstream_hook() -> None:
    seen: list[UsageEvent] = []
    result = await make(usage=seen.append).run(req())
    assert isinstance(result.usage, UsageLedger)
    assert result.usage.events == seen
    assert "query_pipeline" in result.usage.counts_by_operation()
    pipeline_event = next(e for e in seen if e.operation == "query_pipeline")
    assert pipeline_event.extra["cache_status"] == CACHE_DISABLED
    assert pipeline_event.input_tokens and pipeline_event.output_tokens


async def test_a_cache_hit_still_records_one_usage_event() -> None:
    pipeline = make(cache=FakeCache())
    await pipeline.run(req())
    hit = await pipeline.run(req())
    assert hit.usage.counts_by_operation() == {"query_pipeline": 1}
    assert hit.usage.events[0].extra["cache_status"] == CACHE_EXACT_HIT


async def test_a_failing_event_sink_and_a_failing_usage_hook_do_not_break_the_query() -> None:
    async def bad_sink(event: Any) -> None:
        raise RuntimeError("socket closed")

    def bad_hook(event: UsageEvent) -> None:
        raise RuntimeError("meter down")

    result = await make(usage=bad_hook).run(req(), on_event=bad_sink)
    assert result.sources


async def test_a_pipeline_is_reusable_across_requests() -> None:
    pipeline = make()
    one, two = await pipeline.run(req()), await pipeline.run(req(tier="deep"))
    assert one.tier == "standard" and two.tier == "deep"
    assert one.usage is not two.usage


# ---------------------------------------------------------------------------
# run_stages
# ---------------------------------------------------------------------------


async def test_run_stages_runs_in_order_and_times_each_stage() -> None:
    order: list[str] = []

    async def first(ctx: QueryContext) -> QueryContext:
        order.append("first")
        return replace(ctx, extra={**ctx.extra, "seen": ["first"]})

    async def second(ctx: QueryContext) -> QueryContext:
        order.append("second")
        return replace(ctx, extra={**ctx.extra, "seen": [*ctx.extra["seen"], "second"]})

    timings: dict[str, float] = {}
    events: list[Any] = []

    async def sink(event: Any) -> None:
        events.append(event)

    out = await run_stages(
        QueryContext(query="q"), [first, ("two", second)], on_event=sink, timings=timings
    )
    assert order == ["first", "second"] and out.extra["seen"] == ["first", "second"]
    assert set(timings) == {"first", "two"}
    assert [e["stage"] for e in events] == ["first", "two"]


async def test_run_stages_rejects_a_stage_that_does_not_return_a_context() -> None:
    async def broken(ctx: QueryContext) -> Any:
        return None

    with pytest.raises(TypeError, match="broken"):
        await run_stages(QueryContext(query="q"), [broken])


async def test_run_stages_does_not_swallow_a_configuration_error() -> None:
    async def misconfigured(ctx: QueryContext) -> QueryContext:
        raise UnknownStrategyError("no such strategy")

    with pytest.raises(UnknownStrategyError):
        await run_stages(QueryContext(query="q"), [misconfigured])


async def test_a_stage_configuration_error_propagates_out_of_run() -> None:
    async def tier_table_gone(run: Any) -> StrategyOutcome:
        raise UnknownStrategyError("table changed")

    table = {**DEFAULT_STRATEGIES, "self_rag": tier_table_gone}
    assert (await make(strategies=table).run(req())).warnings == ["retrieval_failed"], (
        "a runtime strategy error is fail-open"
    )
    with pytest.raises(UnknownStrategyError):
        await make(strategies={"hybrid": run_hybrid}).run(req())


# ---------------------------------------------------------------------------
# Contract fix 1: one pinned-chunk flag
# ---------------------------------------------------------------------------


def test_the_pinned_flag_is_one_shared_constant() -> None:
    from legal_rag.pipeline.stages.rerank import PINNED_FLAG

    assert PINNED_KEY == "is_pinned" and PINNED_BY_KEY == "pinned_by"
    assert PINNED_FLAG == PINNED_KEY


def test_merge_pinned_chunks_sets_the_shared_pin_flag() -> None:
    found = RetrievedChunk("hit", "text", 0.4, {"document_id": "d1"})
    looked_up = RetrievedChunk("pin", "text", 0.97, {"document_id": "d1"})
    new = RetrievedChunk("new", "text", 0.97, {"document_id": "d2"})
    merged = merge_pinned_chunks([found], [looked_up, new])
    by_id = {c.id: c for c in merged}
    assert by_id["hit"].metadata[PINNED_KEY] is True
    assert "new" in by_id and merged[0].id == "new"


def test_the_context_budget_reserves_a_pinned_chunk_before_a_higher_scored_one() -> None:
    words = " ".join(["word"] * 30)
    strong = RetrievedChunk("strong", words, 0.9)
    pinned = RetrievedChunk("pinned", words, 0.1, {PINNED_KEY: True})
    budget = ContextBudget(total_tokens=50, reserve_tokens=10, max_chunks=5, max_chunk_tokens=40)
    chosen = select_within_budget([strong, pinned], budget, whitespace_token_counter)
    assert [c.id for _, c, _, _ in chosen.kept] == ["pinned"]
    plain = replace(pinned, metadata={})
    chosen = select_within_budget([strong, plain], budget, whitespace_token_counter)
    assert [c.id for _, c, _, _ in chosen.kept] == ["strong"]


# ---------------------------------------------------------------------------
# Contract fix 2: public helpers with the old names kept
# ---------------------------------------------------------------------------


def test_the_ingestion_error_helper_is_public_and_the_old_name_still_works() -> None:
    assert ingestion_router.unreadable_error_message is ingestion_router._unreadable_error_message


def test_emit_usage_lives_in_the_providers_package_and_is_re_exported() -> None:
    assert _common.emit_usage is usage_module.emit_usage
    assert _common.estimate_tokens is usage_module.estimate_tokens
    seen: list[UsageEvent] = []
    usage_module.emit_usage(seen.append, "op", input_text="abcd", output_text="", tag="x")
    assert seen == [UsageEvent("op", input_tokens=1, output_tokens=None, extra={"tag": "x"})]
    usage_module.emit_usage(None, "op")

    def failing(event: UsageEvent) -> None:
        raise RuntimeError("never raised to the caller")

    usage_module.emit_usage(failing, "op")


def test_ingest_imports_the_public_names() -> None:
    source = (QUERY_PATH.parent / "ingest.py").read_text(encoding="utf-8")
    assert "import DocumentRouter, unreadable_error_message" in source
    assert "from legal_rag.providers.usage import emit_usage" in source
    assert "_unreadable_error_message" not in source


# ---------------------------------------------------------------------------
# Hygiene of query.py
# ---------------------------------------------------------------------------

QUERY_FORBIDDEN = {
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
    "langgraph": re.compile(r"lang_?graph", re.IGNORECASE),
    "em dash": re.compile(f"{chr(0x2014)}|{chr(0x2013)}"),
}


@pytest.mark.parametrize("label", sorted(QUERY_FORBIDDEN))
def test_query_module_has_no_forbidden_text(label: str) -> None:
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(QUERY_PATH.read_text(encoding="utf-8").splitlines(), 1)
        if QUERY_FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found in query.py:\n" + "\n".join(hits)
