"""End to end: every finished stage wired through ``QueryPipeline``, with fakes only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from legal_rag.pipeline import (
    ANSWER_SCOPE_FIELDS,
    CACHE_DISABLED,
    CACHE_EXACT_HIT,
    CACHE_MISS,
    CACHE_SEMANTIC_HIT,
    PINNED_KEY,
    PipelineRequest,
    QueryPipeline,
    QueryPipelineConfig,
    jurisdictions,
)
from legal_rag.pipeline import query as query_module
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeReranker
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.types import RetrievedChunk

QUERY = "what notice must a landlord give before eviction under state law"
BASE_FILTERS = {"tenant": "acme"}


def chunk(cid: str, text: str, score: float = 0.8, **meta: Any) -> RetrievedChunk:
    meta.setdefault("document_id", f"doc-{cid}")
    meta.setdefault("title", f"Title {cid}")
    return RetrievedChunk(id=cid, text=text, score=score, metadata=dict(meta))


def corpus() -> list[RetrievedChunk]:
    """Six chunks that share words with ``QUERY``, so a word-overlap reranker scores them."""
    return [
        chunk(
            f"c{i}",
            f"A landlord must give written notice before eviction, rule {i} of the state law.",
        )
        for i in range(6)
    ]


class RecordingRetriever:
    """Returns fixed chunks and records every call, including the filters it was given."""

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "filters": filters, "sub_queries": sub_queries})
        return list(self.chunks)


class FlakyRetriever(RecordingRetriever):
    """Raises while ``failing`` is set, then behaves like ``RecordingRetriever``."""

    def __init__(self, chunks: Sequence[RetrievedChunk], *, failing: bool = True) -> None:
        super().__init__(chunks)
        self.failing = failing

    async def retrieve(self, query: str, **kwargs: Any) -> list[RetrievedChunk]:
        if self.failing:
            self.calls.append({"query": query, "filters": kwargs.get("filters")})
            raise RuntimeError("index unreachable")
        return await super().retrieve(query, **kwargs)


class RecordingReranker:
    """Word-overlap reranker that remembers every document it was asked to score."""

    def __init__(self) -> None:
        self.inner = FakeReranker()
        self.documents: list[str] = []

    async def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None):
        self.documents.extend(documents)
        return await self.inner.rerank(query, documents, top_k=top_k)


class RecordingFetcher:
    """A chunk fetcher that records the ``filters`` it receives. It stores nothing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def fetch_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        scope: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"ids": list(chunk_ids), "scope": scope, "filters": filters})
        return []


def build(retriever: Any = None, reranker: Any = None, **kwargs: Any) -> QueryPipeline:
    return QueryPipeline(
        retriever=retriever if retriever is not None else RecordingRetriever(corpus()),
        reranker=reranker if reranker is not None else FakeReranker(),
        **kwargs,
    )


def request(**kwargs: Any) -> PipelineRequest:
    kwargs.setdefault("query", QUERY)
    return PipelineRequest(**kwargs)


def cached_pipeline(retriever: Any, **kwargs: Any) -> tuple[QueryPipeline, FakeCache, FakeCache]:
    semantic_store, exact_store = FakeCache(), FakeCache()
    pipeline = build(
        retriever,
        semantic_cache=SemanticCache(FakeEmbedder(), semantic_store),
        cache=exact_store,
        **kwargs,
    )
    return pipeline, semantic_store, exact_store


# ---------------------------------------------------------------------------
# (a) standard tier happy path
# ---------------------------------------------------------------------------


async def test_standard_tier_returns_context_and_sources() -> None:
    retriever = RecordingRetriever(corpus())
    result = await build(retriever).run(request())
    assert result.tier == "standard"
    # The standard tier asks for an adaptive strategy, which runs plain hybrid without a model.
    assert result.strategy == "hybrid"
    assert result.metrics["retrieval"]["strategy_metadata"]["degraded_from"] == "self_rag"
    assert result.cache_status == CACHE_DISABLED
    assert result.warnings == []
    assert result.context and "written notice" in result.context
    assert result.sources and result.chunks
    assert len(retriever.calls) >= 1
    assert result.metrics["retrieval"]["chunk_count"] == 6
    assert {"retrieval", "rerank", "analyze"} <= set(result.metrics)
    assert result.usage.counts_by_operation()["query_pipeline"] == 1


# ---------------------------------------------------------------------------
# (b) deep tier selects multi-hop
# ---------------------------------------------------------------------------


async def test_deep_tier_selects_multi_hop() -> None:
    events: list[Any] = []

    async def sink(event: Any) -> None:
        events.append(event)

    result = await build().run(request(tier="deep"), on_event=sink)
    assert result.tier == "deep"
    assert result.strategy == "multi_hop"
    retrieval = result.metrics["retrieval"]
    assert retrieval["strategy"] == "multi_hop"
    assert "hops_performed" in retrieval["strategy_metadata"]
    started = next(e for e in events if isinstance(e, Mapping) and e["type"] == "pipeline_started")
    assert started["strategy"] == "multi_hop"
    done = [e["stage"] for e in events if isinstance(e, Mapping) and e["type"] == "stage_completed"]
    assert done == ["analyze", "retrieve", "rerank", "context"]


async def test_a_tier_resolver_picks_the_tier_when_the_request_names_none() -> None:
    result = await build().run(request(), tier_resolver=lambda _req: "deep")
    assert (result.tier, result.strategy) == ("deep", "multi_hop")


# ---------------------------------------------------------------------------
# (c) a cache hit short-circuits every stage
# ---------------------------------------------------------------------------


def spy_on_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    for name in ("analyze_stage", "retrieve_stage", "rerank_stage", "context_stage"):
        original = getattr(query_module, name)

        async def spy(
            ctx: Any, *args: Any, _original: Any = original, _name: str = name, **kw: Any
        ):
            calls.append(_name)
            return await _original(ctx, *args, **kw)

        monkeypatch.setattr(query_module, name, spy)
    return calls


async def test_a_semantic_cache_hit_runs_no_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = RecordingRetriever(corpus())
    pipeline, _, _ = cached_pipeline(retriever)
    first = await pipeline.run(request(filters=BASE_FILTERS))
    assert first.cache_status == CACHE_MISS

    stage_calls = spy_on_stages(monkeypatch)
    calls_before = len(retriever.calls)
    second = await pipeline.run(request(filters=BASE_FILTERS))
    assert second.cache_status == CACHE_SEMANTIC_HIT and second.cache_hit
    assert stage_calls == []
    assert len(retriever.calls) == calls_before
    assert second.context == first.context
    assert [c.id for c in second.chunks] == [c.id for c in first.chunks]
    assert second.sources == first.sources


async def test_an_exact_cache_hit_runs_no_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = RecordingRetriever(corpus())
    pipeline = build(retriever, cache=FakeCache())
    await pipeline.run(request())
    stage_calls = spy_on_stages(monkeypatch)
    calls_before = len(retriever.calls)
    second = await pipeline.run(request(query=f"  {QUERY}\n"))
    assert second.cache_status == CACHE_EXACT_HIT
    assert stage_calls == [] and len(retriever.calls) == calls_before


# ---------------------------------------------------------------------------
# (d) the tenant filter reaches the retriever and the chunk fetcher unchanged
# ---------------------------------------------------------------------------


async def test_the_request_filters_reach_the_retriever_and_the_fetcher_unchanged() -> None:
    linked = [
        chunk(
            f"n{i}",
            f"A landlord must give notice before eviction, part {i}.",
            next_chunk_id=f"x{i}",
        )
        for i in range(6)
    ]
    retriever = RecordingRetriever(linked)
    fetcher = RecordingFetcher()
    filters = {"tenant": "acme", "year_from": 2015}
    await build(retriever, fetcher=fetcher).run(request(filters=filters))
    assert retriever.calls and all(c["filters"] == filters for c in retriever.calls)
    assert fetcher.calls, "neighbor expansion should have asked the fetcher for the next chunks"
    assert all(c["filters"] == filters for c in fetcher.calls)
    assert filters == {"tenant": "acme", "year_from": 2015}


# ---------------------------------------------------------------------------
# (e) a named citation is pinned, skips the reranker and survives the gate
# ---------------------------------------------------------------------------

PINNED_TEXT = "Liability attaches to anyone acting with state authority."
CITE_QUERY = "what remedy exists for a landlord eviction notice dispute, see 42 U.S.C. 1983"


async def test_a_named_citation_is_pinned_and_survives_the_gate() -> None:
    async def section_lookup(citation: str) -> list[RetrievedChunk]:
        return [
            chunk(
                "usc-1983",
                PINNED_TEXT,
                0.1,
                document_id="usc-42-1983",
                source_type="statute",
                citation="42 U.S.C. 1983",
            )
        ]

    off_topic = chunk("off", "Zebras graze on open grassland far away.", 0.9)
    statutes = RecordingRetriever([*corpus(), off_topic])
    # A matching plug-in replaces the main retriever, so the corpus goes in through the plug-in.
    plugin = jurisdictions.get(
        "us", statute_retriever=statutes, statute_section_lookup=section_lookup
    )
    reranker = RecordingReranker()
    pipeline = build(RecordingRetriever([]), reranker=reranker, jurisdiction=plugin)
    result = await pipeline.run(request(query=CITE_QUERY, filters={"country_code": "US"}))

    pinned = [c for c in result.chunks if c.id == "usc-1983"]
    assert pinned and pinned[0].metadata[PINNED_KEY] is True
    assert pinned[0].metadata["pinned_by"] == "exact_lookup"
    assert PINNED_TEXT not in reranker.documents, "a pinned chunk must skip the reranker"
    assert reranker.documents, "the other chunks still go through the reranker"
    assert PINNED_TEXT in result.context
    assert "Zebras" not in result.context, "an unpinned off-topic chunk is dropped by the gate"
    assert result.metrics["retrieval"]["pinned_count"] == 1

    # Control: the same chunk found by ordinary search, not pinned, is dropped by the gate.
    plain = chunk("plain", PINNED_TEXT, 0.1)
    control = await build(RecordingRetriever([*corpus(), plain])).run(request(query=CITE_QUERY))
    assert PINNED_TEXT not in control.context


# ---------------------------------------------------------------------------
# (f) a trivial query skips retrieval
# ---------------------------------------------------------------------------


async def test_a_trivial_query_skips_retrieval() -> None:
    retriever = RecordingRetriever(corpus())
    pipeline, semantic_store, exact_store = cached_pipeline(retriever)
    result = await pipeline.run(request(query="hello"))
    assert retriever.calls == []
    assert result.metrics["retrieval"]["skipped"] is True
    assert result.context == "" and result.sources == []
    assert result.retrieval_warning == "no_results"
    assert not semantic_store.data and not exact_store.data, "nothing to cache without sources"


# ---------------------------------------------------------------------------
# (g) fail-open, (h) a failed retrieval is never cached
# ---------------------------------------------------------------------------


async def test_a_retriever_that_raises_still_returns_a_result_with_a_warning() -> None:
    retriever = FlakyRetriever(corpus())
    result = await build(retriever).run(request())
    assert "retrieval_failed" in result.warnings and result.degraded
    assert result.context == "" and result.sources == []
    assert retriever.calls, "the retriever was reached"


async def test_a_failed_retrieval_is_not_cached_and_the_next_run_recovers() -> None:
    retriever = FlakyRetriever(corpus())
    pipeline, semantic_store, exact_store = cached_pipeline(retriever)
    failed = await pipeline.run(request())
    assert "retrieval_failed" in failed.warnings
    assert not semantic_store.data and not exact_store.data

    retriever.failing = False
    recovered = await pipeline.run(request())
    assert recovered.cache_status == CACHE_MISS, "the failure was not replayed from cache"
    assert recovered.sources and recovered.warnings == []
    assert semantic_store.data and exact_store.data

    assert (await pipeline.run(request())).cache_hit


async def test_a_degraded_rerank_is_not_cached() -> None:
    class BrokenReranker:
        async def rerank(self, query: str, documents: Sequence[str], *, top_k: int | None = None):
            raise RuntimeError("rate limited")

    pipeline, semantic_store, exact_store = cached_pipeline(
        RecordingRetriever(corpus()), reranker=BrokenReranker(), fallback_reranker=FakeReranker()
    )
    result = await pipeline.run(request())
    assert "rerank_degraded" in result.warnings and result.sources
    assert not semantic_store.data and not exact_store.data


# ---------------------------------------------------------------------------
# (i) every registered answer-scope field reaches BOTH cache keys
# ---------------------------------------------------------------------------

BASE_REQUEST: dict[str, Any] = {
    "tier": "standard",
    "retrieval_strategy": "hybrid",
    "filters": dict(BASE_FILTERS),
}
SCOPE_VARIANTS: dict[str, dict[str, Any]] = {
    "tier": {"tier": "deep"},
    "retrieval_strategy": {"retrieval_strategy": "crag"},
    "enable_shared_corpus": {"enable_shared_corpus": False},
    "collection": {"collection": "other_collection"},
    "pinned_ids": {"pinned_ids": ["doc-1"]},
    "scope_lock_ids": {"scope_lock_ids": ["doc-1"]},
    "boost_ids": {"boost_ids": ["doc-1"]},
    "filters": {"filters": {"tenant": "globex"}},
    "chat_history": {"chat_history": [{"role": "user", "content": "an earlier question"}]},
}


def test_every_registered_scope_field_has_a_variant() -> None:
    assert set(SCOPE_VARIANTS) == set(ANSWER_SCOPE_FIELDS)


def keys_for(pipeline: QueryPipeline, req: PipelineRequest) -> Any:
    from legal_rag.pipeline import get_tier

    tier = get_tier(req.tier).name
    strategy = req.retrieval_strategy or get_tier(tier).retrieval_strategy
    return pipeline.cache_keys(req, tier=tier, retrieval_strategy=strategy)


@pytest.mark.parametrize("field", sorted(SCOPE_VARIANTS))
def test_a_change_in_each_scope_field_changes_both_cache_keys(field: str) -> None:
    pipeline = build()
    base = keys_for(pipeline, request(**BASE_REQUEST))
    changed = keys_for(pipeline, request(**{**BASE_REQUEST, **SCOPE_VARIANTS[field]}))
    assert changed.semantic_scope != base.semantic_scope, f"{field} missed the semantic key"
    assert changed.exact_key != base.exact_key, f"{field} missed the exact key"
    assert changed.scope.key() != base.scope.key()


@pytest.mark.parametrize("layer", ["semantic", "exact"])
@pytest.mark.parametrize("field", sorted(SCOPE_VARIANTS))
async def test_a_change_in_each_scope_field_misses_each_cache_layer(field: str, layer: str) -> None:
    retriever = RecordingRetriever(corpus())
    stores = {
        "semantic_cache": SemanticCache(FakeEmbedder(), FakeCache())
        if layer == "semantic"
        else None,
        "cache": FakeCache() if layer == "exact" else None,
    }
    pipeline = build(retriever, **stores)
    assert (await pipeline.run(request(**BASE_REQUEST))).cache_status == CACHE_MISS
    again = await pipeline.run(request(**BASE_REQUEST))
    assert again.cache_hit, "the same request must hit"

    variant = await pipeline.run(request(**{**BASE_REQUEST, **SCOPE_VARIANTS[field]}))
    assert variant.cache_status == CACHE_MISS, f"{field} did not change the {layer} cache key"


async def test_the_namespace_separates_two_deployments_sharing_one_store() -> None:
    store = FakeCache()
    one = build(cache=store, config=QueryPipelineConfig(namespace="corpus-a"))
    two = build(cache=store, config=QueryPipelineConfig(namespace="corpus-b"))
    await one.run(request())
    assert (await one.run(request())).cache_hit
    assert (await two.run(request())).cache_status == CACHE_MISS
