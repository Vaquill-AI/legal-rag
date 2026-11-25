"""Foundation slice: tier data, context budget, answer scope, usage ledger, protocols, retriever."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline import (
    ANSWER_SCOPE_FIELDS,
    AnswerScope,
    ComplexityCaps,
    ContextBudget,
    PipelineRequest,
    QueryContext,
    ScanResult,
    SparseVector,
    TierConfig,
    UsageLedger,
    VectorPoint,
    VectorStoreRetriever,
    available_tiers,
    build_answer_scope,
    get_tier,
    load_tiers,
)
from legal_rag.pipeline import protocols as P
from legal_rag.pipeline.tiers import _DATA_FILE
from legal_rag.pipeline.types import request_scope_field_names
from legal_rag.providers import protocols as providers
from legal_rag.providers.fakes import FakeEmbedder, FakeVectorStore
from legal_rag.types import Message, RetrievedChunk, UsageEvent


def run(coro):
    return asyncio.run(coro)


def _chunks(*ids: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(id=i, text=f"text {i}", score=1.0 - n * 0.1) for n, i in enumerate(ids)]


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def test_default_tiers_load_and_validate():
    assert available_tiers() == ("standard", "deep")
    for name, tier in load_tiers().items():
        assert isinstance(tier, TierConfig)
        assert tier.name == name
        tier.validate()


def test_tier_values_follow_the_plan():
    standard, deep = get_tier("standard"), get_tier("deep")
    assert (standard.retrieval_strategy, deep.retrieval_strategy) == ("self_rag", "multi_hop")
    assert (standard.top_k_retrieval, deep.top_k_retrieval) == (100, 200)
    assert (standard.top_k_rerank, deep.top_k_rerank) == (10, 20)
    assert (standard.rerank_input_cap, deep.rerank_input_cap) == (0, 200)
    assert (standard.rerank_min_chunks, deep.rerank_min_chunks) == (3, 5)
    assert (standard.neighbor_chunks, deep.neighbor_chunks) == (2, 3)
    assert deep.max_multi_hop_iterations == 5
    assert standard.rerank_score_threshold == deep.rerank_score_threshold == 0.25
    assert standard.features.use_reranking and deep.features.use_reranking
    assert not standard.features.use_hyde and deep.features.use_hyde
    assert not standard.features.use_multi_hop and deep.features.use_multi_hop
    assert deep.features.use_query_decomposer and not standard.features.use_query_decomposer


def test_unknown_tier_raises_value_error():
    for bad in ("fastest", "optimal", "premium", "", None):
        with pytest.raises(ValueError):
            get_tier(bad)


def test_tier_name_is_case_insensitive():
    assert get_tier(" Deep ").name == "deep"


def test_deep_has_larger_caps_than_standard():
    standard, deep = get_tier("standard"), get_tier("deep")
    assert deep.top_k_retrieval > standard.top_k_retrieval
    assert deep.top_k_rerank > standard.top_k_rerank
    assert deep.neighbor_chunks > standard.neighbor_chunks
    s, d = standard.complexity, deep.complexity
    for level in ("simple", "moderate"):
        assert d.retrieval_cap(level) > s.retrieval_cap(level)
        assert d.rerank_cap(level) > s.rerank_cap(level)
    assert d.neighbor_cap("simple") > s.neighbor_cap("simple")
    assert d.neighbor_cap("moderate") > s.neighbor_cap("moderate")
    assert d.case_law_retrieval_floor > s.case_law_retrieval_floor
    assert d.case_law_rerank_floor > s.case_law_rerank_floor
    assert d.single_document_rerank_floor > s.single_document_rerank_floor


def test_complexity_caps_values_and_uncapped():
    caps = get_tier("standard").complexity
    assert isinstance(caps, ComplexityCaps)
    assert [caps.retrieval_cap(x) for x in ("trivial", "simple", "moderate", "complex")] == [
        0,
        20,
        50,
        None,
    ]
    assert [caps.rerank_cap(x) for x in ("simple", "moderate", "complex")] == [5, 10, None]
    assert [caps.neighbor_cap(x) for x in ("trivial", "simple", "moderate", "complex")] == [
        0,
        0,
        1,
        None,
    ]
    assert caps.retrieval_cap("not-a-level") is None
    assert "case_law_research" in caps.case_law_task_types


def _raw() -> dict[str, Any]:
    with _DATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_unknown_keys_are_rejected_at_every_level():
    raw = _raw()
    bad_tier = copy.deepcopy(raw)
    bad_tier["tiers"]["standard"]["surprise"] = 1
    bad_feature = copy.deepcopy(raw)
    bad_feature["tiers"]["standard"]["features"]["use_surprise"] = True
    bad_complexity = copy.deepcopy(raw)
    bad_complexity["tiers"]["standard"]["complexity"]["surprise"] = 1
    bad_level = copy.deepcopy(raw)
    bad_level["tiers"]["standard"]["complexity"]["retrieval"]["gigantic"] = 5
    bad_top = copy.deepcopy(raw)
    bad_top["extra"] = {}
    for bad in (bad_tier, bad_feature, bad_complexity, bad_level, bad_top):
        with pytest.raises(ValueError):
            load_tiers(bad)


def test_validation_rejects_inconsistent_tiers():
    raw = _raw()
    cases = [
        ("retrieval_strategy", "unknown_strategy"),
        ("top_k_rerank", 500),
        ("rerank_input_cap", 3),
        ("rerank_score_threshold", 2.0),
        ("max_multi_hop_iterations", 9),
    ]
    for key, value in cases:
        bad = copy.deepcopy(raw)
        bad["tiers"]["deep"][key] = value
        with pytest.raises(ValueError):
            load_tiers(bad)
    no_hop = copy.deepcopy(raw)
    no_hop["tiers"]["deep"]["features"]["use_multi_hop"] = False
    with pytest.raises(ValueError):
        load_tiers(no_hop)
    missing = copy.deepcopy(raw)
    del missing["tiers"]["deep"]["top_k_rerank"]
    with pytest.raises(ValueError):
        load_tiers(missing)


def test_custom_tier_file_loads(tmp_path: Path):
    raw = _raw()
    raw["tiers"] = {"quick": raw["tiers"]["standard"]}
    path = tmp_path / "tiers.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert available_tiers(path) == ("quick",)
    assert get_tier("quick", path).top_k_retrieval == 100
    with pytest.raises(ValueError):
        get_tier("standard", path)


def test_tier_data_has_no_provider_or_price_keys():
    text = _DATA_FILE.read_text(encoding="utf-8").lower()
    for word in ("model", "provider", "cost", "price", "plan"):
        assert word not in text


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------


def test_context_budget_defaults():
    budget = ContextBudget()
    assert budget.total_tokens == 48000
    assert budget.reserve_tokens == 1000
    assert budget.max_chunks == 60
    assert budget.max_chunk_tokens == 2000
    assert budget.available_tokens == 47000


def test_context_budget_rejects_bad_values():
    with pytest.raises(ValueError):
        ContextBudget(total_tokens=1000, reserve_tokens=1000)
    with pytest.raises(ValueError):
        ContextBudget(max_chunks=0)


# ---------------------------------------------------------------------------
# AnswerScope
# ---------------------------------------------------------------------------


def _scope_inputs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "tier": "standard",
        "retrieval_strategy": "self_rag",
        "enable_shared_corpus": True,
        "collection": None,
        "pinned_ids": [],
        "scope_lock_ids": [],
        "boost_ids": [],
        "filters": {},
        "chat_history": [],
    }
    base.update(overrides)
    return base


def test_answer_scope_raises_on_a_missing_field():
    for name in ANSWER_SCOPE_FIELDS:
        inputs = _scope_inputs()
        del inputs[name]
        with pytest.raises(TypeError, match=name):
            AnswerScope.build(**inputs)


def test_answer_scope_raises_on_an_unknown_field():
    with pytest.raises(TypeError, match="surprise"):
        AnswerScope.build(**_scope_inputs(surprise="x"))


def test_every_request_field_is_registered_or_exempt():
    assert request_scope_field_names() <= set(ANSWER_SCOPE_FIELDS)
    assert {f.name for f in fields(PipelineRequest)} >= request_scope_field_names()


def test_two_scopes_differing_in_any_one_field_have_different_keys():
    base = AnswerScope.build(**_scope_inputs())
    changed = {
        "tier": "deep",
        "retrieval_strategy": "multi_hop",
        "enable_shared_corpus": False,
        "collection": "other",
        "pinned_ids": ["d1"],
        "scope_lock_ids": ["c1"],
        "boost_ids": ["b1"],
        "filters": {"court": "ca9"},
        "chat_history": [{"role": "user", "content": "earlier question"}],
    }
    assert set(changed) == set(ANSWER_SCOPE_FIELDS)
    keys = {base.key()}
    for name, value in changed.items():
        key = AnswerScope.build(**_scope_inputs(**{name: value})).key()
        assert key != base.key(), name
        keys.add(key)
    assert len(keys) == len(ANSWER_SCOPE_FIELDS) + 1


def test_answer_scope_normalizes_absent_values_and_id_order():
    a = AnswerScope.build(**_scope_inputs(collection="  ", filters={"court": None, "tags": []}))
    b = AnswerScope.build(**_scope_inputs())
    assert a.key() == b.key()
    ids1 = AnswerScope.build(**_scope_inputs(pinned_ids=["b", "a", "a"]))
    ids2 = AnswerScope.build(**_scope_inputs(pinned_ids=["a", "b"]))
    assert ids1.key() == ids2.key()
    assert build_answer_scope(**_scope_inputs()).key() == b.key()


def test_nested_filter_changes_reach_the_key():
    one = AnswerScope.build(**_scope_inputs(filters={"date": {"from": 2000, "to": 2010}}))
    two = AnswerScope.build(**_scope_inputs(filters={"date": {"from": 2000, "to": 2011}}))
    off = AnswerScope.build(**_scope_inputs(filters={"include_repealed": False}))
    assert one.key() != two.key()
    assert off.key() != AnswerScope.build(**_scope_inputs()).key()


def test_chat_history_digest_accepts_messages_and_mappings():
    as_message = AnswerScope.build(**_scope_inputs(chat_history=[Message("user", "hi")]))
    as_mapping = AnswerScope.build(
        **_scope_inputs(chat_history=[{"role": "user", "content": "hi"}])
    )
    assert as_message.key() == as_mapping.key()


def test_scope_from_request_and_context_agree():
    request = PipelineRequest(
        query="q",
        filters={"court": "ca9"},
        pinned_ids=["d1"],
        collection="c",
        chat_history=[{"role": "user", "content": "hi"}],
    )
    ctx = QueryContext.from_request(request, tier="deep", retrieval_strategy="multi_hop")
    from_request = AnswerScope.from_request(request, tier="deep", retrieval_strategy="multi_hop")
    assert ctx.answer_scope().key() == from_request.key()
    assert ctx.pinned_ids == ["d1"] and ctx.collection == "c"


def test_query_context_defaults_and_warning_dedup():
    ctx = QueryContext(query="q")
    assert ctx.retrieved == [] and ctx.reranked == [] and ctx.context_chunks == []
    assert ctx.retrieval_warning is None and ctx.extra == {} and ctx.metrics == {}
    ctx.add_warning("degraded")
    ctx.add_warning("degraded")
    assert ctx.warnings == ["degraded"]


# ---------------------------------------------------------------------------
# UsageLedger
# ---------------------------------------------------------------------------


def test_usage_ledger_collects_events():
    seen: list[UsageEvent] = []
    ledger = UsageLedger(downstream=seen.append)
    ledger(UsageEvent(operation="embed", provider="p1", input_tokens=10))
    ledger.record(UsageEvent(operation="generate", provider="p2", input_tokens=5, output_tokens=7))
    ledger(UsageEvent(operation="embed"))
    assert len(ledger) == 3 and len(seen) == 3
    assert ledger.total_input_tokens == 15 and ledger.total_output_tokens == 7
    assert ledger.counts_by_operation() == {"embed": 2, "generate": 1}
    assert ledger.counts_by_provider() == {"p1": 1, "p2": 1}
    assert callable(ledger)
    ledger.clear()
    assert ledger.events == []


# ---------------------------------------------------------------------------
# Protocols and the default retriever
# ---------------------------------------------------------------------------


class RecordingStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def search(self, vector, *, top_k, filters=None):
        self.calls.append({"top_k": top_k, "filters": filters})
        return self.chunks[:top_k]


class HybridStore(RecordingStore):
    async def search_hybrid(self, vector, sparse, *, top_k, filters=None):
        self.calls.append({"top_k": top_k, "filters": filters, "sparse": sparse})
        return self.chunks[:top_k]


class FakeSparse:
    def encode_documents(self, texts):
        return [SparseVector([1], [1.0]) for _ in texts]

    def encode_query(self, text):
        return SparseVector([2], [0.5])


def test_vector_store_retriever_returns_store_chunks():
    chunks = _chunks("a", "b", "c")
    retriever = VectorStoreRetriever(FakeEmbedder(), FakeVectorStore(chunks))
    result = run(retriever.retrieve("what is a tort", top_k=2))
    assert [c.id for c in result] == ["a", "b"]
    assert isinstance(retriever, P.Retriever)


def test_vector_store_retriever_passes_filters_through_unchanged():
    store = RecordingStore(_chunks("a"))
    retriever = VectorStoreRetriever(FakeEmbedder(), store)
    filters = {"court": "ca9", "tenant": "t-1"}
    run(retriever.retrieve("q", top_k=5, filters=filters))
    assert store.calls == [{"top_k": 5, "filters": filters}]
    assert store.calls[0]["filters"] is filters


def test_vector_store_retriever_merges_sub_queries_with_rank_fusion():
    store = RecordingStore(_chunks("a", "b"))
    retriever = VectorStoreRetriever(FakeEmbedder(), store)
    result = run(retriever.retrieve("q", top_k=5, sub_queries=["q", "other", " "]))
    assert len(store.calls) == 2
    assert [c.id for c in result] == ["a", "b"]
    assert run(retriever.retrieve("q", top_k=0)) == []


def test_vector_store_retriever_uses_hybrid_search_when_available():
    hybrid = HybridStore(_chunks("a"))
    run(VectorStoreRetriever(FakeEmbedder(), hybrid, sparse=FakeSparse()).retrieve("q", top_k=3))
    assert hybrid.calls[0]["sparse"].indices == [2]
    plain = RecordingStore(_chunks("a"))
    run(VectorStoreRetriever(FakeEmbedder(), plain, sparse=FakeSparse()).retrieve("q", top_k=3))
    assert "sparse" not in plain.calls[0]


class MinimalLanguageModel:
    async def generate(self, messages, *, max_tokens=None, temperature=None):
        return ""

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        raise NotImplementedError


class MinimalReranker:
    async def rerank(self, query, documents, *, top_k=None):
        return []


class MinimalCache:
    async def get(self, key):
        return None

    async def set(self, key, value, *, ttl_seconds=None):
        return None

    async def delete(self, key):
        return None


class MinimalWriter:
    async def upsert(self, points):
        return None

    async def delete(self, ids):
        return None


class MinimalScanner:
    def scan(self, text):
        return ScanResult(safe=True)


class MinimalLookup:
    async def lookup(self, query):
        return []


class MinimalWeb:
    async def search(self, query, top_k):
        return []


class MinimalFetcher:
    async def fetch_by_ids(self, chunk_ids, *, scope=None):
        return []


class MinimalPlugin:
    condenser_key = "demo"

    def matches(self, filters):
        return bool(filters)

    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        return []

    def adjusters(self):
        return []

    def exact_lookups(self):
        return [MinimalLookup()]

    def label_for(self, chunk):
        return chunk.id


def test_every_protocol_is_satisfied_by_a_minimal_fake():
    pairs = [
        (P.Retriever, VectorStoreRetriever(FakeEmbedder(), FakeVectorStore())),
        (P.SparseEncoder, FakeSparse()),
        (P.VectorWriter, MinimalWriter()),
        (P.HybridVectorStore, HybridStore([])),
        (P.InjectionScanner, MinimalScanner()),
        (P.ExactLookup, MinimalLookup()),
        (P.WebSearch, MinimalWeb()),
        (P.JurisdictionPlugin, MinimalPlugin()),
        (providers.LanguageModel, MinimalLanguageModel()),
        (providers.Embedder, FakeEmbedder()),
        (providers.Reranker, MinimalReranker()),
        (providers.VectorStore, FakeVectorStore()),
        (providers.Cache, MinimalCache()),
    ]
    for protocol, impl in pairs:
        assert isinstance(impl, protocol), protocol.__name__
    assert not isinstance(object(), P.Retriever)
    assert not isinstance(object(), P.JurisdictionPlugin)


def test_reexported_contracts_are_the_existing_ones():
    from legal_rag.retrieval.accumulators import neighbor
    from legal_rag.retrieval.accumulators.base import ChunkFetcher

    assert P.ChunkFetcher is ChunkFetcher
    assert P.Embedder is providers.Embedder
    assert P.LanguageModel is providers.LanguageModel
    assert P.UsageHook is providers.UsageHook
    assert P.TokenCounter == neighbor.TokenCounter
    assert run(MinimalFetcher().fetch_by_ids(["a"])) == []


def test_scan_result_and_vector_types():
    result = ScanResult(safe=False, findings=("override",), sanitized_text="clean")
    assert not result.safe and result.findings == ("override",)
    with pytest.raises(ValueError):
        SparseVector(indices=[1, 2], values=[1.0])
    point = VectorPoint(id="p", vector=[0.1], sparse=SparseVector([1], [1.0]), payload={"k": 1})
    assert point.payload == {"k": 1}


def test_type_aliases_are_usable():
    async def resolver_sink(event: Any) -> None:
        return None

    def resolve(request: PipelineRequest) -> str:
        return "deep" if request.query.startswith("deep") else "standard"

    def adjust(query, chunks, filters):
        return list(chunks)

    sink: P.EventSink = resolver_sink
    tier_resolver: P.TierResolver = resolve
    adjuster: P.ChunkAdjuster = adjust
    assert run(sink({"hop": 1})) is None
    assert tier_resolver(PipelineRequest(query="deep dive")) == "deep"
    assert adjuster("q", _chunks("a"), {})[0].id == "a"
