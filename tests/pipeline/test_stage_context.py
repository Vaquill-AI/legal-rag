"""Context assembly stage: budget, neighbors, dedupe, order, labels, confidence and fail-open."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline import ContextBudget, QueryContext
from legal_rag.pipeline.stages import context as context_module
from legal_rag.pipeline.stages.context import (
    ContextStageConfig,
    context_stage,
    default_label_for,
    filters_scope,
    resolve_pipeline_confidence,
    select_within_budget,
    tiktoken_token_counter,
    whitespace_token_counter,
)
from legal_rag.pipeline.tiers import _DATA_FILE
from legal_rag.types import RetrievedChunk, UsageEvent

count = whitespace_token_counter


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def chunk(cid: str, score: float = 0.8, n_words: int = 20, **meta: Any) -> RetrievedChunk:
    meta.setdefault("title", f"Doc {cid}")
    meta.setdefault("document_id", f"doc-{cid}")
    return RetrievedChunk(id=cid, text=words(n_words, cid), score=score, metadata=meta)


def make_ctx(chunks: Sequence[RetrievedChunk], **kwargs: Any) -> QueryContext:
    kwargs.setdefault("tier", "standard")
    return QueryContext(query="what is the rule", reranked=list(chunks), **kwargs)


def context_of(ctx: QueryContext) -> str:
    return ctx.context


class FakeFetcher:
    """Chunk fetcher with a store. Declares ``filters`` so the stage passes them."""

    def __init__(self, store: Sequence[RetrievedChunk] = ()) -> None:
        self.store = {c.id: c for c in store}
        self.calls: list[dict[str, Any]] = []

    async def fetch_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        scope: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"ids": list(chunk_ids), "scope": scope, "filters": filters})
        return [self.store[i] for i in chunk_ids if i in self.store]


class ScopeOnlyFetcher:
    """A fetcher written against the plain protocol: no ``filters`` keyword."""

    def __init__(self, store: Sequence[RetrievedChunk] = ()) -> None:
        self.store = {c.id: c for c in store}
        self.scopes: list[str | None] = []

    async def fetch_by_ids(
        self, chunk_ids: Sequence[str], *, scope: str | None = None
    ) -> list[RetrievedChunk]:
        self.scopes.append(scope)
        return [self.store[i] for i in chunk_ids if i in self.store]


def chain(*ids: str) -> list[RetrievedChunk]:
    """Chunks linked into a chain by prev and next ids, each from its own document."""
    out = []
    for n, cid in enumerate(ids):
        meta: dict[str, Any] = {}
        if n > 0:
            meta["prev_chunk_id"] = ids[n - 1]
        if n < len(ids) - 1:
            meta["next_chunk_id"] = ids[n + 1]
        out.append(chunk(cid, 0.9, 10, **meta))
    return out


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_total_budget_is_respected_and_best_chunk_at_the_tail_survives():
    chunks = [chunk(f"c{i}", score=0.3 + i * 0.02, n_words=100) for i in range(30)]
    budget = ContextBudget(
        total_tokens=1200, reserve_tokens=200, max_chunks=60, max_chunk_tokens=500
    )
    ctx = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=budget), token_counter=count
    )
    assert 0 < len(ctx.context_chunks) < 30
    assert count(context_of(ctx)) <= budget.available_tokens
    ids = [c.id for c in ctx.context_chunks]
    assert "c29" in ids, "the best chunk sits last after a reorder and must not be evicted"
    assert ids == sorted(ids, key=lambda i: int(i[1:])), "kept chunks keep their input order"
    assert ctx.metrics["context_budget"]["chunks_dropped"] == 30 - len(ids)


async def test_per_chunk_limit_truncates_and_flags():
    chunks = [chunk(f"c{i}", 0.9 - i * 0.1, n_words=1000) for i in range(3)]
    budget = ContextBudget(max_chunk_tokens=200)
    ctx = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=budget), token_counter=count
    )
    assert len(ctx.context_chunks) == 3
    for kept in ctx.context_chunks:
        assert count(kept.text) <= 200
        assert kept.metadata["truncated"] is True
        assert kept.text.endswith("...")
    assert ctx.metrics["context_budget"]["chunks_truncated"] == 3
    assert chunks[0].text.endswith("c0999"), "the caller's chunk is never mutated"
    assert "truncated" not in chunks[0].metadata


async def test_max_chunks_keeps_the_best_scored_not_the_first():
    chunks = [chunk(f"c{i}", score=i / 100, n_words=5) for i in range(20)]
    budget = ContextBudget(max_chunks=5)
    ctx = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=budget), token_counter=count
    )
    assert [c.id for c in ctx.context_chunks] == ["c15", "c16", "c17", "c18", "c19"]


async def test_reserve_tokens_are_honoured():
    chunks = [chunk(f"c{i}", 0.9 - i * 0.01, n_words=50) for i in range(15)]
    loose = ContextBudget(total_tokens=1000, reserve_tokens=0)
    tight = ContextBudget(total_tokens=1000, reserve_tokens=900)
    wide = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=loose), token_counter=count
    )
    narrow = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=tight), token_counter=count
    )
    assert count(context_of(narrow)) <= tight.available_tokens == 100
    assert len(narrow.context_chunks) < len(wide.context_chunks)
    assert narrow.metrics["context_budget"]["tokens_available"] == 100


def test_select_within_budget_uses_the_constants_not_stale_numbers():
    budget = ContextBudget()
    assert (budget.total_tokens, budget.reserve_tokens) == (48000, 1000)
    assert (budget.max_chunks, budget.max_chunk_tokens) == (60, 2000)
    picked = select_within_budget([chunk(f"c{i}", n_words=3) for i in range(80)], budget, count)
    assert len(picked.kept) == 60


async def test_pinned_documents_are_reserved_and_supporting_sources_capped():
    pinned = chunk("pin", score=0.05, n_words=40, document_id="dp")
    others = [chunk(f"o{i}", 0.9 - i * 0.01, 40) for i in range(10)]
    budget = ContextBudget(total_tokens=400, reserve_tokens=0)
    ctx = await context_stage(
        make_ctx([*others, pinned], pinned_ids=["dp"]),
        config=ContextStageConfig(budget=budget),
        token_counter=count,
    )
    ids = {c.id for c in ctx.context_chunks}
    assert "pin" in ids, "a low scored pinned chunk must not be evicted by higher scored ones"

    roomy = await context_stage(
        make_ctx([*others, pinned], pinned_ids=["dp"]),
        config=ContextStageConfig(doc_scoped_supporting_source_cap=3),
        token_counter=count,
    )
    assert len(roomy.context_chunks) == 4, "one pinned chunk plus three supporting sources"


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------


async def test_neighbors_are_added_and_filters_reach_the_fetcher():
    a, b = chunk("a", 0.9, 10, next_chunk_id="b"), chunk("b", 0.9, 10)
    fetcher = FakeFetcher([b])
    ctx = make_ctx([a], filters={"tenant": "t1"})
    out = await context_stage(ctx, fetcher=fetcher, token_counter=count)
    assert out is ctx
    assert fetcher.calls == [
        {"ids": ["b"], "scope": '{"tenant":"t1"}', "filters": {"tenant": "t1"}}
    ]
    assert "b0" in context_of(ctx) and "a0" in context_of(ctx)
    assert ctx.sources[0]["grounding_text"].endswith(b.text)
    assert ctx.sources[0]["text"] == a.text
    assert ctx.metrics["neighbors"]["fetched"] == 1


async def test_a_plain_protocol_fetcher_receives_the_scope_string():
    a, b = chunk("a", 0.9, 10, next_chunk_id="b"), chunk("b", 0.9, 10)
    fetcher = ScopeOnlyFetcher([b])
    ctx = await context_stage(
        make_ctx([a], filters={"tenant": "t1", "kind": "x"}), fetcher=fetcher, token_counter=count
    )
    assert fetcher.scopes == [filters_scope({"kind": "x", "tenant": "t1"})]
    assert "b0" in context_of(ctx)


async def test_explicit_scope_overrides_the_derived_one():
    a, b = chunk("a", 0.9, 10, next_chunk_id="b"), chunk("b", 0.9, 10)
    fetcher = FakeFetcher([b])
    await context_stage(
        make_ctx([a], filters={"tenant": "t1"}),
        fetcher=fetcher,
        config=ContextStageConfig(scope="tenant:t1"),
    )
    assert fetcher.calls[0]["scope"] == "tenant:t1"


async def test_complexity_caps_the_neighbor_count_from_tier_data():
    chunks = chain("a1", "a2", "a3", "a4")[:3]
    store = chain("a1", "a2", "a3", "a4")

    async def run(complexity: str | None) -> tuple[QueryContext, FakeFetcher]:
        fetcher = FakeFetcher(store)
        ctx = make_ctx(chunks)
        if complexity:
            ctx.metrics["query_analysis"] = {"complexity": complexity}
        return await context_stage(ctx, fetcher=fetcher, token_counter=count), fetcher

    def grounding_of(ctx: QueryContext, cid: str) -> str:
        return next(s for s in ctx.sources if s["chunk_id"] == cid)["grounding_text"]

    simple, simple_fetcher = await run("simple")
    assert simple_fetcher.calls == [], "standard tier caps simple queries at zero neighbors"
    assert grounding_of(simple, "a1") == chunks[0].text

    moderate, _ = await run("moderate")
    first = grounding_of(moderate, "a1")
    assert chunks[1].text in first and chunks[2].text not in first
    assert moderate.metrics["neighbors"] == {
        "requested": 1,
        "fetched": 1,
        "capped": True,
        "complexity": "moderate",
    }

    complex_, _ = await run(None)
    assert chunks[2].text in grounding_of(complex_, "a1")
    assert complex_.metrics["neighbors"]["requested"] == 2


async def test_deep_tier_uses_its_own_caps():
    chunks = chain("a1", "a2", "a3")
    ctx = make_ctx(chunks, tier="deep")
    ctx.extra["complexity"] = "simple"
    out = await context_stage(ctx, fetcher=FakeFetcher(chunks), token_counter=count)
    assert out.metrics["neighbors"]["requested"] == 1


async def test_low_scores_boost_the_neighbor_window():
    chunks = chain("a1", "a2", "a3", "a4")[:3]
    for c in chunks:
        c.score = 0.1
    ctx = make_ctx(chunks, tier="standard")
    ctx.extra["complexity"] = "moderate"
    out = await context_stage(ctx, fetcher=FakeFetcher(chunks), token_counter=count)
    assert out.metrics["neighbors"]["requested"] == 3
    no_boost = make_ctx(chunks)
    no_boost.extra["complexity"] = "moderate"
    out = await context_stage(
        no_boost,
        fetcher=FakeFetcher(chunks),
        config=ContextStageConfig(adaptive_neighbor_boost=0),
    )
    assert out.metrics["neighbors"]["requested"] == 1


async def test_neighbors_never_push_the_context_past_the_budget():
    a = chunk("a", 0.9, 50, next_chunk_id="big")
    big = chunk("big", 0.9, 2000)
    budget = ContextBudget(total_tokens=400, reserve_tokens=0)
    ctx = await context_stage(
        make_ctx([a]),
        fetcher=FakeFetcher([big]),
        config=ContextStageConfig(budget=budget),
        token_counter=count,
    )
    assert count(context_of(ctx)) <= 400
    assert ctx.sources[0]["grounding_text"] == a.text


async def test_a_failing_fetcher_still_yields_the_context():
    class Boom:
        async def fetch_by_ids(self, chunk_ids, *, scope=None):
            raise RuntimeError("store down")

    a = chunk("a", 0.9, 10, next_chunk_id="b")
    ctx = await context_stage(make_ctx([a]), fetcher=Boom(), token_counter=count)
    assert "a0" in context_of(ctx)


# ---------------------------------------------------------------------------
# Dedupe, remap, order, labels
# ---------------------------------------------------------------------------


async def test_dedupe_collapses_sources_and_remaps_citation_numbers():
    chunks = [
        chunk("c1", 0.9, citation="1 U.S. 1", title="Alpha v. Beta", document_id="d1"),
        chunk("c2", 0.8, title="Gamma Act", document_id="d2"),
        chunk("c3", 0.7, citation="1 U.S. 1", title="Alpha v. Beta", document_id="d1"),
        chunk("c4", 0.6, title="Alpha v. Beta", document_id="d9"),
    ]
    ctx = await context_stage(make_ctx(chunks), token_counter=count)
    context = context_of(ctx)
    assert [s["title"] for s in ctx.sources] == ["Alpha v. Beta", "Gamma Act"]
    assert [c.metadata["source_index"] for c in ctx.context_chunks] == [1, 2, 1, 1]
    assert context.count("[Source 1: Alpha v. Beta]") == 3
    assert context.count("[Source 2: Gamma Act]") == 1
    table = context.split("--- END SOURCE INDEX ---")[0]
    assert "[1] Alpha v. Beta" in table and "[2] Gamma Act" in table and "[3]" not in table
    assert [s["source_index"] for s in ctx.sources] == [1, 2]
    assert ctx.metrics["source_counts"]["sources"] == 2


async def test_pinned_source_order_is_stable_behind_the_flag():
    x, y = chunk("x", 0.9, document_id="dx"), chunk("y", 0.7, document_id="dy")
    p = chunk("p", 0.5, document_id="dp")

    async def order(chunks: list[RetrievedChunk], enabled: bool) -> list[str]:
        ctx = make_ctx(chunks, pinned_ids=["dp"])
        cfg = ContextStageConfig(stable_pinned_order=enabled)
        await context_stage(ctx, config=cfg, token_counter=count)
        return [s["document_id"] for s in ctx.sources]

    assert await order([x, p, y], True) == ["dp", "dx", "dy"]
    assert await order([y, x, p], True) == ["dp", "dy", "dx"]
    assert (await order([x, p, y], True))[0] == (await order([y, p, x], True))[0] == "dp"
    assert await order([x, p, y], False) == ["dx", "dp", "dy"]


async def test_scope_locked_documents_are_ordered_first_too():
    chunks = [chunk("x", 0.9, document_id="dx"), chunk("l", 0.5, document_id="dl")]
    ctx = make_ctx(chunks, scope_lock_ids=["dl"])
    await context_stage(ctx, token_counter=count)
    assert [s["document_id"] for s in ctx.sources] == ["dl", "dx"]


async def test_label_for_override_is_used_in_headers_and_index():
    chunks = [chunk("c1", 0.9), chunk("c2", 0.8)]
    ctx = await context_stage(
        make_ctx(chunks), label_for=lambda c: f"LABEL-{c.id}", token_counter=count
    )
    context = context_of(ctx)
    assert "[Source 1: LABEL-c1]" in context and "[Source 2: LABEL-c2]" in context
    assert "[2] LABEL-c2" in context


async def test_a_broken_label_for_falls_back_to_the_default_label():
    def broken(_: RetrievedChunk) -> str:
        raise ValueError("bad plugin")

    ctx = await context_stage(make_ctx([chunk("c1")]), label_for=broken, token_counter=count)
    assert "[Source 1: Doc c1]" in context_of(ctx)


def test_default_label_is_title_or_filename_without_prefix_stripping():
    title = "FA/3810/2014 of Acme Vs Beta"
    assert default_label_for(RetrievedChunk("c", "t", metadata={"title": title})) == title
    assert default_label_for(RetrievedChunk("c", "t", metadata={"filename": "a.pdf"})) == "a.pdf"
    court_year = RetrievedChunk(
        "c", "t", metadata={"title": "A v. B", "court": "9th Cir.", "year": 2020}
    )
    assert default_label_for(court_year) == "A v. B (9th Cir., 2020)"
    assert default_label_for(RetrievedChunk("c", "t")) == "Unknown"


async def test_source_enricher_runs_on_the_source_dicts_and_feeds_the_label():
    bare = RetrievedChunk(id="c1", text=words(10), score=0.9, metadata={"document_id": "d1"})
    seen: list[int] = []

    async def enrich(sources: list[dict[str, Any]]) -> None:
        seen.append(len(sources))
        for source in sources:
            source["filename"] = "enriched.pdf"

    ctx = await context_stage(make_ctx([bare]), source_enricher=enrich, token_counter=count)
    assert seen == [1]
    assert ctx.sources[0]["filename"] == "enriched.pdf"
    assert "[Source 1: enriched.pdf]" in context_of(ctx)


async def test_own_documents_are_kept_apart_from_legal_authority():
    chunks = [
        chunk("law", 0.9, source_type="statute"),
        chunk("mine", 0.8, source_type="upload"),
    ]
    ctx = await context_stage(make_ctx(chunks), token_counter=count)
    context = context_of(ctx)
    assert context.index("USER-SUPPLIED DOCUMENTS") < context.index("[Source 2: Doc mine]")
    assert context.index("LEGAL AUTHORITY") < context.index("[Source 1: Doc law]")


# ---------------------------------------------------------------------------
# Hygiene, coverage, confidence, warnings, usage
# ---------------------------------------------------------------------------


async def test_junk_sources_are_dropped_and_numbering_stays_consistent():
    chunks = [
        chunk("good", 0.9, url="https://www.law.cornell.edu/x"),
        chunk("junk", 0.8, url="https://instagram.com/p/abc"),
    ]
    ctx = await context_stage(make_ctx(chunks), token_counter=count)
    assert [s["chunk_id"] for s in ctx.sources] == ["good"]
    assert "Doc junk" not in context_of(ctx)
    assert ctx.metrics["source_counts"]["dropped_by_hygiene"] == 1


async def test_hygiene_never_empties_the_context():
    chunks = [chunk("j1", 0.9, url="https://instagram.com/a")]
    ctx = await context_stage(make_ctx(chunks), token_counter=count)
    assert ctx.sources and "j10" in context_of(ctx)
    assert "source_hygiene_kept_all" in ctx.warnings


async def test_coverage_notice_when_own_documents_were_requested_but_missing():
    chunks = [chunk("law", 0.9, source_type="statute")]
    ctx = make_ctx(chunks, pinned_ids=["my-contract"])
    ctx.query = "what does my contract say about notice"
    await context_stage(ctx, token_counter=count)
    assert context_of(ctx).startswith("[RETRIEVAL NOTICE")
    assert ctx.metrics["coverage_notice"] is True

    plain = await context_stage(make_ctx(chunks), token_counter=count)
    assert not context_of(plain).startswith("[RETRIEVAL NOTICE")
    assert plain.metrics["coverage_notice"] is False


async def test_confidence_fields_are_present():
    chunks = [chunk(f"c{i}", 0.7 - i * 0.05) for i in range(6)]
    ctx = await context_stage(make_ctx(chunks), token_counter=count)
    confidence = ctx.metrics["confidence"]
    assert {"overall_score", "level", "score_spread", "requires_review", "factors"} <= set(
        confidence
    )
    assert 0.0 <= ctx.metrics["multi_signal_confidence"] <= 1.0
    assert ctx.metrics["pipeline_confidence"] == ctx.metrics["multi_signal_confidence"]


async def test_confidence_falls_back_when_the_tier_does_not_enable_it():
    data = json.loads(Path(_DATA_FILE).read_text(encoding="utf-8"))
    data["tiers"]["standard"]["features"]["use_multi_signal_confidence"] = False
    ctx = await context_stage(
        make_ctx([chunk("c1")]), config=ContextStageConfig(tiers=data), token_counter=count
    )
    assert ctx.metrics["multi_signal_confidence"] is None
    assert ctx.metrics["confidence"] is None
    assert ctx.metrics["pipeline_confidence"] == 0.35


def test_resolve_pipeline_confidence():
    assert resolve_pipeline_confidence(0.72, 3) == 0.72
    assert resolve_pipeline_confidence(None, 0) == 0.0
    assert resolve_pipeline_confidence(None, 4) == 0.35


async def test_retrieval_warnings():
    none = await context_stage(QueryContext(query="q"))
    assert none.retrieval_warning == "no_results"
    assert context_of(none) == "" and none.sources == [] and none.context_chunks == []
    assert none.metrics["pipeline_confidence"] == 0.0

    two = await context_stage(make_ctx([chunk("a"), chunk("b")]), token_counter=count)
    assert two.retrieval_warning == "limited_results"

    three = await context_stage(make_ctx([chunk("a"), chunk("b"), chunk("c")]), token_counter=count)
    assert three.retrieval_warning is None


async def test_falls_back_to_retrieved_when_nothing_was_reranked():
    ctx = QueryContext(query="q", retrieved=[chunk("r1")])
    await context_stage(ctx, token_counter=count)
    assert [c.id for c in ctx.context_chunks] == ["r1"]


async def test_unknown_tier_does_not_break_the_stage():
    ctx = await context_stage(make_ctx([chunk("a")], tier="bogus"), token_counter=count)
    assert "a0" in context_of(ctx)


async def test_usage_hook_gets_one_context_event():
    events: list[UsageEvent] = []
    ctx = await context_stage(make_ctx([chunk("a")]), usage=events.append, token_counter=count)
    assert [e.operation for e in events] == ["context_assembly"]
    assert events[0].input_tokens == ctx.metrics["context_tokens"]

    def bad_hook(_: UsageEvent) -> None:
        raise RuntimeError("meter down")

    await context_stage(make_ctx([chunk("a")]), usage=bad_hook, token_counter=count)


# ---------------------------------------------------------------------------
# Non-empty guarantee and fail-open
# ---------------------------------------------------------------------------


async def test_a_tiny_budget_still_yields_one_chunk():
    budget = ContextBudget(total_tokens=10, reserve_tokens=5, max_chunks=3, max_chunk_tokens=50)
    chunks = [chunk(f"c{i}", 0.9 - i * 0.1, n_words=100) for i in range(4)]
    ctx = await context_stage(
        make_ctx(chunks), config=ContextStageConfig(budget=budget), token_counter=count
    )
    assert len(ctx.context_chunks) == 1
    assert ctx.context_chunks[0].id == "c0"
    assert "c00" in context_of(ctx)
    assert ctx.retrieval_warning == "limited_results"


async def test_a_failing_token_counter_fails_open_with_a_context():
    def broken(_: str) -> int:
        raise RuntimeError("tokenizer unavailable")

    chunks = [chunk("a", 0.9), chunk("b", 0.8)]
    ctx = await context_stage(make_ctx(chunks), token_counter=broken)
    assert "a0" in context_of(ctx) and "b0" in context_of(ctx)
    assert "context_assembly_failed" in ctx.warnings
    assert [c.id for c in ctx.context_chunks] == ["a", "b"]
    assert ctx.sources and ctx.retrieval_warning == "limited_results"


async def test_a_failing_enricher_still_yields_the_context():
    async def boom(_: list[dict[str, Any]]) -> None:
        raise RuntimeError("lookup down")

    ctx = await context_stage(make_ctx([chunk("a")]), source_enricher=boom, token_counter=count)
    assert "a0" in context_of(ctx)
    assert "source_enrichment_failed" in ctx.warnings


def test_filters_scope_is_stable_and_empty_means_none():
    assert filters_scope({}) is None and filters_scope(None) is None
    assert filters_scope({"b": 1, "a": 2}) == filters_scope({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_token_counters():
    assert whitespace_token_counter("") == 0
    assert whitespace_token_counter("one two  three\nfour") == 4
    pytest.importorskip("tiktoken")
    try:
        counter = tiktoken_token_counter()
    except Exception:  # noqa: BLE001 - the encoding file may need a download this run cannot do
        pytest.skip("tiktoken encoding unavailable offline")
    assert counter("hello world") > 0 and counter("") == 0


# ---------------------------------------------------------------------------
# Hygiene scan
# ---------------------------------------------------------------------------

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
    "em dash": re.compile(f"{chr(0x2014)}|{chr(0x2013)}"),
}


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_context_module_has_no_forbidden_text(label: str):
    source = Path(context_module.__file__).read_text(encoding="utf-8")
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), start=1)
        if FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found:\n" + "\n".join(hits)
