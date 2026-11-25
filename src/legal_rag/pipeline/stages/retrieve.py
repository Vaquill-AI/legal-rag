"""Retrieval stage: choose a strategy, run it, and leave ranked candidates on the context.

``retrieve_stage`` is ``async def stage(ctx) -> ctx`` with its collaborators passed as keyword
arguments. It reads the tier data (retrieval depth, complexity caps, feature flags), picks a
strategy from a table, and fills ``ctx.retrieved``.

Strategies are entries in ``DEFAULT_STRATEGIES``: ``hybrid``, ``self_rag``, ``crag``, ``hyde``,
``multi_query`` and ``multi_hop``. Pass your own table through ``strategies=`` to add or replace one.
An unknown strategy name raises ``UnknownStrategyError`` before anything is retrieved.

What the stage reads from ``ctx.extra`` (the analyze stage writes these):

- ``query_analysis``: mapping with ``complexity``, ``should_retrieve``, ``should_decompose`` and
  ``legal_task_type``.
- ``condensed_query``: the query rewritten for search. Falls back to ``ctx.query``.
- ``condensed_sub_queries``: per-topic sub-queries to search alongside the main query.
- ``query_expansion``: mapping with ``applied`` and ``variations``.
- ``requires_retrieval``: ``False`` skips retrieval.
- ``force_retrieval_strategy``: beats every other choice. A retry sets it to ``hybrid``.

Tenancy is not a stage concern. ``ctx.filters`` reaches every retriever call unchanged. The only
additions are ids under the key names you choose in ``ScopeFilterKeys``, and none by default.
Implement your ``Retriever`` so it applies the tenant filter itself.

Every failure inside the stage is fail-open: the error is logged and the input context comes back
with a ``retrieval_failed`` warning. A bad configuration (unknown strategy or tier) still raises.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import (
    EventSink,
    ExactLookup,
    JurisdictionPlugin,
    LanguageModel,
    Retriever,
    WebSearch,
)
from legal_rag.pipeline.tiers import TierConfig, get_tier
from legal_rag.pipeline.types import PINNED_BY_KEY, PINNED_KEY, ContextBudget, QueryContext
from legal_rag.providers.protocols import UsageHook
from legal_rag.retrieval.strategies.crag import CRAGConfig, CRAGCorrector, RetrievalQuality
from legal_rag.retrieval.strategies.crag_source_mixer import SourceMixer, SourceMixerConfig
from legal_rag.retrieval.strategies.gap_analysis import LLMGapAnalyzer
from legal_rag.retrieval.strategies.hyde import HyDEConfig, HyDEGenerator
from legal_rag.retrieval.strategies.multi_hop import MultiHopConfig, MultiHopRetriever
from legal_rag.retrieval.strategies.multi_query import MultiQueryConfig, MultiQueryGenerator
from legal_rag.retrieval.strategies.self_rag import SelfRAGConfig, SelfRAGController
from legal_rag.types import RetrievedChunk, UsageEvent

_log = get_logger("pipeline.retrieve")

EXTRA_QUERY_ANALYSIS = "query_analysis"
EXTRA_CONDENSED_QUERY = "condensed_query"
EXTRA_SUB_QUERIES = "condensed_sub_queries"
EXTRA_QUERY_EXPANSION = "query_expansion"
EXTRA_REQUIRES_RETRIEVAL = "requires_retrieval"
EXTRA_FORCE_STRATEGY = "force_retrieval_strategy"

WARNING_NO_RESULTS = "no_results"
WARNING_LIMITED_RESULTS = "limited_results"
WARNING_RETRIEVAL_FAILED = "retrieval_failed"

SKIPPED_BY_STRATEGY = "retrieval_skipped_by_strategy"
"""Strategy metadata flag: the strategy decided retrieval was not needed, so an empty result is
deliberate and must not trigger the no-results retry."""

_TRIVIAL_LEVELS = ("trivial", "simple")
_MULTI_HOP_EXEMPT = ("multi_hop", "multi_query")

StrategyOutcome = tuple[list[RetrievedChunk], dict[str, Any]]


class UnknownStrategyError(ValueError):
    """The requested retrieval strategy is not in the strategy table."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeFilterKeys:
    """Filter keys used to hand scope to the retrievers. ``None`` means the value is not passed.

    The stage never invents a key. With the defaults, retrievers receive ``ctx.filters`` as given
    and are expected to know the scope some other way.
    """

    pinned_ids: str | None = None
    scope_lock_ids: str | None = None
    boost_ids: str | None = None
    collection: str | None = None


def _default_multi_hop_by_tier() -> Mapping[str, MultiHopConfig]:
    return MappingProxyType(
        {"deep": MultiHopConfig(min_chunks_per_hop=25, sufficient_chunks_threshold=80)}
    )


@dataclass(frozen=True)
class RetrieveConfig:
    """Tuning for the retrieval stage. Every number here is a default you can change."""

    tiers: Path | str | Mapping[str, Any] | None = None
    """Tier table source for ``get_tier``. ``None`` is the built-in table.
    A path is read on every call, so load a custom table once and pass the parsed mapping."""
    scope_keys: ScopeFilterKeys = field(default_factory=ScopeFilterKeys)
    self_rag: SelfRAGConfig = field(default_factory=SelfRAGConfig)
    crag: CRAGConfig = field(default_factory=CRAGConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    multi_hop: MultiHopConfig = field(
        default_factory=lambda: MultiHopConfig(min_chunks_per_hop=8, sufficient_chunks_threshold=16)
    )
    multi_hop_by_tier: Mapping[str, MultiHopConfig] = field(
        default_factory=_default_multi_hop_by_tier
    )
    """Per-tier multi-hop shape by tier name. ``max_hops`` always comes from the tier."""
    use_expansion_variations: bool = True
    """Search every query-expansion variation in parallel when the analyzer applied expansion."""
    re_retrieval: bool = True
    """Self-RAG and CRAG look for information gaps and retrieve once more."""
    re_retrieval_min_chunks: int = 5
    re_retrieval_min_score: float = 0.3
    retry_empty_with_hybrid: bool = True
    """When a non-hybrid strategy returns nothing, retry once with hybrid."""
    web_results: int = 5
    mix_token_budget: int = ContextBudget().total_tokens
    mix_primary_ratio: float = 0.6
    boost_chunks_per_id: int = 5
    pinned_depth_per_item: int = 6
    pinned_depth_max: int = 150
    limited_results_max: int = 2
    """At most this many chunks (and at least one) sets ``limited_results``."""


DEFAULT_RETRIEVE_CONFIG = RetrieveConfig()


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def deduplicate_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop duplicate chunks by ``id``, keeping the highest-scored copy, best score first.

    Ties keep first-seen order.
    """
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        current = seen.get(chunk.id)
        if current is None or chunk.score > current.score:
            seen[chunk.id] = chunk
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


def _dedup_keep_order(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop duplicates by ``id``. The best-scored copy takes the first occurrence's position."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        current = best.get(chunk.id)
        if current is None or chunk.score > current.score:
            best[chunk.id] = chunk
    return list(best.values())


def _pin(chunk: RetrievedChunk, source: str) -> RetrievedChunk:
    return replace(chunk, metadata={**chunk.metadata, PINNED_KEY: True, PINNED_BY_KEY: source})


def _pin_first(
    pinned: Sequence[RetrievedChunk], rest: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Pinned chunks first. A pinned copy replaces any copy of the same id in ``rest``."""
    head = _dedup_keep_order(pinned)
    taken = {c.id for c in head}
    return head + [c for c in rest if c.id not in taken]


def _tag_source(name: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        c if "retriever" in c.metadata else replace(c, metadata={**c.metadata, "retriever": name})
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Fan-out over named retrievers
# ---------------------------------------------------------------------------


class FanOutRetriever:
    """Searches several named retrievers at once and merges the lists in declared order.

    Use it for a query that has to see two bodies of text, for example the user's own documents
    and a reference corpus. List the source whose chunks should reach the reranker first before
    the others. A chunk found twice keeps its best score at its first position. Each chunk is
    tagged with ``metadata["retriever"]``. One failing source is logged and skipped. Every source
    failing raises the first error.
    """

    def __init__(self, sources: Mapping[str, Retriever]) -> None:
        if not sources:
            raise ValueError("FanOutRetriever needs at least one named retriever")
        self._sources = dict(sources)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._sources)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        names = list(self._sources)
        results = await asyncio.gather(
            *(
                self._sources[n].retrieve(
                    query, top_k=top_k, filters=filters, sub_queries=sub_queries
                )
                for n in names
            ),
            return_exceptions=True,
        )
        merged: list[RetrievedChunk] = []
        failures: list[Exception] = []
        for name, result in zip(names, results, strict=True):
            if isinstance(result, Exception):
                _log.warning("fan_out_source_failed", source=name, error=type(result).__name__)
                failures.append(result)
            elif isinstance(result, BaseException):
                raise result
            else:
                merged.extend(_tag_source(name, result))
        if failures and len(failures) == len(names):
            raise failures[0]
        return _dedup_keep_order(merged)


def _compose_retriever(
    retriever: Retriever | Mapping[str, Retriever],
    plugin: JurisdictionPlugin | None,
) -> Retriever:
    """The retriever the strategies use.

    A mapping becomes a fan-out. A matching jurisdiction plug-in replaces a single retriever and
    joins a mapping as the source named ``jurisdiction``.
    """
    if isinstance(retriever, Mapping):
        sources = dict(retriever)
        if plugin is not None:
            sources["jurisdiction"] = plugin
        return FanOutRetriever(sources)
    return plugin if plugin is not None else retriever


# ---------------------------------------------------------------------------
# Planning: strategy, depth and skip decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalPlan:
    """What the stage will do for one context, decided before any search."""

    strategy: str
    top_k: int
    scope_locked: bool
    forced: bool
    skip_reason: str | None = None


def has_explicit_scope(ctx: QueryContext) -> bool:
    """True when the user pinned documents or locked the search to specific ids.

    An explicit scope is a stronger signal than any heuristic, so it overrides every
    "skip retrieval" rule.
    """
    return bool(ctx.pinned_ids or ctx.scope_lock_ids)


def _analysis(ctx: QueryContext) -> Mapping[str, Any]:
    value = ctx.extra.get(EXTRA_QUERY_ANALYSIS)
    return value if isinstance(value, Mapping) else {}


def plan_retrieval(ctx: QueryContext, tier: TierConfig, config: RetrieveConfig) -> RetrievalPlan:
    """Pick the strategy and retrieval depth, or a reason to skip retrieval.

    Strategy order: ``force_retrieval_strategy``, then ``ctx.retrieval_strategy`` (already resolved
    by the caller from the request or the tier), then the tier default.
    A query the analyzer wants decomposed is moved to ``multi_hop`` when the tier allows it, unless
    the strategy was forced.
    Depth is the tier depth, lowered by the complexity cap. A cap of ``0`` skips retrieval, except
    under an explicit scope, where the cap is ignored.
    """
    analysis = _analysis(ctx)
    locked = has_explicit_scope(ctx)
    forced_name = ctx.extra.get(EXTRA_FORCE_STRATEGY)
    strategy = str(forced_name or ctx.retrieval_strategy or tier.retrieval_strategy)
    if (
        not forced_name
        and analysis.get("should_decompose")
        and strategy not in _MULTI_HOP_EXEMPT
        and tier.features.use_multi_hop
    ):
        _log.info("adaptive_strategy_override", original=strategy, new="multi_hop")
        strategy = "multi_hop"

    would_skip = None
    if not ctx.extra.get(EXTRA_REQUIRES_RETRIEVAL, True):
        would_skip = "requires_retrieval"
    elif not analysis.get("should_retrieve", True):
        would_skip = "query_analysis"

    complexity = str(analysis.get("complexity", "complex"))
    task_type = str(analysis.get("legal_task_type", "general"))
    caps = tier.complexity
    cap = caps.retrieval_cap(complexity)
    if cap == 0:
        would_skip = would_skip or "complexity_cap"
        if locked:
            cap = None
    skip = None if locked else would_skip
    if locked and would_skip:
        _log.warning("explicit_scope_overrides_skip", reason=would_skip, query_len=len(ctx.query))
    if (
        cap is not None
        and caps.case_law_retrieval_floor
        and task_type in caps.case_law_task_types
        and complexity not in _TRIVIAL_LEVELS
        and cap < caps.case_law_retrieval_floor
    ):
        cap = caps.case_law_retrieval_floor
    top_k = tier.top_k_retrieval if cap is None else min(tier.top_k_retrieval, cap)
    if len(ctx.pinned_ids) > 1:
        # Enough depth for every pinned document to have candidates in the pool.
        top_k = max(
            top_k,
            min(len(ctx.pinned_ids) * config.pinned_depth_per_item, config.pinned_depth_max),
        )
    return RetrievalPlan(strategy, top_k, locked, bool(forced_name), skip)


def _scoped_filters(
    ctx: QueryContext, keys: ScopeFilterKeys, *, lock: bool = True
) -> Mapping[str, Any]:
    """``ctx.filters`` plus any scope ids under the configured key names.

    Returns ``ctx.filters`` itself when there is nothing to add.
    """
    added: dict[str, Any] = {}
    if lock and keys.pinned_ids and ctx.pinned_ids:
        added[keys.pinned_ids] = list(ctx.pinned_ids)
    if lock and keys.scope_lock_ids and ctx.scope_lock_ids:
        added[keys.scope_lock_ids] = list(ctx.scope_lock_ids)
    if keys.collection and ctx.collection:
        added[keys.collection] = ctx.collection
    return {**ctx.filters, **added} if added else ctx.filters


# ---------------------------------------------------------------------------
# The run: what a strategy runner receives
# ---------------------------------------------------------------------------


@dataclass
class RetrievalRun:
    """Everything a strategy runner needs for one retrieval."""

    query: str
    top_k: int
    retriever: Retriever
    filters: Mapping[str, Any]
    tier: TierConfig
    ctx: QueryContext
    config: RetrieveConfig
    llm: LanguageModel | None = None
    web_search: WebSearch | None = None
    usage: UsageHook | None = None
    on_event: EventSink | None = None
    sub_queries: Sequence[str] = ()
    expansion: Sequence[str] = ()
    native_sub_queries: bool = False
    """The retriever does its own fan-out (a jurisdiction plug-in), so give it ``sub_queries``."""
    scope_locked: bool = False


StrategyRunner = Callable[[RetrievalRun], Awaitable[StrategyOutcome]]


async def _search(run: RetrievalRun, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    return await run.retriever.retrieve(query, top_k=top_k or run.top_k, filters=run.filters)


async def _search_many(
    run: RetrievalRun, queries: Sequence[str]
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Search every query in parallel and deduplicate. One failure is skipped, all failing raises."""
    results = await asyncio.gather(*(_search(run, q) for q in queries), return_exceptions=True)
    flat: list[RetrievedChunk] = []
    errors: list[Exception] = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(result)
            _log.warning("parallel_query_failed", error=type(result).__name__)
        elif isinstance(result, BaseException):
            raise result
        else:
            flat.extend(result)
    if errors and len(errors) == len(results):
        raise errors[0]
    deduped = deduplicate_chunks(flat)
    return deduped, {
        "num_queries": len(queries),
        "total_before_dedup": len(flat),
        "chunks_after_dedup": len(deduped),
        "query_errors": len(errors),
    }


async def _degrade_to_hybrid(run: RetrievalRun, wanted: str) -> StrategyOutcome:
    chunks, meta = await run_hybrid(run)
    return chunks, {**meta, "degraded_from": wanted, "degraded_reason": "no_language_model"}


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


async def run_hybrid(run: RetrievalRun) -> StrategyOutcome:
    """Plain search. Condenser sub-queries or expansion variations are searched in parallel."""
    sub_queries = [q for q in run.sub_queries if q and q != run.query]
    if sub_queries and run.native_sub_queries:
        chunks = await run.retriever.retrieve(
            run.query, top_k=run.top_k, filters=run.filters, sub_queries=sub_queries
        )
        return chunks, {"native_sub_queries": len(sub_queries)}
    if sub_queries:
        chunks, meta = await _search_many(run, [run.query, *sub_queries])
        _log.info("hybrid_condenser_parallel", queries=meta["num_queries"], chunks=len(chunks))
        return chunks, {"condenser_decomposition": True, **meta}
    variations = [q for q in run.expansion if q and q != run.query]
    if variations:
        chunks, meta = await _search_many(run, [run.query, *variations])
        _log.info("hybrid_expansion_parallel", queries=meta["num_queries"], chunks=len(chunks))
        return chunks, {"expansion_parallel": True, **meta}
    return await _search(run, run.query), {}


async def _gap_follow_up(
    run: RetrievalRun, chunks: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], str] | None:
    """One gap analysis. Returns the follow-up query and its chunks, or ``None`` if none needed."""
    if run.llm is None:
        return None
    gap = await LLMGapAnalyzer(run.llm, usage_hook=run.usage)(run.query, chunks)
    if gap.has_sufficient_info or not gap.suggested_followup:
        return None
    return await _search(run, gap.suggested_followup), gap.suggested_followup


async def run_self_rag(run: RetrievalRun) -> StrategyOutcome:
    """Ask first whether retrieval is needed. Retrieve once more when the first pass is thin."""
    if run.llm is None:
        return await _degrade_to_hybrid(run, "self_rag")
    controller = SelfRAGController(run.llm, run.config.self_rag, usage_hook=run.usage)
    assessment = await controller.process(query=run.query)
    meta: dict[str, Any] = {
        "decision": assessment.retrieval_decision.value,
        "retrieval_performed": assessment.retrieval_performed,
        "assessment_time_ms": assessment.total_time_ms,
        "requested_count": assessment.requested_count,
    }
    # A failed assessment is not a decision, so it retrieves.
    if assessment.success and not assessment.retrieval_performed:
        _log.info("self_rag_skip_retrieval", decision=meta["decision"])
        return [], {**meta, SKIPPED_BY_STRATEGY: True}
    chunks = await _search(run, run.query)
    re_retrieved = False
    cfg = run.config
    if cfg.re_retrieval and chunks:
        average = sum(c.score for c in chunks) / len(chunks)
        if len(chunks) < cfg.re_retrieval_min_chunks or average < cfg.re_retrieval_min_score:
            follow = await _gap_follow_up(run, chunks)
            if follow is not None:
                chunks = deduplicate_chunks(chunks + follow[0])
                re_retrieved = True
    return chunks, {**meta, "re_retrieved": re_retrieved}


async def apply_crag_correction(
    run: RetrievalRun,
    chunks: list[RetrievedChunk],
    *,
    allow_web: bool = True,
    allow_re_retrieval: bool = True,
) -> StrategyOutcome:
    """Grade already retrieved chunks, drop the irrelevant ones and fill gaps.

    Works on the output of any strategy. A thin result is searched once more from a gap-analysis
    follow-up. When it is still thin and a ``WebSearch`` is available (and the tier enables web
    search), web chunks are added and mixed in under a token budget. The base chunks are never
    discarded. Needs a language model.
    """
    if run.llm is None:
        return chunks, {"degraded_reason": "no_language_model"}
    web_enabled = allow_web and run.web_search is not None and run.tier.features.enable_web_search
    corrector = CRAGCorrector(
        run.llm,
        replace(run.config.crag, enable_external_search=web_enabled),
        usage_hook=run.usage,
    )
    result = await corrector.evaluate_and_correct(run.query, chunks)
    corrected = result.corrected_chunks
    quality = result.quality
    thin = (
        "external_search_recommended" in result.corrections_applied
        or "supplement_externally_recommended" in result.corrections_applied
        or quality in (RetrievalQuality.LOW, RetrievalQuality.PARTIAL)
    )

    re_retrieved = False
    if thin and allow_re_retrieval and run.config.re_retrieval:
        follow = await _gap_follow_up(run, corrected)
        if follow is not None:
            corrected = deduplicate_chunks(corrected + follow[0])
            re_retrieved = True
            again = await corrector.evaluate_and_correct(run.query, corrected)
            corrected, quality = again.corrected_chunks, again.quality
            thin = quality in (RetrievalQuality.LOW, RetrievalQuality.PARTIAL)

    web_chunks: list[RetrievedChunk] = []
    if thin and web_enabled and run.web_search is not None:
        try:
            found = await run.web_search.search(run.query, run.config.web_results)
            web_chunks = [
                c
                if "source_type" in c.metadata
                else replace(c, metadata={**c.metadata, "source_type": "web_search"})
                for c in found
            ]
        except Exception as error:  # noqa: BLE001 - web search is a best-effort supplement
            _log.error("crag_web_search_failed", error=type(error).__name__)

    scores = result.relevance_scores
    relevance = sum(scores) / len(scores) if scores else 0.0
    mixing: dict[str, Any] | None = None
    final = corrected
    if corrected and web_chunks:
        mixer = SourceMixer(
            SourceMixerConfig(
                max_total_tokens=run.config.mix_token_budget,
                primary_token_ratio=run.config.mix_primary_ratio,
            )
        )
        mixed = mixer.mix(corrected, web_chunks, primary_confidence=relevance or 0.5)
        final = mixed.primary_chunks + mixed.supplementary_chunks
        mixing = {
            "strategy": mixed.mix_strategy.value,
            "total_tokens": mixed.total_tokens,
            "truncation_applied": mixed.truncation_applied,
            **mixed.source_summary,
        }
    elif web_chunks:
        final = web_chunks

    return final, {
        "quality": quality.value,
        "original_count": len(chunks),
        "corrected_count": len(corrected),
        "re_retrieved": re_retrieved,
        "web_chunks_count": len(web_chunks),
        "web_search_executed": bool(web_chunks),
        "final_count": len(final),
        "corrections_applied": list(result.corrections_applied),
        "avg_relevance": relevance,
        "evaluation_time_ms": result.evaluation_time_ms,
        "source_mixing": mixing,
    }


async def run_crag(run: RetrievalRun) -> StrategyOutcome:
    """Search once, then grade and correct the result."""
    if run.llm is None:
        return await _degrade_to_hybrid(run, "crag")
    chunks = await _search(run, run.query)
    return await apply_crag_correction(run, chunks)


async def run_hyde(run: RetrievalRun) -> StrategyOutcome:
    """Search with a hypothetical answer, which sits closer to the documents than the question."""
    if run.llm is None:
        return await _degrade_to_hybrid(run, "hyde")
    result = await HyDEGenerator(run.llm, run.config.hyde, usage_hook=run.usage).generate(run.query)
    search_query = result.primary_hypothesis if result.success else run.query
    if not result.success:
        _log.warning("hyde_generation_failed", fallback="original_query")
    chunks = await _search(run, search_query)
    return chunks, {
        "hypothesis_generated": result.success,
        "hypothesis_length": len(result.primary_hypothesis),
        "generation_time_ms": result.generation_time_ms,
    }


async def run_multi_query(run: RetrievalRun) -> StrategyOutcome:
    """Search several phrasings of the question in parallel and merge."""
    if run.llm is None:
        return await _degrade_to_hybrid(run, "multi_query")
    generated = await MultiQueryGenerator(
        run.llm, run.config.multi_query, usage_hook=run.usage
    ).generate(run.query)
    queries = generated.all_queries if generated.success else [run.query]
    if not generated.success:
        _log.warning("multi_query_generation_failed", fallback="single_query")
    chunks, meta = await _search_many(run, queries)
    return chunks, {
        "queries_generated": generated.success,
        "generation_time_ms": generated.generation_time_ms,
        **meta,
    }


def _multi_hop_config(run: RetrievalRun) -> MultiHopConfig:
    base = run.config.multi_hop_by_tier.get(run.tier.name, run.config.multi_hop)
    return replace(base, max_hops=run.tier.max_multi_hop_iterations)


async def run_multi_hop(run: RetrievalRun) -> StrategyOutcome:
    """Retrieve, look for gaps, retrieve again. Works without a model using rule-based follow-ups."""
    analyzer = LLMGapAnalyzer(run.llm, usage_hook=run.usage) if run.llm is not None else None

    async def hop_search(query: str, k: int) -> list[RetrievedChunk]:
        return await _search(run, query, k)

    result = await MultiHopRetriever(
        hop_search, _multi_hop_config(run), gap_analyzer=analyzer
    ).retrieve(run.query, on_hop_event=run.on_event)
    _log.info("multi_hop_complete", hops=result.total_hops, early_stop=result.completed_early)
    return list(result.chunks), {
        "hops_performed": result.total_hops,
        "completed_early": result.completed_early,
        "total_chunks": result.total_chunks_retrieved,
        "retrieval_time_ms": result.retrieval_time_ms,
        "reasoning_chain": {
            "total_hops": result.total_hops,
            "completed_early": result.completed_early,
            "hops": [
                {
                    "hop_number": hop.hop_number,
                    "query": hop.query,
                    "chunks_retrieved": hop.num_chunks_retrieved,
                    "reasoning": hop.reasoning,
                    "action": hop.action.value,
                }
                for hop in result.hops
            ],
        },
    }


DEFAULT_STRATEGIES: Mapping[str, StrategyRunner] = MappingProxyType(
    {
        "hybrid": run_hybrid,
        "self_rag": run_self_rag,
        "crag": run_crag,
        "hyde": run_hyde,
        "multi_query": run_multi_query,
        "multi_hop": run_multi_hop,
    }
)


def resolve_strategy(
    name: str, table: Mapping[str, StrategyRunner] | None = None
) -> StrategyRunner:
    """Look a strategy up by name. An unknown name raises ``UnknownStrategyError``."""
    source = DEFAULT_STRATEGIES if table is None else table
    runner = source.get(name)
    if runner is None:
        raise UnknownStrategyError(
            f"Unknown retrieval strategy {name!r}. Available: {sorted(source)}"
        )
    return runner


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


async def _additive_layers(
    run: RetrievalRun, strategy: str, chunks: list[RetrievedChunk], meta: dict[str, Any]
) -> list[RetrievedChunk]:
    """Extra recall passes (HyDE, multi-query) merged into the pool.

    The base strategy keeps running as chosen. The reranker sorts the larger pool later.
    Skipped for a locked scope, where widening recall is wrong, when no language model is
    available, and for a strategy that is already that technique. A failed layer is dropped.
    """
    features = run.tier.features
    specs: list[tuple[str, StrategyRunner]] = []
    if features.use_hyde and strategy != "hyde":
        specs.append(("hyde", run_hyde))
    if features.use_multi_query and strategy != "multi_query":
        specs.append(("multi_query", run_multi_query))
    if not specs or run.scope_locked or run.llm is None or meta.get(SKIPPED_BY_STRATEGY):
        return chunks
    results = await asyncio.gather(*(fn(run) for _, fn in specs), return_exceptions=True)
    for (name, _), result in zip(specs, results, strict=True):
        if isinstance(result, Exception):
            _log.warning(
                f"{name}_layer_failed", base_strategy=strategy, error=type(result).__name__
            )
            continue
        if isinstance(result, BaseException):
            raise result
        extra_chunks, extra_meta = result
        before = len(chunks)
        chunks = deduplicate_chunks(chunks + extra_chunks)
        meta[f"{name}_layer"] = {**extra_meta, "added": len(chunks) - before}
    return chunks


async def _exact_pins(lookups: Sequence[ExactLookup], query: str) -> list[RetrievedChunk]:
    """Run every exact lookup in parallel. A failed lookup is logged and skipped."""
    if not lookups:
        return []
    results = await asyncio.gather(*(lk.lookup(query) for lk in lookups), return_exceptions=True)
    found: list[RetrievedChunk] = []
    for lookup, result in zip(lookups, results, strict=True):
        if isinstance(result, Exception):
            _log.warning("exact_lookup_failed", lookup=type(lookup).__name__)
        elif isinstance(result, BaseException):
            raise result
        else:
            found.extend(result)
    return [_pin(c, "exact_lookup") for c in _dedup_keep_order(found)]


async def _boost_pins(run: RetrievalRun) -> list[RetrievedChunk]:
    """Fetch the chunks of ``boost_ids`` through the retriever. They join the results, no lock."""
    keys = run.config.scope_keys
    ids = list(run.ctx.boost_ids)
    if not ids or not keys.boost_ids:
        return []
    filters = {**_scoped_filters(run.ctx, keys, lock=False), keys.boost_ids: ids}
    try:
        found = await run.retriever.retrieve(
            run.query, top_k=run.config.boost_chunks_per_id * len(ids), filters=filters
        )
    except Exception as error:  # noqa: BLE001 - boosting is best effort
        _log.warning("boost_fetch_failed", error=type(error).__name__)
        return []
    return [_pin(c, "boost") for c in found]


def _emit_usage(hook: UsageHook | None, strategy: str, query: str, chunks: int) -> None:
    if hook is None:
        return
    try:
        hook(
            UsageEvent(
                operation="retrieve",
                input_tokens=max(1, len(query) // 4),
                extra={"strategy": strategy, "chunks": chunks},
            )
        )
    except Exception:  # noqa: BLE001 - metering must never break retrieval
        return


def _warning_for(count: int, config: RetrieveConfig) -> str | None:
    if count == 0:
        return WARNING_NO_RESULTS
    if count <= config.limited_results_max:
        return WARNING_LIMITED_RESULTS
    return None


def _expansion_queries(ctx: QueryContext, config: RetrieveConfig) -> list[str]:
    expansion = ctx.extra.get(EXTRA_QUERY_EXPANSION)
    if not config.use_expansion_variations or not isinstance(expansion, Mapping):
        return []
    variations = [str(v) for v in expansion.get("variations") or ()]
    return variations if expansion.get("applied") and len(variations) > 1 else []


async def retrieve_stage(
    ctx: QueryContext,
    *,
    retriever: Retriever | Mapping[str, Retriever],
    llm: LanguageModel | None = None,
    exact_lookups: Sequence[ExactLookup] = (),
    web_search: WebSearch | None = None,
    jurisdiction: JurisdictionPlugin | None = None,
    usage: UsageHook | None = None,
    on_event: EventSink | None = None,
    config: RetrieveConfig = DEFAULT_RETRIEVE_CONFIG,
    strategies: Mapping[str, StrategyRunner] | None = None,
) -> QueryContext:
    """Retrieve candidate chunks for ``ctx`` and return a context with ``retrieved`` filled in.

    ``retriever`` is one ``Retriever`` or a mapping of named retrievers that are searched
    together (see ``FanOutRetriever``). ``llm`` powers every strategy except ``hybrid`` and
    ``multi_hop``; without it those strategies fall back to ``hybrid`` and say so in their metadata.
    ``exact_lookups`` run first and their chunks are pinned at the front of the result with
    ``metadata[PINNED_KEY] = True`` (``"is_pinned"``), so the rerank and context stages exempt them
    from filtering.
    A matching ``jurisdiction`` plug-in contributes its retriever and its exact lookups.
    ``on_event`` receives multi-hop progress events.

    The result's ``metrics["retrieval"]`` records the strategy used, the depth, the timing and the
    strategy metadata. ``retrieval_warning`` is ``no_results`` or ``limited_results`` when the
    result is empty or thin. A deliberate skip leaves ``retrieved`` empty with no warning.

    Raises:
        UnknownStrategyError: the chosen strategy is not in the table.
        ValueError: ``ctx.tier`` is not in the tier table.
    """
    table = DEFAULT_STRATEGIES if strategies is None else strategies
    tier = get_tier(ctx.tier, config.tiers)
    plan = plan_retrieval(ctx, tier, config)
    runner = resolve_strategy(plan.strategy, table)
    if plan.skip_reason is not None:
        _log.info("retrieve_skipped", reason=plan.skip_reason)
        return replace(
            ctx,
            retrieved=[],
            metrics={
                **ctx.metrics,
                "retrieval": {"skipped": True, "skip_source": plan.skip_reason},
            },
        )
    try:
        return await _execute(
            ctx,
            plan,
            tier,
            runner,
            retriever=retriever,
            llm=llm,
            exact_lookups=exact_lookups,
            web_search=web_search,
            jurisdiction=jurisdiction,
            usage=usage,
            on_event=on_event,
            config=config,
            table=table,
        )
    except Exception as error:  # noqa: BLE001 - fail open: the input context comes back
        _log.error(
            "retrieve_stage_failed",
            strategy=plan.strategy,
            error_type=type(error).__name__,
            exc_info=True,
        )
        ctx.add_warning(WARNING_RETRIEVAL_FAILED)
        return ctx


async def _execute(
    ctx: QueryContext,
    plan: RetrievalPlan,
    tier: TierConfig,
    runner: StrategyRunner,
    *,
    retriever: Retriever | Mapping[str, Retriever],
    llm: LanguageModel | None,
    exact_lookups: Sequence[ExactLookup],
    web_search: WebSearch | None,
    jurisdiction: JurisdictionPlugin | None,
    usage: UsageHook | None,
    on_event: EventSink | None,
    config: RetrieveConfig,
    table: Mapping[str, StrategyRunner],
) -> QueryContext:
    started = time.monotonic()
    query = str(ctx.extra.get(EXTRA_CONDENSED_QUERY) or ctx.query)
    plugin = (
        jurisdiction if jurisdiction is not None and jurisdiction.matches(ctx.filters) else None
    )
    sub_queries = [str(q) for q in ctx.extra.get(EXTRA_SUB_QUERIES) or ()]
    run = RetrievalRun(
        query=query,
        top_k=plan.top_k,
        retriever=_compose_retriever(retriever, plugin),
        filters=_scoped_filters(ctx, config.scope_keys),
        tier=tier,
        ctx=ctx,
        config=config,
        llm=llm,
        web_search=web_search,
        usage=usage,
        on_event=on_event,
        sub_queries=sub_queries,
        expansion=[] if sub_queries else _expansion_queries(ctx, config),
        native_sub_queries=plugin is not None and not isinstance(retriever, Mapping),
        scope_locked=plan.scope_locked,
    )

    lookups = [*exact_lookups, *(plugin.exact_lookups() if plugin is not None else ())]
    pinned = await _exact_pins(lookups, query)

    used = plan.strategy
    chunks, meta = await runner(run)
    if (
        not chunks
        and used != "hybrid"
        and not plan.forced
        and config.retry_empty_with_hybrid
        and not meta.get(SKIPPED_BY_STRATEGY)
        and "hybrid" in table
    ):
        _log.info("retrieve_empty_retry", from_strategy=used, to_strategy="hybrid")
        chunks, retry_meta = await table["hybrid"](run)
        meta = {**retry_meta, "fallback_from": used}
        used = "hybrid"

    chunks = await _additive_layers(run, used, chunks, meta)
    chunks = _pin_first([*pinned, *await _boost_pins(run)], chunks)

    elapsed_ms = (time.monotonic() - started) * 1000
    sources = Counter(c.metadata["retriever"] for c in chunks if "retriever" in c.metadata)
    summary: dict[str, Any] = {
        "strategy": used,
        "top_k": plan.top_k,
        "scope_locked": plan.scope_locked,
        "chunk_count": len(chunks),
        "pinned_count": sum(1 for c in chunks if c.metadata.get(PINNED_KEY)),
        "time_ms": round(elapsed_ms, 2),
        "strategy_metadata": meta,
    }
    if used != plan.strategy:
        summary["requested_strategy"] = plan.strategy
    if sources:
        summary["sources"] = dict(sources)
    _emit_usage(usage, used, query, len(chunks))
    _log.debug("retrieve_stage_complete", strategy=used, chunks=len(chunks), top_k=plan.top_k)
    return replace(
        ctx,
        retrieved=chunks,
        retrieval_warning=_warning_for(len(chunks), config),
        metrics={**ctx.metrics, "retrieval": summary},
    )


__all__ = [
    "DEFAULT_RETRIEVE_CONFIG",
    "DEFAULT_STRATEGIES",
    "EXTRA_CONDENSED_QUERY",
    "EXTRA_FORCE_STRATEGY",
    "EXTRA_QUERY_ANALYSIS",
    "EXTRA_QUERY_EXPANSION",
    "EXTRA_REQUIRES_RETRIEVAL",
    "EXTRA_SUB_QUERIES",
    "SKIPPED_BY_STRATEGY",
    "WARNING_LIMITED_RESULTS",
    "WARNING_NO_RESULTS",
    "WARNING_RETRIEVAL_FAILED",
    "FanOutRetriever",
    "RetrievalPlan",
    "RetrievalRun",
    "RetrieveConfig",
    "ScopeFilterKeys",
    "StrategyOutcome",
    "StrategyRunner",
    "UnknownStrategyError",
    "apply_crag_correction",
    "deduplicate_chunks",
    "has_explicit_scope",
    "plan_retrieval",
    "resolve_strategy",
    "retrieve_stage",
    "run_crag",
    "run_hybrid",
    "run_hyde",
    "run_multi_hop",
    "run_multi_query",
    "run_self_rag",
]
