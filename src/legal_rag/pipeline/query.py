"""Query orchestrator: tier, two caches, the four stages in order, and the result you read.

``QueryPipeline.run`` resolves a tier, looks the question up in two caches, and on a miss runs the
stages ``analyze``, ``retrieve``, ``rerank`` and ``context`` in that order.
A stage is ``async def stage(ctx) -> ctx``, so there is no graph engine: the runner is a loop, a
cache hit returns before the loop starts, and a stage that returns something other than a
``QueryContext`` is a loud ``TypeError``.

Caches. Layer 1 is the semantic cache (a paraphrase of an earlier question hits). Layer 2 is an
exact cache keyed by the SHA-256 of the normalized question. Both are keyed by the same
``AnswerScope`` plus the caller's ``namespace``, so everything that can change the answer for the
same words (tier, strategy, filters, scope ids, history) is in both keys.
The scope is built by ``AnswerScope.from_request``, which raises on a missing or unknown field.
That guard is deliberate: a filter that reaches only one key is a wrong answer served from cache.
A result is cached only when it has sources and carries no degraded warning (``retrieval_failed``,
``rerank_failed``, ``rerank_degraded`` and friends), so a transient failure is never replayed.

Tenancy is not a pipeline concept. ``request.filters`` is opaque: it reaches your ``Retriever``
and your ``ChunkFetcher`` unchanged and is part of both cache keys, so two tenants never share a
cache entry as long as the tenant filter is in ``request.filters``. Your retriever and chunk
fetcher must apply the tenant filter themselves. Nothing here knows who the caller is.

Failure model. Each stage is fail-open and returns a usable context after any runtime error. A
configuration error is not a runtime error and propagates: an unknown tier raises ``ValueError``,
an unknown retrieval strategy raises ``UnknownStrategyError``, both before any cache or provider
is touched.

No-results retry. When retrieval ran, found nothing and the strategy was not already plain
``hybrid``, the pipeline runs ``retrieve``, ``rerank`` and ``context`` once more with
``force_retrieval_strategy`` set to ``hybrid``, which always returns the top results.

Usage. Every stage and the pipeline itself write ``UsageEvent`` objects to a per-run
``UsageLedger``, which forwards each one to your ``UsageHook``. The ledger is on the result.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import (
    Cache,
    ChunkAdjuster,
    ChunkFetcher,
    EventSink,
    ExactLookup,
    JurisdictionPlugin,
    LanguageModel,
    Reranker,
    Retriever,
    TierResolver,
    TokenCounter,
    UsageHook,
    WebSearch,
)
from legal_rag.pipeline.stages.analyze import WARNING_ANALYZE_FAILED, AnalyzeConfig, analyze_stage
from legal_rag.pipeline.stages.context import (
    ContextStageConfig,
    LabelFor,
    SourceEnricher,
    context_stage,
)
from legal_rag.pipeline.stages.rerank import RerankStageConfig, rerank_stage
from legal_rag.pipeline.stages.retrieve import (
    DEFAULT_RETRIEVE_CONFIG,
    EXTRA_FORCE_STRATEGY,
    WARNING_NO_RESULTS,
    WARNING_RETRIEVAL_FAILED,
    RetrieveConfig,
    StrategyRunner,
    resolve_strategy,
    retrieve_stage,
)
from legal_rag.pipeline.tiers import DEFAULT_TIER, TierConfig, get_tier
from legal_rag.pipeline.types import AnswerScope, PipelineRequest, QueryContext, UsageLedger
from legal_rag.providers.usage import emit_usage
from legal_rag.retrieval.caching.semantic import SemanticCache
from legal_rag.retrieval.rerank.base import BaseReranker
from legal_rag.types import RetrievedChunk

_log = get_logger("legal_rag.pipeline.query")

StageFn = Callable[[QueryContext], Awaitable[QueryContext]]
"""One stage with its collaborators already bound: ``async def stage(ctx) -> ctx``."""

NamedStage = tuple[str, StageFn]

CACHE_MISS = "miss"
CACHE_SEMANTIC_HIT = "semantic_hit"
CACHE_EXACT_HIT = "exact_hit"
CACHE_DISABLED = "disabled"
"""``QueryResult.cache_status`` values. ``disabled`` means no cache was configured."""

WARNING_TIER_RESOLUTION_FAILED = "tier_resolution_failed"

DEGRADED_WARNINGS: frozenset[str] = frozenset(
    {
        WARNING_RETRIEVAL_FAILED,
        WARNING_ANALYZE_FAILED,
        "rerank_failed",
        "rerank_degraded",
        "context_assembly_failed",
        "neighbor_expansion_failed",
        "source_enrichment_failed",
    }
)
"""Warnings that mark a result as degraded. A degraded result is returned but never cached."""

_RETRY_STRATEGY = "hybrid"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryPipelineConfig:
    """The stage configs and the cache and retry switches, in one bundle.

    ``namespace`` prefixes every cache key, so two deployments (or two corpus versions) that share
    one cache store never see each other's entries. ``tiers`` is the tier table source for the
    stages that read tier data (see ``legal_rag.pipeline.tiers.get_tier``); a per-stage config that
    sets its own table keeps it. ``degraded_warnings`` lists the warnings that keep a result out of
    both caches.
    """

    namespace: str = "legal_rag"
    tiers: Path | str | Mapping[str, Any] | None = None
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)
    retrieve: RetrieveConfig = DEFAULT_RETRIEVE_CONFIG
    rerank: RerankStageConfig = field(default_factory=RerankStageConfig)
    context: ContextStageConfig = field(default_factory=ContextStageConfig)
    exact_cache_ttl_seconds: int = 3600
    retry_no_results_with_hybrid: bool = True
    degraded_warnings: frozenset[str] = DEGRADED_WARNINGS

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("QueryPipelineConfig.namespace must be a non-empty string")
        if self.exact_cache_ttl_seconds <= 0:
            raise ValueError("QueryPipelineConfig.exact_cache_ttl_seconds must be positive")


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKeys:
    """The two cache keys of one question, both derived from the same ``AnswerScope``.

    ``semantic_scope`` is the scope string handed to ``SemanticCache``: a paraphrase can only hit
    an entry stored under exactly this scope. ``exact_key`` is the exact cache key, which also
    covers the query text.
    """

    scope: AnswerScope
    semantic_scope: str
    exact_key: str


def normalize_query(query: str) -> str:
    """Trim and collapse whitespace. Case is kept: a citation or a name can depend on it."""
    return " ".join(query.split())


def semantic_scope_key(namespace: str, scope: AnswerScope) -> str:
    """Scope string for the semantic cache: the namespace plus the scope digest."""
    return f"{namespace}:{scope.key()}"


def exact_cache_key(namespace: str, query: str, scope: AnswerScope) -> str:
    """Key for the exact cache: the namespace plus the SHA-256 of the query and the scope."""
    blob = json.dumps(
        {"query": normalize_query(query), "scope": scope.key()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"{namespace}:query:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"


def build_cache_keys(
    namespace: str,
    request: PipelineRequest,
    *,
    tier: str,
    retrieval_strategy: str,
) -> CacheKeys:
    """Both cache keys of a request, with the already resolved tier and strategy."""
    scope = AnswerScope.from_request(request, tier=tier, retrieval_strategy=retrieval_strategy)
    return CacheKeys(
        scope=scope,
        semantic_scope=semantic_scope_key(namespace, scope),
        exact_key=exact_cache_key(namespace, request.query, scope),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """What the pipeline hands back.

    ``context`` is the text for the answering model and ``sources`` the numbered sources it cites.
    ``chunks`` are the chunks that made it into the context. ``warnings`` holds degraded-result
    markers; ``retrieval_warning`` is ``no_results`` or ``limited_results`` when the result is thin.
    ``strategy`` is the retrieval strategy that ran. ``cache_status`` is one of ``miss``,
    ``semantic_hit``, ``exact_hit`` or ``disabled``. On a cache hit ``metrics`` are the stored
    metrics (JSON round-tripped) and ``usage`` holds only the pipeline's own event.
    """

    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    tier: str = DEFAULT_TIER
    strategy: str = "hybrid"
    cache_status: str = CACHE_MISS
    retrieval_warning: str | None = None
    usage: UsageLedger = field(default_factory=UsageLedger)

    @property
    def cache_hit(self) -> bool:
        """True when the result came from either cache."""
        return self.cache_status in (CACHE_SEMANTIC_HIT, CACHE_EXACT_HIT)

    @property
    def degraded(self) -> bool:
        """True when a stage reported a degraded-result warning."""
        return any(w in DEGRADED_WARNINGS for w in self.warnings)


def _jsonable(value: Any) -> Any:
    """A JSON round-tripped copy. Values JSON cannot hold become strings."""
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _payload_from_result(result: QueryResult) -> dict[str, Any]:
    return _jsonable(
        {
            "context": result.context,
            "sources": result.sources,
            "chunks": [
                {"id": c.id, "text": c.text, "score": c.score, "metadata": c.metadata}
                for c in result.chunks
            ],
            "warnings": result.warnings,
            "metrics": result.metrics,
            "tier": result.tier,
            "strategy": result.strategy,
            "retrieval_warning": result.retrieval_warning,
        }
    )


def _result_from_payload(payload: Any, *, cache_status: str) -> QueryResult | None:
    """Rebuild a result from a cached payload, or ``None`` when the payload is not usable."""
    try:
        if not isinstance(payload, Mapping) or not isinstance(payload["context"], str):
            return None
        sources = payload["sources"]
        if not isinstance(sources, list) or not sources:
            return None
        chunks = [
            RetrievedChunk(
                id=str(c["id"]),
                text=str(c.get("text", "")),
                score=float(c.get("score", 0.0)),
                metadata=dict(c.get("metadata") or {}),
            )
            for c in payload.get("chunks") or []
        ]
        warning = payload.get("retrieval_warning")
        return QueryResult(
            context=payload["context"],
            sources=[dict(s) for s in sources],
            chunks=chunks,
            warnings=[str(w) for w in payload.get("warnings") or []],
            metrics=dict(payload.get("metrics") or {}),
            tier=str(payload.get("tier") or DEFAULT_TIER),
            strategy=str(payload.get("strategy") or "hybrid"),
            cache_status=cache_status,
            retrieval_warning=str(warning) if warning else None,
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


class _Events:
    """Wraps the caller's event sink so a failing sink can never break a query."""

    def __init__(self, sink: EventSink | None) -> None:
        self._sink = sink

    @property
    def sink(self) -> EventSink | None:
        """The guarded sink to hand to the stages, or ``None`` when the caller gave none."""
        return self._guarded if self._sink is not None else None

    async def _guarded(self, event: Any) -> None:
        if self._sink is None:
            return
        try:
            await self._sink(event)
        except Exception as error:  # noqa: BLE001 - progress events are best effort
            _log.warning("event_sink_failed", error_type=type(error).__name__)

    async def emit(self, kind: str, **data: Any) -> None:
        await self._guarded({"type": kind, **data})


async def run_stages(
    ctx: QueryContext,
    stages: Sequence[NamedStage | StageFn],
    *,
    on_event: EventSink | None = None,
    timings: dict[str, float] | None = None,
) -> QueryContext:
    """Run stages in order, feeding each the context the previous one returned.

    A stage is ``async def stage(ctx) -> ctx`` or a ``(name, stage)`` pair. Every stage returns a
    whole context, so there is nothing to merge: the next stage simply reads what was returned.
    Nothing is caught here. A stage that fails open does so itself, and one that raises a
    configuration error propagates. A stage that returns anything but a ``QueryContext`` raises
    ``TypeError``. ``timings`` collects elapsed milliseconds per stage name, and ``on_event``
    receives a ``stage_completed`` event after each stage.
    """
    events = _Events(on_event)
    for item in stages:
        if isinstance(item, tuple):
            name, stage = item
        else:
            stage = item
            name = getattr(stage, "__name__", "stage")
        started = time.monotonic()
        result = await stage(ctx)
        if not isinstance(result, QueryContext):
            raise TypeError(f"stage {name!r} returned {type(result).__name__}, not a QueryContext")
        ctx = result
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        if timings is not None:
            timings[name] = round(timings.get(name, 0.0) + elapsed_ms, 2)
        await events.emit("stage_completed", stage=name, elapsed_ms=elapsed_ms)
    return ctx


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class QueryPipeline:
    """Answer-context pipeline: cache lookup, then analyze, retrieve, rerank and context.

    Build it once with your providers and reuse it for every request; it holds no per-query state.
    ``retriever`` and ``reranker`` are required. ``llm`` powers the model-backed parts of the
    stages (condensation, expansion, the adaptive strategies, the corrective filter); without it
    the stages stay rule-based and the adaptive strategies fall back to ``hybrid``.
    ``fetcher`` loads neighbor chunks. ``jurisdiction`` is an opt-in plug-in that applies to a
    request when its ``matches(request.filters)`` says so. ``exact_lookups`` pin named authorities.
    ``semantic_cache`` is layer 1 and ``cache`` backs layer 2, the exact cache; leave either out to
    skip that layer. ``usage`` receives every ``UsageEvent`` the run produces.
    ``strategies`` replaces the retrieval strategy table.
    """

    def __init__(
        self,
        *,
        retriever: Retriever | Mapping[str, Retriever],
        reranker: BaseReranker | Reranker,
        fallback_reranker: BaseReranker | Reranker | None = None,
        llm: LanguageModel | None = None,
        fetcher: ChunkFetcher | None = None,
        jurisdiction: JurisdictionPlugin | None = None,
        exact_lookups: Sequence[ExactLookup] = (),
        web_search: WebSearch | None = None,
        adjusters: Sequence[ChunkAdjuster] = (),
        semantic_cache: SemanticCache | None = None,
        cache: Cache | None = None,
        usage: UsageHook | None = None,
        token_counter: TokenCounter | None = None,
        label_for: LabelFor | None = None,
        source_enricher: SourceEnricher | None = None,
        identity_check: Callable[[str], bool] | None = None,
        strategies: Mapping[str, StrategyRunner] | None = None,
        config: QueryPipelineConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._fallback_reranker = fallback_reranker
        self._llm = llm
        self._fetcher = fetcher
        self._jurisdiction = jurisdiction
        self._exact_lookups = tuple(exact_lookups)
        self._web_search = web_search
        self._adjusters = tuple(adjusters)
        self._semantic = semantic_cache
        self._cache = cache
        self._usage = usage
        self._token_counter = token_counter
        self._label_for = label_for
        self._source_enricher = source_enricher
        self._identity_check = identity_check
        self._strategies = strategies
        cfg = config or QueryPipelineConfig()
        if cfg.tiers is not None:
            if cfg.retrieve.tiers is None:
                cfg = replace(cfg, retrieve=replace(cfg.retrieve, tiers=cfg.tiers))
            if cfg.context.tiers is None:
                cfg = replace(cfg, context=replace(cfg.context, tiers=cfg.tiers))
        self._config = cfg

    @property
    def config(self) -> QueryPipelineConfig:
        return self._config

    # -- planning and keys -------------------------------------------------

    async def _resolve_tier(
        self, request: PipelineRequest, tier_resolver: TierResolver | None
    ) -> tuple[TierConfig, list[str]]:
        """The tier for a request: the request's own, else the resolver's, else the default.

        A resolver that raises falls back to the default tier and adds a warning. A tier name that
        is not in the table raises ``ValueError``, because that is a setup error.
        """
        warnings: list[str] = []
        name: str | None = request.tier
        if not name and tier_resolver is not None:
            try:
                resolved = tier_resolver(request)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                name = str(resolved) if resolved else None
            except Exception as error:  # noqa: BLE001 - a broken resolver must not fail the query
                _log.warning("tier_resolver_failed", error_type=type(error).__name__)
                warnings.append(WARNING_TIER_RESOLUTION_FAILED)
        return get_tier(name or DEFAULT_TIER, self._config.tiers), warnings

    def cache_keys(
        self, request: PipelineRequest, *, tier: str, retrieval_strategy: str
    ) -> CacheKeys:
        """The two cache keys for a request, given the resolved tier and strategy."""
        return build_cache_keys(
            self._config.namespace, request, tier=tier, retrieval_strategy=retrieval_strategy
        )

    # -- cache -------------------------------------------------------------

    async def _lookup(self, request: PipelineRequest, keys: CacheKeys) -> QueryResult | None:
        """Layer 1 (semantic), then layer 2 (exact). Both fail open to a miss."""
        if self._semantic is not None:
            hit = _result_from_payload(
                await self._semantic.get(request.query, keys.semantic_scope),
                cache_status=CACHE_SEMANTIC_HIT,
            )
            if hit is not None:
                return hit
        if self._cache is not None:
            try:
                raw = await self._cache.get(keys.exact_key)
                if raw:
                    hit = _result_from_payload(
                        json.loads(raw.decode("utf-8")), cache_status=CACHE_EXACT_HIT
                    )
                    if hit is not None:
                        return hit
            except Exception as error:  # noqa: BLE001 - a broken cache is a miss
                _log.warning("exact_cache_get_failed", error_type=type(error).__name__)
        return None

    def _cacheable(self, ctx: QueryContext) -> bool:
        """Cache only a result that has sources and no degraded warning."""
        if not ctx.sources:
            return False
        return not any(w in self._config.degraded_warnings for w in ctx.warnings)

    async def _store(self, request: PipelineRequest, keys: CacheKeys, result: QueryResult) -> None:
        payload = _payload_from_result(result)
        if self._semantic is not None:
            await self._semantic.set(request.query, payload, keys.semantic_scope)
        if self._cache is not None:
            try:
                await self._cache.set(
                    keys.exact_key,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    ttl_seconds=self._config.exact_cache_ttl_seconds,
                )
            except Exception as error:  # noqa: BLE001 - failing to cache never fails the query
                _log.warning("exact_cache_set_failed", error_type=type(error).__name__)

    # -- stages ------------------------------------------------------------

    def _stages(
        self,
        tier: TierConfig,
        filters: Mapping[str, Any],
        usage: UsageHook,
        on_event: EventSink | None,
    ) -> dict[str, StageFn]:
        """The four stages with their collaborators bound, looked up at call time by name."""
        cfg = self._config
        plugin = self._jurisdiction
        matched = plugin if plugin is not None and plugin.matches(filters) else None
        label_for = self._label_for or (matched.label_for if matched is not None else None)
        rerank_cfg = cfg.rerank if cfg.rerank.tier is not None else replace(cfg.rerank, tier=tier)

        async def analyze(ctx: QueryContext) -> QueryContext:
            return await analyze_stage(
                ctx,
                llm=self._llm,
                usage=usage,
                config=cfg.analyze,
                tier=tier,
                plugin=matched,
                identity_check=self._identity_check,
            )

        async def retrieve(ctx: QueryContext) -> QueryContext:
            return await retrieve_stage(
                ctx,
                retriever=self._retriever,
                llm=self._llm,
                exact_lookups=self._exact_lookups,
                web_search=self._web_search,
                jurisdiction=plugin,
                usage=usage,
                on_event=on_event,
                config=cfg.retrieve,
                strategies=self._strategies,
            )

        async def rerank(ctx: QueryContext) -> QueryContext:
            return await rerank_stage(
                ctx,
                reranker=self._reranker,
                fallback_reranker=self._fallback_reranker,
                llm=self._llm,
                adjusters=self._adjusters,
                jurisdiction=plugin,
                usage=usage,
                on_event=on_event,
                config=rerank_cfg,
            )

        async def context(ctx: QueryContext) -> QueryContext:
            return await context_stage(
                ctx,
                fetcher=self._fetcher,
                token_counter=self._token_counter,
                label_for=label_for,
                source_enricher=self._source_enricher,
                usage=usage,
                config=cfg.context,
            )

        return {"analyze": analyze, "retrieve": retrieve, "rerank": rerank, "context": context}

    def _wants_retry(self, ctx: QueryContext) -> bool:
        """True when retrieval ran, found nothing and plain ``hybrid`` has not run yet."""
        if not self._config.retry_no_results_with_hybrid:
            return False
        if ctx.retrieval_warning != WARNING_NO_RESULTS or ctx.extra.get(EXTRA_FORCE_STRATEGY):
            return False
        if WARNING_RETRIEVAL_FAILED in ctx.warnings:
            return False
        retrieval = ctx.metrics.get("retrieval")
        if not isinstance(retrieval, Mapping) or retrieval.get("skipped"):
            return False
        meta = retrieval.get("strategy_metadata")
        ran_hybrid = retrieval.get("strategy") == _RETRY_STRATEGY or (
            isinstance(meta, Mapping) and bool({"degraded_from", "fallback_from"} & set(meta))
        )
        if ran_hybrid:
            return False
        table = self._strategies
        return table is None or _RETRY_STRATEGY in table

    # -- run ---------------------------------------------------------------

    async def run(
        self,
        request: PipelineRequest,
        *,
        tier_resolver: TierResolver | None = None,
        on_event: EventSink | None = None,
    ) -> QueryResult:
        """Answer-context for one request.

        Order: resolve the tier and strategy (configuration errors raise here), check the semantic
        cache, check the exact cache, then run the stages, retry once with ``hybrid`` on no
        results, and cache the result if it is complete.
        ``on_event`` receives dict events with a ``type`` key: ``pipeline_started``, ``cache_hit``,
        ``stage_completed``, ``retry``, ``cache_stored`` and ``pipeline_completed``. The stages add
        their own progress events, such as multi-hop steps. A sink that raises is ignored.

        Raises:
            ValueError: the query is blank, or the tier is not in the tier table.
            UnknownStrategyError: the requested or tier default strategy is not in the table.
        """
        if not request.query or not request.query.strip():
            raise ValueError("PipelineRequest.query must not be blank")
        started = time.monotonic()
        tier, tier_warnings = await self._resolve_tier(request, tier_resolver)
        strategy = request.retrieval_strategy or tier.retrieval_strategy
        resolve_strategy(strategy, self._strategies)
        keys = self.cache_keys(request, tier=tier.name, retrieval_strategy=strategy)
        ledger = UsageLedger(downstream=self._usage)
        events = _Events(on_event)
        await events.emit("pipeline_started", tier=tier.name, strategy=strategy)

        caching = self._semantic is not None or self._cache is not None
        hit = await self._lookup(request, keys) if caching else None
        if hit is not None:
            hit.usage = ledger
            hit.tier = tier.name
            hit.metrics["pipeline"] = {}
            await events.emit("cache_hit", layer=hit.cache_status)
            return await self._finish(request, hit, ledger, events, started)

        ctx = QueryContext.from_request(request, tier=tier.name, retrieval_strategy=strategy)
        for warning in tier_warnings:
            ctx.add_warning(warning)
        stages = self._stages(tier, request.filters, ledger, events.sink)
        timings: dict[str, float] = {}

        ctx = await run_stages(
            ctx, [("analyze", stages["analyze"])], on_event=events.sink, timings=timings
        )
        pre_retrieval_warnings = list(ctx.warnings)
        body = [(name, stages[name]) for name in ("retrieve", "rerank", "context")]
        ctx = await run_stages(ctx, body, on_event=events.sink, timings=timings)

        if self._wants_retry(ctx):
            first_pass = {
                key: ctx.metrics[key] for key in ("retrieval", "rerank") if key in ctx.metrics
            }
            await events.emit("retry", reason=WARNING_NO_RESULTS, strategy=_RETRY_STRATEGY)
            ctx = replace(
                ctx,
                retrieved=[],
                reranked=[],
                context_chunks=[],
                context="",
                sources=[],
                retrieval_warning=None,
                warnings=pre_retrieval_warnings,
                extra={**ctx.extra, EXTRA_FORCE_STRATEGY: _RETRY_STRATEGY},
                metrics={
                    **ctx.metrics,
                    "no_results_retry": {"from_strategy": strategy, "first_pass": first_pass},
                },
            )
            ctx = await run_stages(ctx, body, on_event=events.sink, timings=timings)

        result = self._result_from_context(
            ctx, tier=tier.name, strategy=strategy, ledger=ledger, caching=caching
        )
        result.metrics["pipeline"] = {"stages_ms": timings}
        if caching and self._cacheable(ctx):
            await self._store(request, keys, result)
            await events.emit("cache_stored")
        return await self._finish(request, result, ledger, events, started)

    def _result_from_context(
        self,
        ctx: QueryContext,
        *,
        tier: str,
        strategy: str,
        ledger: UsageLedger,
        caching: bool,
    ) -> QueryResult:
        retrieval = ctx.metrics.get("retrieval")
        ran = retrieval.get("strategy") if isinstance(retrieval, Mapping) else None
        meta = retrieval.get("strategy_metadata") if isinstance(retrieval, Mapping) else None
        if isinstance(meta, Mapping) and meta.get("degraded_from"):
            ran = _RETRY_STRATEGY  # an adaptive strategy without a model ran plain hybrid
        return QueryResult(
            context=ctx.context,
            sources=list(ctx.sources),
            chunks=list(ctx.context_chunks),
            warnings=list(ctx.warnings),
            metrics=dict(ctx.metrics),
            tier=tier,
            strategy=str(ran or ctx.retrieval_strategy or strategy),
            cache_status=CACHE_MISS if caching else CACHE_DISABLED,
            retrieval_warning=ctx.retrieval_warning,
            usage=ledger,
        )

    async def _finish(
        self,
        request: PipelineRequest,
        result: QueryResult,
        ledger: UsageLedger,
        events: _Events,
        started: float,
    ) -> QueryResult:
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        result.metrics.setdefault("pipeline", {})["elapsed_ms"] = elapsed_ms
        emit_usage(
            ledger,
            "query_pipeline",
            input_text=request.query,
            output_text=result.context,
            tier=result.tier,
            strategy=result.strategy,
            cache_status=result.cache_status,
            chunks=len(result.chunks),
        )
        await events.emit(
            "pipeline_completed",
            cache_status=result.cache_status,
            chunks=len(result.chunks),
            warnings=list(result.warnings),
            elapsed_ms=elapsed_ms,
        )
        return result


__all__ = [
    "CACHE_DISABLED",
    "CACHE_EXACT_HIT",
    "CACHE_MISS",
    "CACHE_SEMANTIC_HIT",
    "DEGRADED_WARNINGS",
    "WARNING_TIER_RESOLUTION_FAILED",
    "CacheKeys",
    "NamedStage",
    "QueryPipeline",
    "QueryPipelineConfig",
    "QueryResult",
    "StageFn",
    "build_cache_keys",
    "exact_cache_key",
    "normalize_query",
    "run_stages",
    "semantic_scope_key",
]
