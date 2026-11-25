# ruff: noqa: BLE001
"""Query understanding stage: analysis, domain check, enrichment, expansion and condensation.

``analyze_stage`` is ``async def stage(ctx) -> ctx``. It never raises: on any unexpected error it
logs, adds ``WARNING_ANALYZE_FAILED`` and returns the input context, so downstream stages must treat a missing key as
"no analysis, retrieve with the raw query".

It writes these flat keys into ``ctx.extra`` (see ``ANALYZE_EXTRA_KEYS``):

- ``query_analysis``: intent, complexity, ``is_in_domain``, ``should_retrieve`` and friends.
- ``expanded_query``, ``query_expansion``, ``enriched_query``: optional query rewrites.
- ``original_query``, ``condensed_query``, ``condensed_sub_queries``, ``query_type``,
  ``condensation_confidence``, ``requires_retrieval``, ``requested_response_language``.
- ``web_search_suggestion``, ``temporal_recency_gap``, ``temporal_requested_year``,
  ``recency_intent``, ``condenser_key``.

Explicit scope (``pinned_ids`` or ``scope_lock_ids``) overrides every heuristic that would skip
retrieval: the identity hook, the general-knowledge check, the text-processing intent, the
analyzer verdict and the condenser's ``requires_retrieval``.
Use ``retrieval_wanted(ctx)`` downstream instead of re-reading those keys.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import JurisdictionPlugin, LanguageModel, UsageHook
from legal_rag.pipeline.tiers import TierConfig, get_tier
from legal_rag.pipeline.types import QueryContext
from legal_rag.retrieval.conversation.condenser import QueryCondenser, QueryType
from legal_rag.retrieval.conversation.intent_router import IntentRouter, UserIntent
from legal_rag.retrieval.conversation.query_analyzer import QueryAnalyzer, get_query_analyzer
from legal_rag.retrieval.conversation.query_enricher import QueryEnricher, get_query_enricher
from legal_rag.retrieval.conversation.query_expander import QueryExpander
from legal_rag.retrieval.routing.query_classification import (
    detect_temporal_recency_gap,
    is_general_knowledge_query,
)
from legal_rag.types import Message, UsageEvent

_log = get_logger("pipeline.analyze")
_T = TypeVar("_T")

WARNING_ANALYZE_FAILED = "analyze_failed"

ANALYZE_EXTRA_KEYS: tuple[str, ...] = (
    "query_analysis",
    "web_search_suggestion",
    "temporal_recency_gap",
    "temporal_requested_year",
    "recency_intent",
    "expanded_query",
    "query_expansion",
    "enriched_query",
    "original_query",
    "condensed_query",
    "condensed_sub_queries",
    "requested_response_language",
    "query_type",
    "condensation_confidence",
    "requires_retrieval",
    "condenser_key",
)
"""Every ``ctx.extra`` key this stage writes."""

DOMAIN_CHECK_SYSTEM_PROMPT = (
    "You are a legal domain classifier. Given a user query, respond with ONLY 'yes' or 'no'. "
    "Answer 'yes' if the query relates to law, legal proceedings, statutes, regulations, court "
    "cases, legal notices, tax notices, compliance, contracts, or any topic where searching a "
    "legal corpus of court judgments and statutes would be helpful. Answer 'no' only for clearly "
    "non-legal queries like greetings, general knowledge, coding, recipes, etc."
)

_RECENCY_INTENT = re.compile(
    r"\b(recent|latest|new|current|contemporary|emerging|modern)\b.*"
    r"\b(development|decision|judgment|ruling|trend|position|case|amendment|change)s?\b",
    re.IGNORECASE,
)
_RECENCY_WINDOW = re.compile(r"\b(last|past|previous)\s+(\d+)\s+(year|month)", re.IGNORECASE)


@dataclass(frozen=True)
class AnalyzeConfig:
    """Tuning for the analyze stage.

    ``corpus_cutoff_year``: the last year your corpus covers. A query that names a later year sets
    ``temporal_recency_gap`` and ``web_search_suggestion``. ``None`` turns the check off, because
    the right year is a fact about your corpus. Only years 2020 to 2099 are recognized.
    ``domain_check_min_words``: when the rule-based analyzer says "do not retrieve" for a query at
    least this long, one cheap model call double-checks before retrieval is skipped.
    ``text_processing_min_confidence``: minimum intent confidence to treat a message as "transform
    my pasted text" and skip retrieval.
    ``user_intent_min_confidence``: below this, ``user_intent`` is reported as ``None``.
    """

    corpus_cutoff_year: int | None = None
    domain_check_min_words: int = 15
    domain_check_max_chars: int = 500
    domain_check_system_prompt: str = DOMAIN_CHECK_SYSTEM_PROMPT
    text_processing_min_confidence: float = 0.7
    user_intent_min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.domain_check_min_words < 1 or self.domain_check_max_chars < 1:
            raise ValueError("AnalyzeConfig domain check limits must be positive")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def has_explicit_scope(ctx: QueryContext) -> bool:
    """True when the caller pinned documents or locked the search to specific ids.

    Explicit scope is a stronger signal than any classifier: the intent is "search those", never
    "skip retrieval because of what the message looks like". ``boost_ids`` is not a scope.
    """
    return bool(ctx.pinned_ids or ctx.scope_lock_ids)


def retrieval_wanted(ctx: QueryContext) -> bool:
    """Whether retrieval should run, from the analyze outputs. Explicit scope always wins."""
    if has_explicit_scope(ctx):
        return True
    analysis = ctx.extra.get("query_analysis") or {}
    if analysis.get("should_retrieve") is False:
        return False
    return ctx.extra.get("requires_retrieval", True) is not False


def detect_recency_gap(query: str, cutoff_year: int | None) -> tuple[bool, int | None]:
    """``(gap, requested_year)``: the query names a year after ``cutoff_year``. ``None`` disables."""
    _, year = detect_temporal_recency_gap(query)
    return (cutoff_year is not None and year is not None and year > cutoff_year), year


def detect_recency_intent(query: str) -> bool:
    """True for "recent developments" or "last 5 years" phrasing (recent cases inside the corpus)."""
    return bool(_RECENCY_INTENT.search(query) or _RECENCY_WINDOW.search(query))


def _emit(usage: UsageHook | None, event: UsageEvent) -> None:
    if usage is None:
        return
    try:
        usage(event)
    except Exception as exc:
        _log.warning("usage_hook_failed", error=str(exc))


async def llm_legal_domain_check(
    query: str,
    llm: LanguageModel,
    *,
    usage: UsageHook | None = None,
    config: AnalyzeConfig | None = None,
) -> bool:
    """Cheap yes/no call: does this query need a legal corpus search?

    An empty answer counts as "yes" (a wasted search is cheaper than a missed one).
    Reasoning-token headroom for the yes/no call is the adapter's concern, not this function's.
    """
    cfg = config or AnalyzeConfig()
    text = query[: cfg.domain_check_max_chars]
    answer = await llm.generate(
        [Message("system", cfg.domain_check_system_prompt), Message("user", text)]
    )
    verdict = (answer or "").strip().lower()
    _emit(
        usage,
        UsageEvent(
            operation="legal_domain_check",
            input_tokens=(len(cfg.domain_check_system_prompt) + len(text)) // 4,
            output_tokens=max(1, len(verdict) // 4),
        ),
    )
    return verdict.startswith("yes") or not verdict


def _default(
    given: _T | None, llm: LanguageModel | None, build: Callable[[LanguageModel], _T]
) -> _T | None:
    """``given`` when set, else one built from ``llm``, else ``None`` (rule-based only)."""
    if given is not None:
        return given
    return build(llm) if llm is not None else None


def _history_as_dicts(history: Sequence[Mapping[str, Any] | Message]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for turn in history:
        if isinstance(turn, Message):
            turns.append({"role": turn.role, "content": turn.content})
        elif isinstance(turn, Mapping):
            turns.append(dict(turn))
    return turns


def _skip_updates(
    query: str, intent: str, *, web_search: bool = False, **analysis: Any
) -> dict[str, Any]:
    """Outputs for a short-circuit: no retrieval, no model calls, raw query passed through."""
    return {
        "query_analysis": {
            "intent": intent,
            "complexity": "trivial",
            "is_in_domain": False,
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "should_expand": False,
            "should_decompose": False,
            "legal_entities": [],
            "is_general_knowledge": False,
            "should_retrieve": False,
            "user_intent": intent,
            **analysis,
        },
        "web_search_suggestion": web_search,
        "temporal_recency_gap": False,
        "temporal_requested_year": None,
        "recency_intent": False,
        "expanded_query": None,
        "query_expansion": {"applied": False},
        "enriched_query": None,
        "original_query": query,
        "condensed_query": query,
        "condensed_sub_queries": [],
        "requested_response_language": None,
        "query_type": QueryType.STANDALONE.value,
        "condensation_confidence": 1.0,
        "requires_retrieval": False,
        "condenser_key": None,
    }


# ---------------------------------------------------------------------------
# Condensation
# ---------------------------------------------------------------------------


def _condense_passthrough(
    original: str, query: str, query_type: QueryType, confidence: float = 1.0
) -> dict[str, Any]:
    return {
        "original_query": original,
        "condensed_query": query,
        "condensed_sub_queries": [],
        "requested_response_language": None,
        "query_type": query_type.value,
        "condensation_confidence": confidence,
        "requires_retrieval": True,
    }


async def _condense(
    ctx: QueryContext,
    history: list[dict[str, Any]],
    *,
    base_query: str,
    tier: TierConfig,
    scoped: bool,
    condenser: QueryCondenser | None,
    key: str | None,
) -> dict[str, Any]:
    """Rewrite a context-dependent query into a standalone one. Fail-open to the raw query."""
    original = ctx.query
    if not tier.features.use_query_condensation:
        return _condense_passthrough(original, base_query, QueryType.STANDALONE)
    # Pure scope: retrieval is hard-filtered to the pinned ids, so the raw query works and a
    # rewrite only adds latency and drift. With the shared corpus also on, phrasing matters.
    if scoped and not ctx.enable_shared_corpus:
        return _condense_passthrough(original, base_query, QueryType.FOLLOW_UP)
    if not history or condenser is None:
        return _condense_passthrough(original, base_query, QueryType.STANDALONE)
    try:
        result = await condenser.condense(base_query, history, scope=key or "")
    except Exception as exc:
        _log.error("condense_failed", error=str(exc))
        return _condense_passthrough(original, original, QueryType.STANDALONE, 0.5)
    return {
        "original_query": original,
        "condensed_query": result.condensed_query,
        "condensed_sub_queries": list(result.sub_queries),
        "requested_response_language": result.response_language,
        "query_type": result.query_type.value,
        "condensation_confidence": result.confidence,
        # A conversational or clarification verdict never beats an explicit scope.
        "requires_retrieval": bool(result.requires_retrieval or scoped),
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


async def _understand(
    ctx: QueryContext,
    *,
    tier: TierConfig,
    cfg: AnalyzeConfig,
    llm: LanguageModel | None,
    analyzer: QueryAnalyzer,
    condenser: QueryCondenser | None,
    expander: QueryExpander | None,
    enricher: QueryEnricher,
    router: IntentRouter,
    usage: UsageHook | None,
    plugin: JurisdictionPlugin | None,
    identity_check: Callable[[str], bool] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = ctx.query
    history = _history_as_dicts(ctx.chat_history)
    scoped = has_explicit_scope(ctx)

    if identity_check is not None and not scoped and identity_check(query):
        return _skip_updates(query, "identity"), {"short_circuit": "identity"}

    if is_general_knowledge_query(query, chat_history=history):
        if not scoped:
            updates = _skip_updates(
                query, "general_knowledge", web_search=True, is_general_knowledge=True
            )
            return updates, {"short_circuit": "general_knowledge"}
        _log.info("general_knowledge_overridden_explicit_scope", query=query[:50])

    routing = await router.route(query)
    if (
        routing.user_intent == UserIntent.TEXT_PROCESSING
        and routing.confidence >= cfg.text_processing_min_confidence
    ):
        if not scoped:
            updates = _skip_updates(
                query,
                "text_processing",
                complexity="simple",
                legal_task_type="general",
                has_pasted_content=True,
                content_word_count=routing.content_word_count,
            )
            return updates, {"short_circuit": "text_processing"}
        _log.info("text_processing_overridden_explicit_scope", query=query[:50])

    analysis = analyzer.analyze(query, history, intent_routing=routing)
    should_retrieve = analysis.should_retrieve
    complexity = analysis.complexity.value
    notes: dict[str, Any] = {"short_circuit": None, "domain_check": None}

    if scoped and not should_retrieve:
        should_retrieve = True
        complexity = "moderate" if complexity == "trivial" else complexity
        notes["domain_check"] = "skipped_scope"
    elif not should_retrieve and len(query.split()) >= cfg.domain_check_min_words:
        # The rule-based analyzer can miss a plain-language question about pasted legal text.
        # Fail open: when the check cannot run, retrieve anyway.
        if llm is None:
            should_retrieve, notes["domain_check"] = True, "unavailable"
        else:
            try:
                should_retrieve = await llm_legal_domain_check(query, llm, usage=usage, config=cfg)
                notes["domain_check"] = "ran"
            except Exception as exc:
                _log.warning("legal_domain_check_failed", error=str(exc))
                should_retrieve, notes["domain_check"] = True, "failed"
        if should_retrieve and complexity == "trivial":
            complexity = "moderate"

    # Pinned documents establish domain relevance themselves ("tell me about this file").
    is_in_domain = analysis.is_in_domain or bool(ctx.pinned_ids)
    gap, requested_year = detect_recency_gap(query, cfg.corpus_cutoff_year)
    updates: dict[str, Any] = {
        "query_analysis": {
            "intent": analysis.primary_intent.value,
            "complexity": complexity,
            "is_in_domain": is_in_domain,
            "is_ambiguous": analysis.is_ambiguous,
            "ambiguity_score": analysis.ambiguity_score,
            "should_expand": analysis.should_expand_query,
            "should_decompose": analysis.should_decompose,
            "legal_entities": analysis.legal_entities_found,
            "is_general_knowledge": False,
            "should_retrieve": should_retrieve,
            "legal_task_type": analysis.legal_task_type.value,
            "user_intent": (
                routing.user_intent.value
                if routing.confidence >= cfg.user_intent_min_confidence
                else None
            ),
            "has_pasted_content": routing.content_text is not None,
            "content_word_count": routing.content_word_count,
        },
        "web_search_suggestion": gap,
        "temporal_recency_gap": gap,
        "temporal_requested_year": requested_year,
        "recency_intent": detect_recency_intent(query),
        "expanded_query": None,
        "query_expansion": {"applied": False},
    }

    # Expansion is a tier feature, and is skipped for scoped chats: retrieval is hard-filtered, so
    # a rewrite adds latency and can turn "this case" into an unrelated query.
    if (
        analysis.should_expand_query
        and is_in_domain
        and tier.features.use_query_expansion
        and not scoped
        and expander is not None
    ):
        try:
            expansion = await expander.expand(
                query=query, ambiguity_score=analysis.ambiguity_score, is_legal=is_in_domain
            )
            if expansion.expansion_applied and expansion.expanded_queries:
                updates["expanded_query"] = expansion.expanded_queries[0]
                updates["query_expansion"] = {
                    "applied": True,
                    "variations": expansion.expanded_queries,
                    "original": query,
                }
        except Exception as exc:
            _log.warning("query_expansion_failed", error=str(exc))

    enriched: str | None = None
    try:
        result = enricher.enrich(query=query, is_in_domain=is_in_domain)
        enriched = result.enriched_query if result.enrichment_applied else None
    except Exception as exc:
        _log.warning("query_enrichment_failed", error=str(exc))
    updates["enriched_query"] = enriched

    key = plugin.condenser_key if plugin is not None and plugin.matches(ctx.filters) else None
    condensed = await _condense(
        ctx,
        history,
        base_query=updates["expanded_query"] or enriched or query,
        tier=tier,
        scoped=scoped,
        condenser=condenser,
        key=key,
    )
    updates.update(condensed, condenser_key=key)
    if scoped:
        updates["requires_retrieval"] = True
    return updates, notes


async def analyze_stage(
    ctx: QueryContext,
    *,
    llm: LanguageModel | None = None,
    analyzer: QueryAnalyzer | None = None,
    condenser: QueryCondenser | None = None,
    expander: QueryExpander | None = None,
    enricher: QueryEnricher | None = None,
    intent_router: IntentRouter | None = None,
    usage: UsageHook | None = None,
    config: AnalyzeConfig | None = None,
    tier: TierConfig | None = None,
    plugin: JurisdictionPlugin | None = None,
    identity_check: Callable[[str], bool] | None = None,
) -> QueryContext:
    """Understand the query: classify it, rewrite it for retrieval and decide if retrieval runs.

    ``llm`` builds any of ``condenser``, ``expander`` and ``intent_router`` you leave out, and
    runs the domain check. Without it the stage stays rule-based: no condensation, no expansion.
    ``tier`` defaults to the tier named by ``ctx.tier``. ``use_query_condensation`` and
    ``use_query_expansion`` gate the two model-backed rewrites.
    ``identity_check`` is an optional hook for "questions about the assistant itself": when it
    returns True (and nothing is scoped) retrieval is skipped. No default text ships.
    ``plugin`` supplies ``condenser_key``, the jurisdiction profile name. The condenser keeps one
    prompt, so the key scopes its cache and is reported in ``ctx.extra["condenser_key"]``; build
    the condenser with its own ``system_prompt`` for a different profile.
    Returns a new context. On any error it returns ``ctx`` itself with ``WARNING_ANALYZE_FAILED``
    added, so the answer is marked degraded instead of silently running on the raw query.
    """
    try:
        updates, notes = await _understand(
            ctx,
            tier=tier if tier is not None else get_tier(ctx.tier),
            cfg=config if config is not None else AnalyzeConfig(),
            llm=llm,
            analyzer=analyzer if analyzer is not None else get_query_analyzer(),
            condenser=_default(condenser, llm, lambda m: QueryCondenser(m, usage_hook=usage)),
            expander=_default(expander, llm, lambda m: QueryExpander(m, usage_hook=usage)),
            enricher=enricher if enricher is not None else get_query_enricher(),
            router=intent_router if intent_router is not None else IntentRouter(llm),
            usage=usage,
            plugin=plugin,
            identity_check=identity_check,
        )
    except Exception as exc:
        _log.error("analyze_stage_failed", error=str(exc), error_type=type(exc).__name__)
        ctx.add_warning(WARNING_ANALYZE_FAILED)
        return ctx
    return dataclasses.replace(
        ctx,
        extra={**ctx.extra, **updates},
        metrics={**ctx.metrics, "analyze": notes},
    )


__all__ = [
    "ANALYZE_EXTRA_KEYS",
    "DOMAIN_CHECK_SYSTEM_PROMPT",
    "WARNING_ANALYZE_FAILED",
    "AnalyzeConfig",
    "analyze_stage",
    "detect_recency_gap",
    "detect_recency_intent",
    "has_explicit_scope",
    "llm_legal_domain_check",
    "retrieval_wanted",
]
