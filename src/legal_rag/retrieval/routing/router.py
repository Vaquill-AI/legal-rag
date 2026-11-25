"""The unified router: prefilter, shortlist, classify, tiered decision.

A pure-ish async function of (RoutingInput, RouterState) plus an injected taxonomy and two
optional injected providers: an embedder for the candidate shortlist and a language model for
the classifier.

- No embedder: the classifier sees every destination. The shortlist is a latency optimization.
- No language model: the router returns the safe default (answer at low confidence).

Nothing here executes a destination; the caller acts on the returned ``RouteDecision``.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import logging
from collections.abc import Sequence

from legal_rag.providers.protocols import Embedder, LanguageModel
from legal_rag.retrieval.routing import classifier
from legal_rag.retrieval.routing.config import Taxonomy
from legal_rag.retrieval.routing.decision import (
    DEFAULT_POLICY,
    Action,
    DecisionPolicy,
    RouteDecision,
    decide,
)
from legal_rag.retrieval.routing.prefilter import prefilter
from legal_rag.retrieval.routing.state import RouterState, RoutingInput

logger = logging.getLogger("legal_rag.retrieval.routing.router")

_SHORTLIST_K = 6


def _safe_default(taxonomy: Taxonomy, reason: str, source: str = "fallback") -> RouteDecision:
    return RouteDecision(taxonomy.default_intent, 0.3, Action.ANSWER, reason=reason, source=source)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


async def shortlist(
    text: str, taxonomy: Taxonomy, embedder: Embedder, k: int = _SHORTLIST_K
) -> tuple[str, ...]:
    """Rank destinations by best example similarity to the query and keep the top k.

    The default intent is always kept. On any embedder failure every intent is returned.
    """
    try:
        qv = list(await embedder.embed_query(text))
        examples: list[tuple[str, str]] = []
        for d in taxonomy.destinations:
            for ex in d.examples or (d.description,):
                examples.append((d.intent, ex))
        vectors = await embedder.embed_documents([ex for _, ex in examples])
        best: dict[str, float] = {d.intent: 0.0 for d in taxonomy.destinations}
        for (intent, _), vec in zip(examples, vectors, strict=False):
            best[intent] = max(best[intent], _cosine(qv, vec))
        ranked = sorted(best, key=lambda i: -best[i])
        top = ranked[:k]
        if taxonomy.default_intent not in top:
            top.append(taxonomy.default_intent)
        return tuple(top)
    except Exception as exc:
        logger.warning("router_shortlist_failed error=%s", type(exc).__name__)
        return taxonomy.intents()


async def route(
    ri: RoutingInput,
    state: RouterState,
    *,
    taxonomy: Taxonomy,
    embedder: Embedder | None = None,
    llm: LanguageModel | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> RouteDecision:
    """Route one turn. Never raises; the safe default is the taxonomy's default intent."""
    pre = prefilter(ri.instruction, state, taxonomy)
    if pre is not None:
        return pre
    if llm is None:
        return _safe_default(taxonomy, "no classifier configured")

    candidates = (
        await shortlist(ri.routing_text, taxonomy, embedder) if embedder else taxonomy.intents()
    )
    intent, confidence, secondary, reason = await classifier.classify(
        ri, state, candidates, taxonomy=taxonomy, llm=llm
    )
    decision = decide(
        taxonomy,
        intent,
        confidence,
        state,
        secondary_intents=secondary,
        reason=reason,
        source="classifier",
        policy=policy,
    )
    logger.info(
        "router_decision intent=%s confidence=%.3f action=%s precondition=%s",
        decision.intent,
        decision.confidence,
        decision.action.value,
        decision.precondition_note,
    )
    return decision
