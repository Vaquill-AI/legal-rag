"""Rerank stage: rerank, adjust, relevance gate, diversity caps, CRAG filter and reorder.

``rerank_stage`` takes the chunks in ``ctx.retrieved`` and writes the final, ordered list to
``ctx.reranked``. Every step that has a rule worth testing is a pure function in this module:
``_compute_two_bucket_budgets``, ``dynamic_top_k``, ``resolve_rerank_top_k``,
``resolve_gate_threshold``, ``apply_relevance_gate``, ``resolve_document_cap``,
``apply_document_cap``, ``apply_case_cap``, ``detect_contradiction_hints`` and
``split_generated_for_grading``.

Conventions this stage reads from the context (the analyze stage writes them):

- ``ctx.metrics`` or ``ctx.extra`` ``["requested_count"]``: how many results the user asked for.
- ``["query_analysis"]``: a mapping with ``complexity`` and ``legal_task_type``.
- ``["condensed_query"]``: the standalone form of a follow-up question, used to rerank.

Chunk metadata keys: ``document_id`` and ``case_id`` (scope matching and diversity),
``section_number`` (contradiction hints), ``source_type``, ``is_pinned`` and ``is_generated``.
The raw reranker score is snapshotted in ``metadata["rerank_score"]`` before any adjuster runs, and
the relevance gate reads that snapshot, so a demotion can never turn into a deletion.

What the stage writes: ``ctx.reranked`` (final order), ``ctx.extra["alternative_chunks"]`` (the
runners-up, for contrastive explanations), ``ctx.metrics["rerank"]``, ``["crag_filter"]`` and,
when asked for, ``["contradictions"]``.

The stage fails open: any error returns the retrieved chunks unranked, never an empty context.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from legal_rag.chunking.extractors.legal_ner import (
    HybridLegalNERExtractor,
    get_legal_ner_extractor,
)
from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import ChunkAdjuster, EventSink, JurisdictionPlugin
from legal_rag.pipeline.tiers import TierConfig, get_tier
from legal_rag.pipeline.types import PINNED_KEY, QueryContext
from legal_rag.providers.protocols import LanguageModel, Reranker, UsageHook
from legal_rag.retrieval.rerank.base import BaseReranker, RerankOutcome
from legal_rag.retrieval.rerank.cross_encoder import CrossEncoderReranker
from legal_rag.retrieval.rerank.lost_in_middle import LostInMiddleReorderer
from legal_rag.retrieval.strategies.crag import CRAGConfig, CRAGCorrector
from legal_rag.types import RetrievedChunk

_log = get_logger("pipeline.rerank")

PINNED_FLAG = PINNED_KEY
"""Chunk metadata flag for a chunk that must lead the context and survive every filter."""
GENERATED_FLAG = "is_generated"
GENERATED_SOURCE_TYPES: frozenset[str] = frozenset({"generated", "assistant_output"})
_SCOPE_ID_KEYS = ("document_id", "case_id")
_LOW_COMPLEXITY = ("trivial", "simple")

CaseNameExtractor = Callable[[str], Awaitable[list[str]]]
"""``text -> case names``. The default is ``extract_case_names``."""
ChunkPredicate = Callable[[RetrievedChunk], bool]


@dataclass(frozen=True)
class RerankStageConfig:
    """Knobs for the stage. ``None`` on a switch or number means "take the tier's value"."""

    tier: TierConfig | None = None
    use_reranking: bool | None = None
    use_crag: bool | None = None
    use_diversity: bool | None = None
    use_reorder: bool | None = None
    relevance_threshold: float | None = None
    """Gate threshold on the scale of the reranker you inject. A 0-1 cross-encoder and a
    retrieval-scale scorer need different values, so set this when you swap the scorer."""
    fallback_relevance_threshold: float | None = None
    """Threshold on the fallback scorer's scale. ``None`` disables the gate when the fallback
    stands in, because a cross-encoder threshold would empty results on another scale."""
    min_chunks: int | None = None
    min_top_k: int = 5
    max_top_k: int = 40
    top_k_buffer: int = 3
    use_condensed_query: bool = True
    detect_contradictions: bool = False
    downweight_generated: bool = True
    generated_penalty: float = 0.4
    case_law_source_types: frozenset[str] = frozenset({"case_law"})
    case_law_document_cap: int = 2
    landmark_document_cap: int = 4
    per_case_cap: int = 3
    landmark_per_case_cap: int = 5
    landmark_query_pattern: re.Pattern[str] | None = None
    """Queries that match get the higher caps, for multi-opinion judgments. Off by default."""
    case_name_extractor: CaseNameExtractor | None = None
    crag: CRAGConfig | None = None


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _compute_two_bucket_budgets(top_k: int, doc_pool_size: int) -> tuple[int, int]:
    """Split rerank slots between the pinned bucket and the reference bucket.

    Returns ``(pinned_budget, reference_budget)``. Used when the user pinned documents and the
    shared reference corpus is also on, and both pools are non-empty.

    - The pinned bucket targets about 40 percent of ``top_k``, with a floor of 2 chunks.
    - It never exceeds ``top_k // 2 + 1``, so the reference corpus keeps a real voice.
    - It never exceeds ``doc_pool_size``: slots cannot be reserved for chunks that were not found.
    - The reference bucket gets the remainder, with a floor of 1.
    """
    target = max(2, round(top_k * 0.4))
    target = min(target, max(2, top_k // 2 + 1))
    pinned = min(target, doc_pool_size)
    return pinned, max(1, top_k - pinned)


compute_two_bucket_budgets = _compute_two_bucket_budgets


def dynamic_top_k(
    base_top_k: int,
    requested_count: Any,
    *,
    min_top_k: int = 5,
    max_top_k: int = 40,
    buffer: int = 3,
) -> int:
    """The rerank depth after honoring a count the user asked for ("give me 15 cases").

    A valid request becomes ``requested + buffer``, never below ``min_top_k`` or above
    ``max_top_k``. Anything else (missing, zero, negative, not an int) keeps ``base_top_k``.
    """
    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        return base_top_k
    if requested_count <= 0:
        return base_top_k
    return min(max_top_k, max(min_top_k, requested_count + buffer))


def resolve_rerank_top_k(
    tier: TierConfig,
    *,
    requested_count: Any = None,
    complexity: str = "complex",
    legal_task_type: str | None = None,
    pool_has_case_law: bool = False,
    pinned_document_count: int = 0,
    config: RerankStageConfig | None = None,
) -> int:
    """Final rerank depth: dynamic top_k, then the tier's complexity cap and floors.

    The cap comes from tier data (``None`` means uncapped). Floors raise the cap for case-law
    pools and queries and for a single pinned document. A result of 0 means the tier caps this
    complexity at zero, and the stage then skips reranking.
    """
    cfg = config or RerankStageConfig()
    caps = tier.complexity
    top_k = dynamic_top_k(
        tier.top_k_rerank,
        requested_count,
        min_top_k=cfg.min_top_k,
        max_top_k=cfg.max_top_k,
        buffer=cfg.top_k_buffer,
    )
    cap = caps.rerank_cap(complexity)
    case_law_query = (
        legal_task_type in caps.case_law_task_types and complexity not in _LOW_COMPLEXITY
    )
    if cap is not None and (pool_has_case_law or case_law_query):
        cap = max(cap, caps.case_law_rerank_floor)
    if pinned_document_count == 1:
        floor = min(cfg.max_top_k, caps.single_document_rerank_floor)
        top_k = max(top_k, floor)
        if cap is not None:
            cap = max(cap, floor)
    return top_k if cap is None else min(top_k, cap)


def gate_score(chunk: RetrievedChunk) -> float:
    """The score the relevance gate reads: the raw reranker score when it was snapshotted."""
    raw = chunk.metadata.get("rerank_score")
    return float(raw if raw is not None else (chunk.score or 0.0))


def resolve_gate_threshold(
    chunks: Sequence[RetrievedChunk],
    threshold: float,
    *,
    scorer_degraded: bool = False,
) -> tuple[float, str | None]:
    """The threshold to apply, and why the gate was switched off when it was.

    A threshold is calibrated to one scorer's scale. When the top score is below it, or the
    outcome says a fallback scorer stood in, the scores are not on that scale, so the gate is
    disabled (returns 0.0) instead of dropping everything.
    """
    if threshold <= 0:
        return 0.0, None
    if scorer_degraded:
        return 0.0, "fallback_scorer_in_use"
    top = max((gate_score(c) for c in chunks), default=0.0)
    if top < threshold:
        return 0.0, "top_score_below_threshold"
    return threshold, None


def apply_relevance_gate(
    chunks: Sequence[RetrievedChunk],
    *,
    threshold: float,
    min_chunks: int,
    is_exempt: ChunkPredicate | None = None,
) -> list[RetrievedChunk]:
    """Drop chunks whose raw score is below ``threshold``, always keeping ``min_chunks``.

    Exempt chunks (pinned scope) are never dropped. Order is preserved.
    """
    if threshold <= 0 or len(chunks) <= min_chunks:
        return list(chunks)
    exempt = is_exempt or _never
    passing = [c for c in chunks if exempt(c) or gate_score(c) >= threshold]
    if len(passing) >= min_chunks:
        return passing
    floor_ids = {id(c) for c in chunks[:min_chunks]}
    return [c for c in chunks if exempt(c) or id(c) in floor_ids]


def resolve_document_cap(
    base_cap: int,
    legal_task_type: str | None,
    query_text: str,
    *,
    case_law_task_types: Iterable[str] = (),
    landmark_pattern: re.Pattern[str] | None = None,
    case_law_cap: int = 2,
    landmark_cap: int = 4,
) -> int:
    """Per-document chunk cap. Case-law research wants breadth, so it tightens the tier cap."""
    if legal_task_type is not None and legal_task_type in set(case_law_task_types):
        if landmark_pattern is not None and landmark_pattern.search(query_text or ""):
            return min(base_cap, landmark_cap)
        return min(base_cap, case_law_cap)
    return base_cap


def apply_document_cap(
    selected: Sequence[RetrievedChunk],
    alternatives: Sequence[RetrievedChunk],
    *,
    max_per_doc: int,
    is_exempt: ChunkPredicate | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], int]:
    """Cap chunks per document, then backfill from the alternatives.

    Returns ``(selected, alternatives, removed)``. Backfill prefers documents not yet present,
    then documents with fewer kept chunks, then higher score, and obeys the cap too. Exempt chunks
    bypass the cap and do not count toward it.
    """
    exempt = is_exempt or _never
    counts: Counter[str] = Counter()
    kept: list[RetrievedChunk] = []
    removed = 0
    for chunk in selected:
        if exempt(chunk):
            kept.append(chunk)
            continue
        key = _document_key(chunk)
        if counts[key] < max_per_doc:
            counts[key] += 1
            kept.append(chunk)
        else:
            removed += 1
    if not removed:
        return list(selected), list(alternatives), 0

    kept_docs = {_document_key(c) for c in kept}
    pool = sorted(
        alternatives,
        key=lambda c: (
            1 if _document_key(c) in kept_docs else 0,
            counts.get(_document_key(c), 0),
            -c.score,
        ),
    )
    added: list[RetrievedChunk] = []
    for candidate in pool:
        if len(added) >= removed:
            break
        key = _document_key(candidate)
        if exempt(candidate) or counts[key] < max_per_doc:
            added.append(candidate)
            counts[key] += 1
    used = {id(c) for c in added}
    ordered = sorted([*kept, *added], key=lambda c: c.score, reverse=True)
    return ordered, [c for c in alternatives if id(c) not in used], removed


async def extract_case_names(
    text: str, extractor: HybridLegalNERExtractor | None = None
) -> list[str]:
    """Case names cited in ``text``. A thin wrapper over the NER extractor's method."""
    return await (extractor or get_legal_ner_extractor()).extract_case_names(text)


async def apply_case_cap(
    selected: Sequence[RetrievedChunk],
    alternatives: Sequence[RetrievedChunk],
    *,
    max_per_case: int,
    extractor: CaseNameExtractor | None = None,
    is_exempt: ChunkPredicate | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], int]:
    """Cap how many chunks may discuss the same cited case, then backfill from alternatives.

    Several documents can quote one frequently cited case, so a document cap alone leaves the
    context repeating it. Returns ``(selected, alternatives, removed)``. Extraction errors count
    as "no case names", so a failing extractor never changes the result.
    """
    exempt = is_exempt or _never
    extract = extractor or extract_case_names

    async def names_of(chunk: RetrievedChunk) -> list[str]:
        try:
            return list(await extract(chunk.text))
        except Exception as error:  # noqa: BLE001 - extraction is best effort
            _log.warning("case_name_extraction_failed", error_type=type(error).__name__)
            return []

    graded = [c for c in selected if not exempt(c)]
    names = dict(zip(map(id, graded), await asyncio.gather(*(names_of(c) for c in graded))))
    totals: Counter[str] = Counter(n for found in names.values() for n in found)
    over = {name for name, count in totals.items() if count > max_per_case}
    if not over:
        return list(selected), list(alternatives), 0

    kept_count: Counter[str] = Counter()
    kept: list[RetrievedChunk] = []
    excess = 0
    for chunk in selected:
        found = names.get(id(chunk), [])
        if any(n in over and kept_count[n] >= max_per_case for n in found):
            excess += 1
            continue
        kept.append(chunk)
        kept_count.update(found)

    added: list[RetrievedChunk] = []
    for candidate in alternatives:
        if len(added) >= excess:
            break
        found = await names_of(candidate)
        if any(n in over and kept_count[n] >= max_per_case for n in found):
            continue
        added.append(candidate)
        kept_count.update(found)
    used = {id(c) for c in added}
    ordered = sorted([*kept, *added], key=lambda c: c.score, reverse=True)
    return ordered, [c for c in alternatives if id(c) not in used], excess


def detect_contradiction_hints(chunks: Sequence[RetrievedChunk]) -> dict[str, Any] | None:
    """Flag sections that two or more different documents reference.

    Metadata only: it does not read the text. A caller can use it to show a visible warning that
    sources may disagree. Returns ``None`` when nothing is shared.
    """
    if len(chunks) < 2:
        return None
    by_section: dict[str, set[str]] = {}
    for chunk in chunks:
        section = chunk.metadata.get("section_number")
        if section:
            by_section.setdefault(str(section), set()).add(_document_key(chunk))
    shared = {section: docs for section, docs in by_section.items() if len(docs) >= 2}
    if not shared:
        return None
    return {
        "conflicting_sections": list(shared),
        "document_ids": sorted(set().union(*shared.values())),
        "section_count": len(shared),
    }


def is_generated(chunk: RetrievedChunk) -> bool:
    """True for the assistant's own earlier output, which is not authority."""
    meta = chunk.metadata
    return bool(meta.get(GENERATED_FLAG)) or meta.get("source_type") in GENERATED_SOURCE_TYPES


def split_generated_for_grading(
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Return ``(to_grade, generated_to_reappend)``.

    Generated chunks grade highly against a regeneration query (they are the prior answer), so
    grading them can push real authority out. They are split off only when real authority is also
    present, so a generated-only set is graded as is and never emptied.
    """
    generated = [c for c in chunks if is_generated(c)]
    if not generated or len(generated) == len(chunks):
        return list(chunks), []
    return [c for c in chunks if not is_generated(c)], generated


def generated_content_downweight(penalty: float = 0.4) -> ChunkAdjuster:
    """The default adjuster: demote generated chunks below real authority.

    Skipped when every chunk is generated, so a set made only of generated content is not
    reordered. The raw score stays in ``metadata["rerank_score"]`` for the relevance gate.
    """

    def adjust(
        query: str, chunks: Sequence[RetrievedChunk], filters: Mapping[str, Any]
    ) -> list[RetrievedChunk]:
        marked = [is_generated(c) for c in chunks]
        if not any(marked) or all(marked):
            return list(chunks)
        return [
            replace(c, score=(c.score or 0.0) * penalty) if flag else c
            for c, flag in zip(chunks, marked, strict=True)
        ]

    adjust.__name__ = "generated_content_downweight"
    return adjust


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _never(_chunk: RetrievedChunk) -> bool:
    return False


def _flagged(chunk: RetrievedChunk) -> bool:
    return bool(chunk.metadata.get(PINNED_FLAG))


def _document_key(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("document_id") or chunk.id)


def _in_scope(chunk: RetrievedChunk, scope_ids: frozenset[str]) -> bool:
    if not scope_ids:
        return False
    if chunk.id in scope_ids:
        return True
    meta = chunk.metadata
    return any(str(meta[key]) in scope_ids for key in _SCOPE_ID_KEYS if meta.get(key) is not None)


def _lookup(ctx: QueryContext, key: str) -> Any:
    for source in (ctx.metrics, ctx.extra):
        if source.get(key) is not None:
            return source[key]
    return None


def _as_chunk_reranker(
    reranker: BaseReranker | Reranker,
    fallback: BaseReranker | Reranker | None,
    usage: UsageHook | None,
) -> BaseReranker:
    if isinstance(reranker, BaseReranker):
        return reranker
    fallback_chunks = _as_chunk_reranker(fallback, None, usage) if fallback is not None else None
    return CrossEncoderReranker(reranker, fallback=fallback_chunks, usage_hook=usage)


async def _rerank_pool(
    primary: BaseReranker,
    fallback: BaseReranker | None,
    query: str,
    pool: list[RetrievedChunk],
    top_k: int,
) -> RerankOutcome:
    """Rerank one pool. A failing primary hands the pool to the fallback, flagged degraded."""
    if not pool:
        return RerankOutcome(query=query)
    try:
        return await primary.rerank(query, pool, top_k=top_k)
    except Exception as error:
        if fallback is None:
            raise
        _log.warning("rerank_primary_failed", error_type=type(error).__name__)
        outcome = await fallback.rerank(query, pool, top_k=top_k)
        outcome.degraded = True
        outcome.degraded_reason = outcome.degraded_reason or f"error:{type(error).__name__}"
        return outcome


def _adjust(
    query: str,
    chunks: Iterable[RetrievedChunk],
    adjusters: Sequence[ChunkAdjuster],
    filters: Mapping[str, Any],
) -> list[RetrievedChunk]:
    """Run the adjusters in order (a failing one is skipped), then order by score."""
    out = list(chunks)
    for adjuster in adjusters:
        try:
            out = list(adjuster(query, out, filters))
        except Exception as error:  # noqa: BLE001 - one bad adjuster must not empty the context
            _log.warning(
                "rerank_adjuster_failed",
                adjuster=getattr(adjuster, "__name__", type(adjuster).__name__),
                error_type=type(error).__name__,
            )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _snapshot(chunks: Iterable[RetrievedChunk]) -> None:
    for chunk in chunks:
        chunk.metadata["rerank_score"] = chunk.score


def _lead_flagged(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    flagged = [c for c in chunks if _flagged(c)]
    return flagged + [c for c in chunks if not _flagged(c)]


def _plugin_adjusters(plugin: JurisdictionPlugin | None, ctx: QueryContext) -> list[ChunkAdjuster]:
    if plugin is None:
        return []
    try:
        return list(plugin.adjusters()) if plugin.matches(ctx.filters) else []
    except Exception as error:  # noqa: BLE001 - a plugin must not break reranking
        _log.warning("rerank_plugin_failed", error_type=type(error).__name__)
        return []


def _finish(
    ctx: QueryContext,
    reranked: Sequence[RetrievedChunk],
    alternatives: Sequence[RetrievedChunk],
    metrics: Mapping[str, Any],
    warnings: Sequence[str] = (),
) -> QueryContext:
    merged = list(ctx.warnings)
    merged.extend(w for w in warnings if w not in merged)
    return replace(
        ctx,
        reranked=list(reranked),
        extra={**ctx.extra, "alternative_chunks": list(alternatives)},
        metrics={**ctx.metrics, **metrics},
        warnings=merged,
    )


async def _emit(on_event: EventSink | None, event: Mapping[str, Any]) -> None:
    if on_event is None:
        return
    try:
        await on_event(dict(event))
    except Exception as error:  # noqa: BLE001 - progress events are best effort
        _log.warning("rerank_event_failed", error_type=type(error).__name__)


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


async def rerank_stage(
    ctx: QueryContext,
    *,
    reranker: BaseReranker | Reranker,
    fallback_reranker: BaseReranker | Reranker | None = None,
    llm: LanguageModel | None = None,
    adjusters: Sequence[ChunkAdjuster] = (),
    jurisdiction: JurisdictionPlugin | None = None,
    usage: UsageHook | None = None,
    on_event: EventSink | None = None,
    config: RerankStageConfig | None = None,
) -> QueryContext:
    """Rerank ``ctx.retrieved`` into ``ctx.reranked``. Never raises on a retrieval-time failure.

    ``reranker`` is a chunk-level ``BaseReranker`` or a provider ``Reranker`` (wrapped in a
    ``CrossEncoderReranker``). ``fallback_reranker`` stands in when it fails. ``llm`` enables the
    CRAG grade filter when the tier asks for it. ``adjusters`` run in order after the generated
    content downweight and before the plugin's own. A bad ``config.tier`` or an unknown tier name
    raises, because that is a setup error and not a retrieval failure.
    """
    cfg = config or RerankStageConfig()
    tier = cfg.tier or get_tier(ctx.tier)
    try:
        return await _run(
            ctx,
            tier=tier,
            cfg=cfg,
            primary=_as_chunk_reranker(reranker, fallback_reranker, usage),
            fallback=(
                _as_chunk_reranker(fallback_reranker, None, usage)
                if fallback_reranker is not None
                else None
            ),
            llm=llm,
            adjusters=[
                *(
                    [generated_content_downweight(cfg.generated_penalty)]
                    if cfg.downweight_generated
                    else []
                ),
                *adjusters,
                *_plugin_adjusters(jurisdiction, ctx),
            ],
            usage=usage,
            on_event=on_event,
        )
    except Exception as error:  # noqa: BLE001 - fail open: unranked chunks beat no chunks
        _log.error("rerank_stage_failed", error_type=type(error).__name__)
        return _finish(
            ctx,
            ctx.retrieved,
            [],
            {"rerank": {"failed": True, "error_type": type(error).__name__}},
            ["rerank_failed"],
        )


async def _run(
    ctx: QueryContext,
    *,
    tier: TierConfig,
    cfg: RerankStageConfig,
    primary: BaseReranker,
    fallback: BaseReranker | None,
    llm: LanguageModel | None,
    adjusters: Sequence[ChunkAdjuster],
    usage: UsageHook | None,
    on_event: EventSink | None,
) -> QueryContext:
    chunks = [replace(c, metadata=dict(c.metadata)) for c in ctx.retrieved]
    if not chunks:
        return _finish(ctx, [], [], {"rerank": {"input": 0, "selected": 0}})

    analysis = _lookup(ctx, "query_analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    complexity = str(analysis.get("complexity") or "complex")
    task_type = analysis.get("legal_task_type")
    task_type = task_type if isinstance(task_type, str) else None
    condensed = _lookup(ctx, "condensed_query")
    query = condensed.strip() if cfg.use_condensed_query and isinstance(condensed, str) else ""
    query = query or ctx.query
    scope_ids = frozenset(str(i) for i in (*ctx.pinned_ids, *ctx.scope_lock_ids))

    def exempt(chunk: RetrievedChunk) -> bool:
        return _flagged(chunk) or _in_scope(chunk, scope_ids)

    top_k = resolve_rerank_top_k(
        tier,
        requested_count=_lookup(ctx, "requested_count"),
        complexity=complexity,
        legal_task_type=task_type,
        pool_has_case_law=any(
            c.metadata.get("source_type") in cfg.case_law_source_types for c in chunks
        ),
        pinned_document_count=len(set(ctx.pinned_ids)),
        config=cfg,
    )
    skip_rerank = top_k == 0
    if skip_rerank:
        top_k = tier.top_k_rerank
    use_reranking = (
        cfg.use_reranking if cfg.use_reranking is not None else tier.features.use_reranking
    )

    stats: dict[str, Any] = {"input": len(chunks), "top_k": top_k, "reranked": False}
    warnings: list[str] = []
    if not use_reranking or skip_rerank:
        selected, alternatives = chunks[:top_k], chunks[top_k : top_k * 2]
    else:
        selected, alternatives, ranking = await _rank(
            ctx, chunks, query, top_k, tier, cfg, primary, fallback, adjusters, scope_ids
        )
        stats.update(ranking, reranked=True)
        if ranking["degraded"]:
            warnings.append("rerank_degraded")

        threshold = cfg.relevance_threshold
        threshold = tier.rerank_score_threshold if threshold is None else threshold
        degraded_for_gate = bool(ranking["degraded"])
        if degraded_for_gate and cfg.fallback_relevance_threshold:
            threshold, degraded_for_gate = cfg.fallback_relevance_threshold, False
        threshold, skipped = resolve_gate_threshold(
            [c for c in selected if not exempt(c)], threshold, scorer_degraded=degraded_for_gate
        )
        min_chunks = cfg.min_chunks if cfg.min_chunks is not None else tier.rerank_min_chunks
        gated = apply_relevance_gate(
            selected, threshold=threshold, min_chunks=min_chunks, is_exempt=exempt
        )
        stats.update(
            gate_threshold=threshold, gate_skipped=skipped, gate_dropped=len(selected) - len(gated)
        )
        selected = gated

        use_diversity = cfg.use_diversity
        if use_diversity is None:
            use_diversity = tier.features.use_diversity_enforcement
        if use_diversity and not ranking["two_bucket"]:
            selected, alternatives, dropped = await _diversify(
                selected, alternatives, ctx, tier, cfg, task_type, exempt
            )
            stats["diversity_dropped"] = dropped

    for chunk in selected:
        chunk.metadata["selected"] = True
    for chunk in alternatives:
        chunk.metadata["selected"] = False
    extra_metrics: dict[str, Any] = {}
    if cfg.detect_contradictions:
        extra_metrics["contradictions"] = detect_contradiction_hints(selected)

    selected, crag_meta = await _crag_filter(ctx, selected, query, llm, tier, cfg, usage)
    if crag_meta is not None:
        extra_metrics["crag_filter"] = crag_meta

    use_reorder = cfg.use_reorder
    if use_reorder is None:
        use_reorder = tier.features.use_lost_in_middle_reorder
    if use_reorder:
        selected = _reorder(selected)
    stats["selected"] = len(selected)
    await _emit(on_event, {"type": "rerank_complete", **stats})
    return _finish(ctx, selected, alternatives, {"rerank": stats, **extra_metrics}, warnings)


async def _rank(
    ctx: QueryContext,
    chunks: list[RetrievedChunk],
    query: str,
    top_k: int,
    tier: TierConfig,
    cfg: RerankStageConfig,
    primary: BaseReranker,
    fallback: BaseReranker | None,
    adjusters: Sequence[ChunkAdjuster],
    scope_ids: frozenset[str],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], dict[str, Any]]:
    """Rerank and adjust. Returns ``(selected, alternatives, info)``.

    Chunks flagged ``is_pinned`` skip the reranker and lead the result. When the user pinned
    documents and the shared corpus is on, the two pools are reranked separately with reserved
    budgets (pinned first), so a short pinned document is not squeezed out by long reference text.
    Adjusters run on the reference pool only in that mode, since pinned documents are the user's
    own material.
    """
    flagged = [c for c in chunks if _flagged(c)]
    pool = [c for c in chunks if not _flagged(c)]
    slots = max(top_k - len(flagged), 1)
    cap = tier.rerank_input_cap

    def capped(items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return items[:cap] if cap and len(items) > cap else items

    pinned_ids = frozenset(str(i) for i in ctx.pinned_ids)
    pinned_pool = [c for c in pool if _in_scope(c, pinned_ids)]
    reference = [c for c in pool if not _in_scope(c, pinned_ids)]
    two_bucket = bool(pinned_ids) and ctx.enable_shared_corpus and bool(pinned_pool and reference)

    if two_bucket:
        pin_budget, ref_budget = _compute_two_bucket_budgets(slots, len(pinned_pool))
        pin_out, ref_out = await asyncio.gather(
            _rerank_pool(primary, fallback, query, capped(pinned_pool), pin_budget * 2),
            _rerank_pool(primary, fallback, query, capped(reference), ref_budget * 2),
        )
        outcomes = [pin_out, ref_out]
        for outcome in outcomes:
            _snapshot(outcome.chunks)
        ref_ranked = _adjust(query, ref_out.chunks, adjusters, ctx.filters)
        selected = [*pin_out.chunks[:pin_budget], *ref_ranked[:ref_budget]]
        alternatives = [
            *pin_out.chunks[pin_budget : pin_budget * 2],
            *ref_ranked[ref_budget : ref_budget * 2],
        ]
    else:
        outcome = await _rerank_pool(primary, fallback, query, capped(pool), slots * 2)
        outcomes = [outcome]
        _snapshot(outcome.chunks)
        ranked = _adjust(query, outcome.chunks, adjusters, ctx.filters)
        selected, alternatives = ranked[:slots], ranked[slots : slots * 2]

    degraded = any(o.degraded for o in outcomes)
    reason = next((o.degraded_reason for o in outcomes if o.degraded_reason), None)
    info = {
        "two_bucket": two_bucket,
        "pinned_leading": len(flagged),
        "degraded": degraded,
        "degraded_reason": reason,
    }
    return [*flagged, *selected], alternatives, info


async def _diversify(
    selected: list[RetrievedChunk],
    alternatives: list[RetrievedChunk],
    ctx: QueryContext,
    tier: TierConfig,
    cfg: RerankStageConfig,
    task_type: str | None,
    exempt: ChunkPredicate,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], int]:
    """Document cap, then (for case-law tasks) the per-cited-case cap. Pinned chunks are exempt."""
    query_text = str(_lookup(ctx, "condensed_query") or ctx.query)
    case_law_tasks = tier.complexity.case_law_task_types
    max_per_doc = resolve_document_cap(
        tier.max_chunks_per_document,
        task_type,
        query_text,
        case_law_task_types=case_law_tasks,
        landmark_pattern=cfg.landmark_query_pattern,
        case_law_cap=cfg.case_law_document_cap,
        landmark_cap=cfg.landmark_document_cap,
    )
    selected, alternatives, removed = apply_document_cap(
        selected, alternatives, max_per_doc=max_per_doc, is_exempt=exempt
    )
    if task_type is not None and task_type in case_law_tasks:
        landmark = cfg.landmark_query_pattern
        max_per_case = (
            cfg.landmark_per_case_cap
            if landmark is not None and landmark.search(query_text)
            else cfg.per_case_cap
        )
        selected, alternatives, more = await apply_case_cap(
            selected,
            alternatives,
            max_per_case=max_per_case,
            extractor=cfg.case_name_extractor,
            is_exempt=exempt,
        )
        removed += more
    return _lead_flagged(selected), alternatives, removed


async def _crag_filter(
    ctx: QueryContext,
    chunks: list[RetrievedChunk],
    query: str,
    llm: LanguageModel | None,
    tier: TierConfig,
    cfg: RerankStageConfig,
    usage: UsageHook | None,
) -> tuple[list[RetrievedChunk], dict[str, Any] | None]:
    """Grade the final set and drop irrelevant chunks. The result is never empty.

    Skipped without a model, when the tier does not ask for it, and for scoped retrieval (pinned
    documents, a scope lock or a named collection), where the chunks are what the user asked
    about. Flagged chunks the grader drops are restored, generated chunks are graded apart from
    real authority and re-appended, and an empty or failed correction keeps every chunk.
    """
    use_crag = cfg.use_crag if cfg.use_crag is not None else tier.features.use_crag
    scoped = bool(ctx.pinned_ids or ctx.scope_lock_ids or ctx.collection)
    if llm is None or not use_crag or not chunks or scoped:
        return chunks, None
    try:
        to_grade, generated = split_generated_for_grading(chunks)
        result = await CRAGCorrector(llm, cfg.crag, usage_hook=usage).evaluate_and_correct(
            query, to_grade
        )
        graded = list(result.corrected_chunks)
        meta: dict[str, Any] = {
            "quality": result.quality.value,
            "corrections": list(result.corrections_applied),
            "success": result.success,
            "before": len(chunks),
        }
        if not graded:
            meta.update(after=len(chunks), kept_all="filter_emptied")
            return chunks, meta
        kept = {id(c) for c in graded}
        graded += [c for c in chunks if _flagged(c) and id(c) not in kept]
        graded += generated
        meta["after"] = len(graded)
        return _lead_flagged(graded), meta
    except Exception as error:  # noqa: BLE001 - best effort, never empty the context
        _log.warning("crag_filter_failed", error_type=type(error).__name__)
        return chunks, {"success": False, "error_type": type(error).__name__}


def _reorder(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Lost-in-the-middle order. Flagged chunks lead, by score, and skip the U-shape."""
    flagged = sorted((c for c in chunks if _flagged(c)), key=lambda c: c.score, reverse=True)
    rest = [c for c in chunks if not _flagged(c)]
    try:
        rest = list(LostInMiddleReorderer().reorder(rest).reordered_chunks)
    except Exception as error:  # noqa: BLE001 - reordering must not block results
        _log.warning("reorder_failed", error_type=type(error).__name__)
    return flagged + rest


__all__ = [
    "GENERATED_FLAG",
    "GENERATED_SOURCE_TYPES",
    "PINNED_FLAG",
    "CaseNameExtractor",
    "RerankStageConfig",
    "apply_case_cap",
    "apply_document_cap",
    "apply_relevance_gate",
    "compute_two_bucket_budgets",
    "detect_contradiction_hints",
    "dynamic_top_k",
    "extract_case_names",
    "gate_score",
    "generated_content_downweight",
    "is_generated",
    "rerank_stage",
    "resolve_document_cap",
    "resolve_gate_threshold",
    "resolve_rerank_top_k",
    "split_generated_for_grading",
]
