"""Context assembly: the last stage before generation.

Takes the reranked chunks (or the retrieved ones when no rerank ran) and builds what the answering
model reads:

1. Token budgeted selection. The keep-set is chosen by score, never by position, because a
   lost-in-the-middle reorder parks the best chunk at the tail. Oversized chunks are truncated,
   chunks of explicitly pinned documents are reserved first, and the kept chunks are emitted in
   their input order so the reorder survives.
2. Neighbor expansion through a ``ChunkFetcher``, capped by the tier and the query complexity.
3. Source dedupe, with a remap so every ``[Source N]`` in the context resolves to a source that is
   still shown, an optional stable order for pinned documents, a source hygiene filter and an
   optional caller supplied enricher.
4. A coverage notice when the user's own documents were requested and none came back.
5. Multi-signal retrieval confidence.

Conventions this stage reads and writes (``QueryContext`` has no dedicated fields for them):

- Reads the query complexity from ``ctx.metrics`` or ``ctx.extra``, either as
  ``["query_analysis"]["complexity"]`` or as ``["complexity"]``. Missing means ``"complex"``.
- Writes the assembled context string to ``ctx.context``.
- Writes ``ctx.context_chunks``, ``ctx.sources``, ``ctx.retrieval_warning`` and these
  ``ctx.metrics`` keys: ``context_budget``, ``context_tokens``, ``neighbors``, ``source_counts``,
  ``coverage_notice``, ``multi_signal_confidence``, ``confidence`` and ``pipeline_confidence``.
- The user's own documents count as requested when ``ctx.pinned_ids`` is not empty or
  ``ctx.extra["own_documents_requested"]`` is truthy.

Tenancy: this stage never knows who the caller is. Neighbor expansion fetches chunks by id, so a
chunk fetcher that ignores tenant scoping could return another tenant's chunks. The stage forwards
two things to the fetcher: ``scope``, an opaque string derived from ``ctx.filters`` (or
``ContextStageConfig.scope``), and, when the fetcher's ``fetch_by_ids`` declares a ``filters``
keyword, the filters mapping itself. Your fetcher must enforce one of them or bake its own tenant
filter in.

The stage fails open. Any error leaves a non-empty context built from the chunks it was given.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import ChunkFetcher, TokenCounter, UsageHook
from legal_rag.pipeline.tiers import DEFAULT_TIER, TierConfig, get_tier
from legal_rag.pipeline.types import PINNED_KEY, ContextBudget, QueryContext
from legal_rag.retrieval.accumulators.neighbor import NeighborAccumulator, NeighborConfig
from legal_rag.retrieval.rerank.confidence import ConfidenceScorer, RetrievalConfidence
from legal_rag.retrieval.rerank.coverage import (
    DEFAULT_COVERAGE_CONFIG,
    CoverageConfig,
    apply_coverage_notice,
    source_type_of,
)
from legal_rag.retrieval.rerank.source_dedup import (
    build_chunk_to_post_dedup_remap,
    dedupe_sources,
    stabilize_pinned_source_order,
)
from legal_rag.types import RetrievedChunk, UsageEvent
from legal_rag.verification.grounding.source_hygiene import SourceHygiene

# ruff: noqa: BLE001
log = get_logger("pipeline.context")

OWN_DOCUMENTS_REQUESTED_KEY = "own_documents_requested"
DEFAULT_COMPLEXITY = "complex"
UNSCORED_CONFIDENCE = 0.35
"""Confidence reported when chunks exist but no multi-signal score was computed. It is modest on
purpose: there is material, but no grounding verdict."""

SourceEnricher = Callable[[list[dict[str, Any]]], Awaitable[None]]
"""Mutates the source dicts in place, for example to fill a missing filename."""

LabelFor = Callable[[RetrievedChunk], str]
"""``JurisdictionPlugin.label_for`` has this shape."""

_TRUNCATION_MARK = "..."
_TRUNCATION_SAFETY = 0.95
_OWN_DOCUMENTS_HEADER = (
    "=== USER-SUPPLIED DOCUMENTS (facts and work product, NOT legal authority, "
    "never cite as precedent) ==="
)
_AUTHORITY_HEADER = "=== LEGAL AUTHORITY (precedent and statutes, citable as primary law) ==="


# ---------------------------------------------------------------------------
# Config and small helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextStageConfig:
    """Everything the stage tunes. The defaults reproduce the shipped behavior.

    ``budget`` holds the token and chunk limits. ``tiers`` is a tier table source as accepted by
    ``legal_rag.pipeline.tiers.load_tiers``. ``stable_pinned_order`` puts pinned and scope locked
    documents first in the source list so their numbers do not change between turns.
    ``adaptive_neighbor_boost`` extra neighbors are fetched when the mean chunk score is below
    ``adaptive_score_threshold`` (set the boost to 0 to disable). ``scope`` overrides the scope
    string passed to the chunk fetcher.
    """

    budget: ContextBudget = field(default_factory=ContextBudget)
    tiers: Path | str | Mapping[str, Any] | None = None
    stable_pinned_order: bool = True
    section_aware_neighbors: bool = True
    max_section_neighbors: int = 6
    adaptive_neighbor_boost: int = 2
    adaptive_score_threshold: float = 0.4
    doc_scoped_supporting_source_cap: int = 6
    scope: str | None = None
    confidence_scorer: ConfidenceScorer | None = None
    coverage: CoverageConfig = DEFAULT_COVERAGE_CONFIG
    hygiene: SourceHygiene | None = None


def whitespace_token_counter(text: str) -> int:
    """The default token counter: one token per whitespace separated word."""
    return len(text.split()) if text else 0


def tiktoken_token_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """A counter backed by ``tiktoken``. Raises ``ImportError`` when it is not installed."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    return lambda text: len(encoding.encode(text, disallowed_special=())) if text else 0


def default_label_for(chunk: RetrievedChunk) -> str:
    """Neutral label: the chunk's title, else filename, else citation, plus court and year.

    A jurisdiction plugin supplies its own ``label_for`` when titles need parsing.
    """
    meta = chunk.metadata or {}
    label = str(meta.get("title") or meta.get("filename") or meta.get("citation") or "").strip()
    suffix = ", ".join(
        str(part).strip()
        for part in (meta.get("court"), meta.get("year") or meta.get("case_year"))
        if part
    )
    if not label:
        label = "Unknown"
    return f"{label[:100]} ({suffix})" if suffix else label[:120]


def resolve_pipeline_confidence(multi_signal: float | None, n_chunks: int) -> float:
    """Honest answer confidence: the multi-signal score, else 0.0 with no chunks, else 0.35.

    Query rewrite quality must never stand in for it, because it stays high even when retrieval
    found nothing.
    """
    if multi_signal is not None:
        return float(multi_signal)
    return 0.0 if n_chunks <= 0 else UNSCORED_CONFIDENCE


def filters_scope(filters: Mapping[str, Any] | None) -> str | None:
    """Opaque scope string for a chunk fetcher: the filters as sorted compact JSON, or ``None``."""
    if not filters:
        return None
    try:
        return json.dumps(dict(filters), sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(sorted(map(str, filters)))


def _make_labeler(label_for: LabelFor | None) -> LabelFor:
    def label(chunk: RetrievedChunk) -> str:
        if label_for is not None:
            try:
                text = str(label_for(chunk) or "").strip()
                if text:
                    return text
            except Exception as exc:
                log.warning("label_for_failed", error=str(exc))
        return default_label_for(chunk)

    return label


def _accepts_filters(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "filters" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class _FilteredFetcher:
    """Adds the request filters to every ``fetch_by_ids`` call that can take them."""

    def __init__(self, inner: ChunkFetcher, filters: Mapping[str, Any]) -> None:
        self._inner = inner
        self._filters = dict(filters or {})
        self._pass_filters = _accepts_filters(inner.fetch_by_ids)

    async def fetch_by_ids(
        self, chunk_ids: Sequence[str], *, scope: str | None = None
    ) -> list[RetrievedChunk]:
        if self._pass_filters:
            return await self._inner.fetch_by_ids(  # type: ignore[call-arg]
                chunk_ids, scope=scope, filters=self._filters
            )
        return await self._inner.fetch_by_ids(chunk_ids, scope=scope)


def _complexity_of(ctx: QueryContext) -> str:
    for bag in (ctx.metrics, ctx.extra):
        analysis = bag.get("query_analysis")
        if isinstance(analysis, Mapping) and analysis.get("complexity"):
            return str(analysis["complexity"])
        if bag.get("complexity"):
            return str(bag["complexity"])
    return DEFAULT_COMPLEXITY


def _resolve_tier(name: str, source: Path | str | Mapping[str, Any] | None) -> TierConfig:
    try:
        return get_tier(name, source)
    except ValueError as exc:
        log.warning("unknown_tier_using_default", tier=name, error=str(exc))
        return get_tier(DEFAULT_TIER, source)


# ---------------------------------------------------------------------------
# Budgeted selection
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """Result of ``select_within_budget``. ``kept`` is in input order."""

    kept: list[tuple[int, RetrievedChunk, int, int]] = field(default_factory=list)
    """``(input_index, chunk, body_tokens, overhead_tokens)`` per kept chunk."""
    dropped: list[dict[str, Any]] = field(default_factory=list)
    truncated: int = 0

    @property
    def tokens_used(self) -> int:
        return sum(body for _, _, body, _ in self.kept)


def _truncate_to_tokens(
    chunk: RetrievedChunk, tokens: int, limit: int, counter: TokenCounter
) -> tuple[RetrievedChunk, int]:
    """Cut a chunk to at most ``limit`` tokens. The copy is flagged ``truncated`` in metadata."""
    text = chunk.text
    cut = max(int(len(text) * (limit / max(tokens, 1)) * _TRUNCATION_SAFETY), 1)
    shortened = text[:cut] + _TRUNCATION_MARK
    cost = counter(shortened)
    for _ in range(24):  # any counter must end up under the limit, so shrink until it does
        if cost <= limit or cut <= 1:
            break
        cut = max(int(cut * 0.9), 1)
        shortened = text[:cut] + _TRUNCATION_MARK
        cost = counter(shortened)
    meta = {**(chunk.metadata or {}), "truncated": True}
    return replace(chunk, text=shortened, metadata=meta), cost


def select_within_budget(
    chunks: Sequence[RetrievedChunk],
    budget: ContextBudget,
    counter: TokenCounter,
    *,
    overhead_of: Callable[[RetrievedChunk], int] | None = None,
    scope_ids: Sequence[str] = (),
    own_source_types: frozenset[str] = frozenset(),
    supporting_cap: int = 6,
) -> Selection:
    """Choose which chunks fit the budget. Never returns an empty selection for input chunks.

    The cost of a chunk is its body tokens plus ``overhead_of(chunk)`` (the source header and
    index line the renderer adds). Chunks of pinned or scope locked documents, and chunks flagged
    ``PINNED_KEY`` (for example by an exact lookup), are reserved first, in score order. The rest fill the budget by score. When documents are in scope, the number of
    distinct supporting sources is capped so the scoped documents are not drowned out.
    At most ``budget.max_chunks`` are kept. The best chunk is always kept, truncated if needed.
    """
    available = budget.available_tokens
    scope = {str(s) for s in scope_ids}
    items: list[tuple[RetrievedChunk, int, int]] = []
    truncated = 0
    for chunk in chunks:
        tokens = counter(chunk.text or "")
        if tokens > budget.max_chunk_tokens:
            chunk, tokens = _truncate_to_tokens(chunk, tokens, budget.max_chunk_tokens, counter)
            truncated += 1
        items.append((chunk, tokens, overhead_of(chunk) if overhead_of else 0))

    def scoped(chunk: RetrievedChunk) -> bool:
        meta = chunk.metadata or {}
        if meta.get(PINNED_KEY):
            return True
        if not scope:
            return False
        return (
            str(meta.get("document_id") or "") in scope
            or chunk.id in scope
            or str(meta.get("source_type") or "") in own_source_types
        )

    order = sorted(range(len(items)), key=lambda i: (-items[i][0].score, i))
    kept: set[int] = set()
    used = 0

    def try_add(i: int) -> bool:
        nonlocal used
        _, tokens, overhead = items[i]
        if len(kept) < budget.max_chunks and used + tokens + overhead <= available:
            kept.add(i)
            used += tokens + overhead
            return True
        return False

    for i in order:
        if scoped(items[i][0]):
            try_add(i)
    supporting: set[str] = set()
    for i in order:
        if i in kept:
            continue
        chunk = items[i][0]
        key = ""
        if scope and not scoped(chunk):
            meta = chunk.metadata or {}
            key = str(meta.get("document_id") or meta.get("source_id") or meta.get("title") or i)
            if key not in supporting and len(supporting) >= supporting_cap:
                continue
        if try_add(i) and key:
            supporting.add(key)

    if not kept and order:
        best = order[0]
        chunk, tokens, overhead = items[best]
        room = max(available - overhead, 1)
        if tokens > room:
            chunk, tokens = _truncate_to_tokens(chunk, tokens, room, counter)
            truncated += 1
        items[best] = (chunk, tokens, overhead)
        kept.add(best)

    selection = Selection(truncated=truncated)
    for i, (chunk, tokens, overhead) in enumerate(items):
        if i in kept:
            selection.kept.append((i, chunk, tokens, overhead))
        else:
            selection.dropped.append({"index": i, "tokens": tokens, "score": chunk.score})
    return selection


# ---------------------------------------------------------------------------
# Sources, dedupe and rendering
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    chunk: RetrievedChunk
    tokens: int
    overhead: int
    expanded: str
    expanded_tokens: int
    source: dict[str, Any]


@dataclass
class _View:
    sources: list[dict[str, Any]]
    remap: list[int | None]
    labels: list[str]
    context: str
    hygiene_dropped: int = 0
    hygiene_kept_all: bool = False


def _source_dict(chunk: RetrievedChunk, expanded: str) -> dict[str, Any]:
    """One source per chunk. Free-form values stay in ``metadata``."""
    meta = dict(chunk.metadata or {})
    return {
        "chunk_id": chunk.id,
        "document_id": meta.get("document_id"),
        "filename": meta.get("filename") or meta.get("title"),
        "title": meta.get("title"),
        "citation": meta.get("citation"),
        "court": meta.get("court"),
        "jurisdiction": meta.get("jurisdiction"),
        "year": meta.get("year") or meta.get("case_year"),
        "decision_date": meta.get("decision_date"),
        "section_title": meta.get("section_title"),
        "section_number": meta.get("section_number"),
        "page_number": meta.get("page_number"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "char_start": meta.get("char_start"),
        "char_end": meta.get("char_end"),
        "score": chunk.score,
        "text": chunk.text,
        # Server side only: the neighbor expanded text the model saw, for grounding checks.
        "grounding_text": expanded,
        "url": meta.get("url"),
        "pdf_url": meta.get("pdf_url"),
        "file_url": meta.get("file_url"),
        "source_type": meta.get("source_type"),
        "is_shared": meta.get("is_shared", False),
        "selected": True,
        "metadata": meta,
    }


def _shim_chunk(source: Mapping[str, Any]) -> RetrievedChunk:
    """A chunk view of a source dict, so a chunk labeler can label a deduped source."""
    meta = dict(source.get("metadata") or {})
    for key in ("title", "filename", "citation", "court", "year"):
        if source.get(key) and not meta.get(key):
            meta[key] = source[key]
    return RetrievedChunk(
        id=str(source.get("chunk_id") or ""),
        text=str(source.get("text") or ""),
        score=float(source.get("score") or 0.0),
        metadata=meta,
    )


def _is_own_document(chunk: RetrievedChunk, own_types: frozenset[str]) -> bool:
    meta = chunk.metadata or {}
    return source_type_of({"metadata": meta}) in own_types or meta.get("is_corpus") is False


def _index_table(labels: Sequence[str]) -> str:
    lines = ["--- SOURCE INDEX (use these numbers for [N] citations) ---"]
    lines.extend(f"[{i}] {label}" for i, label in enumerate(labels, 1))
    lines.append("--- END SOURCE INDEX ---")
    return "\n".join(lines)


def _render(
    entries: Sequence[_Entry],
    remap: Sequence[int | None],
    labels: Sequence[str],
    own_types: frozenset[str],
) -> str:
    own: list[str] = []
    authority: list[str] = []
    for entry, index in zip(entries, remap, strict=True):
        if index is None:
            continue  # its source was collapsed away, so it has no citable number
        block = f"[Source {index}: {labels[index - 1]}]\n{entry.expanded}"
        (own if _is_own_document(entry.chunk, own_types) else authority).append(block)
    sections = [_index_table(labels)]
    if own and authority:
        sections += [_OWN_DOCUMENTS_HEADER, "\n\n---\n\n".join(own)]
        sections += [_AUTHORITY_HEADER, "\n\n---\n\n".join(authority)]
    elif own:
        sections += [_OWN_DOCUMENTS_HEADER, "\n\n---\n\n".join(own)]
    else:
        sections.append("\n\n---\n\n".join(authority))
    return "\n\n".join(sections)


def _build_view(
    entries: Sequence[_Entry],
    cfg: ContextStageConfig,
    labeler: LabelFor,
    scope_order: list[str],
    hygiene: SourceHygiene,
) -> _View:
    """Dedupe, order, filter and number the sources, then render the context."""
    pre = [e.source for e in entries]
    sources = dedupe_sources(list(pre))
    if cfg.stable_pinned_order:
        sources = stabilize_pinned_source_order(sources, scope_order)
    kept, dropped = hygiene.filter_sources(sources)
    kept_all = False
    if not kept and sources:
        kept, dropped, kept_all = sources, 0, True  # hygiene never empties the context
    remap = build_chunk_to_post_dedup_remap(pre, kept)
    if all(r is None for r in remap):
        kept, remap = list(pre), list(range(1, len(pre) + 1))
    labels = [labeler(_shim_chunk(s)) for s in kept]
    context = _render(entries, remap, labels, cfg.coverage.own_source_types)
    return _View(kept, remap, labels, context, dropped, kept_all)


# ---------------------------------------------------------------------------
# Neighbors, confidence, usage
# ---------------------------------------------------------------------------


def _neighbor_count(
    tier: TierConfig, complexity: str, chunks: Sequence[RetrievedChunk], cfg: ContextStageConfig
) -> tuple[int, bool]:
    """Effective neighbors per side and whether the complexity cap was binding."""
    count = tier.neighbor_chunks if tier.features.use_neighbor_accumulation else 0
    cap = tier.complexity.neighbor_cap(complexity)
    capped = cap is not None and cap < count
    if cap is not None:
        count = min(count, cap)
    if count > 0 and cfg.adaptive_neighbor_boost > 0 and chunks:
        mean = sum(c.score for c in chunks) / len(chunks)
        if mean < cfg.adaptive_score_threshold:
            count += cfg.adaptive_neighbor_boost
    return count, capped


async def _expand(
    ctx: QueryContext,
    chunks: list[RetrievedChunk],
    count: int,
    capped: bool,
    fetcher: ChunkFetcher,
    counter: TokenCounter,
    cfg: ContextStageConfig,
) -> tuple[dict[str, str], int]:
    """Expand chunks with neighbors. Returns ``({chunk_id: expanded_text}, neighbors_fetched)``.

    Dynamic neighbor counts and section extension are off when a complexity cap applies, so the
    cap is a hard cap.
    """
    accumulator = NeighborAccumulator(
        _FilteredFetcher(fetcher, ctx.filters),
        config=NeighborConfig(
            neighbor_count=count,
            dynamic_neighbors=False,
            section_aware=cfg.section_aware_neighbors and not capped,
            max_section_neighbors=cfg.max_section_neighbors,
        ),
        token_counter=counter,
    )
    scope = cfg.scope if cfg.scope is not None else filters_scope(ctx.filters)
    result = await accumulator.accumulate(ctx.query, chunks, scope=scope)
    return {a.chunk_id: a.full_content for a in result.chunks}, result.neighbor_count


def _confidence(
    scorer: ConfidenceScorer, ctx: QueryContext, chunks: list[RetrievedChunk]
) -> RetrievalConfidence | None:
    dense = [c.metadata.get("dense_score") for c in ctx.retrieved]
    sparse = [c.metadata.get("sparse_score") for c in ctx.retrieved]
    usable = bool(ctx.retrieved) and all(isinstance(v, int | float) for v in dense + sparse)
    try:
        return scorer.score(
            chunks=chunks,
            query=ctx.query,
            dense_scores=dense if usable else None,
            sparse_scores=sparse if usable else None,
        )
    except Exception as exc:
        log.warning("multi_signal_confidence_failed", error=str(exc))
        return None


def _emit_usage(usage: UsageHook | None, context_tokens: int, extra: dict[str, Any]) -> None:
    if usage is None:
        return
    try:
        usage(UsageEvent(operation="context_assembly", input_tokens=context_tokens, extra=extra))
    except Exception as exc:
        log.warning("usage_hook_failed", error=str(exc))


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@dataclass
class _Assembly:
    context: str
    context_chunks: list[RetrievedChunk]
    sources: list[dict[str, Any]]
    metrics: dict[str, Any]
    warnings: list[str]
    retrieval_warning: str | None


def _commit(ctx: QueryContext, result: _Assembly) -> None:
    ctx.context_chunks = result.context_chunks
    ctx.sources = result.sources
    ctx.context = result.context
    ctx.metrics.update(result.metrics)
    ctx.retrieval_warning = result.retrieval_warning
    for message in result.warnings:
        ctx.add_warning(message)


def _warning_for(n_chunks: int) -> str | None:
    if n_chunks == 0:
        return "no_results"
    return "limited_results" if n_chunks <= 2 else None


async def _assemble(
    ctx: QueryContext,
    chunks: list[RetrievedChunk],
    *,
    fetcher: ChunkFetcher | None,
    counter: TokenCounter,
    labeler: LabelFor,
    enricher: SourceEnricher | None,
    cfg: ContextStageConfig,
) -> _Assembly:
    budget = cfg.budget
    available = budget.available_tokens
    own_types = cfg.coverage.own_source_types
    warnings: list[str] = []
    scope_order = list(dict.fromkeys([*map(str, ctx.pinned_ids), *map(str, ctx.scope_lock_ids)]))
    tier = _resolve_tier(ctx.tier, cfg.tiers)

    def overhead_of(chunk: RetrievedChunk) -> int:
        label = labeler(chunk)
        return counter(f"[Source 00: {label}]") + counter(f"[00] {label}")

    selection = select_within_budget(
        chunks,
        budget,
        counter,
        overhead_of=overhead_of,
        scope_ids=scope_order,
        own_source_types=own_types,
        supporting_cap=cfg.doc_scoped_supporting_source_cap,
    )
    selected = [chunk for _, chunk, _, _ in selection.kept]

    complexity = _complexity_of(ctx)
    count, capped = _neighbor_count(tier, complexity, selected, cfg)
    expansion: dict[str, str] = {}
    neighbors_fetched = 0
    if count > 0 and fetcher is not None:
        try:
            expansion, neighbors_fetched = await _expand(
                ctx, selected, count, capped, fetcher, counter, cfg
            )
        except Exception as exc:
            log.warning("neighbor_expansion_failed", error=str(exc))
            warnings.append("neighbor_expansion_failed")

    entries: list[_Entry] = []
    for _, chunk, tokens, overhead in selection.kept:
        expanded = expansion.get(chunk.id) or chunk.text
        entries.append(
            _Entry(
                chunk=chunk,
                tokens=tokens,
                overhead=overhead,
                expanded=expanded,
                expanded_tokens=tokens if expanded == chunk.text else counter(expanded),
                source=_source_dict(chunk, expanded),
            )
        )

    # Neighbors may not push the context past the budget: strip them from the lowest scored
    # chunks first. Every chunk already fits on its own, so this always converges.
    total = sum(e.expanded_tokens + e.overhead for e in entries)
    for entry in sorted(entries, key=lambda e: e.chunk.score):
        if total <= available:
            break
        if entry.expanded != entry.chunk.text:
            total -= entry.expanded_tokens - entry.tokens
            entry.expanded, entry.expanded_tokens = entry.chunk.text, entry.tokens
            entry.source["grounding_text"] = entry.chunk.text

    if enricher is not None:
        try:
            await enricher([e.source for e in entries])
        except Exception as exc:
            log.warning("source_enricher_failed", error=str(exc))
            warnings.append("source_enrichment_failed")

    hygiene = cfg.hygiene or SourceHygiene()
    live = list(entries)
    view = _build_view(live, cfg, labeler, scope_order, hygiene)
    while len(live) > 1 and counter(view.context) > available:
        live.remove(min(live, key=lambda e: e.chunk.score))
        view = _build_view(live, cfg, labeler, scope_order, hygiene)
    if view.hygiene_kept_all:
        warnings.append("source_hygiene_kept_all")

    own_requested = bool(ctx.pinned_ids) or bool(ctx.extra.get(OWN_DOCUMENTS_REQUESTED_KEY))
    context, notice = apply_coverage_notice(
        view.context,
        view.sources,
        own_docs_requested=own_requested,
        query=ctx.query,
        docs_pinned=bool(ctx.pinned_ids),
        config=cfg.coverage,
    )

    context_chunks: list[RetrievedChunk] = []
    for entry, index in zip(live, view.remap, strict=True):
        if index is not None:
            meta = {**(entry.chunk.metadata or {}), "source_index": index}
            context_chunks.append(replace(entry.chunk, metadata=meta))
    for number, source in enumerate(view.sources, 1):
        source["source_index"] = number

    confidence: RetrievalConfidence | None = None
    if tier.features.use_multi_signal_confidence:
        confidence = _confidence(cfg.confidence_scorer or ConfidenceScorer(), ctx, context_chunks)
    multi_signal = confidence.overall_score if confidence else None

    used = sum(e.tokens for e in live)
    metrics: dict[str, Any] = {
        "context_budget": {
            "chunks_input": len(chunks),
            "chunks_used": len(live),
            "chunks_dropped": len(chunks) - len(live),
            "chunks_truncated": selection.truncated,
            "tokens_used": used,
            "tokens_available": available,
            "tokens_max": budget.total_tokens,
            "budget_utilization": round(used / available * 100, 1) if available else 0.0,
        },
        "context_tokens": counter(context),
        "neighbors": {
            "requested": count,
            "fetched": neighbors_fetched,
            "capped": capped,
            "complexity": complexity,
        },
        "source_counts": {
            "chunks": len(live),
            "sources": len(view.sources),
            "dropped_by_hygiene": view.hygiene_dropped,
        },
        "coverage_notice": notice,
        "multi_signal_confidence": multi_signal,
        "confidence": confidence.to_dict() if confidence else None,
        "pipeline_confidence": resolve_pipeline_confidence(multi_signal, len(live)),
    }
    return _Assembly(
        context=context,
        context_chunks=context_chunks,
        sources=view.sources,
        metrics=metrics,
        warnings=warnings,
        retrieval_warning=_warning_for(len(live)),
    )


def _fail_open(ctx: QueryContext, chunks: list[RetrievedChunk], cfg: ContextStageConfig) -> None:
    """Best effort context with no counting, no neighbors and no dedupe. Never raises."""
    try:
        limit_chars = cfg.budget.max_chunk_tokens * 4
        kept = chunks[: cfg.budget.max_chunks]
        blocks = [f"[Source {i}]\n{(c.text or '')[:limit_chars]}" for i, c in enumerate(kept, 1)]
        ctx.context_chunks = list(kept)
        ctx.sources = [
            {"chunk_id": c.id, "text": c.text, "score": c.score, "metadata": dict(c.metadata or {})}
            for c in kept
        ]
        ctx.context = "\n\n---\n\n".join(blocks)
        ctx.retrieval_warning = _warning_for(len(kept))
        ctx.metrics["pipeline_confidence"] = resolve_pipeline_confidence(None, len(kept))
    except Exception as exc:
        log.error("context_fail_open_failed", error=str(exc))
    ctx.add_warning("context_assembly_failed")


async def context_stage(
    ctx: QueryContext,
    *,
    fetcher: ChunkFetcher | None = None,
    token_counter: TokenCounter | None = None,
    label_for: LabelFor | None = None,
    source_enricher: SourceEnricher | None = None,
    usage: UsageHook | None = None,
    config: ContextStageConfig | None = None,
) -> QueryContext:
    """Assemble the context the answering model reads. See the module docstring.

    ``fetcher`` loads neighbor chunks; without one, no neighbors are added. ``token_counter``
    defaults to ``whitespace_token_counter``. ``label_for`` names each source (pass
    ``JurisdictionPlugin.label_for``); the default uses the title or filename. ``source_enricher``
    may fill missing fields on the source dicts in place. ``usage`` receives one
    ``context_assembly`` event with the context size.

    The input is ``ctx.reranked``, or ``ctx.retrieved`` when that is empty. Mutates and returns
    ``ctx``. Never raises: on error the context is built from the chunks as given.
    """
    cfg = config or ContextStageConfig()
    chunks = list(ctx.reranked or ctx.retrieved)
    if not chunks:
        ctx.context_chunks = []
        ctx.sources = []
        ctx.context = ""
        ctx.retrieval_warning = "no_results"
        ctx.metrics["context_budget"] = {"chunks_used": 0}
        ctx.metrics["multi_signal_confidence"] = None
        ctx.metrics["pipeline_confidence"] = resolve_pipeline_confidence(None, 0)
        return ctx
    try:
        result = await _assemble(
            ctx,
            chunks,
            fetcher=fetcher,
            counter=token_counter or whitespace_token_counter,
            labeler=_make_labeler(label_for),
            enricher=source_enricher,
            cfg=cfg,
        )
    except Exception as exc:
        log.error("context_assembly_failed", error=str(exc), error_type=type(exc).__name__)
        _fail_open(ctx, chunks, cfg)
        return ctx
    _commit(ctx, result)
    _emit_usage(
        usage,
        int(result.metrics["context_tokens"]),
        {
            "chunks_used": result.metrics["context_budget"]["chunks_used"],
            "neighbors_fetched": result.metrics["neighbors"]["fetched"],
            "sources": result.metrics["source_counts"]["sources"],
        },
    )
    return ctx


__all__ = [
    "DEFAULT_COMPLEXITY",
    "OWN_DOCUMENTS_REQUESTED_KEY",
    "UNSCORED_CONFIDENCE",
    "ContextStageConfig",
    "LabelFor",
    "Selection",
    "SourceEnricher",
    "context_stage",
    "default_label_for",
    "filters_scope",
    "resolve_pipeline_confidence",
    "select_within_budget",
    "tiktoken_token_counter",
    "whitespace_token_counter",
]
