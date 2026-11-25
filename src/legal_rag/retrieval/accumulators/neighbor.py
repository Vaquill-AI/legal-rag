"""Neighbor accumulator: expand retrieved chunks with the chunks around them.

Clauses and sections refer to their neighbors, so a retrieved chunk is more useful with the
text before and after it. Chunks carry their chain in ``metadata["prev_chunk_id"]`` and
``metadata["next_chunk_id"]``. Neighbor chunks are loaded through an injected ``ChunkFetcher``
and may be re-scored by an injected callable so they sit on the same score scale as the
selected chunks.
"""
# ruff: noqa: BLE001, SIM102

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from legal_rag.retrieval.accumulators.base import (
    AccumulatedChunk,
    AccumulatedContext,
    BaseAccumulator,
    ChunkFetcher,
)
from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import RetrievedChunk

logger = get_logger("accumulators.neighbor")

NeighborRescorer = Callable[[str, list[RetrievedChunk]], Awaitable[list[float]]]
TokenCounter = Callable[[str], int]


def _approx_tokens(text: str) -> int:
    return len(text) // 4 if text else 0


def _fit_to_token_budget(
    chunks: list, chunk_tokens: list[int], budget_tokens: int
) -> tuple[list, int]:
    """Trim accumulated chunks so their combined token cost fits budget_tokens.

    Keeps the highest-scored chunks (by ``.score``) whose cost fits, preserving
    the original list order among survivors so a prior Lost-in-the-Middle reorder
    is not disturbed. An over-budget chunk is skipped (not a hard stop) so a
    smaller high-value chunk can still fit. Never returns an empty list for a
    non-empty input -- the single highest-scored chunk is always kept.

    Returns ``(kept_chunks, dropped_count)``.
    """
    if not chunks or sum(chunk_tokens) <= budget_tokens:
        return chunks, 0
    by_score = sorted(range(len(chunks)), key=lambda i: chunks[i].score, reverse=True)
    keep: set[int] = set()
    running = 0
    for i in by_score:
        if running + chunk_tokens[i] <= budget_tokens:
            keep.add(i)
            running += chunk_tokens[i]
    if not keep:  # never drop everything
        keep.add(by_score[0])
    kept = [c for i, c in enumerate(chunks) if i in keep]
    return kept, len(chunks) - len(kept)


@dataclass
class NeighborConfig:
    """Tuning for neighbor expansion."""

    neighbor_count: int = 2
    dynamic_neighbors: bool = True
    section_aware: bool = True
    max_section_neighbors: int = 6
    rescore_threshold: float = 0.0
    budget_tokens: int | None = None


def _meta(chunk: RetrievedChunk, key: str) -> Any:
    return chunk.metadata.get(key)


class NeighborAccumulator(BaseAccumulator):
    """Expands chunks with their neighbors, in reading order, within a token budget."""

    def __init__(
        self,
        fetcher: ChunkFetcher,
        *,
        config: NeighborConfig | None = None,
        rescorer: NeighborRescorer | None = None,
        token_counter: TokenCounter | None = None,
    ):
        cfg = config or NeighborConfig()
        super().__init__(neighbor_count=cfg.neighbor_count, dynamic_neighbors=cfg.dynamic_neighbors)
        self.config = cfg
        self.fetcher = fetcher
        self.rescorer = rescorer
        self.count_tokens = token_counter or _approx_tokens

    async def accumulate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        scope: str | None = None,
        **kwargs: Any,
    ) -> AccumulatedContext:
        started = time.perf_counter()
        if not chunks:
            return AccumulatedContext(query=query)

        count = self.determine_neighbor_count(query)
        known = {c.id: c for c in chunks}
        wanted: set[str] = set()
        for c in chunks:
            section = _meta(c, "section_title")
            self._collect_ids(
                _meta(c, "prev_chunk_id"), count, wanted, known, "prev_chunk_id", section
            )
            self._collect_ids(
                _meta(c, "next_chunk_id"), count, wanted, known, "next_chunk_id", section
            )
        wanted -= set(known)

        fetched: list[RetrievedChunk] = []
        if wanted:
            try:
                fetched = await self.fetcher.fetch_by_ids(sorted(wanted), scope=scope)
            except Exception as exc:
                logger.warning("neighbor_fetch_failed", error=str(exc))
                fetched = []
            if fetched and self.rescorer is not None:
                fetched = await self._rescore(query, fetched)

        pool: dict[str, RetrievedChunk] = {n.id: n for n in fetched}
        pool.update(known)

        accumulated: list[AccumulatedChunk] = []
        seen: set[str] = set()
        for c in chunks:
            if c.id in seen:
                continue
            seen.add(c.id)
            prev = self._walk(c, pool, count, "prev_chunk_id")
            nxt = self._walk(c, pool, count, "next_chunk_id")
            prev_texts = [p.text for p in reversed(prev)]
            next_texts = [n.text for n in nxt]
            full = "\n\n".join([*prev_texts, c.text, *next_texts])
            start, end = _meta(c, "page_start"), _meta(c, "page_end")
            page_range = None
            if start is not None:
                page_range = str(start) if end in (None, start) else f"{start}-{end}"
            accumulated.append(
                AccumulatedChunk(
                    chunk_id=c.id,
                    document_id=str(_meta(c, "document_id") or ""),
                    original_content=c.text,
                    full_content=full,
                    prev_chunks=prev_texts,
                    next_chunks=next_texts,
                    score=c.score,
                    section_title=_meta(c, "section_title"),
                    section_number=_meta(c, "section_number"),
                    page_range=page_range,
                    page_start=start,
                    page_end=end,
                    page_number=_meta(c, "page_number"),
                    char_start=_meta(c, "char_start"),
                    char_end=_meta(c, "char_end"),
                    metadata=dict(c.metadata),
                )
            )

        if self.config.budget_tokens is not None:
            tokens = [self.count_tokens(a.full_content or a.original_content) for a in accumulated]
            accumulated, dropped = _fit_to_token_budget(
                accumulated, tokens, self.config.budget_tokens
            )
            if dropped:
                logger.info("accumulation_budget_enforced", dropped=dropped, kept=len(accumulated))

        combined = self._build_combined_context(accumulated)
        return AccumulatedContext(
            query=query,
            chunks=accumulated,
            combined_context=combined,
            total_chunks=len(accumulated),
            total_tokens=self.count_tokens(combined),
            neighbor_count=len(fetched),
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )

    async def _rescore(self, query: str, neighbors: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Re-score neighbors and drop those below the threshold. Degrades to the input."""
        assert self.rescorer is not None
        try:
            scores = await self.rescorer(query, neighbors)
        except Exception as exc:
            logger.warning("neighbor_rescore_failed", error=str(exc))
            return neighbors
        if len(scores) != len(neighbors):
            return neighbors
        kept = [
            replace(n, score=s)
            for n, s in zip(neighbors, scores)
            if s >= self.config.rescore_threshold
        ]
        return kept

    def _limit(self, count: int, section: Any) -> int:
        if self.config.section_aware and section:
            return max(count, self.config.max_section_neighbors)
        return count

    def _collect_ids(
        self,
        start_id: str | None,
        count: int,
        target: set[str],
        known: dict[str, RetrievedChunk],
        pointer: str,
        anchor_section: str | None,
    ) -> None:
        """Collect neighbor ids, extending past ``count`` while the section is unchanged."""
        current = start_id
        for i in range(self._limit(count, anchor_section)):
            if not current:
                break
            target.add(current)
            chunk = known.get(current)
            if chunk is None:
                break
            if (i + 1) >= count and self.config.section_aware and anchor_section:
                if _meta(chunk, "section_title") != anchor_section:
                    break
            current = _meta(chunk, pointer)

    def _walk(
        self, anchor: RetrievedChunk, pool: dict[str, RetrievedChunk], count: int, pointer: str
    ) -> list[RetrievedChunk]:
        """Neighbors of ``anchor`` in one direction, nearest first."""
        section = _meta(anchor, "section_title")
        out: list[RetrievedChunk] = []
        current = _meta(anchor, pointer)
        for i in range(self._limit(count, section)):
            chunk = pool.get(current) if current else None
            if chunk is None:
                break
            if i >= count and _meta(chunk, "section_title") != section:
                break
            out.append(chunk)
            current = _meta(chunk, pointer)
        return out

    @staticmethod
    def _label(chunk: AccumulatedChunk) -> str:
        meta = chunk.metadata
        for key in ("source", "title", "citation"):
            if meta.get(key):
                return str(meta[key])
        return chunk.section_title or chunk.document_id or chunk.chunk_id

    def _build_combined_context(self, chunks: list[AccumulatedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[Source {i}] {self._label(c)}"
            if c.page_range:
                header += f" (p. {c.page_range})"
            parts.append(f"{header}\n{c.full_content}")
        return "\n\n---\n\n".join(parts)
