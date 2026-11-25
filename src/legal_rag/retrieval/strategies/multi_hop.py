"""Multi-hop retrieval: retrieve, reason about gaps, retrieve again.

1. Retrieve for the query.
2. A gap analyzer looks at the chunks gathered so far (not at a generated answer) and either
   declares them sufficient or proposes a follow-up query.
3. Retrieve for the follow-up and repeat until sufficient or ``max_hops`` is reached.

Early stopping is aggressive: with ``sufficient_chunks_threshold`` small relative to the chunks
fetched per hop, many queries finish in one hop even though more are allowed.
Retrieval is an injected async callable, so any retriever can sit underneath.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from legal_rag.retrieval.strategies._common import fingerprint, safe_error_fields
from legal_rag.retrieval.strategies.gap_analysis import GapAnalyzer, core_terms
from legal_rag.types import RetrievedChunk

logger = logging.getLogger("legal_rag.retrieval.multi_hop")

HopRetrieveFunc = Callable[[str, int], Awaitable[list[RetrievedChunk]]]


@dataclass
class HopEvent:
    """Progress event emitted during multi-hop retrieval, for callers that surface progress."""

    event_type: str
    hop_number: int
    query: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0


HopEventCallback = Callable[[HopEvent], Awaitable[None]]


@dataclass
class HopMessages:
    """Progress wording. Override any field to localise or restyle."""

    started: str = "Starting multi-step retrieval"
    searching: str = "Searching (round {hop})"
    found: str = "Found {count} passages"
    sufficient: str = "Retrieved information is sufficient"
    gaps: str = "Identified {count} gap(s) needing further retrieval"
    complete: str = "Retrieval complete: {total} passages over {hops} round(s)"


class HopAction(StrEnum):
    """What a hop did."""

    RETRIEVE = "retrieve"
    REASON = "reason"
    COMPLETE = "complete"


@dataclass
class Hop:
    """One hop."""

    hop_number: int
    action: HopAction
    query: str
    num_chunks_retrieved: int
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHopConfig:
    """Configuration for multi-hop retrieval."""

    max_hops: int = 3
    min_chunks_per_hop: int = 8
    """The retrieval call asks for twice this many chunks."""
    enable_early_stopping: bool = True
    sufficient_chunks_threshold: int = 5
    use_gap_analysis: bool = True
    """Use the gap analyzer when one is supplied; otherwise follow-ups are rule-based."""
    sufficiency_confidence: float = 0.7
    hop_timeout_seconds: float = 30.0
    messages: HopMessages = field(default_factory=HopMessages)


@dataclass
class MultiHopResult:
    """Outcome of multi-hop retrieval."""

    original_query: str
    query_id: str
    hops: list[Hop]
    chunks: list[RetrievedChunk]
    total_chunks_retrieved: int
    completed_early: bool
    total_hops: int
    retrieval_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiHopRetriever:
    """Iterative retrieval over an injected retrieval callable."""

    def __init__(
        self,
        retrieve_func: HopRetrieveFunc | None = None,
        config: MultiHopConfig | None = None,
        *,
        gap_analyzer: GapAnalyzer | None = None,
    ) -> None:
        self.retrieve_func = retrieve_func
        self.config = config or MultiHopConfig()
        self.gap_analyzer = gap_analyzer
        self.stats: dict[str, Any] = {
            "total_retrievals": 0,
            "early_stops": 0,
            "max_hops_reached": 0,
        }

    async def _emit(
        self,
        callback: HopEventCallback | None,
        event_type: str,
        hop: int,
        query: str,
        message: str,
        progress: float,
        **details: Any,
    ) -> None:
        if callback:
            await callback(HopEvent(event_type, hop, query, message, details, progress))

    async def retrieve(
        self,
        query: str,
        retrieve_func: HopRetrieveFunc | None = None,
        context: dict[str, Any] | None = None,
        on_hop_event: HopEventCallback | None = None,
    ) -> MultiHopResult:
        """Run the loop. A failure returns ``success=False`` with the chunks gathered so far."""
        start = time.monotonic()
        retriever = retrieve_func or self.retrieve_func
        if retriever is None:
            raise ValueError("No retrieve_func provided")
        cfg = self.config
        msgs = cfg.messages
        query_id = fingerprint(query)
        hops: list[Hop] = []
        gathered: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        current = query
        completed_early = False
        max_hops = max(1, cfg.max_hops)
        await self._emit(
            on_hop_event, "multi_hop_started", 0, query, msgs.started, 0.0, max_hops=max_hops
        )
        try:
            for hop_num in range(1, max_hops + 1):
                qhash = hashlib.sha256(current.strip().lower().encode()).hexdigest()
                if qhash in seen_queries:
                    completed_early = True
                    break
                seen_queries.add(qhash)
                progress = hop_num / (max_hops + 1)
                await self._emit(
                    on_hop_event,
                    "multi_hop_iteration",
                    hop_num,
                    current,
                    msgs.searching.format(hop=hop_num),
                    progress,
                    phase="retrieving",
                )
                try:
                    chunks = await asyncio.wait_for(
                        retriever(current, cfg.min_chunks_per_hop * 2),
                        timeout=cfg.hop_timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "multi-hop retrieval timed out query_id=%s hop=%d", query_id, hop_num
                    )
                    completed_early = True
                    break
                new_chunks = [c for c in chunks if c.id not in seen_ids]
                seen_ids.update(c.id for c in new_chunks)
                gathered.extend(new_chunks)
                hops.append(
                    Hop(
                        hop_num,
                        HopAction.RETRIEVE,
                        current,
                        len(new_chunks),
                        metadata={"query_len": len(current)},
                    )
                )
                await self._emit(
                    on_hop_event,
                    "multi_hop_iteration",
                    hop_num,
                    current,
                    msgs.found.format(count=len(new_chunks)),
                    progress,
                    phase="retrieved",
                    chunks_retrieved=len(new_chunks),
                    total_chunks=len(gathered),
                )
                if cfg.enable_early_stopping:
                    if len(gathered) >= cfg.sufficient_chunks_threshold:
                        completed_early = True
                        break
                    if not new_chunks and hop_num > 1:
                        completed_early = True
                        break
                if hop_num >= max_hops:
                    break
                if cfg.use_gap_analysis and self.gap_analyzer is not None and gathered:
                    try:
                        gap = await asyncio.wait_for(
                            self.gap_analyzer(query, gathered), timeout=cfg.hop_timeout_seconds
                        )
                    except TimeoutError:
                        completed_early = True
                        break
                    sufficient = (
                        gap.has_sufficient_info and gap.confidence >= cfg.sufficiency_confidence
                    )
                    message = (
                        msgs.sufficient
                        if gap.has_sufficient_info
                        else msgs.gaps.format(count=len(gap.missing_aspects))
                    )
                    await self._emit(
                        on_hop_event,
                        "multi_hop_gap_analysis",
                        hop_num,
                        current,
                        message,
                        (hop_num + 0.5) / (max_hops + 1),
                        has_sufficient_info=gap.has_sufficient_info,
                        confidence=gap.confidence,
                        missing_aspects=gap.missing_aspects,
                        suggested_followup=gap.suggested_followup,
                    )
                    if sufficient:
                        completed_early = True
                        break
                    current = gap.suggested_followup or query
                else:
                    current = self._followup(query, hop_num, len(new_chunks))
            elapsed = (time.monotonic() - start) * 1000
            self.stats["total_retrievals"] += 1
            self.stats["early_stops" if completed_early else "max_hops_reached"] += 1
            await self._emit(
                on_hop_event,
                "multi_hop_complete",
                len(hops),
                query,
                msgs.complete.format(total=len(gathered), hops=len(hops)),
                1.0,
                total_hops=len(hops),
                total_chunks=len(gathered),
                completed_early=completed_early,
            )
            return MultiHopResult(
                original_query=query,
                query_id=query_id,
                hops=hops,
                chunks=gathered,
                total_chunks_retrieved=len(gathered),
                completed_early=completed_early,
                total_hops=len(hops),
                retrieval_time_ms=elapsed,
                success=True,
                metadata={"max_hops_configured": max_hops, "context_provided": context is not None},
            )
        except Exception as error:  # noqa: BLE001 - return what was gathered
            logger.warning("multi-hop failed query_id=%s %s", query_id, safe_error_fields(error))
            return MultiHopResult(
                original_query=query,
                query_id=query_id,
                hops=hops,
                chunks=gathered,
                total_chunks_retrieved=len(gathered),
                completed_early=False,
                total_hops=len(hops),
                retrieval_time_ms=(time.monotonic() - start) * 1000,
                success=False,
                metadata={"error": str(error)[:160]},
            )

    def _followup(self, original: str, hop_num: int, previous_count: int) -> str:
        """Rule-based follow-up, used when no gap analyzer is available."""
        if previous_count == 0:
            if hop_num == 1:
                return self._simplify(original)
            return core_terms(original)
        if hop_num == 1:
            return f"{original} - detailed explanation"
        if hop_num == 2:
            return f"{original} - relevant examples and cases"
        return original

    @staticmethod
    def _simplify(query: str) -> str:
        """Drop common attribution phrases that narrow a query without adding search terms."""
        simplified = query
        for phrase in (
            " established by",
            " mandated by",
            " required by",
            " specified in",
            " according to",
        ):
            simplified = simplified.replace(phrase, "")
        return " ".join(simplified.split())
