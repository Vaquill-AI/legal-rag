"""Reranker interface over retrieved chunks.

``legal_rag.providers.protocols.Reranker`` scores plain strings and returns index/score pairs.
``BaseReranker`` here works one level up: it takes chunks and returns a ``RerankOutcome`` with
chunks in their new order, so strategies and orderings can be composed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from legal_rag.types import RetrievedChunk


@dataclass
class RerankOutcome:
    """Chunks after reranking."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    original_count: int = 0
    final_count: int = 0
    processing_time_ms: float = 0.0
    degraded: bool = False
    """True when the intended scorer did not run and a fallback ordering stands in for it.

    The scores are then retrieval-scale rather than 0-1 relevance, so a caller that thresholds
    on relevance, or publishes the score as one, must check this flag.
    """
    degraded_reason: str | None = None


class BaseReranker(ABC):
    """Abstract chunk reranker."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        **kwargs: object,
    ) -> RerankOutcome:
        """Return ``chunks`` reordered for ``query``, truncated to ``top_k``."""
