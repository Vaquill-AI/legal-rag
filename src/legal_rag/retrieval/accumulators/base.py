"""
Abstract base class for context accumulators.
Provides interface for expanding chunks with neighbor context.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import RetrievedChunk

logger = get_logger("accumulators.base")


@dataclass
class AccumulatedChunk:
    """A chunk with accumulated neighbor context."""

    # Original chunk info
    chunk_id: str
    document_id: str
    original_content: str

    # Accumulated content (with neighbors)
    full_content: str

    # Context info
    prev_chunks: list[str] = field(default_factory=list)  # Previous chunk contents
    next_chunks: list[str] = field(default_factory=list)  # Next chunk contents

    # Scores (preserved from retrieval)
    score: float = 0.0

    # Section context
    section_title: str | None = None
    section_number: str | None = None

    # For citation
    page_range: str | None = None  # e.g., "12-14"

    # Page positions for PDF navigation (1-based page numbers)
    page_start: int | None = None  # First page of chunk
    page_end: int | None = None  # Last page of chunk (same as page_start for single-page)
    page_number: int | None = None  # Original single page number (backwards compat)

    # Character positions for PDF highlighting
    char_start: int | None = None  # Character offset start in document
    char_end: int | None = None  # Character offset end in document

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccumulatedContext:
    """Result from context accumulation."""

    query: str
    chunks: list[AccumulatedChunk] = field(default_factory=list)

    # Combined context for LLM
    combined_context: str = ""

    # Stats
    total_chunks: int = 0
    total_tokens: int = 0
    neighbor_count: int = 0
    processing_time_ms: float = 0.0


class BaseAccumulator(ABC):
    """
    Abstract base class for context accumulators.

    Implementations should handle:
    - Neighbor chunk retrieval
    - Context merging
    - Deduplication
    - Legal citation formatting
    """

    def __init__(self, neighbor_count: int = 2, dynamic_neighbors: bool = True):
        """
        Initialize accumulator.

        Args:
            neighbor_count: Default number of neighbors (before + after)
            dynamic_neighbors: Adjust based on query complexity
        """
        self.neighbor_count = neighbor_count
        self.dynamic_neighbors = dynamic_neighbors

    @abstractmethod
    async def accumulate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        scope: str | None = None,
        **kwargs: Any,
    ) -> AccumulatedContext:
        """
        Accumulate context from chunks and their neighbors.

        Args:
            query: Original search query
            chunks: Retrieved chunks to expand
            scope: Opaque isolation scope, forwarded to the chunk fetcher
            **kwargs: Provider-specific options

        Returns:
            AccumulatedContext with expanded chunks
        """

    def determine_neighbor_count(self, query: str) -> int:
        """
        Determine optimal neighbor count based on query.

        For legal queries that ask for "full context" or "complete section",
        return more neighbors.

        Args:
            query: Search query

        Returns:
            Number of neighbors to fetch
        """
        if not self.dynamic_neighbors:
            return self.neighbor_count

        query_lower = query.lower()

        # Full context indicators
        full_context_keywords = [
            "full",
            "complete",
            "entire",
            "whole",
            "all of",
            "section",
            "article",
            "clause",
            "provision",
            "context",
            "surrounding",
            "before and after",
        ]

        for keyword in full_context_keywords:
            if keyword in query_lower:
                return max(self.neighbor_count, 4)  # More context

        # Specific reference - less context needed
        specific_keywords = ["quote", "exact", "verbatim", "specific"]
        for keyword in specific_keywords:
            if keyword in query_lower:
                return 1  # Minimal context

        return self.neighbor_count


class ChunkFetcher(Protocol):
    """Fetches chunks by id. ``scope`` is an opaque isolation key the implementation enforces."""

    async def fetch_by_ids(
        self, chunk_ids: Sequence[str], *, scope: str | None = None
    ) -> list[RetrievedChunk]: ...
