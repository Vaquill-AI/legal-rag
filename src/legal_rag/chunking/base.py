"""
Abstract base class for text chunkers.
Provides interface for splitting documents into semantic chunks.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from legal_rag.chunking._logging import get_logger

logger = get_logger("base")


@dataclass
class Chunk:
    """
    Represents a text chunk for RAG.

    This dataclass holds chunk-level information during processing.
    For storage, convert the chunk to whatever record shape your vector store expects.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""

    # Position info
    chunk_index: int = 0
    total_chunks: int = 0  # Total chunks in document
    start_char: int = 0
    end_char: int = 0

    # Document context
    document_id: str = ""
    section_title: str | None = None
    section_number: str | None = None
    section_type: str = "unknown"  # ratio/obiter/facts/etc
    page_number: int | None = None
    page_start: int | None = None  # For multi-page chunks
    page_end: int | None = None

    # Token info
    token_count: int = 0

    # Language info (for multilingual support)
    language: str = "en"  # en/hi/ta/te/etc
    is_code_mixed: bool = False  # Hindi-English mixed

    # Extracted citations in this chunk
    citations: list[dict[str, Any]] = field(default_factory=list)

    # Metadata for legal documents (extensible)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Neighbor references (for accumulator)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


ChunkT = TypeVar("ChunkT", bound="Chunk")


@dataclass
class ChunkerResult:
    """Result from chunking operation."""

    document_id: str
    chunks: Sequence[Chunk] = field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0

    # Stats
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.

    Implementations should handle:
    - Semantic boundary detection
    - Section-aware chunking (for legal docs)
    - Overlap handling
    - Token counting
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100, min_chunk_size: int = 50):
        """
        Initialize chunker with configuration.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        logger.debug(
            "chunker_initialized",
            strategy=self.__class__.__name__,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    @abstractmethod
    async def chunk(
        self, text: str, document_id: str, sections: list[Any] | None = None, **kwargs
    ) -> ChunkerResult:
        """
        Split text into semantic chunks.

        Args:
            text: Full document text
            document_id: Document identifier
            sections: Optional section information for boundary awareness
            **kwargs: Provider-specific options

        Returns:
            ChunkerResult with list of chunks
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """

    def link_chunks(self, chunks: list[ChunkT]) -> list[ChunkT]:
        """
        Link chunks with neighbor references.

        Args:
            chunks: List of chunks to link

        Returns:
            Chunks with prev/next references set
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].id

        return chunks
