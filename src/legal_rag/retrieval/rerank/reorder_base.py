"""
Base class for context reordering strategies.

All reorderers implement the BaseReorderer interface to allow
easy swapping of reordering strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from legal_rag.types import RetrievedChunk


@dataclass
class ReorderingResult:
    """Result of reordering operation."""

    reordered_chunks: list[RetrievedChunk]
    strategy: str
    original_order: list[int]  # Original indices for tracking
    reordering_applied: bool


class BaseReorderer(ABC):
    """
    Base class for chunk reordering strategies.

    Reorderers take a list of retrieved chunks and reorder them
    to optimize LLM performance based on research findings.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize reorderer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def reorder(self, chunks: list[RetrievedChunk]) -> ReorderingResult:
        """
        Reorder chunks using the specific strategy.

        Args:
            chunks: Retrieved chunks with relevance scores

        Returns:
            ReorderingResult with reordered chunks and metadata
        """

    def should_reorder(self, chunks: list[RetrievedChunk]) -> bool:
        """
        Check if reordering is beneficial.

        Override in subclasses for strategy-specific logic.

        Args:
            chunks: Retrieved chunks

        Returns:
            True if reordering should be applied
        """
        # Generally reorder if we have 3+ chunks
        return len(chunks) >= 3
