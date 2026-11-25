"""Retriever interface and result dataclasses.

Nothing here knows about tenants, users or deployments.
Anything a store needs to narrow a search travels in the generic ``filters`` mapping,
which the concrete retriever interprets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from legal_rag.types import RetrievedChunk

RetrieveFunc = Callable[[str], Awaitable[list[RetrievedChunk]]]
"""A bare async retrieval callable: query in, ranked chunks out."""


@dataclass
class RetrievalResult:
    """Result of one retrieval call."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    total_found: int = 0
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    processing_time_ms: float = 0.0
    filters: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkGroup:
    """Chunks that share a document, kept together for source diversity."""

    group_id: str
    group_field: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    avg_score: float = 0.0

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the group."""
        return len(self.chunks)


@dataclass
class GroupedRetrievalResult:
    """Retrieval result organised by document."""

    query: str
    groups: list[ChunkGroup] = field(default_factory=list)
    total_groups: int = 0
    total_chunks: int = 0
    group_by_field: str = "document_id"
    group_size: int = 3
    processing_time_ms: float = 0.0
    filters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """Every chunk from every group, flattened."""
        return [chunk for group in self.groups for chunk in group.chunks]


class BaseRetriever(ABC):
    """Abstract retriever.

    Implementations may run dense search, sparse search or a hybrid of both.
    Scoping (tenant, corpus, jurisdiction and so on) is expressed in ``filters``.
    """

    def __init__(
        self,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        score_threshold: float = 0.0,
    ) -> None:
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.score_threshold = score_threshold

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Return ranked chunks for ``query``."""

    @abstractmethod
    async def retrieve_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return specific chunks by id, for example to fetch neighbours."""
