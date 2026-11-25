"""Protocols the pipeline depends on, and a default retriever built from two of them.

The provider protocols (``LanguageModel``, ``Embedder``, ``Reranker``, ``VectorStore``,
``Cache``, ``UsageHook``) live in ``legal_rag.providers`` and are re-exported here with
``ChunkFetcher`` and ``TokenCounter`` from ``legal_rag.retrieval.accumulators``, so a pipeline
author imports every contract from one place.

Tenancy is not a pipeline concept. Implement ``Retriever`` and ``ChunkFetcher`` so they apply
your tenant filter themselves, and treat ``filters`` as opaque.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from legal_rag.pipeline.types import PipelineRequest, SparseVector, VectorPoint
from legal_rag.providers.protocols import (
    Cache,
    Embedder,
    LanguageModel,
    Reranker,
    UsageHook,
    VectorStore,
)
from legal_rag.retrieval.accumulators.base import ChunkFetcher
from legal_rag.retrieval.accumulators.neighbor import TokenCounter
from legal_rag.retrieval.fusion import reciprocal_rank_fusion
from legal_rag.types import RetrievedChunk

TierResolver = Callable[[PipelineRequest], str]
"""Maps a request to a tier name, for example from a plan lookup you own."""

EventSink = Callable[[Any], Awaitable[None]]
"""Receives progress events (for example multi-hop steps) as they happen."""

ChunkAdjuster = Callable[[str, Sequence[RetrievedChunk], Mapping[str, Any]], list[RetrievedChunk]]
"""``(query, chunks, filters) -> chunks``. Re-scores, boosts or drops chunks after retrieval."""


@runtime_checkable
class Retriever(Protocol):
    """Hybrid or dense search that returns ranked chunks, best first.

    ``sub_queries`` are focused follow-ups from query decomposition. An implementation may
    search each and merge the lists, or ignore them.
    """

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """A sparse (keyword weighted) encoder such as BM25 term hashing."""

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


@runtime_checkable
class VectorWriter(Protocol):
    """The write side of a vector index. ``VectorStore`` is read only."""

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def delete(self, ids: Sequence[str]) -> None: ...


@runtime_checkable
class HybridVectorStore(Protocol):
    """A ``VectorStore`` that can also score a sparse vector in the same call."""

    async def search_hybrid(
        self,
        vector: Sequence[float],
        sparse: SparseVector,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning text for prompt-injection content."""

    safe: bool
    findings: tuple[str, ...] = ()
    sanitized_text: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class InjectionScanner(Protocol):
    """Flags text that tries to instruct the model instead of informing it."""

    def scan(self, text: str) -> ScanResult: ...


@runtime_checkable
class ExactLookup(Protocol):
    """Pins chunks for a named citation or section the query points at directly."""

    async def lookup(self, query: str) -> list[RetrievedChunk]: ...


@runtime_checkable
class WebSearch(Protocol):
    """An optional web search that supplements the corpus when it falls short."""

    async def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class JurisdictionPlugin(Protocol):
    """Opt-in retrieval behavior for one body of law.

    ``matches`` decides from the request filters whether the plugin applies.
    ``condenser_key`` selects the jurisdiction profile the query condenser should use.
    """

    condenser_key: str

    def matches(self, filters: Mapping[str, Any]) -> bool: ...

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...

    def adjusters(self) -> Sequence[ChunkAdjuster]: ...

    def exact_lookups(self) -> Sequence[ExactLookup]: ...

    def label_for(self, chunk: RetrievedChunk) -> str: ...


class VectorStoreRetriever:
    """Default ``Retriever``: embed the query, then search a ``VectorStore``.

    ``filters`` reach the store unchanged, so a tenant filter baked into the store or passed in
    ``filters`` is always applied.
    When ``sparse`` is given and the store implements ``HybridVectorStore``, the query is also
    encoded sparsely and scored in one hybrid call. A store without ``search_hybrid`` is searched
    with the dense vector only.
    When ``sub_queries`` are given, each is searched alongside the main query and the lists are
    merged with reciprocal rank fusion.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        sparse: SparseEncoder | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._sparse = sparse

    async def _search_one(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievedChunk]:
        vector = await self._embedder.embed_query(query)
        if self._sparse is not None and isinstance(self._store, HybridVectorStore):
            return await self._store.search_hybrid(
                vector, self._sparse.encode_query(query), top_k=top_k, filters=filters
            )
        return await self._store.search(vector, top_k=top_k, filters=filters)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        queries = [query]
        for sub in sub_queries or ():
            if sub and sub.strip() and sub not in queries:
                queries.append(sub)
        if len(queries) == 1:
            return await self._search_one(query, top_k, filters)
        ranked = await asyncio.gather(*(self._search_one(q, top_k, filters) for q in queries))
        return reciprocal_rank_fusion(ranked, top_k=top_k)


__all__ = [
    "Cache",
    "ChunkAdjuster",
    "ChunkFetcher",
    "Embedder",
    "EventSink",
    "ExactLookup",
    "HybridVectorStore",
    "InjectionScanner",
    "JurisdictionPlugin",
    "LanguageModel",
    "Reranker",
    "Retriever",
    "ScanResult",
    "SparseEncoder",
    "TierResolver",
    "TokenCounter",
    "UsageHook",
    "VectorStore",
    "VectorStoreRetriever",
    "VectorWriter",
    "WebSearch",
]
