"""Provider protocols.

Every component receives its providers as arguments.
Nothing in legal-rag reads settings, environment variables or secrets.
Implement these protocols with your own clients, or use the fakes in ``legal_rag.providers.fakes`` for tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from legal_rag.types import Message, RerankResult, RetrievedChunk, UsageEvent

T = TypeVar("T", bound=BaseModel)

UsageHook = Callable[[UsageEvent], None]
"""Optional callback for metering. Components call it when it is supplied."""


@runtime_checkable
class LanguageModel(Protocol):
    """A chat model."""

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    async def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T: ...


@runtime_checkable
class Embedder(Protocol):
    """A text embedder."""

    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """A cross-encoder or score-based reranker.

    ``top_k`` of ``None`` or ``0`` means "use the default", which for the bundled rerankers and
    fakes is every document. Ask for an empty result by not calling the reranker.
    """

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]: ...


@runtime_checkable
class VectorStore(Protocol):
    """A vector store. ``filters`` is an opaque mapping the store understands."""

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Cache(Protocol):
    """A key-value cache with optional expiry."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, key: str) -> None:
        """Remove ``key``. Deleting a key that is not present is not an error."""
        ...
