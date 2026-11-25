"""In-memory fakes of each provider protocol, for tests and examples."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from legal_rag.providers.protocols import T
from legal_rag.types import Message, RerankResult, RetrievedChunk


class FakeLanguageModel:
    """Returns canned text. ``structured`` maps a schema name to the instance to return."""

    def __init__(self, reply: str = "", structured: Mapping[str, BaseModel] | None = None) -> None:
        self.reply = reply
        self.structured = dict(structured or {})
        self.calls: list[Sequence[Message]] = []

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.reply

    async def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T:
        self.calls.append(messages)
        found = self.structured.get(schema.__name__)
        if found is None:
            raise KeyError(f"No canned structured reply for {schema.__name__}")
        return found  # type: ignore[return-value]


class FakeEmbedder:
    """Deterministic hash-based embedder with a fixed small dimension.

    Vectors are centred on zero, so unrelated texts score near zero cosine similarity
    and only an identical text scores 1.0. It has no notion of meaning.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimension)]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeReranker:
    """Scores documents by how many query words they contain."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        words = set(query.lower().split())
        scored = [
            RerankResult(index=i, score=float(len(words & set(d.lower().split()))))
            for i, d in enumerate(documents)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k] if top_k else scored


class FakeVectorStore:
    """Returns a fixed list of chunks, ignoring the vector and filters."""

    def __init__(self, chunks: Sequence[RetrievedChunk] = ()) -> None:
        self.chunks = list(chunks)

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeCache:
    """A dictionary-backed cache. Expiry is ignored."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)
