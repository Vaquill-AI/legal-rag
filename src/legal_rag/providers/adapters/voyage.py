"""Thin Voyage AI adapters for the ``Embedder`` and ``Reranker`` protocols.

The ``voyageai`` package is imported lazily, so importing this module costs nothing and the
package is only needed when no client is passed. Pass ``client`` (an async Voyage client, or any
object with the same ``embed`` and ``rerank`` coroutines) to supply your own configured client or a
test double. There are no environment reads, circuit breakers, retries or caches here: wrap the
adapters with your own if you need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legal_rag.providers.errors import RerankRateLimited
from legal_rag.types import RerankResult

DEFAULT_EMBED_MODEL = "voyage-3-large"
DEFAULT_EMBED_DIMENSION = 1024
DEFAULT_RERANK_MODEL = "rerank-2.5"


def _make_client(api_key: str | None, client: Any | None) -> Any:
    if client is not None:
        return client
    if not api_key:
        raise ValueError("Pass api_key or client")
    try:
        import voyageai
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("Install the 'voyageai' package to use the Voyage adapters") from error
    return voyageai.AsyncClient(api_key=api_key)


class VoyageEmbedder:
    """Voyage dense embeddings behind the ``Embedder`` protocol.

    ``dimension`` is only reported to callers that size an index from it. It is not sent to the
    API, so it must match what ``model`` returns.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        model: str = DEFAULT_EMBED_MODEL,
        dimension: int = DEFAULT_EMBED_DIMENSION,
        batch_size: int = 128,
    ) -> None:
        self._client = _make_client(api_key, client)
        self.model = model
        self.dimension = dimension
        self.batch_size = max(1, batch_size)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents in batches, preserving order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            result = await self._client.embed(batch, model=self.model, input_type="document")
            vectors.extend([list(v) for v in result.embeddings])
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query with the query input type."""
        result = await self._client.embed([text], model=self.model, input_type="query")
        return list(result.embeddings[0])


def _retry_after_of(error: Exception) -> float | None:
    headers = getattr(error, "headers", None) or getattr(
        getattr(error, "response", None), "headers", None
    )
    value = headers.get("retry-after") if headers else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_rate_limit(error: Exception) -> bool:
    status = getattr(error, "http_status", None) or getattr(error, "status_code", None)
    return status == 429 or type(error).__name__ == "RateLimitError"


class VoyageReranker:
    """Voyage cross-encoder reranking behind the ``Reranker`` protocol.

    A 429 from the provider is re-raised as ``RerankRateLimited`` so that
    ``legal_rag.retrieval.rerank.CrossEncoderReranker`` can pause and fall back.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        model: str = DEFAULT_RERANK_MODEL,
        truncation: bool = True,
    ) -> None:
        self._client = _make_client(api_key, client)
        self.model = model
        self.truncation = truncation

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Score ``documents`` against ``query``, best first."""
        if not documents:
            return []
        try:
            result = await self._client.rerank(
                query,
                list(documents),
                model=self.model,
                top_k=top_k,
                truncation=self.truncation,
            )
        except Exception as error:
            if _is_rate_limit(error):
                raise RerankRateLimited(_retry_after_of(error), str(error)[:200]) from error
            raise
        return [RerankResult(index=r.index, score=float(r.relevance_score)) for r in result.results]
