"""Small adapters between router callables and the provider protocols.

The router takes ``LanguageModel`` and ``Embedder`` providers directly.
These helpers wrap plain async functions, for callers who have a function rather than a client.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from legal_rag.providers.protocols import T
from legal_rag.types import Message

LlmFn = Callable[[str, str], Awaitable[str]]
EmbedFn = Callable[[str], Awaitable[Sequence[float]]]


class FunctionLanguageModel:
    """Wrap a ``(system, user) -> text`` coroutine function as a ``LanguageModel``."""

    def __init__(self, fn: LlmFn) -> None:
        self._fn = fn

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        system = "\n".join(m.content for m in messages if m.role == "system")
        user = "\n".join(m.content for m in messages if m.role != "system")
        return await self._fn(system, user)

    async def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        max_tokens: int | None = None,
    ) -> T:
        raw = await self.generate(messages, max_tokens=max_tokens)
        return schema.model_validate_json(raw)


class FunctionEmbedder:
    """Wrap a single-text embedding coroutine function as an ``Embedder``."""

    def __init__(self, fn: EmbedFn, dimension: int) -> None:
        self._fn = fn
        self.dimension = dimension

    async def embed_query(self, text: str) -> list[float]:
        return list(await self._fn(text))

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(await self._fn(t)) for t in texts]
