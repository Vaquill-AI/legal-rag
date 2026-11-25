"""Errors that provider adapters raise, so callers can catch them without importing retrieval."""

from __future__ import annotations


class RerankRateLimited(Exception):
    """The provider answered 429. ``retry_after`` is in seconds when the provider said so."""

    def __init__(self, retry_after: float | None = None, detail: str = "") -> None:
        super().__init__(detail or "Rerank rate limited")
        self.retry_after = retry_after
