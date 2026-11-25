"""Rate limiting for a rerank provider: an async token bucket plus a shared back-off pause.

Providers often limit rerank by tokens per minute, and a request costs roughly
``query_tokens x n_documents + sum(document_tokens)``, so the size of the candidate pool,
not the request count, is what runs into the limit.

Two layers, both in-process and both optional:

* A token bucket sized by ``tokens_per_minute``. A burst queues briefly for capacity instead
  of firing every call at once and collecting 429s. Each call reserves an estimate (characters
  divided by ``chars_per_token``) and settles to the provider's reported count afterwards.
  Waiters are served first come, first served.
* A shared pause: a 429 anywhere pauses every caller of the same limiter until ``Retry-After``
  (or a jittered exponential back-off when the header is absent).

The limiter is not cross-process. The provider's own 429 is the authoritative cross-process
signal, so size ``tokens_per_minute`` per process (roughly account limit divided by processes).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

# Defined in ``legal_rag.providers.errors`` so provider adapters can raise it without importing
# retrieval. Re-exported here so the original import path keeps working.
from legal_rag.providers.errors import RerankRateLimited  # noqa: F401

DEFAULT_CHARS_PER_TOKEN = 3.6
"""A slightly low value over-reserves, which only makes a burst queue a little earlier."""

DEFAULT_MAX_REQUEST_TOKENS = 600_000
"""A single reservation is clamped to this so an oversized call cannot wait on a smaller bucket."""


class RateLimitWaitExceeded(Exception):
    """Capacity would not free up before the caller's deadline."""


class RerankDeadlineExceeded(RateLimitWaitExceeded):
    """The call was cut off by the caller's remaining budget, not by a slow provider."""


def estimate_tokens(
    query: str, documents: Sequence[str], chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> int:
    """Rerank accounting on characters: ``query x docs + sum(docs)``."""
    q = len(query or "") / chars_per_token + 1
    docs = sum(len(d or "") for d in documents) / chars_per_token
    return int(q * len(documents) + docs) + 1


def parse_retry_after(value: str | None) -> float | None:
    """``Retry-After`` in delta-seconds form as a float, or ``None`` when absent or malformed."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


@dataclass
class _State:
    tokens: float
    updated: float
    paused_until: float = 0.0
    consecutive_429: int = 0


class RerankRateLimiter:
    """Async token bucket over tokens per minute, with a shared pause after a 429."""

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        max_request_tokens: int = DEFAULT_MAX_REQUEST_TOKENS,
    ) -> None:
        self.rate = max(1.0, tokens_per_minute / 60.0)
        self.max_request_tokens = max_request_tokens
        self.capacity = max(float(tokens_per_minute) / 4.0, float(max_request_tokens))
        self._state = _State(tokens=self.capacity, updated=time.monotonic())
        self._lock = asyncio.Lock()
        self._turn = asyncio.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._state.updated)
        self._state.tokens = min(self.capacity, self._state.tokens + elapsed * self.rate)
        self._state.updated = now

    async def acquire(self, tokens: int, deadline: float) -> int:
        """Reserve ``tokens`` (clamped), waiting at most until ``deadline`` (monotonic seconds).

        Returns the reserved amount, to pass back to ``settle`` or ``refund``.
        Raises ``RateLimitWaitExceeded`` when capacity cannot free up in time.
        """
        want = float(min(max(1, tokens), self.max_request_tokens))
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                async with self._turn:
                    while True:
                        async with self._lock:
                            now = time.monotonic()
                            self._refill(now)
                            wait = max(0.0, self._state.paused_until - now)
                            if wait == 0.0 and self._state.tokens >= want:
                                self._state.tokens -= want
                                return int(want)
                            if wait == 0.0:
                                wait = (want - self._state.tokens) / self.rate
                        if now + wait > deadline:
                            raise RateLimitWaitExceeded(
                                f"needs {wait:.1f}s, {deadline - now:.1f}s left"
                            )
                        await asyncio.sleep(wait + random.uniform(0.0, 0.02))
        except TimeoutError:
            raise RateLimitWaitExceeded("deadline passed while queued") from None

    async def settle(self, reserved: int, actual: int | None) -> None:
        """Correct a reservation to the token count the provider reported."""
        if actual is None:
            return
        async with self._lock:
            self._refill(time.monotonic())
            self._state.tokens = min(self.capacity, self._state.tokens + reserved - actual)
            self._state.consecutive_429 = 0

    async def refund(self, reserved: int) -> None:
        """Return a reservation that was never spent."""
        async with self._lock:
            self._refill(time.monotonic())
            self._state.tokens = min(self.capacity, self._state.tokens + reserved)

    async def pause_after_429(self, retry_after: float | None) -> float:
        """Pause every caller; returns the pause length in seconds. The bucket is drained."""
        async with self._lock:
            self._state.consecutive_429 += 1
            if retry_after is None:
                base = min(16.0, 2.0 ** (self._state.consecutive_429 - 1))
                retry_after = base * random.uniform(0.8, 1.2)
            now = time.monotonic()
            self._state.paused_until = max(self._state.paused_until, now + retry_after)
            self._state.tokens = 0.0
            self._state.updated = now
            return retry_after
