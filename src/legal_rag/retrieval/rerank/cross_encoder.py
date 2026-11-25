"""Cross-encoder reranking over chunks, with rate limiting and a visible fallback.

``CrossEncoderReranker`` wraps any ``legal_rag.providers.protocols.Reranker`` (for example the
Voyage adapter). If the provider fails, is rate limited, or runs out of time, the chunks are
reordered by a fallback reranker instead and the outcome is flagged ``degraded`` so callers
never mistake a fallback ordering for cross-encoder relevance.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace

from legal_rag.providers.protocols import Reranker, UsageHook
from legal_rag.retrieval.rerank._log import get_logger
from legal_rag.retrieval.rerank.base import BaseReranker, RerankOutcome
from legal_rag.retrieval.rerank.rate_limit import (
    RateLimitWaitExceeded,
    RerankDeadlineExceeded,
    RerankRateLimited,
    RerankRateLimiter,
    estimate_tokens,
)
from legal_rag.retrieval.rerank.score_based import ScoreBasedReranker
from legal_rag.types import RetrievedChunk, UsageEvent

logger = get_logger("retrieval.rerank.cross_encoder")


class CrossEncoderReranker(BaseReranker):
    """Rerank chunks with an injected cross-encoder provider."""

    def __init__(
        self,
        reranker: Reranker,
        top_k: int = 5,
        *,
        fallback: BaseReranker | None = None,
        rate_limiter: RerankRateLimiter | None = None,
        usage_hook: UsageHook | None = None,
        default_wait_seconds: float = 10.0,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(top_k=top_k)
        self.reranker = reranker
        self.fallback = fallback or ScoreBasedReranker(top_k=top_k)
        self.rate_limiter = rate_limiter
        self.usage_hook = usage_hook
        self.default_wait_seconds = default_wait_seconds
        self.provider_name = provider_name

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        **kwargs: object,
    ) -> RerankOutcome:
        """Rerank with the provider. ``deadline`` (monotonic seconds) bounds waits and calls."""
        start = time.monotonic()
        k = top_k or self.top_k
        if not chunks:
            return RerankOutcome(query=query)
        deadline = kwargs.get("deadline")
        if not isinstance(deadline, (int, float)):
            deadline = start + self.default_wait_seconds
        documents = [c.text for c in chunks]
        try:
            results = await self._call(query, documents, k, float(deadline))
        except (RerankRateLimited, RateLimitWaitExceeded) as error:
            reason = "deadline" if isinstance(error, RerankDeadlineExceeded) else "rate_limited"
            logger.warning("rerank_rate_limited", error_type=type(error).__name__)
            return await self._degrade(query, chunks, k, start, reason)
        except Exception as error:  # noqa: BLE001 - any provider failure degrades, never raises
            logger.warning("rerank_failed", error_type=type(error).__name__)
            return await self._degrade(query, chunks, k, start, f"error:{type(error).__name__}")

        ordered = [
            replace(chunks[r.index], score=r.score) for r in results if 0 <= r.index < len(chunks)
        ]
        ordered.sort(key=lambda c: c.score, reverse=True)
        ordered = ordered[:k]
        if self.usage_hook is not None:
            with contextlib.suppress(Exception):  # metering must never break reranking
                self.usage_hook(
                    UsageEvent(
                        operation="reranking",
                        provider=self.provider_name,
                        input_tokens=estimate_tokens(query, documents),
                        output_tokens=0,
                        extra={"chunks_reranked": len(chunks)},
                    )
                )
        return RerankOutcome(
            query=query,
            chunks=ordered,
            original_count=len(chunks),
            final_count=len(ordered),
            processing_time_ms=(time.monotonic() - start) * 1000,
        )

    async def _call(self, query: str, documents: list[str], k: int, deadline: float):
        limiter = self.rate_limiter
        estimate = estimate_tokens(query, documents)
        while True:
            reserved = await limiter.acquire(estimate, deadline) if limiter else 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if limiter:
                    await limiter.refund(reserved)
                raise RerankDeadlineExceeded("no time left for the call")
            try:
                results = await asyncio.wait_for(
                    self.reranker.rerank(query, documents, top_k=k), timeout=remaining
                )
            except RerankRateLimited as error:
                if limiter is None:
                    raise
                pause = await limiter.pause_after_429(error.retry_after)
                if time.monotonic() + pause >= deadline:
                    raise
                continue
            except TimeoutError as error:
                if limiter:
                    await limiter.refund(reserved)
                raise RerankDeadlineExceeded("rerank call exceeded the caller budget") from error
            except BaseException:
                if limiter:
                    await limiter.refund(reserved)
                raise
            if limiter:
                await limiter.settle(reserved, reserved)
            return results

    async def _degrade(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        k: int,
        start: float,
        reason: str,
    ) -> RerankOutcome:
        outcome = await self.fallback.rerank(query, chunks, top_k=k)
        outcome.processing_time_ms = (time.monotonic() - start) * 1000
        outcome.degraded = True
        outcome.degraded_reason = reason
        return outcome
