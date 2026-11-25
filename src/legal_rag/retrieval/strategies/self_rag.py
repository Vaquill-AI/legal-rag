"""Self-RAG: adaptive retrieval with self-critique.

1. Decide whether the query needs retrieval at all.
2. Retrieve if so, then generate an answer.
3. Critique the answer and regenerate while it is judged fixable.

Reference: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (2023).
The retrieve and generate steps are caller-supplied async callables, so the controller
holds no retriever and no generator of its own.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, fingerprint, safe_error_fields
from legal_rag.retrieval.strategies.self_rag_prompts import (
    ANSWER_CRITIQUE_PROMPT,
    DEFAULT_RETRIEVE_KEYWORDS,
    DEFAULT_SKIP_KEYWORDS,
    RETRIEVAL_ROUTER_PROMPT,
)
from legal_rag.types import Message

logger = logging.getLogger("legal_rag.retrieval.self_rag")

RetrieveContext = Callable[[str], Awaitable[str]]
GenerateAnswer = Callable[..., Awaitable[str]]


class RetrievalAssessment(BaseModel):
    """Structured output of the retrieval router."""

    decision: str = "uncertain"
    requested_count: int | None = None
    reasoning: str = ""


class AnswerCritique(BaseModel):
    """Structured output of the answer critic."""

    verdict: str = "GOOD"
    reasoning: str = ""


class RetrievalDecision(StrEnum):
    """Whether to retrieve."""

    RETRIEVE = "retrieve"
    SKIP = "skip"
    UNCERTAIN = "uncertain"


class CritiqueResult(StrEnum):
    """Outcome of the answer critique."""

    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    NEEDS_MORE_CONTEXT = "needs_more_context"


@dataclass
class SelfRAGConfig:
    """Configuration for the controller."""

    always_retrieve: bool = False
    enable_critique: bool = True
    max_regeneration_attempts: int = 2
    prefer_retrieval: bool = True
    """When the router is uncertain, retrieve (True) or skip (False)."""
    max_tokens: int = 4096
    critique_context_chars: int = 1000
    router_prompt: str = RETRIEVAL_ROUTER_PROMPT
    critique_prompt: str = ANSWER_CRITIQUE_PROMPT
    retrieve_keywords: tuple[str, ...] = DEFAULT_RETRIEVE_KEYWORDS
    skip_keywords: tuple[str, ...] = DEFAULT_SKIP_KEYWORDS
    """Keyword lists used only when the model-based router fails."""


@dataclass
class SelfRAGResult:
    """What the controller decided and did."""

    query: str
    query_id: str
    retrieval_decision: RetrievalDecision
    retrieval_performed: bool
    num_regenerations: int
    final_critique: CritiqueResult | None
    total_time_ms: float
    success: bool
    requested_count: int | None = None
    answer: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SelfRAGController:
    """Adaptive retrieval and generation with self-reflection."""

    def __init__(
        self,
        llm: LanguageModel,
        config: SelfRAGConfig | None = None,
        *,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SelfRAGConfig()
        self.usage_hook = usage_hook
        self.stats = {"total_queries": 0, "retrieval_performed": 0, "regenerations": 0}

    async def process(
        self,
        query: str,
        retrieve_func: RetrieveContext | None = None,
        generate_func: GenerateAnswer | None = None,
        context: str | None = None,
    ) -> SelfRAGResult:
        """Run the Self-RAG loop. Never raises: failure returns ``success=False``."""
        start = time.monotonic()
        query_id = fingerprint(query)
        regenerations = 0
        retrieved = context
        answer: str | None = None
        requested_count: int | None = None
        try:
            decision, requested_count = await self._assess(query)
            if self.config.always_retrieve:
                decision = RetrievalDecision.RETRIEVE
            if decision is RetrievalDecision.UNCERTAIN:
                decision = (
                    RetrievalDecision.RETRIEVE
                    if self.config.prefer_retrieval
                    else RetrievalDecision.SKIP
                )
            performed = decision is RetrievalDecision.RETRIEVE
            if performed and retrieve_func and not retrieved:
                retrieved = await retrieve_func(query)
            if generate_func:
                answer = await generate_func(query, retrieved)

            critique = CritiqueResult.GOOD
            if self.config.enable_critique and answer:
                critique = await self._critique(query, answer, retrieved or "")
                while (
                    critique is CritiqueResult.NEEDS_IMPROVEMENT
                    and regenerations < self.config.max_regeneration_attempts
                    and generate_func
                ):
                    regenerations += 1
                    self.stats["regenerations"] += 1
                    answer = await generate_func(query, retrieved, previous_answer=answer)
                    critique = await self._critique(query, answer, retrieved or "")

            self.stats["total_queries"] += 1
            self.stats["retrieval_performed"] += int(performed)
            return SelfRAGResult(
                query=query,
                query_id=query_id,
                retrieval_decision=decision,
                retrieval_performed=performed,
                num_regenerations=regenerations,
                final_critique=critique,
                total_time_ms=(time.monotonic() - start) * 1000,
                success=True,
                requested_count=requested_count,
                answer=answer,
                context=retrieved,
                metadata={
                    "always_retrieve": self.config.always_retrieve,
                    "has_context": retrieved is not None,
                    "has_answer": answer is not None,
                },
            )
        except Exception as error:  # noqa: BLE001 - report failure instead of raising
            logger.warning("self-rag failed query_id=%s %s", query_id, safe_error_fields(error))
            return SelfRAGResult(
                query=query,
                query_id=query_id,
                retrieval_decision=RetrievalDecision.UNCERTAIN,
                retrieval_performed=False,
                num_regenerations=regenerations,
                final_critique=None,
                total_time_ms=(time.monotonic() - start) * 1000,
                success=False,
                requested_count=requested_count,
                metadata={"error": str(error)[:160]},
            )

    async def _assess(self, query: str) -> tuple[RetrievalDecision, int | None]:
        """Model-based routing, falling back to keywords when the model call fails."""
        try:
            messages = [
                Message("system", self.config.router_prompt),
                Message("user", f"Query: {query}"),
            ]
            result = await self.llm.generate_structured(
                messages, RetrievalAssessment, max_tokens=self.config.max_tokens
            )
            emit_usage(
                self.usage_hook,
                "self_rag_assessment",
                input_text=self.config.router_prompt + query,
                output_text=result.decision,
            )
            return RetrievalDecision(result.decision.lower()), result.requested_count or None
        except Exception as error:  # noqa: BLE001 - keyword fallback
            logger.warning("self-rag assessment fallback %s", safe_error_fields(error))
            return self._assess_keywords(query), None

    def _assess_keywords(self, query: str) -> RetrievalDecision:
        text = query.lower()
        skip = sum(1 for k in self.config.skip_keywords if k in text)
        retrieve = sum(1 for k in self.config.retrieve_keywords if k in text)
        if retrieve > skip:
            return RetrievalDecision.RETRIEVE
        if skip > retrieve:
            return RetrievalDecision.SKIP
        return RetrievalDecision.UNCERTAIN

    async def _critique(self, query: str, answer: str, context: str) -> CritiqueResult:
        """Model-based critique. A failing critic assumes the answer is good."""
        try:
            shown = context[: self.config.critique_context_chars] or "No context provided"
            user = f"Query: {query}\n\nContext provided:\n{shown}\n\nAnswer to critique:\n{answer}"
            result = await self.llm.generate_structured(
                [Message("system", self.config.critique_prompt), Message("user", user)],
                AnswerCritique,
                max_tokens=self.config.max_tokens,
            )
            emit_usage(
                self.usage_hook,
                "self_rag_critique",
                input_text=self.config.critique_prompt + user,
                output_text=result.verdict,
            )
            verdicts = {
                "GOOD": CritiqueResult.GOOD,
                "NEEDS_IMPROVEMENT": CritiqueResult.NEEDS_IMPROVEMENT,
                "NEEDS_MORE_CONTEXT": CritiqueResult.NEEDS_MORE_CONTEXT,
            }
            return verdicts.get(result.verdict.upper(), CritiqueResult.GOOD)
        except Exception as error:  # noqa: BLE001 - assume acceptable
            logger.warning("self-rag critique fallback %s", safe_error_fields(error))
            return CritiqueResult.GOOD
