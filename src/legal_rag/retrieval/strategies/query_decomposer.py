"""Query decomposition.

Detects when a research query spans several statutory remedies or legal tracks and splits it
into one sub-query per track, for parallel retrieval. The original query is always kept first,
sub-queries are additive, and at most five are produced.

The check is model-based rather than rule-based: "wrongful termination" is several tracks while
"statute of limitations for a Title VII charge" is one, and a pattern match cannot tell them apart.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, model_validator

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, safe_error_fields
from legal_rag.retrieval.strategies.query_decomposer_prompts import (
    DECOMPOSITION_PROMPT,
    DEFAULT_EXAMPLES,
)
from legal_rag.types import Message

logger = logging.getLogger("legal_rag.retrieval.query_decomposer")


class DecompositionAnalysis(BaseModel):
    """Structured output of the decomposition check."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data: Any) -> Any:
        # Models occasionally emit null for a field. Coerce to the safe "no decomposition" default.
        if isinstance(data, dict):
            data = dict(data)
            if data.get("requires_decomposition") is None:
                data["requires_decomposition"] = False
            for key in ("tracks", "sub_queries"):
                if data.get(key) is None:
                    data[key] = []
            if data.get("reasoning") is None:
                data["reasoning"] = ""
        return data

    requires_decomposition: bool = Field(
        default=False,
        description="True if the query spans several distinct provisions, remedies or tracks.",
    )
    tracks: list[str] = Field(default_factory=list, max_length=5)
    sub_queries: list[str] = Field(default_factory=list, max_length=5)
    reasoning: str = ""


@dataclass
class DecompositionResult:
    """Outcome of decomposition. ``all_queries`` is the original followed by the sub-queries."""

    original_query: str
    requires_decomposition: bool
    tracks: list[str]
    sub_queries: list[str]
    all_queries: list[str]
    reasoning: str
    decomposition_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryDecomposer:
    """Split a multi-track legal query into per-track sub-queries."""

    def __init__(
        self,
        llm: LanguageModel,
        *,
        max_tokens: int = 4096,
        min_words: int = 5,
        examples: str = DEFAULT_EXAMPLES,
        prompt_template: str = DECOMPOSITION_PROMPT,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.min_words = min_words
        self.examples = examples
        self.prompt_template = prompt_template
        self.usage_hook = usage_hook

    def _single(self, query: str, reasoning: str, ms: float, *, success: bool, **meta: Any):
        return DecompositionResult(
            original_query=query,
            requires_decomposition=False,
            tracks=[],
            sub_queries=[],
            all_queries=[query],
            reasoning=reasoning,
            decomposition_time_ms=ms,
            success=success,
            metadata=dict(meta),
        )

    async def _analyse(self, prompt: str) -> DecompositionAnalysis | None:
        """Ask once, then once more with a larger token budget. ``None`` if both fail."""
        for budget in (self.max_tokens, self.max_tokens * 2):
            try:
                return await self.llm.generate_structured(
                    [Message("user", prompt)], DecompositionAnalysis, max_tokens=budget
                )
            except Exception as error:  # noqa: BLE001 - truncated or unparseable output
                logger.warning("decomposition attempt failed %s", safe_error_fields(error))
        return None

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose ``query`` if it spans several tracks. Never raises."""
        start = time.monotonic()
        if len(query.split()) < self.min_words:
            return self._single(
                query, "Query too short for multi-track decomposition", 0.0, success=True
            )
        try:
            prompt = self.prompt_template.replace("{query}", query).replace(
                "{examples}", self.examples
            )
            result = await self._analyse(prompt)
            elapsed = (time.monotonic() - start) * 1000
            if result is None:
                return self._single(
                    query,
                    "Decomposition unavailable (structured output failed)",
                    elapsed,
                    success=False,
                )
            emit_usage(
                self.usage_hook,
                "query_decomposition",
                input_text=prompt,
                output_text=" ".join(result.sub_queries),
            )
            if result.requires_decomposition and result.sub_queries:
                seen = {query.strip().lower()}
                unique: list[str] = []
                for sub in result.sub_queries:
                    key = sub.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        unique.append(sub.strip())
                if unique:
                    return DecompositionResult(
                        original_query=query,
                        requires_decomposition=True,
                        tracks=result.tracks,
                        sub_queries=unique,
                        all_queries=[query, *unique],
                        reasoning=result.reasoning,
                        decomposition_time_ms=elapsed,
                        success=True,
                    )
            return self._single(query, result.reasoning, elapsed, success=True)
        except Exception as error:  # noqa: BLE001 - degrade to the original query
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("query decomposition failed %s", safe_error_fields(error))
            return self._single(
                query, f"Decomposition failed: {error!s}", elapsed, success=False, error=str(error)
            )
