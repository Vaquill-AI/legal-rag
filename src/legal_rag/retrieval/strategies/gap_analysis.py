"""Information-gap analysis: is what was retrieved enough, and if not, what to search next.

This is the "reason" step of retrieve-reason-retrieve loops such as multi-hop retrieval.
The default analyzer asks an injected language model and falls back to a simple rule
when the model call fails. Any async callable with the ``GapAnalyzer`` shape can replace it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, safe_error_fields
from legal_rag.types import Message, RetrievedChunk

logger = logging.getLogger("legal_rag.retrieval.gap_analysis")

_STOPWORDS = frozenset(
    [
        "what",
        "the",
        "of",
        "in",
        "and",
        "a",
        "to",
        "for",
        "with",
        "under",
        "about",
        "between",
        "is",
        "are",
        "did",
        "does",
        "how",
        "why",
        "when",
        "where",
    ]
)

SYSTEM_PROMPT = (
    "You are an information sufficiency analyzer for multi-hop retrieval. "
    "Analyze whether the retrieved information is sufficient to answer the query."
)


class InformationGapAnalysis(BaseModel):
    """Structured output of the model-based analyzer."""

    is_sufficient: bool = False
    missing_aspects: list[str] = Field(default_factory=list)
    suggested_followup: str = ""
    confidence: float = 0.0


@dataclass
class GapAnalysisResult:
    """Sufficiency verdict with a suggested follow-up query."""

    has_sufficient_info: bool
    missing_aspects: list[str] = field(default_factory=list)
    suggested_followup: str = ""
    confidence: float = 0.0


GapAnalyzer = Callable[[str, Sequence[RetrievedChunk]], Awaitable[GapAnalysisResult]]


def core_terms(query: str) -> str:
    """The query without stopwords and very short words, or the query itself if nothing is left."""
    kept = [w for w in query.split() if w.lower() not in _STOPWORDS and len(w) > 2]
    return " ".join(kept) if kept else query


class LLMGapAnalyzer:
    """Model-backed gap analyzer."""

    def __init__(
        self,
        llm: LanguageModel,
        *,
        max_chunks: int = 10,
        chunk_chars: int = 500,
        max_tokens: int = 1024,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.max_chunks = max_chunks
        self.chunk_chars = chunk_chars
        self.max_tokens = max_tokens
        self.usage_hook = usage_hook

    async def __call__(
        self, original_query: str, retrieved_chunks: Sequence[RetrievedChunk]
    ) -> GapAnalysisResult:
        if not retrieved_chunks:
            return GapAnalysisResult(
                has_sufficient_info=False,
                missing_aspects=["No information retrieved yet"],
                suggested_followup=core_terms(original_query),
                confidence=0.0,
            )
        snippets = [
            f"[Chunk {i + 1}] {chunk.text[: self.chunk_chars]}"
            for i, chunk in enumerate(retrieved_chunks[: self.max_chunks])
        ]
        user = (
            f'**Original Query**: "{original_query}"\n\n'
            f"**Information Retrieved So Far** (from {len(retrieved_chunks)} chunks):\n"
            + "\n\n".join(snippets)
            + "\n\nDetermine:\n1. Is this information SUFFICIENT to fully answer the query?\n"
            "2. What specific aspects are still MISSING?\n"
            "3. What follow-up query would retrieve the missing information?"
        )
        try:
            result = await self.llm.generate_structured(
                [Message("system", SYSTEM_PROMPT), Message("user", user)],
                InformationGapAnalysis,
                max_tokens=self.max_tokens,
            )
            emit_usage(
                self.usage_hook,
                "gap_analysis",
                input_text=SYSTEM_PROMPT + user,
                output_text=result.suggested_followup,
            )
            return GapAnalysisResult(
                has_sufficient_info=result.is_sufficient,
                missing_aspects=result.missing_aspects,
                suggested_followup=result.suggested_followup or original_query,
                confidence=result.confidence,
            )
        except Exception as error:  # noqa: BLE001 - rule-based fallback
            logger.warning("gap analysis failed, using rule %s", safe_error_fields(error))
            enough = len(retrieved_chunks) >= 3
            return GapAnalysisResult(
                has_sufficient_info=enough,
                missing_aspects=[] if enough else ["Insufficient chunks"],
                suggested_followup=core_terms(original_query),
                confidence=0.7 if enough else 0.3,
            )
