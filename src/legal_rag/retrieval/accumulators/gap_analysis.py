"""
Shared Gap Analysis Utility.

Extracted from multi_hop/retriever.py to be reused by:
- Multi-hop retriever (existing)
- Self-RAG re-retrieval (Phase 1A)
- CRAG re-retrieval before web fallback (Phase 1B)
- Composed strategy (Phase 3A)

Implements the "reason" step in "retrieve-reason-retrieve" pattern.
"""
# ruff: noqa: BLE001

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import Message, UsageEvent

logger = get_logger("accumulators.gap_analysis")


class InformationGapAnalysis(BaseModel):
    """Structured output of the sufficiency check."""

    is_sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    suggested_followup: str | None = None
    confidence: float = 0.5


@dataclass
class GapAnalysisResult:
    """Result of information gap analysis."""

    has_sufficient_info: bool
    missing_aspects: list[str]
    suggested_followup: str
    confidence: float


async def analyze_information_gaps(
    original_query: str,
    retrieved_chunks: list[Any],
    llm: LanguageModel,
    *,
    max_chars_per_chunk: int = 1500,
    max_chunks: int = 8,
    max_tokens: int = 512,
    usage_hook: UsageHook | None = None,
) -> GapAnalysisResult:
    """
    Use LLM to analyze what information is still missing from retrieved chunks.

    Args:
        original_query: The user's original query.
        retrieved_chunks: Chunks retrieved so far (any type with .text, .content or str).
        llm: Injected language model.
        max_chars_per_chunk: Characters of each chunk shown to the model.
        max_chunks: Number of chunks shown to the model.
        max_tokens: Output token budget.
        usage_hook: Optional metering callback.

    Returns:
        GapAnalysisResult with sufficiency assessment and suggested follow-up.
    """
    if not retrieved_chunks:
        return GapAnalysisResult(
            has_sufficient_info=False,
            missing_aspects=["No information retrieved yet"],
            suggested_followup=_simplify_query(original_query),
            confidence=0.0,
        )

    # Build context from retrieved chunks
    context_snippets = []
    max_chars = max_chars_per_chunk
    for i, chunk in enumerate(retrieved_chunks[:max_chunks]):
        content = getattr(chunk, "text", None) or getattr(chunk, "content", None) or str(chunk)
        context_snippets.append(f"[Chunk {i + 1}] {content[:max_chars]}")

    context = "\n\n".join(context_snippets)

    system_prompt = """You are an information sufficiency analyzer for multi-hop RAG retrieval.
Analyze if the retrieved information is sufficient to answer the query."""

    user_prompt = f"""**Original Query**: "{original_query}"

**Information Retrieved So Far** (from {len(retrieved_chunks)} chunks):
{context}

Determine:
1. Is this information SUFFICIENT to fully answer the query?
2. What specific aspects are still MISSING?
3. What follow-up query would retrieve the missing information?"""

    try:
        result = await llm.generate_structured(
            [Message("system", system_prompt), Message("user", user_prompt)],
            InformationGapAnalysis,
            max_tokens=max_tokens,
        )

        if usage_hook is not None:
            usage_hook(
                UsageEvent(
                    operation="gap_analysis",
                    input_tokens=len(system_prompt + user_prompt) // 4,
                    output_tokens=100,
                )
            )

        return GapAnalysisResult(
            has_sufficient_info=result.is_sufficient,
            missing_aspects=result.missing_aspects,
            suggested_followup=result.suggested_followup or original_query,
            confidence=result.confidence,
        )

    except Exception as e:
        logger.warning(
            "gap_analysis_llm_failed",
            error=str(e)[:100],
            fallback="rule_based",
        )

        # Fallback: rule-based logic
        chunks_sufficient = len(retrieved_chunks) >= 3
        return GapAnalysisResult(
            has_sufficient_info=chunks_sufficient,
            missing_aspects=[] if chunks_sufficient else ["Insufficient chunks"],
            suggested_followup=_simplify_query(original_query),
            confidence=0.7 if chunks_sufficient else 0.3,
        )


def _simplify_query(query: str) -> str:
    """Extract core terms from query for reformulation fallback."""
    stopwords = {
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
    }
    words = query.split()
    core_terms = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    return " ".join(core_terms) if core_terms else query
