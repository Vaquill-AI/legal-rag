"""CRAG: Corrective RAG.

Grades each retrieved chunk for relevance with a model, classifies the retrieval as a whole,
and applies a correction:

* HIGH: keep everything.
* MEDIUM: drop the chunks graded below the threshold.
* PARTIAL: keep only relevant chunks and flag that the caller should supplement from elsewhere.
* LOW: keep the best few chunks (when configured to) and recommend alternative retrieval.

Reference: Yan et al., "Corrective Retrieval Augmented Generation" (2024).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, fingerprint, safe_error_fields
from legal_rag.retrieval.strategies.crag_prompts_relevance import RELEVANCE_PROMPT
from legal_rag.types import Message, RetrievedChunk

logger = logging.getLogger("legal_rag.retrieval.crag")


class RelevanceScore(BaseModel):
    """Structured output: one relevance grade between 0 and 1."""

    score: float | None = None


class RetrievalQuality(StrEnum):
    """Overall quality of a retrieved set."""

    HIGH = "high"
    MEDIUM = "medium"
    PARTIAL = "partial"
    LOW = "low"


@dataclass
class KnowledgeGapAnalysis:
    """What the retrieved context covers and what it lacks."""

    covered_aspects: list[str]
    missing_aspects: list[str]
    confidence_score: float
    recommendation: str
    has_partial_knowledge: bool


@dataclass
class CRAGConfig:
    """Thresholds and switches for the corrector."""

    relevance_threshold: float = 0.5
    high_quality_ratio: float = 0.8
    partial_quality_ratio: float = 0.5
    low_quality_ratio: float = 0.3
    enable_external_search: bool = False
    enable_query_decomposition: bool = True
    enable_chunk_filtering: bool = True
    preserve_for_low_quality: bool = True
    preserve_count: int = 3
    max_chunks_to_evaluate: int = 25
    min_chunks_required: int = 1
    chunk_chars: int = 1000
    max_tokens: int = 1024
    relevance_prompt: str = RELEVANCE_PROMPT
    """Grading rubric. Uses ``{query}`` and ``{chunk}`` placeholders."""


@dataclass
class CRAGResult:
    """Outcome of grading and correction."""

    query: str
    original_chunks: list[RetrievedChunk]
    corrected_chunks: list[RetrievedChunk]
    quality: RetrievalQuality
    relevance_scores: list[float]
    corrections_applied: list[str]
    evaluation_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class CRAGCorrector:
    """Grade retrieved chunks with an injected language model and correct the set."""

    def __init__(
        self,
        llm: LanguageModel,
        config: CRAGConfig | None = None,
        *,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or CRAGConfig()
        self.usage_hook = usage_hook
        self.stats: dict[str, Any] = {"evaluated": 0, "chunks_filtered": 0}

    async def evaluate_and_correct(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        context: dict[str, Any] | None = None,
    ) -> CRAGResult:
        """Grade ``chunks`` for ``query`` and return the corrected set. Never raises."""
        start = time.monotonic()
        try:
            if not chunks:
                return self._result(query, [], [], RetrievalQuality.LOW, [], ["no_results"], start)
            to_grade = chunks[: self.config.max_chunks_to_evaluate]
            scores = await self._grade(query, to_grade)
            emit_usage(
                self.usage_hook,
                "crag_evaluation",
                input_text=query * len(to_grade),
                chunks_evaluated=len(to_grade),
            )
            quality = self._assess_quality(scores)
            corrected, corrections = self._apply_corrections(chunks, scores, quality)
            self.stats["evaluated"] += 1
            key = f"{quality.value}_quality"
            self.stats[key] = self.stats.get(key, 0) + 1
            return self._result(query, chunks, corrected, quality, scores, corrections, start)
        except Exception as error:  # noqa: BLE001 - fall back to the original chunks
            logger.warning(
                "crag evaluation failed query_id=%s %s",
                fingerprint(query),
                safe_error_fields(error),
            )
            return self._result(
                query,
                chunks,
                chunks,
                RetrievalQuality.MEDIUM,
                [],
                ["error_fallback"],
                start,
                success=False,
                metadata={"error": str(error)[:160]},
            )

    async def _grade(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        async def grade_one(chunk: RetrievedChunk) -> float:
            # When grading is unavailable, keep the retrieval signal instead of dropping to 0.
            fallback = float(chunk.score)
            try:
                prompt = self.config.relevance_prompt.replace("{query}", query).replace(
                    "{chunk}", chunk.text[: self.config.chunk_chars]
                )
                result = await self.llm.generate_structured(
                    [Message("user", prompt)], RelevanceScore, max_tokens=self.config.max_tokens
                )
                if result is None or result.score is None:
                    return fallback
                return max(0.0, min(1.0, float(result.score)))
            except Exception as error:  # noqa: BLE001 - per-chunk fallback
                logger.warning("crag chunk grading failed %s", safe_error_fields(error))
                return fallback

        return list(await asyncio.gather(*(grade_one(c) for c in chunks)))

    def _assess_quality(self, scores: list[float]) -> RetrievalQuality:
        if not scores:
            return RetrievalQuality.LOW
        relevant = sum(1 for s in scores if s >= self.config.relevance_threshold)
        ratio = relevant / len(scores)
        if ratio >= self.config.high_quality_ratio and relevant >= self.config.min_chunks_required:
            return RetrievalQuality.HIGH
        if ratio >= self.config.partial_quality_ratio:
            return RetrievalQuality.PARTIAL
        if ratio <= self.config.low_quality_ratio:
            return RetrievalQuality.LOW
        return RetrievalQuality.MEDIUM

    def _keep_relevant(
        self, chunks: list[RetrievedChunk], scores: list[float], *, keep_ungraded: bool
    ) -> list[RetrievedChunk]:
        kept: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=False):
            if score >= self.config.relevance_threshold:
                kept.append(chunk)
            else:
                self.stats["chunks_filtered"] += 1
        if keep_ungraded:
            kept.extend(chunks[len(scores) :])
        return kept

    def _apply_corrections(
        self,
        chunks: list[RetrievedChunk],
        scores: list[float],
        quality: RetrievalQuality,
    ) -> tuple[list[RetrievedChunk], list[str]]:
        if quality is RetrievalQuality.HIGH:
            return chunks, ["none_needed"]
        if quality is RetrievalQuality.MEDIUM and self.config.enable_chunk_filtering:
            kept = self._keep_relevant(chunks, scores, keep_ungraded=True)
            return kept, [f"filtered_{len(chunks) - len(kept)}_chunks"]
        if quality is RetrievalQuality.PARTIAL:
            corrections = ["partial_knowledge_detected"]
            if self.config.enable_external_search:
                corrections.append("supplement_externally_recommended")
            return self._keep_relevant(chunks, scores, keep_ungraded=False), corrections
        if quality is RetrievalQuality.LOW:
            corrections = []
            if self.config.enable_external_search:
                corrections.append("external_search_recommended")
            if self.config.enable_query_decomposition:
                corrections.append("query_decomposition_recommended")
            if self.config.preserve_for_low_quality and chunks:
                ranked = sorted(
                    zip(chunks[: len(scores)], scores, strict=False),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                preserved = [c for c, _ in ranked[: self.config.preserve_count]]
                corrections.append(f"preserved_{len(preserved)}_low_quality_chunks")
                return preserved, corrections
            return [], corrections
        return chunks, []

    def _result(
        self,
        query: str,
        original: list[RetrievedChunk],
        corrected: list[RetrievedChunk],
        quality: RetrievalQuality,
        scores: list[float],
        corrections: list[str],
        start: float,
        *,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CRAGResult:
        return CRAGResult(
            query=query,
            original_chunks=original,
            corrected_chunks=corrected,
            quality=quality,
            relevance_scores=scores,
            corrections_applied=corrections,
            evaluation_time_ms=(time.monotonic() - start) * 1000,
            success=success,
            metadata=metadata or {},
        )
