"""Score-based reranker: retrieval scores with light, model-free boosts."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from legal_rag.retrieval.rerank._log import get_logger
from legal_rag.retrieval.rerank.base import BaseReranker, RerankOutcome
from legal_rag.types import RetrievedChunk

logger = get_logger("retrieval.rerank.score_based")


@dataclass
class ScoreBoostConfig:
    """Boost sizes, as fractions added to a multiplier that starts at 1.0."""

    term_overlap: float = 0.2
    """Scaled by the fraction of query terms found in the chunk."""
    title_match: float = 0.15
    """Added when a query term appears in ``metadata["section_title"]``."""
    section_number_present: float = 0.05
    """Added when ``metadata["section_number"]`` is set."""
    exact_phrase: float = 0.25
    """Added when the whole query appears verbatim in the chunk."""
    min_phrase_chars: int = 10


class ScoreBasedReranker(BaseReranker):
    """Reorder chunks by retrieval score multiplied by simple lexical boosts."""

    def __init__(self, top_k: int = 5, boosts: ScoreBoostConfig | None = None) -> None:
        super().__init__(top_k=top_k)
        self.boosts = boosts or ScoreBoostConfig()

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        **kwargs: object,
    ) -> RerankOutcome:
        start = time.monotonic()
        k = top_k or self.top_k
        if not chunks:
            return RerankOutcome(query=query)
        terms = set(query.lower().split())
        scored = [replace(c, score=self._boosted(c, query, terms)) for c in chunks]
        scored.sort(key=lambda c: c.score, reverse=True)
        kept = scored[:k]
        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "rerank_completed", provider="score_based", original=len(chunks), final=len(kept)
        )
        return RerankOutcome(
            query=query,
            chunks=kept,
            original_count=len(chunks),
            final_count=len(kept),
            processing_time_ms=elapsed,
        )

    def _boosted(self, chunk: RetrievedChunk, query: str, terms: set[str]) -> float:
        cfg = self.boosts
        boost = 1.0
        text = chunk.text.lower()
        if terms:
            boost += cfg.term_overlap * len(terms & set(text.split())) / len(terms)
        title = chunk.metadata.get("section_title")
        if title and terms & set(str(title).lower().split()):
            boost += cfg.title_match
        if chunk.metadata.get("section_number"):
            boost += cfg.section_number_present
        phrase = " ".join(query.lower().split())
        if len(phrase) > cfg.min_phrase_chars and phrase in text:
            boost += cfg.exact_phrase
        return (chunk.score or 0.0) * boost
