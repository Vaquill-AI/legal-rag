"""
Confidence Scoring for RAG Retrieval.

Calculates confidence scores for retrieval results based on multiple signals:
- Score spread: High spread indicates clear winner
- Dense/sparse agreement: Agreement between search methods
- Citation grounding: Citations found in retrieved sources
- Semantic consistency: Coherent topics across results

Formula:
    confidence = (
        0.4 * retrieval_score_spread +      # High spread = clear winner
        0.3 * dense_sparse_agreement +       # Agreement between methods
        0.2 * citation_grounding_ratio +     # Citations in sources
        0.1 * semantic_consistency           # Coherent topic
    )
"""

import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from legal_rag.retrieval.rerank._log import get_logger

logger = get_logger("retrieval.confidence")


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""

    HIGH = "high"  # > 0.8: Reliable, can be used directly
    MEDIUM = "medium"  # 0.6-0.8: Usable with caveats
    LOW = "low"  # 0.4-0.6: Requires verification
    VERY_LOW = "very_low"  # < 0.4: Should not be trusted


@dataclass
class RetrievalConfidence:
    """
    Confidence metrics for retrieval results.

    Attributes:
        overall_score: Combined confidence score (0-1)
        level: Confidence level category
        score_spread: How much top results differ from others
        dense_sparse_agreement: Agreement between dense and sparse search
        citation_grounding: Ratio of citations found in sources
        semantic_consistency: Topic coherence across results
        requires_review: Whether human review is recommended
        suggested_action: Recommended action based on confidence
        factors: Detailed breakdown of confidence factors
    """

    overall_score: float
    level: ConfidenceLevel
    retrieval_strength: float = 0.0
    score_spread: float = 0.0
    dense_sparse_agreement: float = 0.0
    citation_grounding: float = 0.0
    semantic_consistency: float = 0.0
    requires_review: bool = False
    suggested_action: str = ""
    factors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": round(self.overall_score, 3),
            "level": self.level.value,
            "retrieval_strength": round(self.retrieval_strength, 3),
            "score_spread": round(self.score_spread, 3),
            "dense_sparse_agreement": round(self.dense_sparse_agreement, 3),
            "citation_grounding": round(self.citation_grounding, 3),
            "semantic_consistency": round(self.semantic_consistency, 3),
            "requires_review": self.requires_review,
            "suggested_action": self.suggested_action,
            "factors": self.factors,
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG retrieval results.

    Uses multiple signals to determine how reliable the retrieval
    results are for generating accurate responses.

    Example:
        scorer = ConfidenceScorer()
        confidence = scorer.score(
            chunks=retrieved_chunks,
            query=query,
            citations=extracted_citations
        )

        if confidence.requires_review:
            print(f"Low confidence: {confidence.suggested_action}")
    """

    # Default weights for confidence factors.
    #
    # retrieval_strength (mean of the top-K clamped chunk scores) is the PRIMARY
    # signal: it measures whether strongly-relevant material was actually
    # retrieved. score_spread alone was a poor proxy -- it measures only how much
    # the #1 chunk beats the rest, so a well-grounded answer with several
    # uniformly-relevant sources (statutes + on-point cases) had LOW spread and
    # scored LOW confidence, exactly backwards (observed: 0.39-0.57 on verified
    # statutory answers). Spread is kept as a secondary "clear winner" signal.
    #
    # dense_sparse_agreement and citation_grounding default to weight 0: many stores fuse
    # dense and sparse results server-side and return a single score, and the scorer is
    # only useful for citation grounding when it is handed the query citations. Both return
    # a neutral constant when their inputs are missing. Raise their weights when you can
    # supply per-method scores and extracted citations.
    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "retrieval_strength": 0.55,
        "score_spread": 0.25,
        "semantic_consistency": 0.20,
        "dense_sparse_agreement": 0.0,
        "citation_grounding": 0.0,
    }

    # How many top chunks feed retrieval_strength. The best few are the material
    # the answer is actually grounded on; the long tail is retrieval padding, so
    # averaging all of it would drag a strong answer down.
    RETRIEVAL_STRENGTH_TOP_K: ClassVar[int] = 5

    # Calibration band for retrieval_strength. The score fed in is the
    # raw cross-encoder rerank relevance, which does NOT span [0, 1]: even a strong,
    # on-point legal match scores ~0.5-0.65 and scores above ~0.7 are rare, while
    # anything below ~0.25 is retrieval noise. Reading that raw number straight
    # into thresholds tuned for a true [0, 1] scale (0.6 medium / 0.8 high)
    # structurally capped confidence at ~0.41 for an EXCELLENT retrieval and left
    # even a pinned authority below "medium" (measured). We min-max
    # normalize the mean top-K score across [NOISE_FLOOR, STRONG_MATCH_SCORE] so a
    # mean at/above STRONG_MATCH_SCORE reads as full strength (1.0) and a mean
    # at/below the noise floor reads as 0.0, mapping the real cross-encoder domain onto
    # the confidence domain WITHOUT inflating genuinely weak retrieval. Tune the
    # band edges if the badge over-/under-trusts. A deliberately boosted
    # chunk (score above 1.0, clamped to 1.0 before averaging) saturates the band.
    RETRIEVAL_STRENGTH_NOISE_FLOOR: ClassVar[float] = 0.25
    STRONG_MATCH_SCORE: ClassVar[float] = 0.55

    # Thresholds for confidence levels
    THRESHOLDS: ClassVar[dict[str, float]] = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        review_threshold: float = 0.6,
    ):
        """
        Initialize confidence scorer.

        Args:
            weights: Custom weights for confidence factors
            review_threshold: Score below which review is recommended
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.review_threshold = review_threshold

    def score(
        self,
        chunks: list[Any],
        query: str,
        citations: list[Any] | None = None,
        dense_scores: list[float] | None = None,
        sparse_scores: list[float] | None = None,
    ) -> RetrievalConfidence:
        """
        Calculate confidence score for retrieval results.

        Args:
            chunks: Retrieved chunks with scores
            query: Original query
            citations: Extracted citations from query/context
            dense_scores: Scores from dense vector search
            sparse_scores: Scores from sparse (BM25) search

        Returns:
            RetrievalConfidence with all metrics
        """
        if not chunks:
            return RetrievalConfidence(
                overall_score=0.0,
                level=ConfidenceLevel.VERY_LOW,
                requires_review=True,
                suggested_action="No relevant documents found. Try rephrasing your query.",
            )

        # Calculate individual factors
        retrieval_strength = self._calculate_retrieval_strength(chunks)
        score_spread = self._calculate_score_spread(chunks)
        dense_sparse_agreement = self._calculate_dense_sparse_agreement(dense_scores, sparse_scores)
        citation_grounding = self._calculate_citation_grounding(chunks, citations)
        semantic_consistency = self._calculate_semantic_consistency(chunks, query)

        # Calculate weighted overall score
        overall_score = (
            self.weights.get("retrieval_strength", 0.0) * retrieval_strength
            + self.weights["score_spread"] * score_spread
            + self.weights["dense_sparse_agreement"] * dense_sparse_agreement
            + self.weights["citation_grounding"] * citation_grounding
            + self.weights["semantic_consistency"] * semantic_consistency
        )

        # Determine confidence level
        level = self._get_confidence_level(overall_score)

        # Determine if review is needed
        requires_review = overall_score < self.review_threshold

        # Generate suggested action
        suggested_action = self._get_suggested_action(
            overall_score, level, score_spread, citation_grounding
        )

        # Compile factors for transparency
        factors = {
            "retrieval_strength": {
                "value": round(retrieval_strength, 3),
                "weight": self.weights.get("retrieval_strength", 0.0),
                "contribution": round(
                    self.weights.get("retrieval_strength", 0.0) * retrieval_strength, 3
                ),
            },
            "score_spread": {
                "value": round(score_spread, 3),
                "weight": self.weights["score_spread"],
                "contribution": round(self.weights["score_spread"] * score_spread, 3),
            },
            "dense_sparse_agreement": {
                "value": round(dense_sparse_agreement, 3),
                "weight": self.weights["dense_sparse_agreement"],
                "contribution": round(
                    self.weights["dense_sparse_agreement"] * dense_sparse_agreement, 3
                ),
            },
            "citation_grounding": {
                "value": round(citation_grounding, 3),
                "weight": self.weights["citation_grounding"],
                "contribution": round(self.weights["citation_grounding"] * citation_grounding, 3),
            },
            "semantic_consistency": {
                "value": round(semantic_consistency, 3),
                "weight": self.weights["semantic_consistency"],
                "contribution": round(
                    self.weights["semantic_consistency"] * semantic_consistency, 3
                ),
            },
            "chunks_analyzed": len(chunks),
            "citations_found": len(citations) if citations else 0,
        }

        logger.info(
            "confidence_score_calculated",
            overall_score=round(overall_score, 3),
            level=level.value,
            requires_review=requires_review,
        )

        return RetrievalConfidence(
            overall_score=overall_score,
            level=level,
            retrieval_strength=retrieval_strength,
            score_spread=score_spread,
            dense_sparse_agreement=dense_sparse_agreement,
            citation_grounding=citation_grounding,
            semantic_consistency=semantic_consistency,
            requires_review=requires_review,
            suggested_action=suggested_action,
            factors=factors,
        )

    def _calculate_retrieval_strength(self, chunks: list[Any]) -> float:
        """Absolute relevance of the best retrieved chunks (mean of top-K clamped
        scores).

        This is the signal ``score_spread`` lacks: it rewards retrieving
        strongly-relevant material even when several sources are uniformly
        relevant (low spread). Reads ``score`` (the ranked score), clamped to
        [0, 1] so a boosted chunk contributes a full 1.0, never more. Placeholder
        scores flagged ``score_estimated`` in metadata are excluded.

        Returns:
            Score between 0-1 (mean of the top-K clamped chunk scores).
        """
        scores: list[float] = []
        for chunk in chunks:
            meta = (
                (getattr(chunk, "metadata", None) or {})
                if not isinstance(chunk, dict)
                else (chunk.get("metadata") or {})
            )
            if isinstance(meta, dict) and meta.get("score_estimated"):
                continue  # an estimated placeholder carries no real relevance
            raw = getattr(chunk, "score", None)
            if raw is None:
                raw = chunk.get("score", 0.0) if isinstance(chunk, dict) else 0.0
            try:
                scores.append(min(max(float(raw or 0.0), 0.0), 1.0))
            except (TypeError, ValueError):
                continue

        if not scores:
            return 0.5  # results exist but carry no usable score: neutral
        scores.sort(reverse=True)
        top = scores[: self.RETRIEVAL_STRENGTH_TOP_K]
        raw_mean = sum(top) / len(top)
        # Min-max normalize the raw cross-encoder domain onto [0, 1] across the
        # [noise_floor, strong_match] band. A strong answer (raw mean ~0.5) then
        # reads as strong instead of under-reporting against [0, 1] thresholds,
        # while weak retrieval (raw mean near the floor) still reads near 0.
        band = self.STRONG_MATCH_SCORE - self.RETRIEVAL_STRENGTH_NOISE_FLOOR
        calibrated = (raw_mean - self.RETRIEVAL_STRENGTH_NOISE_FLOOR) / band
        return min(max(calibrated, 0.0), 1.0)

    def _calculate_score_spread(self, chunks: list[Any]) -> float:
        """
        Calculate score spread - how distinct the top result is.

        A high spread indicates the top result is clearly better,
        suggesting higher confidence. Low spread means results are
        similar, making it harder to determine relevance.

        Returns:
            Score between 0-1 (1 = large spread, high confidence)
        """
        if not chunks or len(chunks) < 2:
            return 0.5  # Neutral if not enough data

        # Extract scores from chunks
        scores = []
        for chunk in chunks:
            if hasattr(chunk, "score"):
                scores.append(chunk.score)
            elif isinstance(chunk, dict) and "score" in chunk:
                scores.append(chunk["score"])

        if len(scores) < 2:
            return 0.5

        # Sort descending
        scores = sorted(scores, reverse=True)

        # Calculate spread between top score and average of rest
        top_score = scores[0]
        rest_avg = statistics.mean(scores[1:]) if len(scores) > 1 else 0

        # Normalize spread to 0-1
        # If top score is significantly higher than rest, spread is high
        if top_score > 0:
            spread = (top_score - rest_avg) / top_score
            return min(max(spread, 0.0), 1.0)

        return 0.5

    def _calculate_dense_sparse_agreement(
        self,
        dense_scores: list[float] | None,
        sparse_scores: list[float] | None,
    ) -> float:
        """
        Calculate agreement between dense and sparse search results.

        High agreement between semantic (dense) and keyword (sparse)
        search suggests the results are genuinely relevant.

        Returns:
            Score between 0-1 (1 = high agreement)
        """
        # No data OR all-zero scores. The hybrid retrievers fuse dense+sparse via
        # a store-side fusion, which returns a single fused score and does NOT
        # surface per-method scores -- so these arrive as all-zeros. Treat that as
        # "no signal" and return the neutral default, NOT a false 1.0 (all-zero in
        # -> avg_diff 0 -> agreement 1.0, which silently inflated confidence).
        if not dense_scores or not sparse_scores or not any(dense_scores) or not any(sparse_scores):
            return 0.7  # Neutral: we don't have real per-method scores

        if len(dense_scores) != len(sparse_scores):
            # Different lengths, normalize
            min_len = min(len(dense_scores), len(sparse_scores))
            dense_scores = dense_scores[:min_len]
            sparse_scores = sparse_scores[:min_len]

        if not dense_scores:
            return 0.7

        try:
            # Normalize scores to 0-1 range
            dense_max = max(dense_scores) if max(dense_scores) > 0 else 1
            sparse_max = max(sparse_scores) if max(sparse_scores) > 0 else 1

            dense_norm = [s / dense_max for s in dense_scores]
            sparse_norm = [s / sparse_max for s in sparse_scores]

            # Calculate correlation (simplified)
            # Using average absolute difference
            diffs = [abs(d - s) for d, s in zip(dense_norm, sparse_norm, strict=False)]
            avg_diff = statistics.mean(diffs)

            # Convert to agreement score (lower diff = higher agreement)
            agreement = 1.0 - min(avg_diff, 1.0)
            return agreement

        except (ValueError, ZeroDivisionError, TypeError) as e:
            logger.debug(
                "dense_sparse_agreement_calc_failed",
                error=str(e),
                fallback_value=0.7,
            )
            return 0.7  # Neutral fallback when calculation fails

    def _calculate_citation_grounding(
        self,
        chunks: list[Any],
        citations: list[Any] | None,
    ) -> float:
        """
        Calculate how well citations are grounded in retrieved sources.

        If the query or context mentions specific cases, this checks
        if those cases appear in the retrieved chunks.

        Returns:
            Score between 0-1 (1 = all citations found in sources)
        """
        if not citations:
            return 0.8  # No citations to verify, neutral-positive

        if not chunks:
            return 0.0

        # Extract text from chunks for searching
        chunk_texts = []
        for chunk in chunks:
            if hasattr(chunk, "text"):
                chunk_texts.append(chunk.text.lower())
            elif hasattr(chunk, "original_content"):
                chunk_texts.append(chunk.original_content.lower())
            elif isinstance(chunk, dict):
                content = chunk.get("content", chunk.get("original_content", ""))
                chunk_texts.append(content.lower())

        combined_text = " ".join(chunk_texts)

        # Count how many citations are found in chunks
        found_count = 0
        for citation in citations:
            # Get citation text to search for
            citation_text = ""
            if hasattr(citation, "full_citation"):
                citation_text = citation.full_citation
            elif isinstance(citation, dict):
                citation_text = citation.get("full_citation", str(citation))
            else:
                citation_text = str(citation)

            # Check if citation appears in chunks
            if citation_text.lower() in combined_text:
                found_count += 1

        # Calculate grounding ratio
        if len(citations) > 0:
            return found_count / len(citations)

        return 0.8

    def _calculate_semantic_consistency(
        self,
        chunks: list[Any],
        query: str,
    ) -> float:
        """
        Calculate semantic consistency of retrieved chunks.

        Checks if chunks are topically coherent (not scattered across
        unrelated subjects), which indicates focused, relevant results.

        Returns:
            Score between 0-1 (1 = highly consistent topics)
        """
        if not chunks or len(chunks) < 2:
            return 0.8  # Single result is inherently consistent

        # Section-title CONCENTRATION, not uniqueness. The prior formula penalized
        # diverse sources (1 - (unique-1)/n -> ~0 when every chunk came from a
        # different section, floored to 0.3), but a legal answer legitimately
        # draws on several distinct provisions (termination + notice + liability)
        # or on statutes + on-point cases -- that diversity is coherent grounding,
        # not scatter, so the old penalty dragged every broad answer down ~0.14.
        # We instead REWARD concentration mildly: the fraction of chunks in the
        # single most-common section, mapped onto [0.6, 1.0]. A focused answer (one
        # dominant section) approaches 1.0; a legitimately broad answer floors at
        # ~0.6 (neutral), never the old 0.3 penalty.
        section_titles = []
        for chunk in chunks:
            title = None
            if isinstance(chunk, dict):
                title = chunk.get("section_title")
            else:
                title = (getattr(chunk, "metadata", None) or {}).get("section_title")

            if title:
                section_titles.append(title.lower())

        if not section_titles:
            return 0.7  # No section info, assume moderate consistency

        concentration = Counter(section_titles).most_common(1)[0][1] / len(section_titles)
        return 0.6 + 0.4 * concentration

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Map score to confidence level."""
        if score >= self.THRESHOLDS["high"]:
            return ConfidenceLevel.HIGH
        elif score >= self.THRESHOLDS["medium"]:
            return ConfidenceLevel.MEDIUM
        elif score >= self.THRESHOLDS["low"]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _get_suggested_action(
        self,
        score: float,
        level: ConfidenceLevel,
        score_spread: float,
        citation_grounding: float,
    ) -> str:
        """Generate suggested action based on confidence analysis."""
        if level == ConfidenceLevel.HIGH:
            return "Results appear reliable. Proceed with response."

        if level == ConfidenceLevel.MEDIUM:
            if score_spread < 0.5:
                return "Multiple relevant results found. Consider reviewing all sources."
            if citation_grounding < 0.5:
                return (
                    "Some citations could not be verified. Cross-reference with original sources."
                )
            return "Results are moderately reliable. Verify critical claims."

        if level == ConfidenceLevel.LOW:
            if score_spread < 0.3:
                return "Results are ambiguous. Consider rephrasing the query for more specific results."
            if citation_grounding < 0.3:
                return "Citations not found in retrieved documents. Consult primary sources."
            return "Low confidence in results. Consult legal professional for verification."

        # VERY_LOW
        return "Results are unreliable. Strongly recommend consulting a legal professional."


# Singleton instance
_default_scorer: ConfidenceScorer | None = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create default confidence scorer instance."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = ConfidenceScorer()
    return _default_scorer
