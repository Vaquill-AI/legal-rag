"""
Diversity-based Reorderer.

Reorders chunks to maximize diversity while maintaining relevance.
Useful for multi-faceted legal queries where different aspects need coverage.

Strategy:
1. Keep highest relevance chunk at top
2. For remaining chunks: maximize diversity
3. Uses MMR (Maximal Marginal Relevance) approach

Based on:
- MMR: Carbonell & Goldstein (1998)
- LangChain's MMR reranking
- "Diversity in Search Results" research
"""

from legal_rag.retrieval.rerank._log import get_logger
from legal_rag.retrieval.rerank.reorder_base import BaseReorderer, ReorderingResult
from legal_rag.types import RetrievedChunk

logger = get_logger("rag.reordering.diversity")


class DiversityReorderer(BaseReorderer):
    """
    Reorders chunks to maximize diversity while maintaining relevance.

    Uses a greedy MMR-like approach:
    1. Select highest relevance chunk first
    2. For each subsequent position, select chunk that:
       - Has high relevance score
       - Is most different from already selected chunks
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize Diversity reorderer.

        Args:
            config: Optional configuration
                - lambda_param: Balance between relevance and diversity (0-1)
                  0 = max diversity, 1 = max relevance (default: 0.5)
                - min_chunks_for_reorder: Minimum chunks needed (default: 3)
        """
        super().__init__(config)
        self.lambda_param = self.config.get("lambda_param", 0.5)
        self.min_chunks = self.config.get("min_chunks_for_reorder", 3)

    def should_reorder(self, chunks: list[RetrievedChunk]) -> bool:
        """Check if reordering is beneficial."""
        return len(chunks) >= self.min_chunks

    def _calculate_similarity(self, chunk1: RetrievedChunk, chunk2: RetrievedChunk) -> float:
        """
        Calculate similarity between two chunks.

        For now, uses simple text overlap. Could be enhanced with:
        - Embedding cosine similarity
        - Semantic similarity models
        - Section/source-based clustering
        """
        # Simple text-based similarity (Jaccard)
        words1 = set(chunk1.text.lower().split())
        words2 = set(chunk2.text.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_mmr_score(
        self, candidate: RetrievedChunk, selected: list[RetrievedChunk]
    ) -> float:
        """
        Calculate MMR score for a candidate chunk.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            candidate: Candidate chunk to score
            selected: Already selected chunks

        Returns:
            MMR score (higher is better)
        """
        relevance = candidate.score

        if not selected:
            return relevance

        # Find maximum similarity to any selected chunk
        max_similarity = max(self._calculate_similarity(candidate, sel) for sel in selected)

        # MMR formula
        mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity

        return mmr_score

    def reorder(self, chunks: list[RetrievedChunk]) -> ReorderingResult:
        """
        Reorder chunks using MMR-based diversity strategy.

        Algorithm:
        1. Select chunk with highest relevance score
        2. For each remaining position:
           - Calculate MMR score for all unselected chunks
           - Select chunk with highest MMR score
        3. Result: Diverse set of chunks covering different aspects

        Args:
            chunks: Retrieved chunks with relevance scores

        Returns:
            ReorderingResult with reordered chunks
        """
        if not self.should_reorder(chunks):
            logger.debug(
                "diversity_reorder_skip",
                chunk_count=len(chunks),
                reason="below_min_threshold",
            )
            return ReorderingResult(
                reordered_chunks=chunks,
                strategy="diversity",
                original_order=list(range(len(chunks))),
                reordering_applied=False,
            )

        # Store original indices
        indexed_chunks = [(i, chunk) for i, chunk in enumerate(chunks)]

        # Greedy MMR selection
        selected = []
        selected_indices = []
        remaining = indexed_chunks.copy()

        # Start with highest relevance chunk
        best_idx = max(range(len(remaining)), key=lambda i: remaining[i][1].score)
        orig_idx, best_chunk = remaining.pop(best_idx)
        selected.append(best_chunk)
        selected_indices.append(orig_idx)

        # Iteratively select chunks with highest MMR score
        while remaining:
            # Calculate MMR score for all remaining chunks
            mmr_scores = [
                (i, orig_idx, chunk, self._calculate_mmr_score(chunk, selected))
                for i, (orig_idx, chunk) in enumerate(remaining)
            ]

            # Select chunk with highest MMR score
            best_idx, best_orig_idx, best_chunk, _ = max(mmr_scores, key=lambda x: x[3])

            selected.append(best_chunk)
            selected_indices.append(best_orig_idx)
            remaining.pop(best_idx)

        logger.debug(
            "diversity_reordered",
            original_count=len(chunks),
            reordered_count=len(selected),
            lambda_param=self.lambda_param,
            relevance_distribution=self._get_relevance_distribution(selected),
        )

        return ReorderingResult(
            reordered_chunks=selected,
            strategy="diversity",
            original_order=selected_indices,
            reordering_applied=True,
        )

    def _get_relevance_distribution(self, chunks: list[RetrievedChunk]) -> dict:
        """Get relevance score distribution for logging."""
        if not chunks:
            return {}

        scores = [c.score for c in chunks]
        return {
            "first": round(scores[0], 3),
            "middle": round(scores[len(scores) // 2], 3),
            "last": round(scores[-1], 3),
            "avg": round(sum(scores) / len(scores), 3),
        }


def get_diversity_reorderer(config: dict | None = None) -> DiversityReorderer:
    """Get or create Diversity reorderer instance."""
    return DiversityReorderer(config)
