"""Reranking, reordering, authority weighting, confidence scoring and source hygiene."""

from legal_rag.retrieval.rerank.authority import (
    AuthorityConfig,
    AuthorityReranker,
    CourtLevel,
    apply_court_boost,
    get_court_level,
)
from legal_rag.retrieval.rerank.base import BaseReranker, RerankOutcome
from legal_rag.retrieval.rerank.confidence import (
    ConfidenceLevel,
    ConfidenceScorer,
    RetrievalConfidence,
)
from legal_rag.retrieval.rerank.coverage import CoverageConfig, apply_coverage_notice
from legal_rag.retrieval.rerank.cross_encoder import CrossEncoderReranker
from legal_rag.retrieval.rerank.diversity import DiversityReorderer
from legal_rag.retrieval.rerank.lost_in_middle import LostInMiddleReorderer
from legal_rag.retrieval.rerank.rate_limit import (
    RateLimitWaitExceeded,
    RerankDeadlineExceeded,
    RerankRateLimited,
    RerankRateLimiter,
)
from legal_rag.retrieval.rerank.reorder_base import BaseReorderer, ReorderingResult
from legal_rag.retrieval.rerank.score_based import ScoreBasedReranker, ScoreBoostConfig
from legal_rag.retrieval.rerank.source_dedup import dedupe_sources

__all__ = [
    "AuthorityConfig",
    "AuthorityReranker",
    "BaseReorderer",
    "BaseReranker",
    "ConfidenceLevel",
    "ConfidenceScorer",
    "CourtLevel",
    "CoverageConfig",
    "CrossEncoderReranker",
    "DiversityReorderer",
    "LostInMiddleReorderer",
    "RateLimitWaitExceeded",
    "ReorderingResult",
    "RerankDeadlineExceeded",
    "RerankOutcome",
    "RerankRateLimited",
    "RerankRateLimiter",
    "RetrievalConfidence",
    "ScoreBasedReranker",
    "ScoreBoostConfig",
    "apply_court_boost",
    "apply_coverage_notice",
    "dedupe_sources",
    "get_court_level",
]
