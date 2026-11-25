"""Context accumulation: expand retrieved chunks and check information sufficiency."""

from legal_rag.retrieval.accumulators.base import (
    AccumulatedChunk,
    AccumulatedContext,
    BaseAccumulator,
    ChunkFetcher,
)
from legal_rag.retrieval.accumulators.gap_analysis import (
    GapAnalysisResult,
    InformationGapAnalysis,
    analyze_information_gaps,
)
from legal_rag.retrieval.accumulators.neighbor import NeighborAccumulator, NeighborConfig

__all__ = [
    "AccumulatedChunk",
    "AccumulatedContext",
    "BaseAccumulator",
    "ChunkFetcher",
    "GapAnalysisResult",
    "InformationGapAnalysis",
    "NeighborAccumulator",
    "NeighborConfig",
    "analyze_information_gaps",
]
