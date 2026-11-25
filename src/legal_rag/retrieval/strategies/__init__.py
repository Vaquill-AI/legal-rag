"""Query-time retrieval strategies: HyDE, multi-query, multi-hop, Self-RAG, CRAG and decomposition."""

from legal_rag.retrieval.strategies.crag import (
    CRAGConfig,
    CRAGCorrector,
    CRAGResult,
    KnowledgeGapAnalysis,
    RetrievalQuality,
)
from legal_rag.retrieval.strategies.crag_source_mixer import (
    MixedContext,
    MixStrategy,
    SourceMixer,
    SourceMixerConfig,
)
from legal_rag.retrieval.strategies.gap_analysis import (
    GapAnalysisResult,
    GapAnalyzer,
    LLMGapAnalyzer,
)
from legal_rag.retrieval.strategies.hyde import (
    HyDEConfig,
    HyDEGenerator,
    HyDEResult,
    HyDETransformer,
    HyDETransformResult,
)
from legal_rag.retrieval.strategies.multi_hop import (
    Hop,
    HopEvent,
    HopMessages,
    MultiHopConfig,
    MultiHopResult,
    MultiHopRetriever,
)
from legal_rag.retrieval.strategies.multi_query import (
    MultiQueryConfig,
    MultiQueryGenerator,
    MultiQueryResult,
    QueryVariations,
    generate_rewrites,
)
from legal_rag.retrieval.strategies.query_decomposer import (
    DecompositionAnalysis,
    DecompositionResult,
    QueryDecomposer,
)
from legal_rag.retrieval.strategies.self_rag import (
    CritiqueResult,
    RetrievalDecision,
    SelfRAGConfig,
    SelfRAGController,
    SelfRAGResult,
)

__all__ = [
    "CRAGConfig",
    "CRAGCorrector",
    "CRAGResult",
    "CritiqueResult",
    "DecompositionAnalysis",
    "DecompositionResult",
    "GapAnalysisResult",
    "GapAnalyzer",
    "Hop",
    "HopEvent",
    "HopMessages",
    "HyDEConfig",
    "HyDEGenerator",
    "HyDEResult",
    "HyDETransformResult",
    "HyDETransformer",
    "KnowledgeGapAnalysis",
    "LLMGapAnalyzer",
    "MixStrategy",
    "MixedContext",
    "MultiHopConfig",
    "MultiHopResult",
    "MultiHopRetriever",
    "MultiQueryConfig",
    "MultiQueryGenerator",
    "MultiQueryResult",
    "QueryDecomposer",
    "QueryVariations",
    "RetrievalDecision",
    "RetrievalQuality",
    "SelfRAGConfig",
    "SelfRAGController",
    "SelfRAGResult",
    "SourceMixer",
    "SourceMixerConfig",
    "generate_rewrites",
]
