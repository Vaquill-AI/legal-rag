"""Retrieval techniques: strategies, fusion, reranking, routing, conversation, caching and streaming.

The base retriever contract and rank fusion are re-exported here.
The larger subpackages are imported on demand, so ``import legal_rag.retrieval`` stays cheap:

- ``legal_rag.retrieval.strategies``: HyDE, multi-query, multi-hop, Self-RAG, CRAG, decomposition
- ``legal_rag.retrieval.rerank``: cross-encoder and score-based reranking, authority, diversity, lost-in-the-middle
- ``legal_rag.retrieval.routing``: query classification and the unified router
- ``legal_rag.retrieval.conversation``: condensation, memory, intent routing, query analysis
- ``legal_rag.retrieval.caching``: semantic cache with payload validation
- ``legal_rag.retrieval.streaming``: server-sent event shapes and response assembly
- ``legal_rag.retrieval.accumulators``: neighbour expansion and gap analysis
"""

from legal_rag.retrieval.base import (
    BaseRetriever,
    ChunkGroup,
    GroupedRetrievalResult,
    RetrievalResult,
)
from legal_rag.retrieval.fusion import reciprocal_rank_fusion, rrf_merge_chunks

__all__ = [
    "BaseRetriever",
    "ChunkGroup",
    "GroupedRetrievalResult",
    "RetrievalResult",
    "reciprocal_rank_fusion",
    "rrf_merge_chunks",
]
