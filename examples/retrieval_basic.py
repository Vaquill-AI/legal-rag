"""Retrieval basics, offline: HyDE, rank fusion, reranking and a semantic cache.

Shows four retrieval pieces working together with the in-memory fakes, so it needs no API keys
and no network:

1. HyDE writes a hypothetical answer passage and that passage is embedded instead of the question.
2. Reciprocal rank fusion merges a dense list and a keyword list into one ranking.
3. A reranker re-scores the fused chunks against the original question.
4. A semantic cache returns the stored ranking when the same question is asked again.

Run it from the repository root with ``python examples/retrieval_basic.py``.
"""

from __future__ import annotations

import asyncio

from legal_rag.providers.fakes import (
    FakeCache,
    FakeEmbedder,
    FakeLanguageModel,
    FakeReranker,
    FakeVectorStore,
)
from legal_rag.retrieval import reciprocal_rank_fusion
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.retrieval.strategies import HyDEGenerator
from legal_rag.types import RetrievedChunk

QUESTION = "can an official be sued for acting under color of state law"
HYPOTHESIS = (
    "A person may sue a state official under 42 U.S.C. 1983 when the official, acting under "
    "color of state law, deprives the person of a federal constitutional right."
)
SECTION_1983 = RetrievedChunk(
    "usc-42-1983",
    "42 U.S.C. 1983. Every person who, under color of state law, deprives another of a constitutional right shall be liable.",
)
SECTION_242 = RetrievedChunk(
    "usc-18-242",
    "18 U.S.C. 242. Whoever, under color of law, willfully deprives a person of rights shall be fined or imprisoned.",
)
MONROE = RetrievedChunk(
    "monroe-v-pape",
    "Monroe v. Pape, 365 U.S. 167 (1961). Section 1983 reaches misuse of power under color of state law.",
)


def show(title: str, chunks: list[RetrievedChunk]) -> None:
    print(title)
    for rank, chunk in enumerate(chunks, start=1):
        print(f"  {rank}. {chunk.id:<14} score={chunk.score:.4f}")


async def retrieve(question: str, cache: SemanticCache) -> list[RetrievedChunk]:
    cached = await cache.get(question)
    if cached is not None:
        print("cache: hit, skipping HyDE, search and rerank")
        return [RetrievedChunk(**row) for row in cached]
    print("cache: miss, running the full pipeline")

    # 1. HyDE: embed an answer-shaped passage instead of the question.
    hyde = HyDEGenerator(FakeLanguageModel(reply=HYPOTHESIS))
    hypothesis = (await hyde.generate(question)).primary_hypothesis
    vector = await FakeEmbedder().embed_query(hypothesis)
    print(f"hyde: embedded {len(vector)}-dim vector for: {hypothesis[:60]}...")

    # 2. Fuse a dense list and a keyword list (the fakes ignore the vector and return fixed lists).
    dense = await FakeVectorStore([SECTION_1983, MONROE, SECTION_242]).search(vector, top_k=3)
    keyword = await FakeVectorStore([MONROE, SECTION_242]).search(vector, top_k=3)
    show("dense list:", dense)
    show("keyword list:", keyword)
    fused = reciprocal_rank_fusion([dense, keyword])
    show("fused with RRF:", fused)

    # 3. Rerank the fused chunks against the original question.
    ranked = await FakeReranker().rerank(question, [chunk.text for chunk in fused], top_k=3)
    reranked = [RetrievedChunk(fused[r.index].id, fused[r.index].text, r.score) for r in ranked]
    show("reranked by word overlap:", reranked)

    # 4. Store the ranking so the next similar question skips all of the above.
    await cache.set(question, [{"id": c.id, "text": c.text, "score": c.score} for c in reranked])
    return reranked


async def main() -> None:
    cache = SemanticCache(FakeEmbedder(), FakeCache())
    await retrieve(QUESTION, cache)
    print()
    again = await retrieve(QUESTION, cache)
    print(f"cache: top chunk on the second call is {again[0].id}; stats {cache.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
