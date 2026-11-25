"""Query pipeline basics, offline: one question through every stage, then an answer and a check.

Builds a ``QueryPipeline`` with the in-memory fakes and the US jurisdiction plug-in, asks one
question, and prints the tier, retrieval strategy, cache status, sources and the start of the
context. It then writes an answer with a canned model and scores it with the hallucination
detector. No API key and no network are needed.

Run it from the repository root with ``python examples/pipeline_basic.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from legal_rag.agents.generation import GenerationConfig, generate_answer_buffered
from legal_rag.pipeline import PipelineRequest, QueryPipeline, jurisdictions
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeLanguageModel, FakeReranker
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.types import RetrievedChunk
from legal_rag.verification.grounding.detector import HallucinationDetector

QUESTION = "what written notice must a landlord give a tenant before starting an eviction"
SECTIONS = [
    RetrievedChunk(
        "tenancy-12",
        "A landlord must give a tenant written notice of at least 14 days before starting an "
        "eviction for unpaid rent.",
        0.82,
        {"title": "Residential Tenancies Act, section 12", "source_type": "statute"},
    ),
    RetrievedChunk(
        "tenancy-13",
        "The written notice must state the amount owed, the date it is due and the tenant's "
        "right to pay and stay.",
        0.74,
        {"title": "Residential Tenancies Act, section 13", "source_type": "statute"},
    ),
    RetrievedChunk(
        "tenancy-40",
        "A court may dismiss an eviction filed without the written notice the landlord must give.",
        0.61,
        {"title": "Residential Tenancies Act, section 40", "source_type": "statute"},
    ),
]
ANSWER = (
    "A landlord must give the tenant written notice of at least 14 days before an eviction for "
    "unpaid rent [Source 1]. The notice must state the amount owed and the due date [Source 2]."
)


class ListRetriever:
    """Stands in for your search index. Apply your tenant filter here, from ``filters``."""

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        return SECTIONS[:top_k]


async def main() -> None:
    index = ListRetriever()
    pipeline = QueryPipeline(
        retriever=index,
        reranker=FakeReranker(),
        jurisdiction=jurisdictions.get("us", statute_retriever=index),
        semantic_cache=SemanticCache(FakeEmbedder(), FakeCache()),
    )
    request = PipelineRequest(QUESTION, filters={"country_code": "US"})
    result = await pipeline.run(request)
    print(f"tier={result.tier} strategy={result.strategy} cache={result.cache_status}")
    print("sources:", [s["metadata"]["title"] for s in result.sources])
    print("context:", result.context[:200].replace("\n", " "))

    again = await pipeline.run(request)
    print(f"second ask: cache={again.cache_status} chunks={len(again.chunks)}")

    config = GenerationConfig(
        QUESTION, result.chunks, context=result.context, tier=result.tier, enable_planning=False
    )
    answer = await generate_answer_buffered(
        config, FakeLanguageModel(reply=ANSWER), detector=HallucinationDetector()
    )
    check = answer.verification
    verdict = (
        "no verdict" if check is None else f"verified={check.verified} score={check.score:.2f}"
    )
    print("answer:", answer.answer)
    print("verification:", verdict)


if __name__ == "__main__":
    asyncio.run(main())
