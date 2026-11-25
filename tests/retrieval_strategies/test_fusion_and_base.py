from collections.abc import Mapping, Sequence
from typing import Any

from legal_rag.retrieval.base import (
    BaseRetriever,
    ChunkGroup,
    GroupedRetrievalResult,
    RetrievalResult,
)
from legal_rag.retrieval.fusion import reciprocal_rank_fusion, rrf_merge_chunks
from legal_rag.types import RetrievedChunk


def chunk(cid: str, score: float = 0.0, **meta: Any) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=f"text {cid}", score=score, metadata=dict(meta))


def test_rrf_rewards_agreement_across_lists():
    a = [chunk("x"), chunk("y"), chunk("z")]
    b = [chunk("y"), chunk("x"), chunk("w")]
    fused = reciprocal_rank_fusion([a, b])
    assert {c.id for c in fused} == {"x", "y", "z", "w"}
    assert {fused[0].id, fused[1].id} == {"x", "y"}
    assert fused[0].score == fused[1].score
    assert fused[0].id == "x"  # tie keeps first-seen order


def test_rrf_scores_follow_the_formula_and_do_not_mutate_inputs():
    a = [chunk("x", 0.9)]
    fused = reciprocal_rank_fusion([a], k=60)
    assert abs(fused[0].score - 1 / 61) < 1e-12
    assert a[0].score == 0.9


def test_rrf_top_k_and_empty():
    assert reciprocal_rank_fusion([]) == []
    lists = [[chunk(str(i)) for i in range(5)]]
    assert len(reciprocal_rank_fusion(lists, top_k=2)) == 2


def test_rrf_merge_ranks_each_source_by_its_own_scores():
    mixed = [
        chunk("a1", 0.9, source="docs"),
        chunk("a2", 0.1, source="docs"),
        chunk("b1", 50.0, source="corpus"),
        chunk("b2", 40.0, source="corpus"),
    ]
    merged = rrf_merge_chunks(mixed)
    assert [c.id for c in merged[:2]] == ["a1", "b1"]  # raw scales do not matter
    assert rrf_merge_chunks([]) == []


class StubRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__(top_k=3)
        self.seen_filters: Mapping[str, Any] | None = None

    async def retrieve(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, Any] | None = None
    ) -> RetrievalResult:
        self.seen_filters = filters
        chunks = [chunk("1"), chunk("2")][: top_k or self.top_k]
        return RetrievalResult(query=query, chunks=chunks, total_found=len(chunks))

    async def retrieve_by_ids(
        self, chunk_ids: Sequence[str], *, filters: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return [chunk(i) for i in chunk_ids]


async def test_base_retriever_carries_generic_filters_only():
    r = StubRetriever()
    out = await r.retrieve("q", filters={"corpus": "statutes"})
    assert r.seen_filters == {"corpus": "statutes"}
    assert out.total_found == 2
    assert [c.id for c in await r.retrieve_by_ids(["9"])] == ["9"]
    fields = set(RetrievalResult.__dataclass_fields__)
    assert not {"user_id", "organization_id", "matter_id"} & fields


def test_grouped_result_flattens():
    g1 = ChunkGroup(group_id="d1", group_field="document_id", chunks=[chunk("1"), chunk("2")])
    g2 = ChunkGroup(group_id="d2", group_field="document_id", chunks=[chunk("3")])
    res = GroupedRetrievalResult(query="q", groups=[g1, g2])
    assert g1.chunk_count == 2
    assert [c.id for c in res.all_chunks] == ["1", "2", "3"]
