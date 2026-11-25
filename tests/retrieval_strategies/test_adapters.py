from types import SimpleNamespace

import pytest

from legal_rag.providers import Embedder, Reranker
from legal_rag.providers.adapters import VoyageEmbedder, VoyageReranker
from legal_rag.retrieval.rerank import CrossEncoderReranker, RerankRateLimited
from legal_rag.types import RetrievedChunk


class FakeVoyageClient:
    def __init__(self) -> None:
        self.embed_calls: list[tuple[list[str], str, str]] = []

    async def embed(self, texts, model, input_type):
        self.embed_calls.append((list(texts), model, input_type))
        return SimpleNamespace(embeddings=[[float(len(t)), 1.0] for t in texts])

    async def rerank(self, query, documents, model, top_k=None, truncation=True):
        order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=i, relevance_score=len(documents[i]) / 100) for i in order
            ][:top_k]
        )


async def test_embedder_batches_and_sets_input_type():
    client = FakeVoyageClient()
    emb = VoyageEmbedder(client=client, batch_size=2, dimension=2)
    assert isinstance(emb, Embedder)
    vectors = await emb.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert [c[2] for c in client.embed_calls] == ["document", "document"]
    assert await emb.embed_query("four") == [4.0, 1.0]
    assert client.embed_calls[-1][2] == "query"


async def test_reranker_maps_results_and_plugs_into_cross_encoder():
    reranker = VoyageReranker(client=FakeVoyageClient())
    assert isinstance(reranker, Reranker)
    ranked = await reranker.rerank("q", ["a", "longer doc", "mid"], top_k=2)
    assert [r.index for r in ranked] == [1, 2]
    assert await reranker.rerank("q", []) == []
    chunks = [
        RetrievedChunk(id="1", text="short"),
        RetrievedChunk(id="2", text="a considerably longer chunk"),
    ]
    out = await CrossEncoderReranker(reranker, top_k=1).rerank("q", chunks)
    assert [c.id for c in out.chunks] == ["2"] and not out.degraded


async def test_reranker_translates_429():
    class Limited:
        async def rerank(self, *args, **kwargs):
            error = RuntimeError("slow down")
            error.http_status = 429  # type: ignore[attr-defined]
            error.headers = {"retry-after": "3"}  # type: ignore[attr-defined]
            raise error

    with pytest.raises(RerankRateLimited) as info:
        await VoyageReranker(client=Limited()).rerank("q", ["d"])
    assert info.value.retry_after == 3.0


def test_adapters_require_a_key_or_client():
    with pytest.raises(ValueError):
        VoyageReranker()
