import pytest

from legal_rag.providers import (
    Cache,
    Embedder,
    LanguageModel,
    Reranker,
    RerankRateLimited,
    VectorStore,
)
from legal_rag.providers.fakes import (
    FakeCache,
    FakeEmbedder,
    FakeLanguageModel,
    FakeReranker,
    FakeVectorStore,
)
from legal_rag.types import Message, RetrievedChunk


def test_fakes_satisfy_protocols():
    assert isinstance(FakeLanguageModel(), LanguageModel)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeVectorStore(), VectorStore)
    assert isinstance(FakeCache(), Cache)


async def test_fakes_behave():
    lm = FakeLanguageModel(reply="ok")
    assert await lm.generate([Message("user", "hi")]) == "ok"
    emb = FakeEmbedder(dimension=4)
    assert len(await emb.embed_query("x")) == 4
    first = await emb.embed_query("x")
    assert first == await emb.embed_query("x")
    assert first != await emb.embed_query("y")
    assert await emb.embed_documents(["x", "y"]) == [first, await emb.embed_query("y")]
    ranked = await FakeReranker().rerank("tax law", ["tax rules", "other", "law of tax"])
    assert [(r.index, r.score) for r in ranked] == [(2, 2.0), (0, 1.0), (1, 0.0)]
    assert [
        r.index for r in await FakeReranker().rerank("tax law", ["a", "tax", "law tax"], top_k=1)
    ] == [2]
    store = FakeVectorStore([RetrievedChunk(id="1", text="a"), RetrievedChunk(id="2", text="b")])
    assert len(await store.search([0.1], top_k=1)) == 1
    cache = FakeCache()
    await cache.set("k", b"v")
    assert await cache.get("k") == b"v"


async def test_fake_cache_delete_removes_a_key_and_leaves_the_others():
    cache = FakeCache()
    await cache.set("a", b"1")
    await cache.set("b", b"2")
    await cache.delete("a")
    assert await cache.get("a") is None
    assert await cache.get("b") == b"2"


async def test_fake_cache_delete_of_a_missing_key_is_not_an_error():
    cache = FakeCache()
    await cache.delete("never-set")
    await cache.set("k", b"v")
    await cache.delete("k")
    await cache.delete("k")
    assert cache.data == {}


def test_a_cache_without_delete_does_not_satisfy_the_protocol():
    class GetSetOnly:
        async def get(self, key):
            return None

        async def set(self, key, value, *, ttl_seconds=None):
            return None

    assert not isinstance(GetSetOnly(), Cache)


def test_rerank_rate_limited_is_one_class_on_every_import_path():
    from legal_rag.providers.errors import RerankRateLimited as from_errors
    from legal_rag.retrieval.rerank import RerankRateLimited as from_rerank
    from legal_rag.retrieval.rerank.rate_limit import RerankRateLimited as from_rate_limit

    assert RerankRateLimited is from_errors is from_rerank is from_rate_limit


def test_rerank_rate_limited_carries_retry_after_and_a_message():
    bare = RerankRateLimited()
    assert bare.retry_after is None
    assert str(bare) == "Rerank rate limited"
    detailed = RerankRateLimited(2.5, "slow down")
    assert detailed.retry_after == 2.5
    assert str(detailed) == "slow down"
    assert isinstance(detailed, Exception)


async def test_voyage_adapter_raises_the_provider_level_error_on_a_429():
    from legal_rag.providers.adapters.voyage import VoyageReranker

    class Limited:
        async def rerank(self, *args, **kwargs):
            error = RuntimeError("slow down")
            error.http_status = 429  # type: ignore[attr-defined]
            error.headers = {"retry-after": "7"}  # type: ignore[attr-defined]
            raise error

    with pytest.raises(RerankRateLimited) as info:
        await VoyageReranker(client=Limited()).rerank("q", ["d"])
    assert info.value.retry_after == 7.0


def test_emit_usage_sets_the_provider_field_instead_of_burying_it_in_extra():
    from legal_rag.providers.usage import emit_usage

    seen = []
    emit_usage(seen.append, "embed", provider="acme", input_text="abcd", tag="x")
    (event,) = seen
    assert event.provider == "acme"
    assert dict(event.extra) == {"tag": "x"}
