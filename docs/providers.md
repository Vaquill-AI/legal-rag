# Providers

Every component in legal-rag receives its model, embedder, reranker, vector store and cache as arguments, and each one is a `typing.Protocol`, so you plug in your own client without subclassing anything.
Nothing in legal-rag reads settings, environment variables or secrets, so your code decides where keys come from and the library never reaches for them.

The protocols live in `legal_rag.providers.protocols`.
They are re-exported from `legal_rag.providers` together with `RerankRateLimited` and `UsageHook`.
All five are `runtime_checkable`, so `isinstance(obj, Embedder)` works, but it only checks that the members exist, not their signatures.
Every method is `async`.

## The protocols

Signatures are copied from `protocols.py`.
`T` is a `TypeVar` bound to `pydantic.BaseModel`.

| Protocol | Members | Its fake in `legal_rag.providers.fakes` |
| --- | --- | --- |
| `LanguageModel` | `async def generate(self, messages: Sequence[Message], *, max_tokens: int \| None = None, temperature: float \| None = None) -> str`<br>`async def generate_structured(self, messages: Sequence[Message], schema: type[T], *, max_tokens: int \| None = None) -> T` | `FakeLanguageModel(reply="", structured=None)` returns `reply` and records every call in `.calls`. `generate_structured` looks up `schema.__name__` in `structured` and raises `KeyError` when there is no canned instance. |
| `Embedder` | `dimension: int`<br>`async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]`<br>`async def embed_query(self, text: str) -> list[float]` | `FakeEmbedder(dimension=8)` hashes the text with SHA-256 into a zero-centred vector. The same text always gives the same vector, and it has no notion of meaning. |
| `Reranker` | `async def rerank(self, query: str, documents: Sequence[str], *, top_k: int \| None = None) -> list[RerankResult]` | `FakeReranker()` scores each document by how many lowercase query words it contains and sorts best first. A `top_k` of `None` or `0` means no limit. |
| `VectorStore` | `async def search(self, vector: Sequence[float], *, top_k: int, filters: Mapping[str, Any] \| None = None) -> list[RetrievedChunk]` | `FakeVectorStore(chunks=())` returns the first `top_k` of its fixed chunks and ignores the vector and the filters. |
| `Cache` | `async def get(self, key: str) -> bytes \| None`<br>`async def set(self, key: str, value: bytes, *, ttl_seconds: int \| None = None) -> None`<br>`async def delete(self, key: str) -> None` | `FakeCache()` keeps a dict in `.data` and ignores `ttl_seconds`. |
| `UsageHook` | `Callable[[UsageEvent], None]`, a plain synchronous callable and not a protocol class. | None. Pass `events.append` or any function. |

Two contract details to honour in your own adapters.
`filters` on `VectorStore.search` is an opaque mapping that only your store has to understand.
`Cache.delete` of a key that is not present must not raise.
A cache that lacks `delete` does not satisfy `Cache`.

## Writing an adapter

An adapter is any class with the right members.
This example writes a toy `Embedder` and a toy `LanguageModel` with no network, checks them against the protocols, and passes them to two components: `HyDEGenerator(llm, config=None, *, usage_hook=None)` and `SemanticCache(embedder, cache, config=None)`.

```python
import asyncio

from legal_rag.providers import Embedder, LanguageModel
from legal_rag.providers.fakes import FakeCache
from legal_rag.retrieval.caching.semantic import SemanticCache
from legal_rag.retrieval.strategies.hyde import HyDEGenerator


class LetterEmbedder:
    """Toy embedder: a normalised letter histogram. Similar spellings give similar vectors."""

    dimension = 26

    def _vec(self, text: str) -> list[float]:
        counts = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                counts[ord(ch) - ord("a")] += 1.0
        total = sum(counts) or 1.0
        return [c / total for c in counts]

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        return self._vec(text)


class CannedModel:
    """Toy language model: no network, one fixed answer."""

    async def generate(self, messages, *, max_tokens=None, temperature=None) -> str:
        return (
            "A claim on a written contract must generally be brought within the period "
            "that the governing statute of limitations sets for contract actions."
        )

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        raise NotImplementedError("this toy model only answers free text")


async def main() -> None:
    embedder, model = LetterEmbedder(), CannedModel()
    print("embedder ok:", isinstance(embedder, Embedder), "| model ok:", isinstance(model, LanguageModel))

    hyde = HyDEGenerator(model)
    result = await hyde.generate("limitation period for a contract claim")
    print("hyde success:", result.success)
    print("hypothesis:", result.primary_hypothesis[:60] + "...")

    cache = SemanticCache(embedder, FakeCache())
    await cache.set("statute of limitations for contract claims", {"answer": "see the statute"})
    print("similar query hit:", await cache.get("statute of limitations for contract claim"))
    print("unrelated query hit:", await cache.get("zzz qqq xxx"))


asyncio.run(main())
```

Output:

```text
embedder ok: True | model ok: True
hyde success: True
hypothesis: A claim on a written contract must generally be brought with...
similar query hit: {'answer': 'see the statute'}
unrelated query hit: None
```

A `Reranker` adapter is shown in the Errors section below.
A `Cache` or a `VectorStore` is one class with the members in the table, so a Redis-backed cache or a hosted vector index is a thin wrapper over your own client.
Return `RetrievedChunk` objects from `VectorStore.search`, with `score` set by your store.

## Errors

`legal_rag.providers.errors` holds the errors that adapters raise, so callers can catch them without importing retrieval.
It has one class today: `RerankRateLimited(retry_after: float | None = None, detail: str = "")`.
`retry_after` is in seconds and is `None` when the provider did not say.
`str(error)` is `detail`, or `"Rerank rate limited"` when `detail` is empty.
The same class is importable from `legal_rag.providers`, `legal_rag.providers.errors` and `legal_rag.retrieval.rerank`, and the tests pin that they are one object.

A reranker adapter should raise `RerankRateLimited` when the provider answers 429, and should not swallow the error or return an empty list.
`CrossEncoderReranker` handles it and returns a `RerankOutcome` with `degraded=True` and a `degraded_reason`, so a degraded rerank stays visible to the caller instead of passing for a real ranking.

```python
import asyncio

from legal_rag.providers import Reranker, RerankRateLimited
from legal_rag.retrieval.rerank import CrossEncoderReranker
from legal_rag.types import RetrievedChunk


class SlowDownReranker:
    """Toy reranker whose provider always answers 429."""

    async def rerank(self, query, documents, *, top_k=None):
        raise RerankRateLimited(retry_after=0.01, detail="provider said 429")


async def main() -> None:
    provider = SlowDownReranker()
    print("reranker ok:", isinstance(provider, Reranker))
    chunks = [RetrievedChunk("1", "tax rules"), RetrievedChunk("2", "other"), RetrievedChunk("3", "law of tax")]
    outcome = await CrossEncoderReranker(provider, top_k=2).rerank("tax law", chunks)
    print("degraded:", outcome.degraded)
    print("reason:", outcome.degraded_reason)
    print("kept:", [c.id for c in outcome.chunks])


asyncio.run(main())
```

Output:

```text
reranker ok: True
degraded: True
reason: rate_limited
kept: ['1', '2']
```

With no `fallback` supplied, this run kept the first two chunks in their input order.
`CrossEncoderReranker` also accepts `fallback`, `rate_limiter`, `usage_hook`, `default_wait_seconds` and `provider_name`.
Check `outcome.degraded` before you cache a ranking.

## Usage events

`legal_rag.providers.usage` has two helpers.
`estimate_tokens(text)` returns `max(1, ceil(len(text) / 4))`, a rough count that is good enough for metering.
`emit_usage(hook, operation, *, provider=None, input_text="", output_text="", **extra)` calls the hook once with a `UsageEvent` and returns `None`.
Token counts are estimated from the text lengths and stay `None` for an empty text, `provider` sets `UsageEvent.provider`, and every other keyword goes into `UsageEvent.extra`.
When `hook` is `None` it does nothing, and an exception raised inside the hook is swallowed so metering never breaks the pipeline.
Components that take a `usage_hook` argument, such as `HyDEGenerator`, call it for you.

```python
import asyncio

from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.providers.usage import emit_usage, estimate_tokens
from legal_rag.retrieval.strategies.hyde import HyDEGenerator
from legal_rag.types import UsageEvent

events: list[UsageEvent] = []

emit_usage(events.append, "demo.generate", input_text="x" * 40, output_text="y" * 10, model="toy")
emit_usage(events.append, "demo.embed", input_text="hello")
emit_usage(None, "demo.ignored", input_text="no hook, nothing happens")


def broken(event: UsageEvent) -> None:
    raise RuntimeError("the meter is down")


emit_usage(broken, "demo.safe", input_text="still returns")

# A component that takes a usage_hook calls it for you.
reply = "A hypothetical passage about the limitation period for contract claims, long enough to keep."
hyde = HyDEGenerator(FakeLanguageModel(reply=reply), usage_hook=events.append)
asyncio.run(hyde.generate("limitation period for a contract claim"))

for event in events:
    print(event)
print("estimate_tokens:", estimate_tokens(""), estimate_tokens("abcd"), estimate_tokens("abcde"))
```

Output:

```text
UsageEvent(operation='demo.generate', provider=None, input_tokens=10, output_tokens=3, extra={'model': 'toy'})
UsageEvent(operation='demo.embed', provider=None, input_tokens=2, output_tokens=None, extra={})
UsageEvent(operation='hyde_generation', provider=None, input_tokens=363, output_tokens=23, extra={})
estimate_tokens: 1 1 2
```

Pass `provider="name"` to `emit_usage` to set `UsageEvent.provider`, which the bundled components leave as `None` because they do not know which backend you plugged in.

## The Voyage adapter

`VoyageEmbedder` and `VoyageReranker` in `legal_rag.providers.adapters` are thin adapters for the `Embedder` and `Reranker` protocols.
Install the `voyage` extra, which pulls in `voyageai>=0.3`: `pip install "legal-rag[voyage]"`.
The `voyageai` package is imported lazily, only when you do not pass a `client`.
There are no environment reads, circuit breakers, retries or caches in the adapter, so wrap it with your own if you need them.

The snippet below was not run, because it needs a real key and network access.

```python
from legal_rag.providers.adapters import VoyageEmbedder, VoyageReranker

api_key = "YOUR_VOYAGE_KEY"  # your code supplies the key, legal-rag never reads it from the environment

embedder = VoyageEmbedder(api_key, model="voyage-3-large", dimension=1024, batch_size=128)
reranker = VoyageReranker(api_key, model="rerank-2.5", truncation=True)
```

| Class | Parameters (after `api_key`, all keyword-only) | Defaults |
| --- | --- | --- |
| `VoyageEmbedder` | `client`, `model`, `dimension`, `batch_size` | `None`, `"voyage-3-large"`, `1024`, `128` |
| `VoyageReranker` | `client`, `model`, `truncation` | `None`, `"rerank-2.5"`, `True` |

`api_key` defaults to `None`, and passing neither `api_key` nor `client` raises `ValueError("Pass api_key or client")`.
`client` may be an async Voyage client or any object with the same `embed` and `rerank` coroutines, which is how you inject a configured client or a test double.
`dimension` is only the value reported as `embedder.dimension`, since it is not sent to the API.
`embed_documents` sends texts in batches of `batch_size` with `input_type="document"`, `embed_query` sends `input_type="query"`, and `batch_size` below 1 is raised to 1.
`VoyageReranker` returns `[]` for an empty document list and turns a 429 into `RerankRateLimited`, detected by `http_status` or `status_code` equal to 429, or an exception class named `RateLimitError`.
It reads `retry-after` from the error's `headers`, or from `error.response.headers`, and any other exception is re-raised unchanged.

## Shared data types

`Message`, `RerankResult`, `RetrievedChunk` and `UsageEvent` are defined in `legal_rag.types` and re-exported from `legal_rag`.
They carry no tenancy keys and no product fields, so anything deployment-specific goes in `metadata` or `extra`.

| Type | Fields | Mutability |
| --- | --- | --- |
| `Message` | `role: str`, `content: str` | frozen dataclass |
| `RetrievedChunk` | `id: str`, `text: str`, `score: float = 0.0`, `metadata: dict[str, Any] = {}` | mutable dataclass |
| `RerankResult` | `index: int` (position in the input list), `score: float` | frozen dataclass |
| `UsageEvent` | `operation: str`, `provider: str \| None = None`, `input_tokens: int \| None = None`, `output_tokens: int \| None = None`, `extra: Mapping[str, Any] = {}` | frozen dataclass |

`Message.role` is a plain string, and the types module does not restrict its values.

## Tests

The provider tests are in `tests/test_providers.py`.
They check that each fake satisfies its protocol, that the fakes behave as described above, that `FakeCache.delete` tolerates a missing key, that a cache without `delete` is rejected, that `RerankRateLimited` is one class on every import path, and that `VoyageReranker` turns a 429 into `RerankRateLimited` with `retry_after` read from the headers.
Run them with `python -m pytest tests/test_providers.py`.
