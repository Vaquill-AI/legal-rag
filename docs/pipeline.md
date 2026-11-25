# Pipeline

The pipeline component turns a question into a numbered, token-budgeted context with sources, and turns a document into embedded, written chunks.
`QueryPipeline` runs `analyze`, `retrieve`, `rerank` and `context` behind a two-layer cache, `IngestPipeline` runs extract to write, and both are plain async code with no graph engine and no tenant concept.

Every snippet on this page runs as written against the in-memory fakes in `legal_rag.providers.fakes`.
Swap the fakes for your own `Retriever`, `Reranker`, `Embedder`, `LanguageModel` and `Cache`.

## The query flow

```text
PipelineRequest
  |
  v
resolve tier       request.tier, else tier_resolver(request), else DEFAULT_TIER ("standard")
resolve strategy   request.retrieval_strategy, else the tier's retrieval_strategy
build cache keys   AnswerScope.from_request (raises on a missing or unknown field)
  |                an unknown tier or strategy raises here, before any cache or provider is used
  v
semantic cache  ---- hit ----> QueryResult(cache_status="semantic_hit")
  | miss
  v
exact cache     ---- hit ----> QueryResult(cache_status="exact_hit")
  | miss
  v
analyze -> retrieve -> rerank -> context
              |
              +-- no results and the strategy was not hybrid: run retrieve, rerank and context
                  once more with the strategy forced to hybrid
  |
  v
store in both caches, only if there are sources and no degraded warning
  |
  v
QueryResult
```

A cache hit returns before any stage runs.
`cache_status` is `miss`, `semantic_hit`, `exact_hit` or `disabled` (no cache configured).
`on_event` receives `pipeline_started`, `cache_hit`, `stage_completed`, `retry`, `cache_stored` and `pipeline_completed`, and a sink that raises is ignored.

## The stages

Each stage is `async def stage(ctx, **collaborators) -> ctx` and is importable on its own (`analyze_stage`, `retrieve_stage`, `rerank_stage`, `context_stage`).
Each is fail-open on a runtime error and raises on a configuration error.

| Stage | Reads | Writes | On a runtime failure |
| --- | --- | --- | --- |
| analyze | `query`, `chat_history`, `pinned_ids`, `scope_lock_ids`, tier flags `use_query_condensation` and `use_query_expansion` | the `extra` keys in `ANALYZE_EXTRA_KEYS` (`query_analysis`, `condensed_query`, `condensed_sub_queries`, `query_expansion`, `requires_retrieval`, ...) and `metrics["analyze"]` | returns the input context with `WARNING_ANALYZE_FAILED` added and logs |
| retrieve | `tier`, `retrieval_strategy`, `filters`, the scope ids, and from `extra`: `condensed_query`, `condensed_sub_queries`, `query_analysis`, `query_expansion`, `requires_retrieval`, `force_retrieval_strategy` | `retrieved`, `retrieval_warning` (`no_results` or `limited_results`), `metrics["retrieval"]` | returns the input context with the warning `retrieval_failed`; an unknown strategy raises `UnknownStrategyError`, an unknown tier raises `ValueError` |
| rerank | `retrieved`, `pinned_ids`, `scope_lock_ids`, `enable_shared_corpus`, `filters`, and from `extra` or `metrics`: `condensed_query`, `query_analysis`, `requested_count` | `reranked`, `extra["alternative_chunks"]`, `metrics["rerank"]`, `metrics["crag_filter"]` | returns the retrieved chunks unranked with `rerank_failed`; a fallback reranker standing in adds `rerank_degraded` |
| context | `reranked` (else `retrieved`), `pinned_ids`, `filters` (forwarded to your chunk fetcher as its scope), the query complexity | `context`, `context_chunks`, `sources`, `retrieval_warning`, `metrics` keys `context_budget`, `context_tokens`, `neighbors`, `source_counts`, `coverage_notice`, `confidence`, `pipeline_confidence` | builds a plain numbered context from the chunks it was given and adds `context_assembly_failed` |

`neighbor_expansion_failed` and `source_enrichment_failed` are also degraded warnings.
`run_stages(ctx, stages)` is the loop `QueryPipeline` uses: it runs `(name, stage)` pairs in order and raises `TypeError` if a stage does not return a `QueryContext`.

## Tiers

A tier is data in `pipeline/tiers.default.json`, so tuning is a data change.
The tier names no model, provider or price.

| | standard | deep |
| --- | --- | --- |
| `retrieval_strategy` | `self_rag` | `multi_hop` |
| `top_k_retrieval` / `top_k_rerank` | 100 / 10 | 200 / 20 |
| `rerank_input_cap` (0 means no cap) | 0 | 200 |
| `rerank_score_threshold` / `rerank_min_chunks` | 0.25 / 3 | 0.25 / 5 |
| `neighbor_chunks` | 2 | 3 |
| `max_multi_hop_iterations` | 3 | 5 |
| retrieval depth for a `simple` / `moderate` query | 20 / 50 | 50 / 100 |
| extra features | none | HyDE, multi-query, multi-hop, query decomposer, iterative retrieval and others |

The effective depth is `min(tier depth, complexity cap)`, so a simple question on `standard` asks the retriever for 20 chunks, not 100.
A cap of 0 on a `trivial` query skips retrieval, except when the request pins documents or locks the scope.
The strategies are `hybrid`, `self_rag`, `crag`, `hyde`, `multi_query` and `multi_hop`.
Without an `llm`, every strategy except `hybrid` and `multi_hop` runs as `hybrid` and says so in its metadata.
Flags such as `use_self_rag` or `use_hallucination_detection` are labels that no stage reads directly.

Precedence is `request.tier`, then `tier_resolver(request)` (sync or async), then `DEFAULT_TIER`.
A resolver that raises falls back to the default tier and adds the warning `tier_resolution_failed`.
A tier name that is not in the table raises `ValueError`.
Your own tier data has the shape `{"tiers": {name: {...}}}`, every scalar key is required, and an unknown key raises `ValueError`.
Pass it as `QueryPipelineConfig(tiers=...)`, as a dict or a path to a JSON file.
A path is re-read on every call, so read a file once with `json.load` and pass the dict.
Tier names ignore case: keys are lowercased when the table loads, and two keys that differ only in case raise `ValueError`.

```python
import asyncio

from legal_rag.pipeline import (
    PipelineRequest,
    QueryPipeline,
    QueryPipelineConfig,
    VectorStoreRetriever,
    available_tiers,
    get_tier,
)
from legal_rag.providers.fakes import FakeEmbedder, FakeReranker, FakeVectorStore
from legal_rag.types import RetrievedChunk

for name in available_tiers():
    t = get_tier(name)
    print(name, t.retrieval_strategy, t.top_k_retrieval, t.top_k_rerank, t.neighbor_chunks, t.max_multi_hop_iterations)

MY_TIERS = {"tiers": {"fast": {
    "description": "Shallow and cheap", "retrieval_strategy": "hybrid",
    "top_k_retrieval": 20, "top_k_rerank": 5, "rerank_input_cap": 0, "rerank_min_chunks": 2,
    "rerank_score_threshold": 0.0, "neighbor_chunks": 0, "max_multi_hop_iterations": 1,
    "max_chunks_per_document": 3, "target_latency_ms": 1000, "quality_target": 0.8,
    "features": {"use_reranking": True},
}}}
store = FakeVectorStore([RetrievedChunk("c1", "Written notice is required before an eviction.", 0.9)])
index = VectorStoreRetriever(FakeEmbedder(), store)


def broken_resolver(request: PipelineRequest) -> str:
    raise RuntimeError("plan lookup is down")


async def main() -> None:
    ask = PipelineRequest("is written notice required before an eviction")
    pipe = QueryPipeline(retriever=index, reranker=FakeReranker())
    print((await pipe.run(ask)).tier)                                                      # default
    print((await pipe.run(ask, tier_resolver=lambda r: "deep")).tier)                      # resolver
    print((await pipe.run(PipelineRequest(ask.query, tier="standard"), tier_resolver=lambda r: "deep")).tier)  # request wins
    fallback = await pipe.run(ask, tier_resolver=broken_resolver)
    print(fallback.tier, fallback.warnings)
    try:
        await pipe.run(PipelineRequest(ask.query, tier="enterprise"))
    except ValueError as error:
        print("raised:", str(error)[:42])

    mine = QueryPipeline(retriever=index, reranker=FakeReranker(), config=QueryPipelineConfig(tiers=MY_TIERS))
    result = await mine.run(PipelineRequest(ask.query, tier="fast"))
    print(result.tier, result.strategy)


asyncio.run(main())
```

```text
standard self_rag 100 10 2 3
deep multi_hop 200 20 3 5
standard
deep
standard
standard ['tier_resolution_failed']
raised: Unknown tier 'enterprise'. Available tiers
fast hybrid
```

## Quick start

This builds a `QueryPipeline` with fakes and the US plug-in, asks the same question twice, and shows the miss and both kinds of hit.
`examples/pipeline_basic.py` extends it with answer generation and a grounding check.

```python
import asyncio

from legal_rag.pipeline import PipelineRequest, QueryPipeline, VectorStoreRetriever, jurisdictions
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeReranker, FakeVectorStore
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.types import RetrievedChunk

CHUNKS = [
    RetrievedChunk("s4", "A landlord must give a tenant written notice of fourteen days before an eviction.", 0.8, {"title": "Lease Act, section 4"}),
    RetrievedChunk("s9", "The written notice must state the amount owed and the date it is due.", 0.7, {"title": "Lease Act, section 9"}),
]


async def main() -> None:
    index = VectorStoreRetriever(FakeEmbedder(), FakeVectorStore(CHUNKS))
    pipeline = QueryPipeline(
        retriever=index,
        reranker=FakeReranker(),
        jurisdiction=jurisdictions.get("us", statute_retriever=index),
        semantic_cache=SemanticCache(FakeEmbedder(), FakeCache()),  # layer 1
        cache=FakeCache(),  # layer 2
    )
    request = PipelineRequest("what written notice must a landlord give before an eviction", filters={"country_code": "US"})
    first = await pipeline.run(request)
    print(first.cache_status, first.tier, first.strategy, [s["metadata"]["title"] for s in first.sources])
    second = await pipeline.run(request)
    print(second.cache_status, second.cache_hit)

    exact_only = QueryPipeline(retriever=index, reranker=FakeReranker(), cache=FakeCache())
    await exact_only.run(request)
    print((await exact_only.run(request)).cache_status)


asyncio.run(main())
```

```text
miss standard hybrid ['Lease Act, section 4', 'Lease Act, section 9']
semantic_hit True
exact_hit
```

`strategy` is `hybrid` although `standard` asks for `self_rag`, because no `llm` was given (see the known limits).
The semantic layer is checked first, so with both layers configured a repeat is a `semantic_hit`.
Configure only `cache=` to get exact hits.

## The two-layer cache and AnswerScope

Layer 1 is `SemanticCache` (a paraphrase of an earlier question hits).
Layer 2 is an exact cache keyed by the SHA-256 of the whitespace-normalized query and the scope, with `exact_cache_ttl_seconds` (default 3600).
Case is kept in the key because a citation or a name can depend on it.
Both layers use the same `AnswerScope` plus `QueryPipelineConfig.namespace`, so nothing that can change the answer is missing from either key.

`ANSWER_SCOPE_FIELDS` is a closed list: `tier`, `retrieval_strategy`, `enable_shared_corpus`, `collection`, `pinned_ids`, `scope_lock_ids`, `boost_ids`, `filters` and `chat_history` (the last six turns, as a digest).
An ignored filter is a wrong answer served from cache, so `AnswerScope.build` raises `TypeError` for a field that is missing and for a field that is not on the list.
A test fails when a new `PipelineRequest` field is added without being registered or exempted.
`None`, blank strings and empty containers normalize to "absent", so an unset field and its default share a key.
`request.extra` and `query` are not in the scope: put every answer-changing input in `filters` or a scope field.
A result is cached only when it has sources and no degraded warning, so a transient failure is never replayed.

```python
from legal_rag.pipeline import ANSWER_SCOPE_FIELDS, AnswerScope, PipelineRequest, build_cache_keys

print(ANSWER_SCOPE_FIELDS)


def keys(**request_fields):
    request = PipelineRequest("notice period", **request_fields)
    return build_cache_keys("demo", request, tier="standard", retrieval_strategy="hybrid")


print("filter reaches both keys:", keys(filters={"state": "NY"}).exact_key != keys(filters={"state": "TX"}).exact_key)
print("blank filter is unset   :", keys(filters={"state": " "}).exact_key == keys().exact_key)

everything = dict.fromkeys(ANSWER_SCOPE_FIELDS)  # None is allowed for every field
AnswerScope.build(**everything)
try:
    AnswerScope.build(**{k: v for k, v in everything.items() if k != "filters"})
except TypeError as error:
    print(error)
try:
    AnswerScope.build(**everything, court_level="appellate")
except TypeError as error:
    print(error)
```

```text
('tier', 'retrieval_strategy', 'enable_shared_corpus', 'collection', 'pinned_ids', 'scope_lock_ids', 'boost_ids', 'filters', 'chat_history')
filter reaches both keys: True
blank filter is unset   : True
AnswerScope.build: missing=['filters'] unknown=[]
AnswerScope.build: missing=[] unknown=['court_level']
```

## Tenancy is the caller's job

The library has no user, organization or project concept.
`request.filters` is opaque: it reaches your `Retriever` and your `ChunkFetcher` unchanged and is part of both cache keys.
Put the tenant in `filters`, apply it in your retriever, and give the shared cache store a `namespace` that names the deployment or corpus version.
The chunk fetcher receives `scope` (the filters as sorted JSON) and, if `fetch_by_ids` declares a `filters` keyword, the filters mapping, so it can enforce the tenant too.
The ids in `pinned_ids`, `scope_lock_ids`, `boost_ids` and `collection` reach your retriever only if you name filter keys for them in `RetrieveConfig(scope_keys=ScopeFilterKeys(...))`.
On the ingest side the same idea is `payload_extra` (see below).

```python
import asyncio

from legal_rag.pipeline import PipelineRequest, QueryPipeline, QueryPipelineConfig
from legal_rag.providers.fakes import FakeCache, FakeReranker
from legal_rag.types import RetrievedChunk

CORPUS = [
    RetrievedChunk("a1", "Tenant A memo: written notice is fourteen days.", 0.9, {"tenant_id": "a"}),
    RetrievedChunk("b1", "Tenant B memo: written notice is thirty days.", 0.9, {"tenant_id": "b"}),
]


class TenantIndex:
    """The pipeline never filters by tenant. Your retriever does."""

    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        return [c for c in CORPUS if c.metadata["tenant_id"] == (filters or {}).get("tenant_id")][:top_k]


async def main() -> None:
    pipeline = QueryPipeline(
        retriever=TenantIndex(),
        reranker=FakeReranker(),
        cache=FakeCache(),  # one store shared by every tenant
        config=QueryPipelineConfig(namespace="deploy-1-corpus-7"),
    )
    for tenant in ("a", "b", "a"):
        result = await pipeline.run(PipelineRequest("how many days of written notice", filters={"tenant_id": tenant}))
        print(tenant, result.cache_status, "|", result.chunks[0].text)

    # request.extra is not part of the key, so a tenant id there would NOT separate the caches.
    keys = [pipeline.cache_keys(PipelineRequest("q", extra=e), tier="standard", retrieval_strategy="hybrid") for e in ({}, {"tenant_id": "b"})]
    print("extra changes the key:", keys[0].exact_key != keys[1].exact_key)


asyncio.run(main())
```

```text
a miss | Tenant A memo: written notice is fourteen days.
b miss | Tenant B memo: written notice is thirty days.
a exact_hit | Tenant A memo: written notice is fourteen days.
extra changes the key: False
```

## IngestPipeline

`IngestPipeline` runs extract, clean, classify, chunk, optional contextualize (needs an `llm`), optional injection scan (needs a `scanner`), embed in batches, and write in batches with rollback.
`ingest(source, document_id=..., metadata=...)` takes bytes, a path, text (a plain `str` is text, not a path) or an `IngestorResult`.
`payload_extra` is copied into every point last, so per-document metadata can never overwrite it, and keys in `RESERVED_PAYLOAD_KEYS` are refused.
`on_status(stage, message)` receives the names in `INGEST_STAGES`, and a failing callback never fails the ingest.

| Situation | Outcome |
| --- | --- |
| parser failed, unreadable text, chunker failed, scanner rejected the document | returns `IngestResult(success=False, status="failed", error=...)` |
| empty document | returns `success=True` with `status="no_text_content"` and zero chunks |
| embedder failed after retries or returned the wrong count | raises `IngestEmbeddingError` |
| vector of the wrong dimension | raises `EmbeddingDimensionError` |
| a write batch failed | raises `IngestWriteError` after deleting what was written (`rolled_back`, `orphaned_ids`) |

With a scanner, flagged chunks are dropped and counted in `chunks_excluded`; `IngestOptions(strict_scan=True)` rejects the whole document instead.
Re-ingesting a `document_id` is the caller's to manage: delete the old points first.

```python
import asyncio

from legal_rag.ingestion.base import IngestorResult
from legal_rag.pipeline import IngestEmbeddingError, IngestOptions, IngestPipeline, IngestWriteError, ScanResult
from legal_rag.providers.fakes import FakeEmbedder

TEXT = "\n\n".join(
    f"Section {n}. Notice rule {n}.\n\nA landlord must give written notice of {n + 10} days before "
    f"an eviction in case type {n}. The notice must state the amount owed and the date it is due. " * 6
    for n in range(1, 21)
)


class MemoryWriter:
    """A VectorWriter: upsert and delete. It can fail on a chosen batch."""

    def __init__(self, fail_on_batch=None):
        self.points, self.fail_on_batch, self.batches = {}, fail_on_batch, 0

    async def upsert(self, points):
        self.batches += 1
        if self.batches == self.fail_on_batch:
            raise RuntimeError("index is down")
        self.points.update({p.id: p for p in points})

    async def delete(self, ids):
        for i in ids:
            self.points.pop(i, None)


class DownEmbedder(FakeEmbedder):
    async def embed_documents(self, texts):
        raise RuntimeError("embedding service is down")


class RejectEverything:
    def scan(self, text):
        return ScanResult(safe=False, findings=("instruction_override",))


async def locked_pdf(content, filename, document_id):
    return IngestorResult(document_id=document_id, filename=filename, content="", success=False, error="encrypted PDF")


async def main() -> None:
    once = IngestOptions(retry_attempts=1)
    writer = MemoryWriter()
    stages = []

    async def on_status(stage, message):
        stages.append(stage)

    ok = IngestPipeline(embedder=FakeEmbedder(), writer=writer, payload_extra={"tenant_id": "a"}, on_status=on_status, options=once)
    result = await ok.ingest(TEXT, document_id="lease-act", metadata={"title": "Lease Act"})
    point = next(iter(writer.points.values()))
    print(result.success, result.status, result.chunks_stored, point.payload["tenant_id"], point.payload["title"])
    print(stages)
    print((await ok.ingest("   ", document_id="blank")).status)  # success, zero chunks

    # A problem with the document RETURNS success=False.
    parser = IngestPipeline(embedder=FakeEmbedder(), writer=MemoryWriter(), extractor=locked_pdf)
    print((await parser.ingest(b"%PDF", document_id="d1")).error)
    scanner = IngestPipeline(embedder=FakeEmbedder(), writer=MemoryWriter(), scanner=RejectEverything())
    print((await scanner.ingest(TEXT, document_id="d2")).error)

    # A problem with a provider RAISES.
    try:
        await IngestPipeline(embedder=DownEmbedder(), writer=MemoryWriter(), options=once).ingest(TEXT, document_id="d3")
    except IngestEmbeddingError as error:
        print(type(error).__name__, error)
    flaky = MemoryWriter(fail_on_batch=2)
    small_batches = IngestOptions(write_batch_size=1, write_concurrency=1, retry_attempts=1)
    try:
        await IngestPipeline(embedder=FakeEmbedder(), writer=flaky, options=small_batches).ingest(TEXT, document_id="d4")
    except IngestWriteError as error:
        print(type(error).__name__, "rolled_back =", error.rolled_back, "points left =", len(flaky.points))


asyncio.run(main())
```

```text
True completed 20 a Lease Act
['extracting', 'extracted', 'chunked', 'embedded', 'stored', 'completed']
no_text_content
encrypted PDF
All 20 chunks were flagged by the injection scanner
IngestEmbeddingError Embedding failed: embedding service is down
IngestWriteError rolled_back = True points left = 0
```

## Writing a JurisdictionPlugin

A plug-in implements the `JurisdictionPlugin` protocol: `condenser_key`, `matches(filters)`, async `retrieve(...)`, `adjusters()`, `exact_lookups()` and `label_for(chunk)`.
Register a factory by name in `legal_rag.pipeline.jurisdictions` (`register`, `get`, `available`, `unregister`).
`register` rejects a taken name unless `replace=True`, `get` raises `UnknownJurisdictionError` for an unknown name and `TypeError` if the factory does not return a plug-in.
`QueryPipeline(jurisdiction=plugin)` applies the plug-in to a request only when `plugin.matches(request.filters)` is true.
For such a request the plug-in's retriever replaces the main one, its exact lookups run first, its adjusters run after the reranker and its `label_for` names the sources.
The built-in `"us"` plug-in matches `country_code` or `jurisdiction` set to a US value and takes injected case, statute and lookup backends.

```python
import asyncio
from dataclasses import replace

from legal_rag.pipeline import PINNED_BY_KEY, PINNED_KEY, PipelineRequest, QueryPipeline, VectorStoreRetriever, jurisdictions
from legal_rag.providers.fakes import FakeEmbedder, FakeReranker, FakeVectorStore
from legal_rag.types import RetrievedChunk


class SectionLookup:
    """An ExactLookup: the query names an authority, so pin it."""

    async def lookup(self, query):
        if "section 12" not in query.lower():
            return []
        return [RetrievedChunk("s12", "Definitions apply to this Act.", 0.5, {"cite": "s. 12"})]


class DemoRegion:
    """A JurisdictionPlugin for an imaginary region."""

    condenser_key = "DEMO"

    def __init__(self, source):
        self._source = source

    def matches(self, filters):
        return str(filters.get("jurisdiction", "")).lower() == "demo"

    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        return await self._source.retrieve(query, top_k=top_k, filters=filters)

    def adjusters(self):
        def demote_repealed(query, chunks, filters):
            return [replace(c, score=c.score * 0.1) if c.metadata.get("status") == "repealed" else c for c in chunks]

        return [demote_repealed]

    def exact_lookups(self):
        return [SectionLookup()]

    def label_for(self, chunk):
        return f"DEMO {chunk.metadata.get('cite', chunk.id)}"


def index():
    chunks = [
        RetrievedChunk("old", "Written notice of seven days is required before an eviction.", 0.9, {"cite": "s. 2", "status": "repealed"}),
        RetrievedChunk("new", "Written notice of fourteen days is required before an eviction.", 0.8, {"cite": "s. 3"}),
    ]
    return VectorStoreRetriever(FakeEmbedder(), FakeVectorStore(chunks))


async def main() -> None:
    jurisdictions.register("demo", lambda *, source: DemoRegion(source))
    print(jurisdictions.available())
    plugin = jurisdictions.get("demo", source=index())
    pipeline = QueryPipeline(retriever=index(), reranker=FakeReranker(), jurisdiction=plugin)
    result = await pipeline.run(PipelineRequest("written notice before an eviction, and section 12", filters={"jurisdiction": "demo"}))
    print(result.context.splitlines()[1:4])
    for c in result.chunks:
        m = c.metadata
        print(c.id, "score", round(c.score, 2), "rerank_score", m.get("rerank_score"), "pinned", m.get(PINNED_KEY, False), m.get(PINNED_BY_KEY))


asyncio.run(main())
```

```text
('demo', 'us')
['[1] DEMO s. 12', '[2] DEMO s. 3', '[3] DEMO s. 2']
s12 score 0.5 rerank_score None pinned True exact_lookup
new score 4.0 rerank_score 4.0 pinned False None
old score 0.4 rerank_score 4.0 pinned False None
```

## Pinned chunks and the raw rerank score

`PINNED_KEY` (`"is_pinned"`) is the one chunk-metadata flag every stage and plug-in reads and writes.
A pinned chunk skips the reranker, is exempt from the relevance gate and the diversity caps, leads the result and is reserved first in the context budget.
Exact lookups pin their chunks with `PINNED_KEY` and `PINNED_BY_KEY` (`"pinned_by"`, diagnostics only), as `s12` shows above.
The rerank stage snapshots the raw reranker score in `metadata["rerank_score"]` before any adjuster runs, and the relevance gate reads that snapshot.
`old` above was demoted to a score of 0.4 but still passed the gate on its raw 4.0, so a demotion can never become a deletion.

## Known limits

- `QueryResult.strategy` is `hybrid` when an adaptive strategy ran without a model, while the cache keys use the strategy that was requested. Read `metrics["retrieval"]["strategy_metadata"]` for `degraded_from`.
- A matching plug-in replaces a single main retriever, so the plug-in must reach every source you want. Pass `retriever` as a mapping of named retrievers and the plug-in joins it as the source `jurisdiction` instead.
- The tier threshold 0.25 is calibrated to a 0 to 1 cross-encoder. Set `RerankStageConfig.relevance_threshold` for another scorer. When a fallback reranker stands in, the gate is off unless `fallback_relevance_threshold` is set.
- A failed `analyze` stage adds `WARNING_ANALYZE_FAILED`, which counts as degraded, so the answer is not cached. The query then runs unchanged, without condensation or expansion.
- A degraded or empty result is returned but never cached.

```python
import asyncio

from legal_rag.pipeline import PipelineRequest, QueryPipeline, UnknownStrategyError, VectorStoreRetriever
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeReranker, FakeVectorStore
from legal_rag.types import RetrievedChunk


class DownIndex:
    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        raise ConnectionError("index unreachable")


async def main() -> None:
    ask = PipelineRequest("written notice before an eviction")
    index = VectorStoreRetriever(FakeEmbedder(), FakeVectorStore([RetrievedChunk("s4", "Written notice is required.", 0.8)]))

    # Standard asks for self_rag, which needs a model. With no llm it runs hybrid and says so.
    result = await QueryPipeline(retriever=index, reranker=FakeReranker()).run(ask)
    print(result.strategy, result.metrics["retrieval"]["strategy_metadata"])

    # A runtime failure is fail-open: a result with a warning, never cached.
    cache = FakeCache()
    down = QueryPipeline(retriever=DownIndex(), reranker=FakeReranker(), cache=cache)
    failed = await down.run(ask)
    print(failed.warnings, repr(failed.context), failed.degraded, len(cache.data))

    # A configuration error raises.
    try:
        await down.run(PipelineRequest(ask.query, retrieval_strategy="telepathy"))
    except UnknownStrategyError as error:
        print("raised:", str(error)[:40])


asyncio.run(main())
```

```text
hybrid {'degraded_from': 'self_rag', 'degraded_reason': 'no_language_model'}
['retrieval_failed'] '' True 0
raised: Unknown retrieval strategy 'telepathy'.
```

## Where the tests live

All under `tests/pipeline/`, run with `python -m pytest tests/pipeline`.

- `test_foundation.py`: tier loading and validation, `AnswerScope`, the protocols, `VectorStoreRetriever`.
- `test_query_pipeline.py`: tier precedence, both cache layers, the no-results retry, usage, `run_stages`.
- `test_pipeline_integration.py`: end to end per tier, filters reaching the retriever, and one cache-key test per scope field.
- `test_stage_analyze.py`, `test_stage_retrieve.py`, `test_stage_rerank.py`, `test_stage_context.py`: one file per stage.
- `test_ingest.py`: the ingest outcomes, rollback and payload rules.
- `test_jurisdiction_us.py`: the US plug-in.
- `test_hygiene.py`: the package must stay free of product, tenancy and vendor names.
