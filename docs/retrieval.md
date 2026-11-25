# Retrieval

The `legal_rag.retrieval` package holds the query-time half of a legal RAG system: rank fusion, reranking, adaptive retrieval strategies, query routing, conversation handling, a semantic cache and streaming event shapes.
Every component receives its providers (language model, embedder, reranker, vector store, cache) as arguments, so the package runs against any backend and, in tests, against the in-memory fakes.

## Install

```bash
pip install "legal-rag[retrieval]"          # adds tiktoken, used for token counting when present
pip install "legal-rag[retrieval,voyage]"   # also adds voyageai, for VoyageEmbedder and VoyageReranker
```

The `voyage` extra is only needed for `legal_rag.providers.adapters.VoyageEmbedder` and `VoyageReranker`.
Everything else takes the provider protocols in `legal_rag.providers`: `LanguageModel`, `Embedder`, `Reranker`, `VectorStore`, `Cache`.

## Package map

`import legal_rag.retrieval` stays cheap.
It re-exports `BaseRetriever`, `RetrievalResult`, `ChunkGroup`, `GroupedRetrievalResult`, `reciprocal_rank_fusion` and `rrf_merge_chunks`.
The larger subpackages are imported on demand.

| Subpackage | What it does | Main entry points |
| --- | --- | --- |
| `fusion` (top level) | Merges ranked lists without score calibration. | `reciprocal_rank_fusion`, `rrf_merge_chunks` |
| `strategies` | Query-time retrieval strategies. | `HyDEGenerator`, `HyDETransformer`, `MultiQueryGenerator`, `generate_rewrites`, `MultiHopRetriever`, `SelfRAGController`, `CRAGCorrector`, `QueryDecomposer`, `SourceMixer`, `LLMGapAnalyzer` |
| `rerank` | Reranking, reordering, authority weighting, confidence scoring, source hygiene. | `CrossEncoderReranker`, `ScoreBasedReranker`, `AuthorityReranker`, `DiversityReorderer`, `LostInMiddleReorderer`, `ConfidenceScorer`, `RerankRateLimiter`, `dedupe_sources`, `apply_coverage_notice` |
| `routing` | Decides what a turn should be routed to. | `route`, `shortlist`, `decide`, `DecisionPolicy`, `Taxonomy`, `Destination`, `example_taxonomy` |
| `conversation` | Condensing, memory, intent routing and query analysis. | `QueryCondenser`, `ConversationManager`, `IntentRouter`, `QueryAnalyzer`, `QueryExpander`, `QueryEnricher`, `RetrievalDecisionMaker`, `RetrievalLoopDetector`, `split_message` |
| `caching` | Semantic cache and payload validation. | `SemanticCache`, `SemanticCacheConfig`, `ValidationMetadata`, `validate_payload`, `PayloadSchema` |
| `streaming` | Server-sent event shapes and response assembly. | `ChatEventStream`, `thinking_event`, `sources_event`, `chunk_event`, `done_event`, `error_event`, `get_stream_context` |
| `accumulators` | Expands retrieved chunks and checks whether the context is sufficient. | `NeighborAccumulator`, `ChunkFetcher`, `analyze_information_gaps` |

Two name clashes to know about.
`conversation.HyDEResult` is an alias of `HyDETransformResult`, not the `HyDEResult` that `HyDEGenerator.generate` returns.
`conversation.RetrievalDecision` and `strategies.RetrievalDecision` are different types.

## Retrieval strategies

Each strategy takes its language model through the constructor and returns a result instead of raising when the model fails.
That result has `success=False` and carries the input forward (the original query, the original chunks, or for multi-hop the chunks gathered so far).
`hybrid` is the baseline the others adapt, and it has no class in this package.

| Strategy | What it does | Class | Providers it takes | You still wire |
| --- | --- | --- | --- | --- |
| `hybrid` | Searches the store and merges dense and keyword lists with reciprocal rank fusion. | none, compose `reciprocal_rank_fusion` | none | `Embedder`, `VectorStore`, optionally a `Reranker` |
| `self_rag` | Decides whether to retrieve at all, then critiques the answer and regenerates up to `max_regeneration_attempts`. | `SelfRAGController` | `LanguageModel` | `retrieve_func` (query to context string) and `generate_func` |
| `crag` | Grades each retrieved chunk for relevance, classifies the retrieval as a whole and filters or preserves chunks. | `CRAGCorrector` | `LanguageModel` | the retrieval that produces the chunks |
| `hyde` | Writes a hypothetical answer passage so that the passage is embedded instead of the question. | `HyDEGenerator`, `HyDETransformer` | `LanguageModel` | `Embedder`, `VectorStore` |
| `multi_query` | Generates query variations; `all_queries` includes the original when `include_original` is set. | `MultiQueryGenerator` | `LanguageModel` | `Embedder`, `VectorStore`, then fuse the lists |
| `multi_hop` | Retrieves in rounds, deduplicates chunks across hops and stops early when enough has been found. | `MultiHopRetriever` | optional `gap_analyzer` (`LLMGapAnalyzer` takes a `LanguageModel`) | `retrieve_func(query, top_k)` |

`QueryDecomposer` (a `LanguageModel`) splits a multi-track question into sub-queries.
`SourceMixer` shares a token budget between primary and supplementary chunks, using a `MixStrategy` of `PRIMARY_LED`, `SUPPLEMENTARY_LED` or `BALANCED`.
`CRAGConfig.enable_external_search` is off by default.

```python
import asyncio

from legal_rag.providers.fakes import FakeLanguageModel, FakeVectorStore
from legal_rag.retrieval.strategies import (
    MultiHopConfig,
    MultiHopRetriever,
    MultiQueryGenerator,
    QueryVariations,
)
from legal_rag.types import RetrievedChunk

store = FakeVectorStore([RetrievedChunk(f"c{i}", f"chunk {i}") for i in range(4)])


async def search(query: str, top_k: int) -> list[RetrievedChunk]:
    return await store.search([0.0], top_k=top_k)


async def main() -> None:
    llm = FakeLanguageModel(
        structured={
            "QueryVariations": QueryVariations(
                queries=["limitation period for breach of contract claims", "statute of limitations written contract"]
            )
        }
    )
    variants = await MultiQueryGenerator(llm).generate("how long do I have to sue for breach of contract")
    print("multi_query:", variants.success, variants.all_queries)

    hop = MultiHopRetriever(search, MultiHopConfig(max_hops=2, sufficient_chunks_threshold=3))
    result = await hop.retrieve("who is liable under section 1983")
    print("multi_hop:", result.success, result.total_hops, [c.id for c in result.chunks])


asyncio.run(main())
```

`FakeLanguageModel.structured` maps a schema class name to the instance to return.
A schema with no canned reply raises `KeyError` inside the fake.
`MultiQueryGenerator` treats that as a model failure and returns the original query alone with `success=False`.

## Quick start: fusion and reranking

`reciprocal_rank_fusion` scores each chunk as the sum of `1 / (k + rank)` over the lists it appears in, with `k=60` by default.
It keeps the first copy of each chunk and replaces its `score` with the fused score.
There are two reranker layers.
A provider `Reranker` takes strings and returns `RerankResult(index, score)`.
`CrossEncoderReranker` wraps a provider, takes chunks and returns a `RerankOutcome`.

```python
import asyncio

from legal_rag.providers.fakes import FakeReranker
from legal_rag.retrieval import reciprocal_rank_fusion
from legal_rag.retrieval.rerank import CrossEncoderReranker
from legal_rag.types import RetrievedChunk

dense = [
    RetrievedChunk("a", "Section 1983 civil action for deprivation of rights"),
    RetrievedChunk("b", "Monroe v. Pape on color of state law"),
    RetrievedChunk("c", "Criminal penalty for deprivation of rights"),
]
keyword = [
    RetrievedChunk("b", "Monroe v. Pape on color of state law"),
    RetrievedChunk("c", "Criminal penalty for deprivation of rights"),
]

fused = reciprocal_rank_fusion([dense, keyword], top_k=3)
for chunk in fused:
    print("fused", chunk.id, round(chunk.score, 4))


async def main() -> None:
    query = "color of state law"
    # The provider protocol scores plain strings and returns index and score pairs.
    scored = await FakeReranker().rerank(query, [c.text for c in fused], top_k=2)
    print("provider order", [fused[r.index].id for r in scored])

    # CrossEncoderReranker wraps a provider and works on chunks.
    outcome = await CrossEncoderReranker(FakeReranker(), top_k=2).rerank(query, fused)
    print("outcome", [c.id for c in outcome.chunks], "degraded:", outcome.degraded)


asyncio.run(main())
```

Output:

```text
fused b 0.0325
fused c 0.032
fused a 0.0164
provider order ['b', 'c']
outcome ['b', 'c'] degraded: False
```

When the provider is rate limited or a deadline passes, `CrossEncoderReranker` falls back to `ScoreBasedReranker` and sets `RerankOutcome.degraded` and `degraded_reason`.
Check `degraded` before you treat a score as a relevance score.

## Quick start: semantic cache

`SemanticCache(embedder, cache, config)` stores entries in the injected `Cache` as one JSON blob per scope.
A scope is any string you choose (a tenant, a project), or `None` for a shared scope, and one scope never sees another's entries.
Results must be JSON serializable.
The defaults are a similarity threshold of 0.85, a TTL of 3600 seconds and 200 entries per scope.

```python
import asyncio

from legal_rag.providers.fakes import FakeCache, FakeEmbedder
from legal_rag.retrieval.caching import SemanticCache, SemanticCacheConfig


async def main() -> None:
    cache = SemanticCache(FakeEmbedder(), FakeCache(), SemanticCacheConfig(similarity_threshold=0.85))
    question = "what is the limitation period for a contract claim"

    print("first lookup:", await cache.get(question, scope="tenant-a"))
    await cache.set(question, [{"id": "usc-1", "score": 0.9}], scope="tenant-a")
    print("second lookup:", await cache.get(question, scope="tenant-a"))
    print("other scope:", await cache.get(question, scope="tenant-b"))
    print(cache.get_stats())


asyncio.run(main())
```

Output:

```text
first lookup: None
second lookup: [{'id': 'usc-1', 'score': 0.9}]
other scope: None
{'hits': 1, 'misses': 2, 'sets': 1, 'errors': 0, 'hit_rate': '33.3%', 'total_queries': 3}
```

`FakeEmbedder` has no notion of meaning, so only an identical question hits.
A real embedder also hits on close paraphrases.
Set `require_validated=True` to serve only entries stored with a passing `ValidationMetadata`.
`invalidate(scope)` drops a scope.

## Routing

`route(routing_input, state, taxonomy=..., embedder=None, llm=None, policy=DEFAULT_POLICY)` returns a `RouteDecision` and never executes anything.
You supply the `Taxonomy`, a tuple of `Destination` records, each with an intent id, a description, examples, an `ActionCost` and the state it needs (`needs_documents`, `min_documents`, `requires`).
`example_taxonomy()` shows the shape.
The router first applies a deterministic prefilter for an empty message, a pure greeting or an explicit `skill_slug`.
Without a language model it returns the taxonomy's default intent at confidence 0.3 with `Action.ANSWER`.
Otherwise it optionally shortlists the destinations by embedding similarity (`shortlist` keeps the top 6 plus the default and returns every intent if the embedder fails), asks the model for JSON with `intent` and `confidence`, and passes the result to `decide`.
`decide` first enforces hard preconditions, so a destination that needs two documents degrades to the default intent when fewer are pinned.
It then maps confidence to an `Action` using `DecisionPolicy`.
The execute threshold rises with cost (0.55 reversible, 0.70 moderate, 0.80 expensive, 0.85 destructive).
An expensive or destructive destination below that bar but at or above `confirm_floor` (0.60) gets `Action.CONFIRM`.
Anything at or above `offer_floor` (0.35) gets `Action.ANSWER_AND_OFFER`, and anything lower gets `Action.ANSWER`.

```python
import asyncio
import json

from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.retrieval.routing import (
    PinnedDoc,
    RouterState,
    RoutingInput,
    decide,
    example_taxonomy,
    route,
)

taxonomy = example_taxonomy()


async def main() -> None:
    llm = FakeLanguageModel(reply=json.dumps({"intent": "compare", "confidence": 0.9}))
    ask = RoutingInput(instruction="how do these two rules differ", full_message="how do these two rules differ")

    one_doc = RouterState(pinned_docs=(PinnedDoc("d1"),))
    two_docs = RouterState(pinned_docs=(PinnedDoc("d1"), PinnedDoc("d2")))

    for label, state in (("one document", one_doc), ("two documents", two_docs)):
        decision = await route(ask, state, taxonomy=taxonomy, llm=llm)
        print(label, "->", decision.intent, decision.action.value, decision.source)

    greeting = RoutingInput(instruction="hello", full_message="hello")
    print("greeting ->", (await route(greeting, RouterState(), taxonomy=taxonomy, llm=llm)).source)
    print("no llm ->", (await route(ask, two_docs, taxonomy=taxonomy)).reason)

    # decide() alone maps a scored intent to an action with no model call.
    print("0.40 ->", decide(taxonomy, "compare", 0.40, two_docs).action.value)
    print("0.65 ->", decide(taxonomy, "compare", 0.65, two_docs).action.value)


asyncio.run(main())
```

Output:

```text
one document -> answer answer classifier
two documents -> compare execute classifier
greeting -> prefilter
no llm -> no classifier configured
0.40 -> answer_and_offer
0.65 -> answer_and_offer
```

## Design rules

- Every provider is injected.
  Components take a `LanguageModel`, `Embedder`, `Reranker`, `VectorStore` or `Cache` as an argument.
  A `usage_hook` callback is optional where metering applies.
- No environment reads.
  Nothing in `retrieval/` or `providers/` reads settings, environment variables or secrets.
  The Voyage adapters take an `api_key` or a ready `client`.
- Fail open.
  Strategies, the router, the semantic cache and `shortlist` catch provider failures and return a safe value.
  The safe value is the original query, the original chunks, a cache miss or the default intent.
  The reranker reports a fallback through `degraded`.
- No tenancy keys.
  Isolation is an opaque `scope` string or a free-form `filters` mapping that your store understands.
- The router decides and the caller acts.
  `route` returns a `RouteDecision` and never runs a destination.

## Tests

- `tests/retrieval_strategies/` covers the strategies, HyDE, fusion and the base retriever, reranking and the Voyage adapters.
  A hygiene test there scans the retrieval and adapter sources for environment reads and tenancy keys.
- `tests/retrieval_routing/` covers the routing engine and rules, conversation handling and scoped general questions.
  It also covers the accumulators, the cache and the streaming events.
- `examples/retrieval_basic.py` runs HyDE, fusion, reranking and the semantic cache together with the fakes and needs no keys.

Run one file at a time with `python -m pytest tests/retrieval_strategies/test_rerank.py`.
