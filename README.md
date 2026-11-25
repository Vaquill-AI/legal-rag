# legal-rag

Retrieval techniques for legal AI, from ingestion to verified answers, in one Python package.

legal-rag is the retrieval layer we use for legal question answering, with the production wiring taken out.
Every model, embedder, reranker, vector store and cache is an interface you pass in.
Nothing reads an environment variable, a settings file or a secret.
The test suite and every example run offline against in-memory fakes.

Built and maintained by [Vaquill](https://vaquill.ai), a legal data lab for AI.

> **Status: early release (0.0.x).**
> The techniques are tested and the guides run as written, but the interfaces can still change before 0.1.
> There is no PyPI release yet, so install from GitHub.

## What is in it

| Component | What it covers | Guide |
| --- | --- | --- |
| `ingestion` | Format dispatch, OCR cleanup, PDF, Office and email extraction, audio and image fallbacks | [docs/ingestion.md](docs/ingestion.md) |
| `chunking` | Section-aware, recursive, semantic and multilingual chunking, contextual headers, precedent tracking | [docs/chunking.md](docs/chunking.md) |
| `retrieval` | HyDE, multi-query, multi-hop, Self-RAG, CRAG, query decomposition, RRF, reranking, MMR, routing, conversation memory, caching | [docs/retrieval.md](docs/retrieval.md) |
| `verification` | Citation extraction and grounding, fabricated-case checks, claim checks, RAGAS-style evaluation, retrieval metrics | [docs/verification.md](docs/verification.md) |
| `agents` | Plan, validate, write, refine and verify answer generation, plus tiered agentic search tools | [docs/agents.md](docs/agents.md) |
| `pipeline` | The query and ingest pipelines as plain async stages, standard and deep tiers, a two-layer cache, jurisdiction plug-ins | [docs/pipeline.md](docs/pipeline.md) |
| `mcp_bridge` | Turns any OpenAPI document into MCP tools: catalogue, one-request dispatch, result mapping | [docs/mcp_bridge.md](docs/mcp_bridge.md) |
| `providers` | The protocols for the language model, embedder, reranker, store and cache, plus in-memory fakes | [docs/providers.md](docs/providers.md) |

The components build on each other.
The pipeline composes ingestion, chunking and retrieval, and the agents use the verification checks.
You can also use any one of them alone.

## Install

```bash
pip install "legal-rag[chunking] @ git+https://github.com/Vaquill-AI/legal-rag.git"
```

The core package depends only on `pydantic`.
Each component brings its own dependencies through an extra:

| Extra | Adds |
| --- | --- |
| `chunking` | `tiktoken` and `semchunk` for exact token counts and semantic splitting |
| `retrieval` | `tiktoken` |
| `ingestion` | `tiktoken` |
| `ingestion-office` | DOCX, XLSX, PPTX, XLS, MSG, HTML and RTF extraction |
| `voyage` | A Voyage AI embedder and reranker adapter |
| `mcp` | The MCP SDK, only for `ToolResult.to_mcp()` |
| `all` | `chunking`, `retrieval`, `verification`, `ingestion` and `mcp` |
| `dev` | The test and lint tools |

`ingestion-pdf-agpl` adds PDF layout extraction through PyMuPDF.
PyMuPDF is AGPL-3.0 or commercially licensed.
It is a separate extra, it is never imported by default, and it is not part of `all`.
Check its licence before you ship it in a product you distribute.

To work on the code, clone the repository and install the development extras:

```bash
git clone https://github.com/Vaquill-AI/legal-rag.git
cd legal-rag
python -m venv .venv
.venv/bin/python -m pip install -e ".[all,dev]"
.venv/bin/python -m pytest
```

## A first question

This is [examples/pipeline_basic.py](examples/pipeline_basic.py) in short.
It sends one question through the query pipeline with the US jurisdiction plug-in and a semantic cache.
It then writes an answer and checks it against the sources.
The index, the reranker, the embedder and the model are fakes, so it needs no key and no network.

```python
import asyncio

from legal_rag.pipeline import PipelineRequest, QueryPipeline, jurisdictions
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeReranker
from legal_rag.retrieval.caching import SemanticCache
from legal_rag.types import RetrievedChunk


class MyIndex:
    """Stands in for your search index. Apply your tenant filter from `filters`."""

    async def retrieve(self, query, *, top_k, filters=None, sub_queries=None):
        return [
            RetrievedChunk(
                "tenancy-12",
                "A landlord must give a tenant written notice of at least 14 days "
                "before starting an eviction for unpaid rent.",
                0.82,
                {"title": "Residential Tenancies Act, section 12", "source_type": "statute"},
            )
        ][:top_k]


async def main():
    index = MyIndex()
    pipeline = QueryPipeline(
        retriever=index,
        reranker=FakeReranker(),  # swap in your reranker
        jurisdiction=jurisdictions.get("us", statute_retriever=index),
        semantic_cache=SemanticCache(FakeEmbedder(), FakeCache()),
    )
    request = PipelineRequest(
        "what written notice must a landlord give", filters={"country_code": "US"}
    )
    result = await pipeline.run(request)
    print(result.tier, result.strategy, result.cache_status)
    print([source["metadata"]["title"] for source in result.sources])


asyncio.run(main())
```

The complete script runs as shipped and prints the tier, the retrieval strategy, the cache status, the sources, a semantic cache hit on the second ask, and a verification score for the written answer:

```bash
python examples/pipeline_basic.py
```

There is one runnable example per component.
See [examples/README.md](examples/README.md).

## Bring your own providers

The library asks you for five small interfaces, defined in [src/legal_rag/providers/protocols.py](src/legal_rag/providers/protocols.py):

- `LanguageModel` for generation.
- `Embedder` for dense vectors.
- `Reranker` for scoring candidates.
- `VectorStore` for retrieval.
- `Cache` for the exact and semantic answer caches.

An optional `UsageHook` receives a usage event (operation, provider and estimated tokens) from the code that calls a provider, so you can meter cost without the library knowing about your billing.
The in-memory fakes in `legal_rag.providers.fakes` implement all of them, which is why the tests need nothing external.
The [providers guide](docs/providers.md) shows how to write an adapter, and `legal_rag.providers.adapters.voyage` is a worked one.

## Design rules

- **Providers are parameters.** The library never constructs a client for you.
- **No hidden configuration.** No component reads environment variables, settings files or secrets.
- **Rules live in data.** Jurisdiction rules, prompts and vocabularies are data files and plug-ins, not code. The default path is US-first, and other jurisdictions are opt-in data packs.
- **Fail open, and say so.** A pipeline stage that fails hands the context on with a warning. A degraded answer is never written to the cache.
- **The caller owns tenancy, billing and auth.** The library takes opaque filters and a cache namespace. A cache key with a missing or unknown filter raises, so one tenant's answer cannot be served to another.
- **Documentation is tested.** Every code block in the guides runs offline against the fakes.

## Known limits

We would rather you read these here than find them in production.
Each guide ends with its own list.

- The hallucination detector counts a partially overlapping claim as supported, so one grounded claim plus one partial claim scores 1.0.
- Once any sentence in an answer matches a claim pattern, sentences that match none are not checked.
- A verification result is evidence for review, not a legal opinion, and not a substitute for reading the cited authority.
- The retrieval strategies that need a language model degrade to plain hybrid retrieval when none is supplied.
  `QueryResult.strategy` then reports `hybrid`, while cache keys use the strategy that was requested.
- There are no published benchmark numbers yet.

## Layout

```text
legal-rag/
  src/legal_rag/
    ingestion/         format dispatch, extraction, OCR cleanup, index config
    chunking/          section-aware, semantic and contextual chunking, precedent tracking
    retrieval/         strategies, rerank, routing, conversation, caching, streaming
    verification/      citations, grounding, evaluation
    agents/            answer generation actors and agentic search tools
    pipeline/          query and ingest pipelines, tiers, jurisdiction plug-ins
    mcp_bridge/        OpenAPI document to MCP tool catalogue, dispatch and result mapping
    providers/         protocols, adapters and in-memory fakes
  docs/                one guide per component
  tests/               one folder per component, plus repo-wide hygiene checks
  examples/            one runnable example per component
```

## Contributing

Bug reports and pull requests are welcome.
Read [CONTRIBUTING.md](CONTRIBUTING.md) first.
Report a security issue as described in [SECURITY.md](SECURITY.md), not in a public issue.

## License

Apache-2.0. See [LICENSE](LICENSE).
