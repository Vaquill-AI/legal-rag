# Chunking

The `legal_rag.chunking` package splits legal documents into retrieval-sized chunks along their own structure (chapters, sections, articles, orders) instead of at arbitrary character counts.
Every chunk carries a token count, a detected document type, a section label and links to its neighbours, and chunking itself makes no network or model calls.

## Install

```bash
pip install "legal-rag[chunking]"   # tiktoken>=0.7 and semchunk>=2.2
```

The core install depends only on `pydantic`.
`tiktoken` is a soft dependency: without it (or when it cannot load its encoding file offline) token counts fall back to a whitespace approximation, so chunk sizes are approximate but chunking still works.
`semchunk` is an optional extra needed only by `SemchunkLegalChunker`.
`import legal_rag.chunking` and every other chunker work without it, and building `SemchunkLegalChunker()` without it raises an `ImportError` that names `pip install "legal-rag[chunking]"`.

```python
import sys

sys.modules["tiktoken"] = None  # simulate: tiktoken is not installed

from legal_rag.chunking.tokens import count_tokens, get_encoding

print(type(get_encoding()).__name__, count_tokens("The court affirmed the judgment."))
```

This prints `ApproximateEncoding 5`, one token per whitespace-delimited word.

```python
import sys

sys.modules["semchunk"] = None  # simulate: semchunk is not installed

import legal_rag.chunking as chunking

try:
    chunking.SemchunkLegalChunker()
except ImportError as error:
    print(error)
```

This prints `SemchunkLegalChunker needs the optional 'semchunk' package. Install it with: pip install "legal-rag[chunking]"`.

## What is in the package

| Class | Kind | Use it when | Needs |
| --- | --- | --- | --- |
| `RecursiveLegalChunker` | Section-aware, recursive splitter. Recommended default. | You want true overlap, clean splits and metadata for any of the 12 document types. | Nothing |
| `SemchunkLegalChunker` (aliases `SemanticChunker`, `SectionAwareChunker`) | Legacy splitter built on `semchunk`, judgment sections only. | You want semchunk's boundaries and a `section` value from the judgment detector. | `semchunk` |
| `EmailChunker` | Conversation-aware splitter for email text. | The input is an email thread with headers and quoted replies. | Nothing |
| `DocumentTypeClassifier`, `UniversalSectionDetector` | Pattern-based detection used by the recursive chunker. | You only need the document type or the section boundaries. | Nothing |
| `MultilingualPipeline`, `LanguageDetector`, `CrossLingualSearcher` | Not a chunker. Detects language and embeds text. | You embed chunks in several languages. | An `Embedder` |
| `ContextualRetrievalEnhancer`, `SimplifiedContextEnhancer` | Not chunkers. Prepend context to chunk text, from a model or from rules. | You want contextual retrieval before embedding. | A `LanguageModel` (model version only) |
| `PrecedentTracker` | Not a chunker. Checks whether a cited case is still good law. | You post-process citations found in chunks. | Nothing |

There is no embedding-based "semantic" chunker.
`SemanticChunker` is only an alias of the semchunk chunker, no chunker takes an embedder, and text with no recognised heading becomes one `UNKNOWN` section.

## The chunk

`chunker.chunk(text, document_id, ...)` is async and returns a `ChunkerResult` (`chunks`, `total_chunks`, `total_tokens`, `avg_chunk_size`, `min_chunk_size`, `max_chunk_size`, `processing_time_ms`).
The recursive chunker returns `LegalChunk` objects, which extend `Chunk` with these fields.

| Field | Meaning |
| --- | --- |
| `content`, `chunk_index`, `total_chunks`, `document_id` | Text (header included), position and owner. |
| `token_count` | Tokens in `content`, including the header and the overlap prefix, so it can exceed `chunk_size`. |
| `document_type` | A `DocumentType` such as `STATUTE`, `JUDGMENT`, `CONTRACT`. |
| `section_type_universal`, `section_type` | The section as a `LegalSectionType` enum and as its string value. |
| `section_title` | The full heading token of the section, up to the end of its number, for example `"Section 12-101."` or `"Sec. 3.4(b)"`. |
| `contextual_header` | The metadata header already prepended to `content`. Empty unless you pass `document_metadata`. |
| `section_priority`, `section_confidence`, `hierarchy_level` | Retrieval boost (0 to 100), detection confidence and nesting level. |
| `prev_chunk_id`, `next_chunk_id` | Neighbour links, set for every chunk that has a neighbour. |

Defaults come from `ChunkingConfig`: `chunk_size=512`, `chunk_overlap=100`, `min_chunk_size=50` tokens.
An explicit `chunk_overlap=0` turns overlap off and `min_chunk_size=0` keeps every piece, and only `None` selects the default.

## Quick start

```python
import asyncio

from legal_rag.chunking import RecursiveLegalChunker

STATUTE = """\
CHAPTER 12. CONSUMER DATA PROTECTION

Section 101. Definitions.
In this chapter: (1) "Personal data" means information that identifies, relates to, or is
reasonably linkable to an individual consumer. (2) "Controller" means a person that, alone or
jointly with others, determines the purposes and means of processing personal data.

Section 102. Consumer rights.
A consumer may, by written request to a controller, (1) confirm whether the controller is
processing the consumer's personal data, (2) obtain a copy of that data, and (3) require the
controller to delete personal data provided by or obtained about the consumer.
"""


async def main() -> None:
    chunker = RecursiveLegalChunker(chunk_size=80, chunk_overlap=15, min_chunk_size=10)
    result = await chunker.chunk(
        STATUTE,
        document_id="demo-ch12",
        document_metadata={"title": "Consumer Data Protection Act"},
    )
    print(f"{result.total_chunks} chunks, {result.total_tokens} tokens in total")
    for chunk in result.chunks:
        print(
            f"[{chunk.chunk_index}] {chunk.token_count:>3} tokens"
            f" | doc type: {chunk.document_type.value}"
            f" | section: {chunk.section_type_universal.value}"
            f" | label: {chunk.section_title!r}"
            f" | prev: {chunk.prev_chunk_id is not None}"
            f" | next: {chunk.next_chunk_id is not None}"
        )
    first, second = result.chunks[0], result.chunks[1]
    print("first.next_chunk_id == second.id:", first.next_chunk_id == second.id)
    print(second.contextual_header.strip())


asyncio.run(main())
```

With `tiktoken` installed the output is:

```text
3 chunks, 190 tokens in total
[0]  23 tokens | doc type: statute | section: chapter | label: 'CHAPTER 12' | prev: False | next: True
[1]  77 tokens | doc type: statute | section: section | label: 'Section 101.' | prev: True | next: True
[2]  90 tokens | doc type: statute | section: section | label: 'Section 101.' | prev: True | next: False
first.next_chunk_id == second.id: True
Act: Consumer Data Protection Act
Section: SECTION
```

Later chunks start with the tail of the previous chunk (the true overlap), which with the header is why `token_count` can pass 80.
The `Section` headings share one label because adjacent headings of the same type form one section run.

## Document types and pattern packs

`DocumentTypeClassifier.classify(text, threshold=0.5)` scores the first 5,000 characters against weighted regular expressions and returns `(DocumentType, confidence)`.
Below the threshold it returns `UNKNOWN`, and an `UNKNOWN` document is split only with generic preamble, definitions, `ARTICLE`/`SECTION`/`PART` and `SCHEDULE`/`ANNEX`/`APPENDIX`/`EXHIBIT` patterns.
`UniversalSectionDetector.detect(text, document_type=None)` then picks the heading patterns for that type.
`RecursiveLegalChunker.chunk(..., document_type=...)` skips classification when you already know the type.

Classification patterns beyond the built-in ones live in `src/legal_rag/chunking/data/jurisdiction_patterns.json`, grouped into named packs.
The file lists `default_packs`, which is `["us"]`, so the default path is US-first and every other pack is opt-in.
Packs add classification patterns only, and the section heading patterns are fixed in code.
The other files in `chunking/data` hold prompt vocabulary, language data, legal concepts, seed precedents and US document vocabulary.

```python
import asyncio

from legal_rag.chunking import DocumentType, RecursiveLegalChunker, get_document_type_classifier
from legal_rag.chunking.document_types import available_pattern_packs, load_pattern_packs

CONTRACT = """MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is made and entered into on January 1, 2024, between Acme Corp and Beta LLC.

ARTICLE I DEFINITIONS
"Confidential Information" means all non-public information.
"""

assert "us" in available_pattern_packs()
assert load_pattern_packs().keys() == load_pattern_packs(["us"]).keys()  # default is ["us"]

doc_type, confidence = get_document_type_classifier().classify(CONTRACT)
print(doc_type, round(confidence, 2))


async def main() -> None:
    result = await RecursiveLegalChunker().chunk(
        CONTRACT, "nda-1", document_type=DocumentType.CONTRACT
    )
    print(result.total_chunks, {c.document_type.value for c in result.chunks})


asyncio.run(main())
```

To use a non-default pack, classify with `get_document_type_classifier(["us", "<pack>"])` (call `available_pattern_packs()` for the names) and pass the result as `document_type=`.
`RecursiveLegalChunker` itself takes no `jurisdictions` argument and always uses the default detector.

## Injecting a model or an embedder

Components receive their providers as arguments, and nothing in `legal_rag` reads settings, environment variables or secrets.
The protocols are in `legal_rag.providers.protocols`, and in-memory fakes for tests are in `legal_rag.providers.fakes`.

| Component | Parameter | Protocol |
| --- | --- | --- |
| `ContextualRetrievalEnhancer(llm, cache=None, *, max_tokens=100, max_concurrent=5, ...)` | `llm`, `cache` | `LanguageModel` (`generate`, `generate_structured`), `Cache` (`get`, `set`, `delete`) |
| `MultilingualPipeline(embedder, config=None, cross_lingual_embedder=None, language_detector=None)`, `CrossLingualSearcher(embedder=...)` | `embedder` | `Embedder` (`dimension`, `embed_documents`, `embed_query`) |
| `SemchunkLegalChunker(citation_extractor=...)` | `citation_extractor` | any object with `extract(text)` |
| `HybridLegalNERExtractor(token_classifier=...)` | `token_classifier` | async `token_classification(text)` |
| `extract_us_legal_metadata(text, llm=None)` | `llm` | `LanguageModel` |

`ContextualRetrievalEnhancer.enhance_chunks` returns one string per chunk, the generated context followed by the chunk text, and does not modify the chunks.
If the model call fails, the string starts with a `[Section: ...]` label instead.
Contexts are cached under the prompt version plus content hashes of the document and of the chunk, so re-chunking the same text hits the cache even though every chunk gets a new random `id`.

```python
import asyncio

from legal_rag.chunking import RecursiveLegalChunker
from legal_rag.chunking.contextual_retrieval import ContextualRetrievalEnhancer
from legal_rag.chunking.multilingual import MultilingualPipeline
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeLanguageModel

TEXT = 'ARTICLE I TERM\n\nThe Receiving Party shall protect the information for two years.\n\n'
TEXT += "ARTICLE II LAW\n\nThis Agreement is governed by the laws of Delaware.\n"


async def main() -> None:
    result = await RecursiveLegalChunker(chunk_size=20, chunk_overlap=3, min_chunk_size=3).chunk(
        TEXT, "nda-1"
    )
    llm = FakeLanguageModel(reply="Article II, a term clause.")  # use your own LanguageModel
    cache = FakeCache()  # optional; use your own Cache
    enhancer = ContextualRetrievalEnhancer(llm, cache, max_tokens=100, max_concurrent=5)
    texts = await enhancer.enhance_chunks(list(result.chunks), TEXT, "nda-1")
    print(f"{len(texts)} chunks, {len(llm.calls)} model calls, {len(cache.data)} cached contexts")

    pipeline = MultilingualPipeline(FakeEmbedder(dimension=6))  # use your own Embedder
    one = await pipeline.embed("The court affirmed the judgment.")
    print(one.language, one.dimension, one.embedder_used.value)


asyncio.run(main())
```

## Known limits

- Case-name extraction drops a leading signal word, so `strip_leading_signal_words("Under Armour")` returns `"Armour"`, and `"Under Brown v. Board of Education (1954)"` yields `"Brown v. Board"`.
- A run of same-type headings shares the first heading's `section_title`.
- `chunk_size=0` is not a valid size and is replaced by the default, unlike `chunk_overlap` and `min_chunk_size`, where `0` is a real value.
- When a section splits into several pieces, a piece under `min_chunk_size` tokens is skipped, so joining chunks may not reproduce the source text.
- `SemchunkLegalChunker` is the older path: it detects judgment sections only and labels chunks with `chunk.section` (a `LegalSection`) instead of `section_type_universal`.
- `PrecedentTracker` knows only its seed list (`data/precedent_data.json`) plus `register_overruling` calls, so an unknown citation is not proof of good law.

## Tests and example

- Tests: `tests/chunking/` (`test_chunkers.py`, `test_data_driven.py`, `test_injected_providers.py`, `test_case_names_and_default_headings.py`, `test_chunking_regressions.py`), run with `python -m pytest tests/chunking`.
- Example: `examples/chunking_basic.py`.
