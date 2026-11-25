# Examples

Each example is a short, self-contained script.
They use inline sample text and the public API only.
None of them needs an API key, a network connection or a file download.

Run them from the repository root with the project's virtual environment.

| Example | What it shows | Command | Extra to install |
| --- | --- | --- | --- |
| `chunking_basic.py` | Splitting a statute excerpt with `RecursiveLegalChunker`: token counts, detected document type, section labels and prev/next links on each chunk. | `python examples/chunking_basic.py` | None required. `legal-rag[chunking]` adds tiktoken for exact token counts (a whitespace approximation is used without it). |
| `ingestion_basic.py` | Extracting text from in-memory `.txt`, `.html` and `.docx` files with `DocumentRouter` and reading the unified `IngestorResult`. | `python examples/ingestion_basic.py` | None for `.txt`. `legal-rag[ingestion-office]` for `.html` (beautifulsoup4) and `.docx` (python-docx). The `.docx` step is skipped with a note when python-docx is missing. |
| `verification_basic.py` | Extracting citations from an answer, flagging a fabricated case that no retrieved source supports, and checking each case citation with a `CaseResolver`. | `python examples/verification_basic.py` | None. The `verification` extra has no dependencies. |
| `retrieval_basic.py` | HyDE query expansion, reciprocal rank fusion of a dense and a keyword list, reranking the fused chunks, and a semantic cache that misses and then hits, all with the fake providers. | `python examples/retrieval_basic.py` | None. The fakes need no keys or network. |
| `mcp_bridge_basic.py` | Building MCP tool catalogue entries from an inline OpenAPI document with `build_catalogue`, dispatching a statute lookup as one HTTP request on a fake client with `call_operation`, and mapping a 200 to text content and a 404 to `isError` with `to_tool_result`. | `python examples/mcp_bridge_basic.py` | None. The client is a fake, and the bridge imports neither `mcp` nor `httpx`. |
| `pipeline_basic.py` | Running one question through `QueryPipeline` with the US jurisdiction plug-in and a semantic cache (tier, strategy, cache status, sources, context, then a cache hit), then writing an answer with `generate_answer_buffered` and scoring it with `HallucinationDetector`. | `python examples/pipeline_basic.py` | None. The providers are fakes and the model reply is canned. |

To install an extra from a checkout:

```bash
python -m pip install -e ".[chunking]"
python -m pip install -e ".[ingestion-office]"
```
