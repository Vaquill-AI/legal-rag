# Ingestion

The ingestion component turns files (plain text, Office documents, email, PDFs, images and audio) into one unified `IngestorResult` with extracted text, page data and metadata.
`DocumentRouter` picks a parser per file, falls back to a second parser when the first fails or returns unreadable text, and every external service (OCR, vision, transcription) is passed in through small protocols instead of being read from settings.

## Install

```bash
pip install "legal-rag[ingestion]"            # base extra (tiktoken)
pip install "legal-rag[ingestion-office]"     # Office, HTML, RTF, XML, .msg and image helpers
pip install "legal-rag[ingestion-pdf-agpl]"   # PDF layout parser (AGPL, read the warning below)
```

`ingestion-office` installs python-docx, openpyxl, python-pptx, xlrd, olefile, extract-msg, beautifulsoup4, striprtf, defusedxml, pillow and pillow-heif.
Each of these libraries is imported inside a `try` block, so a missing library never breaks `import legal_rag.ingestion`.
Importing the package loads beautifulsoup4, extract-msg and olefile when they are installed, and importing `MultiFormatIngestor` also loads the Office and image libraries that are installed.
At parse time it either produces a per-document `success=False` result with an install hint, or, for HTML and image sizing, a reduced fallback.

Warning: `ingestion-pdf-agpl` installs PyMuPDF and PyMuPDF4LLM, which are AGPL or commercially licensed.
legal-rag does not bundle or relicense them.
Check that your licence covers your use before you install this extra in a distributed product.
Importing `legal_rag.ingestion` never loads them; the layout module (`legal_rag.ingestion.pymupdf4llm_ingestor`) is imported only when a PDF is routed to it.

## Supported formats

| Formats | Handler | Needs |
| --- | --- | --- |
| `.txt`, `.md`, `.markdown`, `.csv`, `.json` | `TextExtractor` | nothing beyond the base install |
| `.html`, `.htm` | `TextExtractor` | `ingestion-office` (beautifulsoup4); without it, tags are stripped with a regex |
| `.rtf` | `TextExtractor` | `ingestion-office` (striprtf) |
| `.xml` | `TextExtractor` | `ingestion-office` (defusedxml, XXE safe) |
| `.docx`, `.dotx`, `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.xls`, `.pptx` | `OfficeExtractor` | `ingestion-office` |
| `.doc`, `.ppt` | `OfficeExtractor` | `ingestion-office`; best effort only, `.doc` is much better with a `DocumentConverter` |
| `.eml` | `EmailFileIngestor` | nothing beyond the base install |
| `.msg` | `EmailFileIngestor` | `ingestion-office` (extract-msg) |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.webp`, `.heic`, `.heif` | `ImageExtractor` | an `OcrClient` or `VisionClient` (see below); pillow for transcoding |
| `.mp3`, `.wav`, `.m4a` | `AudioIngestor` | a `TranscriptionClient` |
| `.pdf` | layout parser, then OCR parser if registered | `ingestion-pdf-agpl` for the layout parser; an `OcrClient` for scans |
| `.epub`, `.xps` | routed to the layout parser by the format map | `ingestion-pdf-agpl` |

`.odt`, `.ods` and `.odp` appear in the `DocumentType` enum but are not handled: the router answers `Unsupported file format`.

## Quick start

This runs with no extras, no keys and no files on disk.

```python
import asyncio

from legal_rag.ingestion import DocumentRouter

OPINION = (
    "SUPREME COURT OF THE UNITED STATES\n\n"
    "Marbury v. Madison, 5 U.S. (1 Cranch) 137 (1803)\n\n"
    "It is emphatically the province and duty of the judicial department "
    "to say what the law is.\n"
)


async def main() -> None:
    router = DocumentRouter()
    routed = await router.route_and_parse(
        file_content=OPINION.encode("utf-8"),
        filename="opinion.txt",
        document_id="doc-1",
    )
    doc = routed.ingestor_result
    print("parser:", routed.parser_used.value)
    print("success:", doc.success, "error:", doc.error)
    print("type:", doc.document_type.value, "pages:", doc.page_count)
    print("chars:", doc.total_chars)
    print("title:", doc.title, "sections:", len(doc.sections))
    print("metadata:", doc.metadata)
    print("reasoning:", routed.routing_decision.reasoning)
    print(doc.content.splitlines()[0])


asyncio.run(main())
```

Output:

```text
parser: multi_format
success: True error: None
type: txt pages: 1
chars: 177
title: None sections: 0
metadata: {'lines': 6}
reasoning: Format: txt | Parser: multi_format
SUPREME COURT OF THE UNITED STATES
```

`route_and_parse` takes `file_content` (bytes) or `file_path`, plus `filename`, `document_id`, an optional `force_parser` (a `ParserType`) and optional pre-detected `characteristics`.
It returns a `RoutingResult` with `ingestor_result`, `parser_used`, `fallback_used`, `routing_decision` and `total_time_ms`.
Parse failures are reported as `ingestor_result.success == False` with `error` set, not raised.
`route_and_parse_document(...)` is a one-call wrapper that builds a default router when you do not pass one.

The package exports the router and its result types, the base types and `EmailFileIngestor`.
`MultiFormatIngestor`, `MistralOCRIngestor`, `AudioIngestor` and `ImageExtractor` are imported from their submodules, as the snippets below do, so the package import stays light.

## OCR cleanup and the text-quality gate

Extractors clean their own output.
`TextExtractor`, `OfficeExtractor` and `ImageExtractor` call `clean_ocr_artifacts`, and `MultiFormatIngestor` puts the cleaned text in `IngestorResult.content` (with the same text in `text_clean`).
`clean_ocr_artifacts(text, preserve_structure=True, return_stats=False)` removes image placeholders, OCR confidence markers, page markers, running page footers, replacement characters and zero-width characters, normalizes Unicode and whitespace, and collapses duplicated blocks.

```python
from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

raw = "Holding text ⇒ picture [527 x 52] intentionally omitted ⇐ continues here."
cleaned, stats = clean_ocr_artifacts(raw, return_stats=True)
print(repr(cleaned))
print(stats.artifacts_removed, stats.original_length, stats.cleaned_length)
print(str(stats))
# 'Holding text continues here.'
# 1 73 28
# Cleaned 1 artifacts (61.6% reduction, 73 → 28 chars)
```

The quality gate judges the extracted string, not the file.
`assess_extraction_quality(text)` looks at the first 2000 characters and returns a `TextQuality` with a `verdict` (`TextQualityVerdict`), `is_readable` and the ratios behind the decision.

| Verdict | Meaning | Readable |
| --- | --- | --- |
| `readable` | looks like human text | yes |
| `too_short` | fewer than 50 characters after stripping, nothing to judge | yes |
| `glyph_noise` | under 80% printable characters (control bytes, unmapped glyphs) | no |
| `mojibake` | over 10% U+FFFD replacement characters | no |
| `symbol_mapped` | over 30% Private Use Area characters | no |
| `drm_scrambled` | mostly ASCII letters with almost no spaces or very long words | no |

```python
from legal_rag.ingestion.text_quality import assess_extraction_quality, is_text_readable

good = "The court held that the statute applies to every party before it in this case."
noise = "\x00\x01\x02\x03\x04\x05\x06\x07" * 20
scrambled = "xqzvkjhgfdsapoiuytrewqmnbvcxzlkjhgfdsa" * 10

for label, text in [("good", good), ("noise", noise), ("scrambled", scrambled)]:
    quality = assess_extraction_quality(text)
    print(label, quality.verdict.value, quality.is_readable, round(quality.printable_ratio, 2))

print(is_text_readable(good))
# good readable True 1.0
# noise glyph_noise False 0.0
# scrambled drm_scrambled False 1.0
# True
```

Inside the router the gate applies only to output from the PDF layout parser and the OCR parser, and only when at least 100 characters were extracted.
Unreadable layout-parser text triggers the fallback to the OCR parser when one is registered.
If the last parser in the chain still returns unreadable text, the result is replaced by `success=False`, empty `content` and an actionable `error` built by `unreadable_error_message(verdict, ocr_attempted)`.
Text, Office, email and image output is never gated.

## Supply your own OCR, vision or transcription

Nothing in the package reads settings or environment variables.
You pass objects that satisfy the protocols in `legal_rag.ingestion.protocols`, all of which are `runtime_checkable`.

| Protocol | Shape | Used by |
| --- | --- | --- |
| `OcrClient` | `ocr(content, filename, *, include_images=False) -> OcrResult \| None` (blocking) | `MistralOCRIngestor`, `ImageExtractor`, `OfficeExtractor` |
| `VisionClient` | `async describe_image(image, mime_type, *, prompt) -> str` | `ImageExtractor`, `OfficeExtractor`, `DocumentRouter` (handwriting) |
| `TranscriptionClient` | `model_name` attribute and `async transcribe(audio_path, *, language=None) -> TranscriptionResult` | `AudioIngestor` |
| `DocumentConverter` | `async to_docx(content, filename) -> bytes \| None` | `OfficeExtractor` (legacy `.doc`) |
| `ObjectStore` | `put_object(key, data, *, content_type) -> str` | `MistralOCRIngestor` (hoists inline images) |
| `SyncBreaker`, `AsyncBreaker`, `ExtractionGate`, `ErrorReporter` | optional circuit breaking, concurrency gate and error hook | OCR client config, `AudioIngestor`, layout parser |

An `OcrResult` is a dict with `text`, `pages` (each `{"page_number", "text", "char_count"}`) and `page_count`.
Return `None` when the service found no text.

Register an OCR parser on the router to enable scanned-PDF handling.
`MistralOCRIngestor` is the generic ingestor around any `OcrClient`, despite its name; `MistralOcrClient(MistralOcrConfig(api_key=...))` is the bundled reference client for one hosted OCR API.

```python
import asyncio

from legal_rag.ingestion import DocumentRouter, ParserType
from legal_rag.ingestion.mistral_ocr import MistralOCRIngestor
from legal_rag.ingestion.protocols import OcrClient, OcrResult


class MyOcr:
    def __init__(self, text: str) -> None:
        self.text = text

    def ocr(
        self, content: bytes, filename: str, *, include_images: bool = False
    ) -> OcrResult | None:
        page = {"page_number": 1, "text": self.text, "char_count": len(self.text)}
        return {"text": self.text, "pages": [page], "page_count": 1}


GOOD = "The appellant argues that the lower court erred in admitting the statement. " * 3
NOISE = "\x00\x01\x02\x03\x04\x05\x06\x07" * 20


async def run(label: str, text: str) -> None:
    client = MyOcr(text)
    assert isinstance(client, OcrClient)
    router = DocumentRouter(parser_factories={ParserType.OCR: lambda: MistralOCRIngestor(client)})
    routed = await router.route_and_parse(
        file_content=b"%PDF-1.4 scanned",
        filename="scan.pdf",
        document_id="scan-1",
        force_parser=ParserType.OCR,
    )
    doc = routed.ingestor_result
    print(label, "| success:", doc.success, "| sections:", len(doc.sections))
    print("  error:", (doc.error or "none")[:70])


async def main() -> None:
    await run("good", GOOD)
    await run("noise", NOISE)


asyncio.run(main())
```

```text
good | success: True | sections: 1
  error: none
noise | success: False | sections: 0
  error: This PDF has no extractable text layer. It was exported in a way that
```

`force_parser` skips routing and is used here only so the example needs no PDF library.
In normal use, leave it out: once an OCR parser is registered, the router sends scanned PDFs to it after the layout pre-flight check, and uses it as the fallback when the layout parser fails or returns unreadable text.
The OCR ingestor returns your client's text as is, one `DocumentSection` per page, and does not call `clean_ocr_artifacts`.

Vision and transcription clients go into the `MultiFormatIngestor` that the router uses for images and audio.
Pass a ready-made instance under `ParserType.MULTI_FORMAT`:

```python
import asyncio
from pathlib import Path

from legal_rag.ingestion import DocumentRouter, ParserType
from legal_rag.ingestion.audio_ingestor import AudioIngestor
from legal_rag.ingestion.extractors.image import ImageExtractor
from legal_rag.ingestion.multi_format import MultiFormatIngestor
from legal_rag.ingestion.protocols import TranscriptionResult


class MyTranscriber:
    model_name = "my-asr"

    async def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        text = "The witness stated that the meeting took place on the second floor. " * 2
        return TranscriptionResult(text=text, language=language or "en", duration_seconds=20.0)


class MyVision:
    async def describe_image(self, image: bytes, mime_type: str, *, prompt: str) -> str:
        return "Handwritten note: payment due on the first of each month."


async def main() -> None:
    multi = MultiFormatIngestor(
        audio_ingestor=AudioIngestor(MyTranscriber()),
        image_extractor=ImageExtractor(vision_client=MyVision()),
    )
    router = DocumentRouter(parsers={ParserType.MULTI_FORMAT: multi})
    for name, data in [("hearing.mp3", b"fake-audio-bytes"), ("note.png", b"not-really-a-png")]:
        doc = (await router.route_and_parse(file_content=data, filename=name)).ingestor_result
        print(name, doc.success, doc.content[:40])
    bare = await DocumentRouter().route_and_parse(file_content=b"abc", filename="hearing.mp3")
    print("no provider:", bare.ingestor_result.error)


asyncio.run(main())
```

```text
hearing.mp3 True The witness stated that the meeting took
note.png True Handwritten note: payment due on the fir
no provider: No audio ingestor configured
```

`ImageExtractor` tries the vision client first and falls back to the OCR client.
Pass `vision_client` to `DocumentRouter(...)` to enable page-by-page handwriting OCR for scanned PDFs, used when you call `route_and_parse(..., handwritten=True)`.

## Known limits

- `title` is `None` and `sections` is empty for `.txt`, `.html`, `.docx`, `.pptx` and the other extractor-backed formats. `MultiFormatIngestor` never fills them. Email results carry a title and sections, and OCR and layout results carry per-page or per-heading sections.
- The HTML `<title>` is returned in `metadata["title"]`, never in `IngestorResult.title`, which stays `None`. Without beautifulsoup4 the regex fallback recovers no title.
- PPTX extraction reads slide shape text, tables and speaker notes. Notes follow the slide text under a `Speaker notes:` heading and are also in `slides[i]["notes"]`.
- A document that carries images but no text gives no text unless you supply a client. An image with no `OcrClient` or `VisionClient` returns `success=False` and an `error` that names both. An image that a configured client reads as blank still returns `success=True` with empty `content` and a note in `metadata`, so check `total_chars`.
- Without a registered OCR parser, a scanned PDF is not recovered. The layout parser's only fallback is the router's default `MultiFormatIngestor`, which has no PDF handler and answers `No PDF ingestor configured`.
- Page counts for formats without pages are estimates of 3000 characters per page, and `pages` are synthesized at paragraph boundaries.
- `.doc` and `.ppt` use a byte-level fallback that can return partial text. Convert them to `.docx` and `.pptx`, or supply a `DocumentConverter`.
- `.odt`, `.ods` and `.odp` are not supported. Unknown extensions return `Unsupported file format`.
- `OcrClient.ocr` is a blocking call. `MistralOCRIngestor`, `ImageExtractor` and `OfficeExtractor` run it with `asyncio.to_thread`, so the client must be safe to call from a worker thread. A client whose `ocr` is a coroutine function is not supported.
- A `MultiFormatIngestor` built without `pdf_ingestor` rejects every `.pdf` with that same error, so pass one when you use it outside the router.

## Tests

- `tests/ingestion/test_ingestion.py` covers the protocols, the router, the quality gate, the OCR, audio, image and email ingestors and the layout module's lazy import, using the fakes in `tests/ingestion/conftest.py` (`FakeOcrClient`, `FakeVisionClient`, `FakeTranscriber`, `FakeObjectStore`).
- `tests/ingestion/test_extraction_paths.py` builds real DOCX, XLSX, PPTX, HTML, RTF, MSG and PDF fixtures in a temporary directory and asserts on the extracted text. A format whose library is missing is skipped.
- `examples/ingestion_basic.py` is a runnable tour of the router on text, HTML and DOCX.

```bash
python -m pytest tests/ingestion
```
