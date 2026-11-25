"""Ingestion pipeline: extract, clean, classify, chunk, contextualize, scan, embed, write.

``IngestPipeline`` is the orchestration only. Every step reuses a component that already lives
in ``legal_rag.ingestion``, ``legal_rag.chunking`` or ``legal_rag.providers``:

1. extract text with a ``DocumentRouter`` (or your own extractor callable),
2. reject text that is empty or unreadable before anything is chunked,
3. clean OCR artifacts, then classify the document type,
4. chunk with a legal-aware chunker,
5. optionally prepend model-written context to each chunk (needs an ``llm``),
6. optionally scan each chunk for prompt-injection content (needs a ``scanner``),
7. embed in batches, with retry, and check the vector dimension,
8. write points in batches, and delete every id already written if any batch fails.

Tenancy is not a pipeline concept. The library never invents a user, organization or project
key. A deployer who needs one passes it in ``payload_extra``, which is copied into every point's
payload last, so a per-document ``metadata`` mapping can never overwrite it. The ``writer`` you
pass decides which collection the points land in.

Which failures return and which raise:

- A problem with the document returns ``IngestResult(success=False)``: the parser failed, the
  text is unreadable, the chunker failed, or the scanner rejected the document.
- A problem with a provider raises: ``IngestEmbeddingError``, ``EmbeddingDimensionError`` or
  ``IngestWriteError``. Your queue can retry those. Nothing is left half written, because a
  failed write rolls back first.
- An empty document is a success with status ``no_text_content`` and zero chunks.

``on_status`` is told about each stage as it finishes (see ``INGEST_STAGES``). A failing status
callback never fails the ingest.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

from legal_rag.chunking import BaseChunker, get_email_chunker, get_recursive_legal_chunker
from legal_rag.chunking.base import Chunk
from legal_rag.chunking.contextual_retrieval import ContextualRetrievalEnhancer
from legal_rag.chunking.document_types import DocumentType, get_document_type_classifier
from legal_rag.ingestion.base import IngestorResult
from legal_rag.ingestion.extractors.base import (
    estimate_pages_from_text,
    synthesize_pages_from_text,
)
from legal_rag.ingestion.router import DocumentRouter, unreadable_error_message
from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts, deduplicate_repeated_text
from legal_rag.ingestion.text_quality import assess_extraction_quality
from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import (
    Cache,
    Embedder,
    InjectionScanner,
    LanguageModel,
    ScanResult,
    SparseEncoder,
    UsageHook,
    VectorWriter,
)
from legal_rag.pipeline.types import VectorPoint
from legal_rag.providers.usage import emit_usage

_log = get_logger("legal_rag.pipeline.ingest")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Public constants and types
# ---------------------------------------------------------------------------

STAGE_EXTRACTING = "extracting"
STAGE_EXTRACTED = "extracted"
STAGE_CHUNKED = "chunked"
STAGE_CONTEXTUALIZED = "contextualized"
STAGE_SCANNED = "scanned"
STAGE_EMBEDDED = "embedded"
STAGE_STORED = "stored"
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"

INGEST_STAGES: tuple[str, ...] = (
    STAGE_EXTRACTING,
    STAGE_EXTRACTED,
    STAGE_CHUNKED,
    STAGE_CONTEXTUALIZED,
    STAGE_SCANNED,
    STAGE_EMBEDDED,
    STAGE_STORED,
    STAGE_COMPLETED,
    STAGE_FAILED,
)
"""Every stage name ``on_status`` can receive, in the order a full ingest reports them.
``contextualized`` and ``scanned`` are skipped when no ``llm`` or ``scanner`` was given, and
``failed`` replaces whatever stage did not finish."""

STATUS_COMPLETED = "completed"
STATUS_NO_TEXT_CONTENT = "no_text_content"
STATUS_FAILED = "failed"

RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "document_id",
        "text",
        "chunk_index",
        "total_chunks",
        "page_number",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
        "section_title",
        "section_number",
        "section_type",
        "language",
        "prev_chunk_id",
        "next_chunk_id",
        "citations",
    }
)
"""Payload keys the pipeline writes itself. ``payload_extra`` may not use them, and a per-document
``metadata`` key with one of these names is ignored."""

StatusCallback = Callable[[str, str], Awaitable[None]]
"""``async def on_status(stage: str, message: str) -> None``."""

Extractor = Callable[[bytes, str, str], Awaitable[IngestorResult]]
"""``async def extract(content, filename, document_id) -> IngestorResult``. A custom stand-in for
the document router."""

IngestSource = bytes | bytearray | str | PathLike[str] | IngestorResult
"""What ``IngestPipeline.ingest`` accepts: file bytes, a file path, already extracted text (a
plain ``str`` is TEXT, not a path), or an ``IngestorResult`` you built yourself."""

_SOURCE_TYPES = (bytes, bytearray, str, PathLike, IngestorResult)
_EMAIL_SUFFIXES = frozenset({".eml", ".msg"})
_PLACEHOLDER_TITLES = frozenset(
    {"(anonymous)", "anonymous", "untitled", "untitled document", "unknown", "document", ""}
)
_NO_TEXT_MIN_CHARS = 10
_READABILITY_MIN_CHARS = 100


class IngestError(Exception):
    """Base class for the failures ``IngestPipeline.ingest`` raises."""


class IngestEmbeddingError(IngestError):
    """The embedder failed after its retries. Nothing was written."""


class EmbeddingDimensionError(IngestError, ValueError):
    """An embedding does not have the dimension the index expects. Nothing was written."""


class IngestWriteError(IngestError):
    """A batch write failed. ``rolled_back`` says whether the already written ids were deleted.

    When ``rolled_back`` is ``False``, ``orphaned_ids`` lists the ids that may still be in the
    index, so the caller can sweep them.
    """

    def __init__(
        self,
        message: str,
        *,
        rolled_back: bool,
        orphaned_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.rolled_back = rolled_back
        self.orphaned_ids = list(orphaned_ids)


@dataclass(frozen=True)
class IngestOptions:
    """Tuning for one ``IngestPipeline``. Every value has a safe default.

    ``contextualize`` is ``None`` to contextualize when an ``llm`` was given, ``True`` to require
    one, ``False`` to never do it. ``vector_dimension`` is the length every vector must have, for
    example the vector size of the target collection. It defaults to ``embedder.dimension``.
    ``strict_scan`` rejects the whole document when any chunk is
    flagged, instead of dropping the flagged chunks. ``retry_on`` lists the exceptions worth
    retrying for the embedder and the writer.
    """

    embed_batch_size: int = 64
    write_batch_size: int = 100
    write_concurrency: int = 4
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    contextualize: bool | None = None
    strict_scan: bool = False
    vector_dimension: int | None = None

    def __post_init__(self) -> None:
        for name in ("embed_batch_size", "write_batch_size", "write_concurrency", "retry_attempts"):
            if getattr(self, name) < 1:
                raise ValueError(f"IngestOptions.{name} must be at least 1")
        if self.retry_base_delay < 0 or self.retry_max_delay < 0:
            raise ValueError("IngestOptions retry delays must not be negative")
        if self.vector_dimension is not None and self.vector_dimension < 1:
            raise ValueError("IngestOptions.vector_dimension must be at least 1")


@dataclass
class DocumentRecord:
    """What the pipeline learned about one document. A small stand-in for a database row."""

    document_id: str
    filename: str = ""
    title: str | None = None
    document_type: str | None = None
    page_count: int = 0
    char_count: int = 0
    parser: str | None = None
    ocr_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityFinding:
    """One chunk the injection scanner flagged, and what it matched."""

    chunk_index: int
    findings: tuple[str, ...] = ()


@dataclass
class IngestResult:
    """Outcome of one ``ingest`` call.

    ``status`` is ``completed``, ``no_text_content`` (a success with zero chunks) or ``failed``.
    ``chunks_excluded`` counts chunks the scanner flagged and the pipeline dropped.
    ``scan_skipped`` is true when a scanner was given but raised, and strict mode was off.
    """

    document_id: str
    success: bool
    status: str
    chunks_stored: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    chunks_excluded: int = 0
    security_findings: list[SecurityFinding] = field(default_factory=list)
    scan_skipped: bool = False
    contextualized: bool = False
    error: str | None = None
    document: DocumentRecord | None = None
    processing_time_ms: float = 0.0

    @property
    def no_text_content(self) -> bool:
        """True for a document that was readable but held no text to index."""
        return self.status == STATUS_NO_TEXT_CONTENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_embedding_dimensions(
    vectors: Sequence[Sequence[float]],
    expected_dim: int,
    *,
    label: str = "the index",
) -> None:
    """Raise ``EmbeddingDimensionError`` when any vector is not ``expected_dim`` long.

    Run before the first write, so a mismatch never reaches the index.
    """
    for i, vector in enumerate(vectors):
        actual = len(vector)
        if actual != expected_dim:
            raise EmbeddingDimensionError(
                f"Embedding dimension mismatch at index {i}: expected {expected_dim} for "
                f"{label}, got {actual}. Check the embedder dimension against the vector size "
                "of the collection."
            )


async def _retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    retry_on: tuple[type[BaseException], ...],
    label: str,
    document_id: str,
) -> T:
    """Run ``operation``, retrying ``retry_on`` errors with exponential backoff."""
    attempt = 1
    while True:
        try:
            return await operation()
        except retry_on as exc:
            if attempt >= attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            _log.warning(
                "ingest_retry",
                step=label,
                document_id=document_id,
                attempt=attempt,
                wait_seconds=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
            attempt += 1


def _clean_text(text: str) -> tuple[str, int]:
    """Strip OCR artifacts and repeated blocks. Returns the text and the characters dropped."""
    cleaned = clean_ocr_artifacts(text)
    if isinstance(cleaned, tuple):
        cleaned = cleaned[0]
    cleaned, removed = deduplicate_repeated_text(cleaned)
    return cleaned, removed


def _jsonable(value: Any) -> Any:
    """Convert enums, dates, paths, sets and nested containers to plain JSON-friendly values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((_jsonable(v) for v in value), key=str)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class _ScanOutcome:
    """What the scan step hands back: the kept chunks and texts, the findings, or an error."""

    chunks: list[Chunk]
    texts: list[str]
    findings: list[SecurityFinding] = field(default_factory=list)
    skipped: bool = False
    error: str | None = None


def _scan_all(scanner: InjectionScanner, texts: Sequence[str]) -> list[ScanResult]:
    return [scanner.scan(text) for text in texts]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Turn a document into indexed points.

    Args:
        embedder: Dense embedder. Its ``dimension`` is the dimension every vector must have.
        writer: Write side of the vector index. It decides the target collection.
        router: A ``DocumentRouter`` used to extract text. Defaults to a plain
            ``DocumentRouter()`` when neither a router nor an extractor is given.
        extractor: An async callable ``(content, filename, document_id) -> IngestorResult``
            used instead of a router. Give a router or an extractor, not both.
        chunker: Any ``BaseChunker``. Defaults to the recursive legal chunker, or the email
            chunker for ``.eml`` and ``.msg`` files.
        sparse_encoder: Optional keyword encoder. Its vectors are attached to each point.
            It is synchronous and runs in a worker thread.
        scanner: Optional injection scanner, synchronous, run in a worker thread.
        llm: Optional language model for contextual retrieval.
        cache: Optional cache for the generated context.
        usage_hook: Optional metering callback, given one event per embedding batch.
        on_status: Optional async ``(stage, message)`` callback.
        payload_extra: Keys copied into every point's payload, for example a tenant key.
            The pipeline never adds such keys itself. Keys in ``RESERVED_PAYLOAD_KEYS`` are
            refused.
        options: ``IngestOptions``.

    An instance holds no per-document state, so one pipeline can run many ingests at once.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        writer: VectorWriter,
        router: DocumentRouter | None = None,
        extractor: Extractor | None = None,
        chunker: BaseChunker | None = None,
        sparse_encoder: SparseEncoder | None = None,
        scanner: InjectionScanner | None = None,
        llm: LanguageModel | None = None,
        cache: Cache | None = None,
        usage_hook: UsageHook | None = None,
        on_status: StatusCallback | None = None,
        payload_extra: Mapping[str, Any] | None = None,
        options: IngestOptions | None = None,
    ) -> None:
        if router is not None and extractor is not None:
            raise ValueError("Pass a router or an extractor, not both")
        self._options = options or IngestOptions()
        if self._options.contextualize and llm is None:
            raise ValueError("IngestOptions.contextualize needs an llm")
        extra = dict(payload_extra or {})
        clash = sorted(set(extra) & RESERVED_PAYLOAD_KEYS)
        if clash:
            raise ValueError(f"payload_extra uses reserved payload keys: {clash}")

        self._embedder = embedder
        self._writer = writer
        self._router = router
        self._extractor = extractor
        self._chunker = chunker
        self._sparse = sparse_encoder
        self._scanner = scanner
        self._llm = llm
        self._cache = cache
        self._usage_hook = usage_hook
        self._on_status = on_status
        self._payload_extra = extra

    # -- public API ----------------------------------------------------------

    async def ingest(
        self,
        source: IngestSource,
        *,
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest one document.

        Args:
            source: File bytes, a path, extracted text, or an ``IngestorResult``.
            document_id: Id stored on every chunk. Re-ingesting the same id is the caller's
                to manage: delete the old points first.
            metadata: Free-form document metadata. ``filename`` names the file for the
                extractor, and ``title`` feeds the chunk headers. Every key lands in each
                chunk's payload, except the ones in ``RESERVED_PAYLOAD_KEYS``.

        Returns:
            An ``IngestResult``. See the module docstring for what returns and what raises.
        """
        if not isinstance(source, _SOURCE_TYPES):
            raise TypeError(
                "source must be bytes, a path, text or an IngestorResult, "
                f"got {type(source).__name__}"
            )
        started = time.monotonic()
        meta = dict(metadata or {})
        record = DocumentRecord(
            document_id=document_id,
            filename=str(meta.get("filename") or ""),
            metadata=dict(meta),
        )
        try:
            result = await self._run(source, document_id, meta, record)
        except Exception as exc:
            await self._status(STAGE_FAILED, str(exc))
            raise
        result.processing_time_ms = (time.monotonic() - started) * 1000
        return result

    # -- orchestration -------------------------------------------------------

    async def _run(
        self,
        source: IngestSource,
        document_id: str,
        meta: dict[str, Any],
        record: DocumentRecord,
    ) -> IngestResult:
        # 1. Extract.
        await self._status(STAGE_EXTRACTING, "Extracting text")
        try:
            extracted = await self._extract(source, document_id, meta)
        except Exception as exc:  # noqa: BLE001 - a parser failure is a document outcome
            _log.error("ingest_extract_failed", document_id=document_id, error=str(exc))
            return await self._fail(record, f"Document parsing failed: {exc}")
        if not extracted.success:
            return await self._fail(record, extracted.error or "Document parsing failed")

        content = extracted.content or ""
        record.filename = record.filename or extracted.filename or ""
        record.page_count = extracted.page_count
        record.char_count = len(content)
        record.parser = extracted.parser_name
        record.ocr_used = extracted.ocr_used

        await self._status(STAGE_EXTRACTED, f"Extracted {len(content)} characters")

        # 2. A document with no text is a success with zero chunks.
        if len(content.strip()) < _NO_TEXT_MIN_CHARS:
            return await self._no_text(record, "Document has no extractable text")

        # 2b. Readability gate, before anything is chunked.
        if len(content.strip()) >= _READABILITY_MIN_CHARS:
            quality = assess_extraction_quality(content)
            if not quality.is_readable:
                _log.warning(
                    "ingest_unreadable_content",
                    document_id=document_id,
                    verdict=quality.verdict.value,
                    total_chars=len(content),
                )
                return await self._fail(
                    record,
                    unreadable_error_message(quality.verdict, ocr_attempted=extracted.ocr_used),
                )

        # 3 and 4. Clean, classify and chunk.
        try:
            chunks, doc_payload = await self._chunk(extracted, document_id, meta, record)
        except Exception as exc:  # noqa: BLE001 - a chunker failure is a document outcome
            _log.error("ingest_chunk_failed", document_id=document_id, error=str(exc))
            return await self._fail(record, f"Chunking failed: {exc}")
        if not chunks:
            return await self._no_text(record, "Document produced no chunks")
        await self._status(STAGE_CHUNKED, f"Created {len(chunks)} chunks")

        # 5. Contextualize.
        texts = [chunk.content for chunk in chunks]
        contextualized = False
        if self._llm is not None and self._options.contextualize is not False:
            texts, contextualized = await self._contextualize(chunks, content, document_id)
            if contextualized:
                await self._status(STAGE_CONTEXTUALIZED, "Added chunk context")

        # 6. Security scan.
        excluded = 0
        findings: list[SecurityFinding] = []
        scan_skipped = False
        if self._scanner is not None:
            outcome = await self._scan(self._scanner, chunks, texts)
            if outcome.error is not None:
                rejected = await self._fail(record, outcome.error)
                rejected.security_findings = outcome.findings
                return rejected
            chunks, texts = outcome.chunks, outcome.texts
            excluded, findings, scan_skipped = (
                len(outcome.findings),
                outcome.findings,
                outcome.skipped,
            )
            if not chunks:
                return await self._fail(
                    record, f"All {excluded} chunks were flagged by the injection scanner"
                )
            await self._status(
                STAGE_SCANNED,
                f"Excluded {excluded} flagged chunks" if excluded else "No flagged chunks",
            )

        # 7. Embed (raises on provider failure or a dimension mismatch).
        vectors = await self._embed(texts, document_id)
        sparse_vectors = (
            await asyncio.to_thread(self._sparse.encode_documents, texts) if self._sparse else None
        )
        await self._status(STAGE_EMBEDDED, f"Embedded {len(vectors)} chunks")

        # 8. Write, with rollback.
        points = self._build_points(chunks, vectors, sparse_vectors, doc_payload, document_id)
        await self._write(points, document_id)
        await self._status(STAGE_STORED, f"Stored {len(points)} chunks")
        await self._status(STAGE_COMPLETED, "Ingest complete")
        return IngestResult(
            document_id=document_id,
            success=True,
            status=STATUS_COMPLETED,
            chunks_stored=len(points),
            chunk_ids=[p.id for p in points],
            chunks_excluded=excluded,
            security_findings=findings,
            scan_skipped=scan_skipped,
            contextualized=contextualized,
            document=record,
        )

    # -- steps ---------------------------------------------------------------

    async def _extract(
        self, source: IngestSource, document_id: str, meta: Mapping[str, Any]
    ) -> IngestorResult:
        if isinstance(source, IngestorResult):
            return source
        if isinstance(source, str):
            return IngestorResult(
                document_id=document_id,
                filename=str(meta.get("filename") or "document.txt"),
                content=source,
                page_count=estimate_pages_from_text(source),
                metadata={"source": "pre_parsed"},
            )
        if isinstance(source, PathLike):
            path = Path(source)
            content = await asyncio.to_thread(path.read_bytes)
            filename = str(meta.get("filename") or path.name)
        else:
            content = bytes(source)
            filename = str(meta.get("filename") or "document")

        if self._extractor is not None:
            return await self._extractor(content, filename, document_id)
        if self._router is None:
            self._router = DocumentRouter()
        routed = await self._router.route_and_parse(
            file_content=content, filename=filename, document_id=document_id
        )
        return routed.ingestor_result

    async def _chunk(
        self,
        extracted: IngestorResult,
        document_id: str,
        meta: Mapping[str, Any],
        record: DocumentRecord,
    ) -> tuple[list[Chunk], dict[str, Any]]:
        """Clean, classify and chunk. Returns the chunks and the document-level payload."""
        filename = record.filename
        is_email = self._chunker is None and Path(filename).suffix.lower() in _EMAIL_SUFFIXES

        text = extracted.content
        pages = extracted.pages
        if not is_email:
            # Never let a parser that gave no cleaned text or no page data collapse every chunk
            # onto page 1, or leak running headers and repeated blocks into the chunks. Derive
            # both from the same text that is chunked, so offsets and page ranges line up.
            if not extracted.text_clean and text:
                cleaned, removed = _clean_text(text)
                if cleaned:
                    text, pages = cleaned, None
                    _log.info("ingest_derived_clean_text", document_id=document_id, removed=removed)
            if not pages and text:
                pages = synthesize_pages_from_text(text)

        doc_type: DocumentType | None = None
        if not is_email:
            try:
                found, _confidence = get_document_type_classifier().classify(text)
                if found != DocumentType.UNKNOWN:
                    doc_type = found
            except Exception as exc:  # noqa: BLE001 - classification is best effort
                _log.warning("ingest_classify_failed", document_id=document_id, error=str(exc))

        doc_payload = self._document_payload(extracted, filename, doc_type, meta)
        record.title = doc_payload.get("title")
        record.document_type = doc_payload.get("document_type")

        chunker = self._chunker or (
            get_email_chunker() if is_email else get_recursive_legal_chunker()
        )
        if is_email:
            chunk_result = await chunker.chunk(
                text=extracted.content,
                document_id=document_id,
                sections=extracted.sections,
                document_metadata=doc_payload,
            )
        else:
            chunk_result = await chunker.chunk(
                text=text,
                document_id=document_id,
                sections=extracted.sections,
                pages=pages,
                boxes=extracted.boxes,
                document_metadata=doc_payload,
                document_type=doc_type,
            )
        return list(chunk_result.chunks), doc_payload

    @staticmethod
    def _document_payload(
        extracted: IngestorResult,
        filename: str,
        doc_type: DocumentType | None,
        meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Document-level payload. What the parser found, overridden by the caller's metadata."""
        payload: dict[str, Any] = {}
        if filename:
            payload["filename"] = filename
        title = (extracted.title or "").strip()
        if title.lower() in _PLACEHOLDER_TITLES:
            title = ""
        title = title or filename
        if title:
            payload["title"] = title
        if doc_type is not None:
            payload["document_type"] = doc_type.value
        if extracted.date:
            payload["date"] = extracted.date
        if extracted.parties:
            payload["parties"] = list(extracted.parties)
        if extracted.jurisdiction:
            payload["jurisdiction"] = extracted.jurisdiction
        if extracted.ocr_used:
            # Absent means unknown, never "text layer": OCR text has its own error profile.
            payload["extraction_method"] = "ocr"

        for key, value in meta.items():
            if key in RESERVED_PAYLOAD_KEYS or value is None:
                continue
            payload[str(key)] = value
        # The chunk headers measure the title, so it must be a string.
        if "title" in payload:
            payload["title"] = str(payload["title"])
        return payload

    async def _contextualize(
        self, chunks: Sequence[Chunk], full_text: str, document_id: str
    ) -> tuple[list[str], bool]:
        """Prepend model-written context to each chunk. Falls back to the plain chunk text."""
        plain = [chunk.content for chunk in chunks]
        if self._llm is None:
            return plain, False
        try:
            enhancer = ContextualRetrievalEnhancer(self._llm, cache=self._cache)
            texts = await enhancer.enhance_chunks(list(chunks), full_text, document_id)
        except Exception as exc:  # noqa: BLE001 - context is an enhancement, never a requirement
            _log.error("ingest_contextualize_failed", document_id=document_id, error=str(exc))
            return plain, False
        if len(texts) != len(chunks):
            return plain, False
        emit_usage(
            self._usage_hook,
            "ingest.contextualize",
            document_id=document_id,
            chunks=len(chunks),
        )
        return texts, True

    async def _scan(
        self, scanner: InjectionScanner, chunks: Sequence[Chunk], texts: Sequence[str]
    ) -> _ScanOutcome:
        """Scan the text that will be embedded. Flagged chunks are dropped, or rejected in
        strict mode. Stored text is never rewritten, so ``sanitized_text`` is not applied."""
        strict = self._options.strict_scan
        try:
            results = await asyncio.to_thread(_scan_all, scanner, list(texts))
        except Exception as exc:  # noqa: BLE001 - see below
            _log.error("ingest_scan_failed", error=str(exc), error_type=type(exc).__name__)
            if strict:
                return _ScanOutcome(
                    list(chunks),
                    list(texts),
                    error=f"Security scan failed and strict mode is on: {exc}",
                )
            # Not strict: keep going, but say the scan did not happen.
            return _ScanOutcome(list(chunks), list(texts), skipped=True)

        flagged = [i for i, r in enumerate(results) if not r.safe]
        findings = [SecurityFinding(i, tuple(results[i].findings)) for i in flagged]
        if flagged and strict:
            return _ScanOutcome(
                list(chunks),
                list(texts),
                findings=findings,
                error=f"Security threat detected: {len(flagged)} flagged chunks",
            )
        if flagged:
            _log.warning("ingest_scan_flagged", flagged=len(flagged), total=len(chunks))
        drop = set(flagged)
        keep = [i for i in range(len(chunks)) if i not in drop]
        return _ScanOutcome([chunks[i] for i in keep], [texts[i] for i in keep], findings=findings)

    async def _embed(self, texts: Sequence[str], document_id: str) -> list[list[float]]:
        """Embed in batches with retry, then check count and dimension before anything is written."""
        opts = self._options
        vectors: list[list[float]] = []
        for start in range(0, len(texts), opts.embed_batch_size):
            batch = list(texts[start : start + opts.embed_batch_size])
            try:
                embedded = await _retry(
                    lambda b=batch: self._embedder.embed_documents(b),
                    attempts=opts.retry_attempts,
                    base_delay=opts.retry_base_delay,
                    max_delay=opts.retry_max_delay,
                    retry_on=opts.retry_on,
                    label="embed",
                    document_id=document_id,
                )
            except Exception as exc:
                raise IngestEmbeddingError(f"Embedding failed: {exc}") from exc
            if len(embedded) != len(batch):
                raise IngestEmbeddingError(
                    f"Embedder returned {len(embedded)} vectors for {len(batch)} texts"
                )
            emit_usage(
                self._usage_hook,
                "ingest.embed",
                input_text="\n".join(batch),
                document_id=document_id,
                count=len(batch),
            )
            vectors.extend(list(v) for v in embedded)
        expected = opts.vector_dimension or self._embedder.dimension
        _validate_embedding_dimensions(vectors, expected)
        return vectors

    def _build_points(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        sparse_vectors: Sequence[Any] | None,
        doc_payload: Mapping[str, Any],
        document_id: str,
    ) -> list[VectorPoint]:
        """One point per kept chunk. Neighbor links and indexes are recomputed over the kept
        chunks, so a dropped chunk never leaves a dangling reference."""
        total = len(chunks)
        points: list[VectorPoint] = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            payload: dict[str, Any] = {
                **doc_payload,
                "chunk_id": chunk.id,
                "document_id": document_id,
                "text": chunk.content,
                "chunk_index": i,
                "total_chunks": total,
                "page_number": chunk.page_number,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "char_start": chunk.start_char,
                "char_end": chunk.end_char,
                "section_title": chunk.section_title,
                "section_number": chunk.section_number,
                "section_type": chunk.section_type,
                "language": chunk.language,
                "prev_chunk_id": chunks[i - 1].id if i > 0 else None,
                "next_chunk_id": chunks[i + 1].id if i + 1 < total else None,
                "citations": list(chunk.citations or []),
                **self._payload_extra,
            }
            points.append(
                VectorPoint(
                    id=chunk.id,
                    vector=vector,
                    sparse=sparse_vectors[i] if sparse_vectors is not None else None,
                    payload=_jsonable(payload),
                )
            )
        return points

    async def _write(self, points: Sequence[VectorPoint], document_id: str) -> None:
        """Write in batches. If any batch fails, delete every id a batch started writing.

        Ids are recorded BEFORE the upsert is awaited, so a write that lands just as the task is
        cancelled is still in the rollback set. Deleting an id that was never written is a no-op.
        """
        opts = self._options
        batches = [
            list(points[i : i + opts.write_batch_size])
            for i in range(0, len(points), opts.write_batch_size)
        ]
        declared: list[str] = []
        gate = asyncio.Semaphore(opts.write_concurrency)

        async def write_one(batch: list[VectorPoint]) -> None:
            async with gate:
                declared.extend(p.id for p in batch)
                await _retry(
                    lambda: self._writer.upsert(batch),
                    attempts=opts.retry_attempts,
                    base_delay=opts.retry_base_delay,
                    max_delay=opts.retry_max_delay,
                    retry_on=opts.retry_on,
                    label="write",
                    document_id=document_id,
                )

        try:
            outcomes = await asyncio.gather(
                *(write_one(b) for b in batches), return_exceptions=True
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._rollback(declared, document_id))
            raise

        failures = [o for o in outcomes if isinstance(o, BaseException)]
        if not failures:
            return
        for extra in failures[1:]:
            _log.warning(
                "ingest_write_secondary_failure",
                document_id=document_id,
                error=str(extra),
                error_type=type(extra).__name__,
            )
        rolled_back = await self._rollback(declared, document_id)
        raise IngestWriteError(
            f"Write failed for {len(failures)} of {len(batches)} batches: {failures[0]}",
            rolled_back=rolled_back,
            orphaned_ids=[] if rolled_back else declared,
        ) from failures[0]

    async def _rollback(self, ids: Sequence[str], document_id: str) -> bool:
        """Delete ids already written. Returns whether the delete went through."""
        if not ids:
            return True
        opts = self._options
        _log.warning("ingest_rollback", document_id=document_id, ids=len(ids))
        try:
            await _retry(
                lambda: self._writer.delete(list(ids)),
                attempts=opts.retry_attempts,
                base_delay=opts.retry_base_delay,
                max_delay=opts.retry_max_delay,
                retry_on=opts.retry_on,
                label="rollback",
                document_id=document_id,
            )
        except Exception as exc:  # noqa: BLE001 - reported through IngestWriteError
            _log.error(
                "ingest_rollback_failed",
                document_id=document_id,
                orphaned=len(ids),
                sample=list(ids[:10]),
                error=str(exc),
            )
            return False
        return True

    # -- status and results --------------------------------------------------

    async def _status(self, stage: str, message: str) -> None:
        if self._on_status is None:
            return
        try:
            await self._on_status(stage, message)
        except Exception as exc:  # noqa: BLE001 - status reporting is best effort
            _log.warning("ingest_status_callback_failed", stage=stage, error=str(exc))

    async def _fail(self, record: DocumentRecord, message: str) -> IngestResult:
        await self._status(STAGE_FAILED, message)
        return IngestResult(
            document_id=record.document_id,
            success=False,
            status=STATUS_FAILED,
            error=message,
            document=record,
        )

    async def _no_text(self, record: DocumentRecord, message: str) -> IngestResult:
        _log.info("ingest_no_text_content", document_id=record.document_id, reason=message)
        await self._status(STAGE_COMPLETED, message)
        return IngestResult(
            document_id=record.document_id,
            success=True,
            status=STATUS_NO_TEXT_CONTENT,
            document=record,
        )


__all__ = [
    "INGEST_STAGES",
    "RESERVED_PAYLOAD_KEYS",
    "DocumentRecord",
    "EmbeddingDimensionError",
    "Extractor",
    "IngestEmbeddingError",
    "IngestError",
    "IngestOptions",
    "IngestPipeline",
    "IngestResult",
    "IngestSource",
    "IngestWriteError",
    "SecurityFinding",
    "StatusCallback",
]
