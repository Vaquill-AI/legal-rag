"""Small protocols for the services the ingestion stages depend on.

Nothing in ``legal_rag.ingestion`` reads settings, environment variables or secrets.
OCR, speech transcription, vision description, document conversion, object storage and
circuit breaking are passed in through constructor arguments that satisfy these protocols.
Every protocol here is optional: a stage that is given none of them either skips the
related step or reports a clear error in its result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, TypeVar, runtime_checkable

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised by a breaker when it refuses a call because it is open."""


class OcrPage(TypedDict, total=False):
    """One OCR page: 1-based number, markdown text and character count."""

    page_number: int
    text: str
    char_count: int


class OcrResult(TypedDict, total=False):
    """OCR output: full markdown text, per-page text and the page count."""

    text: str
    pages: list[OcrPage]
    page_count: int


@runtime_checkable
class OcrClient(Protocol):
    """Blocking OCR client.

    Async callers (``MistralOCRIngestor``, ``ImageExtractor``, ``OfficeExtractor``) run it
    with ``asyncio.to_thread``, so an implementation must be safe to call from a worker
    thread. A client that is a coroutine function is not supported.

    Returns ``None`` when the service produced no usable text.
    """

    def ocr(
        self, content: bytes, filename: str, *, include_images: bool = False
    ) -> OcrResult | None: ...


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of a speech transcription call."""

    text: str
    language: str | None = None
    duration_seconds: float | None = None


@runtime_checkable
class TranscriptionClient(Protocol):
    """Speech-to-text client for a file on disk."""

    model_name: str

    async def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> TranscriptionResult: ...


@runtime_checkable
class VisionClient(Protocol):
    """Describe or transcribe an image with a vision-capable model.

    ``prompt`` carries the instruction.
    Returns the model's text, or an empty string when it saw nothing.
    """

    async def describe_image(self, image: bytes, mime_type: str, *, prompt: str) -> str: ...


@runtime_checkable
class ObjectStore(Protocol):
    """Blocking object storage used to hoist inline images out of OCR markdown.

    ``put_object`` stores ``data`` under ``key`` and returns the URL the rewritten
    markdown should reference.
    It raises on failure; callers drop the image reference in that case.
    """

    def put_object(self, key: str, data: bytes, *, content_type: str) -> str: ...


@runtime_checkable
class DocumentConverter(Protocol):
    """Convert a legacy or layout-heavy document to DOCX bytes.

    Returns ``None`` when conversion is not possible.
    """

    async def to_docx(self, content: bytes, filename: str) -> bytes | None: ...


@runtime_checkable
class SyncBreaker(Protocol):
    """Circuit breaker around a blocking call. Raises ``CircuitOpenError`` when open."""

    def call_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T: ...


@runtime_checkable
class AsyncBreaker(Protocol):
    """Circuit breaker around an awaitable call. Raises ``CircuitOpenError`` when open."""

    async def call(self, func: Callable[[], Awaitable[T]]) -> T: ...


class NullBreaker:
    """Breaker that always lets the call through. Used when none is supplied."""

    def call_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        return await func()


ErrorReporter = Callable[[str, BaseException, Mapping[str, Any]], None]
"""Optional hook ``(event, error, context)`` for handled errors worth surfacing."""


def report_error(
    reporter: ErrorReporter | None, event: str, error: BaseException, **context: Any
) -> None:
    """Call ``reporter`` when one is set. A failing reporter never breaks ingestion."""
    if reporter is None:
        return
    try:
        reporter(event, error, context)
    except Exception:  # noqa: BLE001 - a reporting hook must not change the outcome
        return


@runtime_checkable
class ExtractionGate(Protocol):
    """Host-wide gate that limits how many heavy extractions run at once.

    ``guard`` returns a context manager.
    It yields a token when a slot was acquired and ``None`` when the wait budget ran out,
    in which case the caller skips the heavy path.
    An implementation may use a semaphore, a lock file or a distributed counter.
    """

    def guard(self, *, limit: int, max_wait_seconds: float) -> Any: ...
