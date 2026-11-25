# ruff: noqa: BLE001, ASYNC230, G201
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Mistral OCR client and ingestor.

Uses the Mistral OCR REST API (standard library HTTP only) for OCR and text extraction.
Returns markdown with table, heading and formatting preserved.

Supported formats:
- Documents: PDF, DOCX, DOC, PPTX, PPT, XLSX, CSV
- Images: JPEG, PNG, TIFF, AVIF

Input methods:
- PDF and images: base64 data URL inline (no upload step)
- Office documents: Files API upload, signed URL, then OCR

Everything that used to come from application settings is carried by
``MistralOcrConfig``: the API key, model, retry limits, rate limit, concurrency cap, an
optional circuit breaker (``SyncBreaker``) and an optional error reporter.
Inline images found in the markdown are hoisted to an ``ObjectStore`` when one is given,
and stripped otherwise.

This module is a reference implementation of the ``OcrClient`` protocol
(``MistralOcrClient``).
Check the provider's current pricing and terms before using it.
"""

import asyncio
import base64
import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.base import (
    BaseIngestor,
    DocumentSection,
    IngestorResult,
)
from legal_rag.ingestion.protocols import (
    CircuitOpenError,
    ErrorReporter,
    NullBreaker,
    ObjectStore,
    OcrClient,
    OcrResult,
    SyncBreaker,
    report_error,
)

logger = get_logger("legal_rag.ingestion.mistral_ocr")

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
OLE2_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


OCR_ENDPOINT = "https://api.mistral.ai/v1/ocr"
FILES_ENDPOINT = "https://api.mistral.ai/v1/files"

# Documented provider limits at the time of writing (verify against the current docs):
#   - 50 MB file size per request (inline base64 AND URL have the same limit)
#   - 1000 pages per request
# No separate inline-base64 limit exists; the 50 MB applies uniformly.
#
# INLINE_SIZE_LIMIT is set 2 MB below the hard ceiling for headroom
# (the JSON + data-URL wrapper adds a few KB; rounding + retries take
# some more).
INLINE_SIZE_LIMIT = 48 * 1024 * 1024  # 48 MB (hard limit is 50 MB)

# We split at 900 pages (not 1000) to leave headroom for off-by-one and
# PyMuPDF rounding edge cases.
MAX_PAGES_PER_REQUEST = 900

# Target byte size per chunk for adaptive splitting. Set 8 MB below the
# 48 MB inline ceiling to leave headroom for the per-chunk PDF container
# overhead that ``insert_pdf`` introduces and for the byte-probe
# granularity (we probe every 5 pages, so a chunk can temporarily grow
# past the target by up to 5 pages' worth before the next probe).
SPLIT_TARGET_BYTES = 40 * 1024 * 1024  # 40 MB


_RETRY_BACKOFF_BASE = 2.0  # seconds (exponential base; jittered at use)


@dataclass
class MistralOcrConfig:
    """Everything the OCR client needs, passed in by the caller.

    Args:
        api_key: Provider API key.
        model: OCR model name. Pin an explicit version in production.
        max_retries: Retries after the first attempt for 429, 5xx and network errors.
        retry_after_max_seconds: Cap on a server-provided ``Retry-After`` wait.
        requests_per_second: Sustained request rate (token bucket).
        burst: Token bucket capacity.
        max_concurrency: Maximum in-flight HTTP calls.
        breaker: Optional circuit breaker. It should not count ``OCRRateLimitError`` or
            ``OCRCredentialError`` as failures (see ``BREAKER_EXCLUDED_EXCEPTIONS``).
        report_error: Optional hook called for handled errors worth surfacing.
    """

    api_key: str
    model: str = "mistral-ocr-latest"
    max_retries: int = 3
    retry_after_max_seconds: float = 60.0
    requests_per_second: float = 1.5
    burst: int = 12
    max_concurrency: int = 6
    breaker: SyncBreaker | None = None
    report_error: ErrorReporter | None = None
    _limiter: Any = field(default=None, init=False, repr=False)
    _semaphore: Any = field(default=None, init=False, repr=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def rate_limiter(self) -> "_TokenBucket":
        with self._lock:
            if self._limiter is None:
                self._limiter = _TokenBucket(rps=self.requests_per_second, burst=self.burst)
            return self._limiter

    def concurrency(self) -> threading.BoundedSemaphore:
        with self._lock:
            if self._semaphore is None:
                self._semaphore = threading.BoundedSemaphore(
                    value=max(1, int(self.max_concurrency))
                )
            return self._semaphore

    def active_breaker(self) -> SyncBreaker:
        return self.breaker or NullBreaker()


# ---------------------------------------------------------------------------
# Global rate limiter (token bucket) + concurrency cap
# ---------------------------------------------------------------------------
#
# The retry loop alone is not enough to prevent 429s. With N concurrent
# workers (or N parallel image OCR calls), every worker can
# blast the API simultaneously, hit 429, then all back off in lockstep
# and re-collide. We fix this with two coordinated limits:
#
#   1. Token-bucket smoothing on requests-per-second so we never sustain
#      more than ``MistralOcrConfig.requests_per_second`` over time.
#   2. A hard concurrency cap on in-flight HTTP calls (semaphore) so a
#      burst of N concurrent callers can't all hold sockets at once.
#
# Both live on the config object, so share one config to share the budget across
# callers in a process. Cross-process coordination is left to the host.


class _TokenBucket:
    """Thread-safe leaky-token-bucket. Blocks `acquire()` until a token
    is available. Refills at ``rps`` tokens/second up to ``burst``."""

    def __init__(self, rps: float, burst: int):
        self._rps = max(0.1, float(rps))
        self._burst = max(1, int(burst))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, name: str = "mistral_ocr") -> None:
        # Loop instead of computing the exact sleep up front so concurrent
        # callers don't all wake at the same instant and stampede.
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(self._burst, self._tokens + elapsed * self._rps)
                    self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                # Need (1 - tokens) tokens; sleep proportionally.
                needed = 1.0 - self._tokens
                wait = needed / self._rps
            # Small jitter so concurrent callers don't wake in lockstep.
            time.sleep(wait + random.uniform(0.0, 0.05))


def _parse_retry_after(header_value: str | None) -> float | None:
    """Parse the Retry-After header. Returns seconds to wait or None if
    the value is missing/unparseable. Both delta-seconds and HTTP-date
    formats are accepted per RFC 7231."""
    if not header_value:
        return None
    s = header_value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(s)
        if target.tzinfo is None:
            # Naive datetime; assume UTC per RFC.

            target = target.replace(tzinfo=UTC)
        from datetime import datetime

        now = datetime.now(UTC)
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return None


class OCRTransientError(Exception):
    """Transient OCR failure (rate limit exhaustion, 5xx, network timeout).

    Callers should retry the whole task rather than surfacing
    "No text could be extracted" to the user.
    """


class OCRPermanentError(Exception):
    """Permanent OCR failure (malformed file, unsupported format, password-protected).

    Callers should mark the job as failed without retrying.
    """


class OCRRateLimitError(Exception):
    """HTTP 429 from Mistral: a THROTTLE, not a fault.

    Deliberately its own type, and deliberately NOT a subclass of
    ``OCRTransientError``, because it is the one signal that must be visible
    to the retry loop while being invisible to the circuit breaker's failure
    accounting (see ``BREAKER_EXCLUDED_EXCEPTIONS``). ``urllib`` reports a 429
    and a 503 as the same ``HTTPError`` class, so a status-code check inside
    the breaker is impossible; the translation has to happen at the transport
    boundary, in ``_post_json``, before ``breaker.call_sync`` ever sees the
    exception.

    ``retry_after`` is the parsed ``Retry-After`` header in seconds, or None.
    """

    def __init__(self, retry_after: float | None = None, detail: str = ""):
        super().__init__(detail or "Mistral OCR rate limited (HTTP 429)")
        self.retry_after = retry_after
        self.detail = detail


class OCRCredentialError(OCRPermanentError):
    """HTTP 401/402 from the provider: the caller's credential or billing is broken.

    A sibling of ``OCRRateLimitError`` in intent. Both are statuses that a naive reading
    files under "the call failed", and both are the dependency answering correctly:

        429  the service is healthy and busy      -> slow down and retry
        401  the key is wrong or revoked          -> stop and alert the operator
        402  the workspace has no active billing  -> stop and alert the operator

    It subclasses ``OCRPermanentError`` so a retrying worker never re-sends a call that
    can only be refused, and it is not an ``OCRTransientError``.
    It is excluded from circuit breaker failure accounting for the same reason a 429 is:
    the breaker asks whether the dependency is unhealthy, and here it is not.
    """

    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(
            detail
            or f"The OCR provider rejected the credential (HTTP {status_code}). "
            "Check the API key and the workspace billing status."
        )
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Why the last OCR attempt on this thread never reached a verdict
# ---------------------------------------------------------------------------
#
# A caller that gets None back from ``mistral_ocr`` cannot tell "the service answered
# and the document is unreadable" from "the service never answered". The first is a
# verdict on the document, the second is an outage.
# The distinction is recorded per thread and consumed on read, so a stale reason is never
# attributed to a later document that took a different path.
# Thread-local is correct because every caller runs one document per worker thread.

_thread_state = threading.local()


def _record_ocr_unavailable(reason: str) -> None:
    """Mark the current thread's OCR attempt as 'never reached the service'."""
    _thread_state.unavailable_reason = reason


def take_ocr_unavailable_reason() -> str | None:
    """Consume-once: why the last ``mistral_ocr()`` call ON THIS THREAD could
    not obtain an answer from the service, or None if it did.

    None means either that OCR succeeded, or that it ran and the response was
    genuinely unusable (empty pages, sludge) -- a verdict on the document.
    A non-None value means no verdict was ever reached.
    """
    reason = getattr(_thread_state, "unavailable_reason", None)
    _thread_state.unavailable_reason = None
    return reason


# ---------------------------------------------------------------------------
# Circuit breaker contract
# ---------------------------------------------------------------------------
#
# A 429 is not a fault and must never count toward opening a breaker: the service is up
# and telling the caller that only its own send rate is wrong, which the token bucket and
# the Retry-After aware backoff already handle.
# ``_post_json`` raises ``OCRRateLimitError`` for a 429 (instead of an ``HTTPError``) so a
# breaker can exclude it by type, and ``OCRCredentialError`` for 401/402 for the same
# reason. 5xx and network errors keep flowing through the breaker: they are evidence the
# dependency is unhealthy.

BREAKER_EXCLUDED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OCRRateLimitError,
    OCRCredentialError,
)
"""Exception types a supplied breaker should not count as failures."""


# ---------------------------------------------------------------------------
# REST helpers (no SDK dependency, matching news/scripts pattern)
# ---------------------------------------------------------------------------


def _as_credential_error(e: BaseException) -> OCRCredentialError | None:
    """Translate a 401/402 ``HTTPError`` into ``OCRCredentialError``, else None.

    Shared by every Mistral call site rather than living only in
    ``_post_json``, because the OCR endpoint is not the only one that can
    refuse us. Office documents go Files-API-upload -> signed-URL -> OCR, so a
    billing lapse rejects the UPLOAD and the request never reaches the OCR
    endpoint at all. Those two helpers used to swallow it into a generic
    "upload failed" and return None, which is the same conflation this class
    exists to prevent: no alert, and the document blamed for the credential.
    """
    if isinstance(e, OCRCredentialError):
        return e
    if isinstance(e, urllib.error.HTTPError) and e.code in (401, 402):
        try:
            detail = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            detail = ""
        return OCRCredentialError(
            status_code=e.code,
            detail=f"{e.code} {e.reason}: {detail}" if detail else "",
        )
    return None


def _post_json(url: str, body: dict, cfg: MistralOcrConfig, timeout: int = 300) -> dict:
    """POST JSON to Mistral API.

    Acquires from the global token bucket + concurrency semaphore before
    sending so that callers across the process share a single rate budget.
    The limiter sleeps the caller until a slot is available; this is the
    correct behavior because each `mistral_ocr()` call runs in a worker
    thread (async callers wrap it with `asyncio.to_thread`).

    A 429 is re-raised as `OCRRateLimitError` HERE, at the transport
    boundary, so it never reaches the circuit breaker as a generic
    `HTTPError`. See ``BREAKER_EXCLUDED_EXCEPTIONS`` for why a throttle must not
    be counted as a fault. Every other status keeps its original exception
    so the breaker's classification is unchanged.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    cfg.rate_limiter().acquire("mistral_ocr_post")
    with cfg.concurrency():
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)  # nosec B310 - fixed provider URL
        except urllib.error.HTTPError as e:
            # Read the body while it is still open: Mistral names the actual
            # problem in it ("Check your subscription on ..."), and by the
            # time this reaches an alert the response is closed.
            credential_error = _as_credential_error(e)
            if credential_error is not None:
                raise credential_error from e
            if e.code != 429:
                raise
            try:
                retry_after = _parse_retry_after(e.headers.get("Retry-After"))
            except Exception:
                retry_after = None
            raise OCRRateLimitError(
                retry_after=retry_after,
                detail=f"{e.code} {e.reason}",
            ) from e
        return json.loads(resp.read().decode("utf-8"))


def _get_signed_url(file_id: str, cfg: MistralOcrConfig) -> str:
    """Get a signed URL for an uploaded file from Mistral Files API."""
    url = f"{FILES_ENDPOINT}/{file_id}/url"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=30)  # nosec B310 - fixed provider URL
            result = json.loads(resp.read().decode("utf-8"))
            return result["url"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(_RETRY_BACKOFF_BASE * (2**attempt))
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise


def _delete_file(file_id: str, cfg: MistralOcrConfig) -> None:
    """Delete an uploaded file from Mistral Files API (cleanup after OCR)."""
    try:
        url = f"{FILES_ENDPOINT}/{file_id}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            method="DELETE",
        )
        urllib.request.urlopen(req, timeout=15)  # nosec B310 - fixed provider URL
    except Exception:
        logger.warning("mistral_ocr_file_cleanup_failed", file_id=file_id)


def _upload_file(file_content: bytes, filename: str, cfg: MistralOcrConfig) -> str:
    """Upload a file to Mistral Files API. Returns file_id."""
    import mimetypes

    boundary = "----LegalRagOcrBoundary"
    mime_type = mimetypes.guess_type(filename)[0] or "application/pdf"

    # Sanitize filename to prevent header injection via quotes or CRLF
    safe_filename = re.sub(r'["\r\n\\]', "_", filename)

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            f"ocr\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode()
        + file_content
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        FILES_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)  # nosec B310 - fixed provider URL
    result = json.loads(resp.read().decode("utf-8"))
    return result["id"]


# ---------------------------------------------------------------------------
# Large PDF splitting (> 50MB)
# ---------------------------------------------------------------------------


def _split_pdf_adaptive(
    src: Any,
    filename: str,
    target_bytes: int = SPLIT_TARGET_BYTES,
    max_pages_per_chunk: int = MAX_PAGES_PER_REQUEST,
) -> list[tuple[int, int, bytes]]:
    """Greedy-pack a PDF into chunks each <= target_bytes AND <= max_pages.

    Walks pages in order. For each page, tentatively appends it to the
    current chunk. If the resulting chunk exceeds ``target_bytes`` OR
    ``max_pages_per_chunk`` AND the chunk already has more than one page,
    the tentative page is removed, the chunk is emitted, and a new chunk
    is started with the same page.

    Guarantees:
    - Every chunk respects BOTH the byte and page limits (except a
      single page that is larger than target on its own, which is
      emitted alone with a warning).
    - Pages are never split across chunks.
    - Order is preserved: chunk N covers pages [N.start, N.end).

    Performance optimization: re-serializing ``tobytes()`` on every page
    append would be O(N^2). We instead check size only every
    ``_PROBE_EVERY`` pages, which is adequate because a single page's
    contribution is small relative to ``target_bytes``.

    Args:
        src: An opened PyMuPDF ``Document`` (caller owns the lifetime).
        filename: For logs only.
        target_bytes: Soft ceiling per chunk in bytes.
        max_pages_per_chunk: Hard ceiling per chunk in pages.

    Returns:
        List of ``(start_page, end_page_exclusive, pdf_bytes)`` tuples.
    """
    import fitz

    total_pages = src.page_count
    chunks: list[tuple[int, int, bytes]] = []
    _PROBE_EVERY = 5  # Check bytes size every N page appends

    current_start = 0
    current_doc = fitz.open()

    def _flush(end_exclusive: int) -> None:
        nonlocal current_doc, current_start
        if current_doc.page_count == 0:
            return
        data = current_doc.tobytes(deflate=True)
        chunks.append((current_start, end_exclusive, data))
        current_doc.close()
        current_doc = fitz.open()
        current_start = end_exclusive

    for page_idx in range(total_pages):
        current_doc.insert_pdf(src, from_page=page_idx, to_page=page_idx)
        pages_in_chunk = current_doc.page_count
        over_pages = pages_in_chunk > max_pages_per_chunk

        # Probe size only every _PROBE_EVERY pages, and always on the
        # final page of the document (to avoid missing the last flush).
        should_probe = (
            (pages_in_chunk % _PROBE_EVERY == 0) or over_pages or (page_idx == total_pages - 1)
        )
        over_size = False
        if should_probe:
            tentative_size = len(current_doc.tobytes(deflate=True))
            over_size = tentative_size > target_bytes

        if (over_size or over_pages) and pages_in_chunk > 1:
            # Remove the just-added page, flush, then re-add to new chunk.
            current_doc.delete_page(pages_in_chunk - 1)
            _flush(page_idx)
            current_doc.insert_pdf(src, from_page=page_idx, to_page=page_idx)
        elif over_size and pages_in_chunk == 1:
            # Single page already exceeds the target. Emit as-is; Mistral's
            # 20 MB absolute ceiling will either accept (if < 20 MB) or
            # raise a hard error (which becomes an OCRPermanentError).
            logger.warning(
                "mistral_ocr_oversized_single_page",
                filename=filename,
                page=page_idx,
                size_mb=round(len(current_doc.tobytes(deflate=True)) / 1024 / 1024, 1),
            )
            _flush(page_idx + 1)

    # Final flush for any trailing pages.
    _flush(total_pages)
    return chunks


def _ocr_large_pdf(
    file_content: bytes,
    filename: str,
    cfg: MistralOcrConfig,
) -> dict | None:
    """Split a large PDF into chunks and OCR each in parallel.

    Routing conditions (any of):
      1. File > 20 MB (inline base64 limit)
      2. Page count > MAX_PAGES_PER_REQUEST (Mistral per-request cap)

    Uses PyMuPDF to split adaptively by target byte size (``SPLIT_TARGET_BYTES``)
    instead of a fixed page count. Fixed-page splitting was wasteful for
    image-heavy scanned PDFs where 40 pages could exceed 20 MB and
    trigger recursive re-splitting through ``mistral_ocr()``. The
    adaptive packer keeps every chunk under the inline limit on the
    first pass, cutting wall time 30-40% on image-heavy documents.

    All-or-nothing semantics: if any chunk fails after its own retry,
    the entire request raises OCRTransientError so the caller can retry
    the whole job. Silent partial success is forbidden because users
    get billed for the full document and have no way to detect missing
    pages.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        import fitz
    except ImportError:
        logger.error("mistral_ocr_large_pdf_no_fitz", filename=filename)
        raise OCRPermanentError("PyMuPDF (fitz) is required for large-PDF OCR but not installed")

    doc = fitz.open(stream=file_content, filetype="pdf")
    total_pages = doc.page_count
    logger.info(
        "mistral_ocr_splitting_large_pdf",
        filename=filename,
        total_pages=total_pages,
        size_mb=round(len(file_content) / 1024 / 1024, 1),
    )

    chunks = _split_pdf_adaptive(doc, filename)
    doc.close()

    if not chunks:
        logger.error("mistral_ocr_split_no_chunks", filename=filename)
        raise OCRPermanentError("PDF contains no pages")

    logger.info(
        "mistral_ocr_adaptive_split",
        filename=filename,
        total_chunks=len(chunks),
        chunk_page_sizes=[end - start for start, end, _ in chunks],
        chunk_byte_sizes_mb=[round(len(data) / 1024 / 1024, 1) for _, _, data in chunks],
    )

    def _ocr_chunk(start: int, end: int, chunk_bytes: bytes) -> tuple[int, dict]:
        """OCR a single chunk. Returns (start_page, result).

        Raises OCRTransientError / OCRPermanentError on failure so that
        ThreadPoolExecutor propagates the exception via future.result().
        """
        chunk_name = f"{filename}_pages_{start + 1}_to_{end}.pdf"
        logger.info("mistral_ocr_processing_chunk", chunk=chunk_name, pages=end - start)
        result = mistral_ocr(chunk_bytes, chunk_name, cfg)
        if result is None:
            raise OCRTransientError(
                f"OCR failed for chunk {chunk_name}. "
                "All-or-nothing policy: aborting the entire document."
            )
        return start, result

    # OCR chunks in parallel (max 3 concurrent to respect Mistral rate limits).
    max_parallel = min(3, len(chunks))
    results_by_start: dict[int, dict] = {}
    first_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_ocr_chunk, start, end, data): (start, end)
            for start, end, data in chunks
        }
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                start_page, chunk_result = future.result()
                for page in chunk_result["pages"]:
                    page["page_number"] = start_page + page["page_number"]
                results_by_start[start_page] = chunk_result
            except (OCRTransientError, OCRPermanentError) as e:
                if first_error is None:
                    first_error = e
                    logger.error(
                        "mistral_ocr_chunk_failed",
                        filename=filename,
                        chunk_start=start,
                        chunk_end=end,
                        error=str(e)[:200],
                    )
                # Keep collecting to drain the pool, but we'll raise below.
            except Exception as e:
                if first_error is None:
                    first_error = OCRTransientError(
                        f"Unexpected error OCRing chunk pages {start + 1}-{end}: {e!s}"
                    )
                    logger.error(
                        "mistral_ocr_chunk_unexpected_error",
                        filename=filename,
                        chunk_start=start,
                        chunk_end=end,
                        exc_info=True,
                    )

    if first_error is not None:
        # Fail the entire document. Silent partial success is forbidden.
        logger.error(
            "mistral_ocr_large_pdf_aborted",
            filename=filename,
            total_chunks=len(chunks),
            successful_chunks=len(results_by_start),
        )
        raise first_error

    if len(results_by_start) != len(chunks):
        # Defensive: should not happen if first_error tracking is correct.
        raise OCRTransientError(
            f"OCR chunk count mismatch: expected {len(chunks)}, got {len(results_by_start)}"
        )

    # Reassemble in page order.
    all_pages = []
    all_text_parts = []
    for start_page in sorted(results_by_start):
        r = results_by_start[start_page]
        all_pages.extend(r["pages"])
        all_text_parts.append(r["text"])

    return {
        "text": "\n\n".join(all_text_parts),
        "pages": all_pages,
        "page_count": len(all_pages),
    }


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------


def _unescape_html_entities(text: str) -> str:
    """Decode HTML entities that Mistral OCR embeds in markdown output.

    Uses stdlib html.unescape for complete entity support including numeric
    entities (&#8212;, &#x2014;, etc.) that the manual approach missed.
    """
    import html as html_mod

    return html_mod.unescape(text)


def _get_image_size_from_b64(b64_str: str) -> tuple[int, int]:
    """Get image dimensions from a base64-encoded image string.

    For Office docs, Mistral returns images without bounding boxes.
    Only decodes the first 32KB of base64 (enough for any image header),
    then uses PIL's lazy loading to read dimensions without decompressing
    the full pixel data. Keeps memory usage constant regardless of image size.

    Returns (width, height) or (0, 0) on failure.
    """
    try:
        import io

        from PIL import Image

        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        # Decode only first 32KB of base64 (enough for PNG IHDR / JPEG SOF)
        # ceil(32768 / 3) * 4 = 43692 base64 chars
        header_b64 = b64_str[:43692]
        img_bytes = base64.b64decode(header_b64 + "==")
        img = Image.open(io.BytesIO(img_bytes))
        return img.size
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Core OCR function
# ---------------------------------------------------------------------------


def mistral_ocr(
    file_content: bytes,
    filename: str,
    cfg: MistralOcrConfig,
    include_images: bool = False,
) -> dict | None:
    """
    OCR a document or image using Mistral REST API.

    Supports PDF, DOCX, PPTX, XLSX, CSV, JPEG, PNG, TIFF, AVIF.

    Routing by format:
    - PDF/images (<= 20MB): base64 data URL inline (fast, no upload step).
    - PDF (> 20MB): split into chunks with PyMuPDF, OCR each, merge results.
    - Office docs (DOCX, PPTX, XLSX): upload to Files API, get signed URL, OCR.

    Mistral hard limits: 50MB file size, 1,000 pages.

    Args:
        include_images: If True, include base64 images in the markdown output.
            Use for OCR extraction (user wants to see images).
            Leave False for RAG pipeline (images bloat the text).

    Returns:
        dict with keys: text, pages, page_count, or None on failure.
    """
    import os

    # Start each document with a clean slate so the reason a caller reads
    # back can only ever describe THIS call.
    _thread_state.unavailable_reason = None

    ext = os.path.splitext(filename)[1].lower()

    # Normalize: strip junk bytes before the file's magic number.
    # Some files arrive with a stray CRLF + UTF-8 BOM written by an upstream
    # tool serving the file in text mode. Strict parsers (including Mistral's
    # OCR API) reject them and return an empty result. PyMuPDF/zipfile are
    # lenient, which is why this goes unnoticed until translation fails.
    _MAGIC_BY_EXT = {
        ".pdf": PDF_MAGIC,
        # OOXML Office formats are ZIP containers (local-file-header magic).
        ".docx": ZIP_MAGIC,
        ".pptx": ZIP_MAGIC,
        ".xlsx": ZIP_MAGIC,
        # Legacy Word 97-2003 .doc is an OLE2/CFB container, NOT ZIP -- matching
        # it against the ZIP magic made every real .doc fail this guard.
        ".doc": OLE2_CFB_MAGIC,
        # NOTE: legacy .ppt is also CFB, but left as ZIP for now (a .pptx
        # mislabeled .ppt still works today); revisit alongside .doc.
        ".ppt": ZIP_MAGIC,
    }
    magic = _MAGIC_BY_EXT.get(ext)
    if magic:
        header_idx = file_content.find(magic)
        if header_idx > 0:
            logger.info(
                "mistral_ocr_stripped_prefix",
                filename=filename,
                ext=ext,
                stripped_bytes=header_idx,
            )
            file_content = file_content[header_idx:]
        elif header_idx < 0:
            logger.warning(
                "mistral_ocr_missing_magic",
                filename=filename,
                ext=ext,
                first_bytes=file_content[:16].hex(),
            )
            raise OCRPermanentError(
                f"File does not look like a valid {ext[1:].upper()}. "
                "The file may be corrupted or mislabeled."
            )

    file_size = len(file_content)

    # Formats that support base64 data URL inline (PDF and images)
    _INLINE_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".avif"}
    # Formats that require file upload first (Office documents)
    _UPLOAD_EXTS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".csv"}

    if ext not in _INLINE_EXTS and ext not in _UPLOAD_EXTS:
        logger.error("mistral_ocr_unsupported_format", filename=filename, ext=ext)
        return None

    is_inline = ext in _INLINE_EXTS
    uploaded_file_id: str | None = None

    # Route to the splitter when either the file exceeds the 20 MB inline
    # limit OR the PDF has more pages than Mistral accepts in a single
    # request. Small-byte large-page PDFs (e.g. text-only case volumes)
    # would otherwise hit Mistral's 1000-page cap and come back empty.
    needs_split = is_inline and file_size > INLINE_SIZE_LIMIT
    if ext == ".pdf" and not needs_split:
        try:
            import fitz

            probe = fitz.open(stream=file_content, filetype="pdf")
            page_count = probe.page_count
            probe.close()
            if page_count > MAX_PAGES_PER_REQUEST:
                logger.info(
                    "mistral_ocr_routing_large_page_count",
                    filename=filename,
                    pages=page_count,
                )
                needs_split = True
        except Exception:
            logger.warning("mistral_ocr_page_probe_failed", filename=filename, exc_info=True)

    if needs_split:
        # A chunked document fails on a worker thread, so the per-chunk
        # recording lands on the wrong thread-local. Re-record on the caller's
        # thread, where the reader is.
        try:
            return _ocr_large_pdf(file_content, filename, cfg)
        except OCRTransientError:
            _record_ocr_unavailable("chunked_transient_failure")
            raise

    if is_inline:
        # PDF/images: base64 data URL (fast, no upload step)
        _EXT_MIME = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tiff": "image/tiff",
            ".avif": "image/avif",
        }
        mime_type = _EXT_MIME.get(ext, "application/pdf")
        file_b64 = base64.b64encode(file_content).decode("utf-8")
        document = {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{file_b64}",
        }
    else:
        # DOCX/PPTX/XLSX: upload to Files API first, then get signed URL
        logger.info("mistral_ocr_uploading_file", filename=filename, ext=ext, size=file_size)
        try:
            uploaded_file_id = _upload_file(file_content, filename, cfg)
            signed_url = _get_signed_url(uploaded_file_id, cfg)
            document = {
                "type": "document_url",
                "document_url": signed_url,
            }
            logger.info("mistral_ocr_file_uploaded", filename=filename, file_id=uploaded_file_id)
        except Exception as e:
            credential_error = _as_credential_error(e)
            if uploaded_file_id:
                _delete_file(uploaded_file_id, cfg)
            if credential_error is not None:
                # A rejected upload is the SAME outage as a rejected OCR call,
                # and it is the only way an Office document sees one. Route it
                # through the retry loop's handler so it alerts and raises
                # exactly like the inline path does.
                logger.error(
                    "mistral_ocr_credential_rejected",
                    filename=filename,
                    status=credential_error.status_code,
                    stage="upload",
                    detail=credential_error.detail[:200],
                )
                report_error(
                    cfg.report_error,
                    "mistral_ocr_credential_rejected",
                    credential_error,
                    status_code=credential_error.status_code,
                    stage="upload",
                    detail=credential_error.detail[:200],
                )
                _record_ocr_unavailable(f"http_{credential_error.status_code}")
                raise credential_error from e
            logger.error("mistral_ocr_upload_failed", filename=filename, error=str(e)[:200])
            _record_ocr_unavailable("upload_failed")
            return None

    # For Office documents (DOCX, PPTX, etc.), Mistral requires
    # include_image_base64=True. If caller doesn't want images, use
    # image_limit=0 to suppress them instead.
    force_base64_images = not is_inline
    body: dict = {
        "model": cfg.model,
        "document": document,
        "include_image_base64": include_images or force_base64_images,
        # NOTE on table_format and extract_header/extract_footer:
        #
        # Tested on real scanned handwritten docs (Adobe Scan 17-Nov-2025.pdf).
        # table_format="html" replaces table content in markdown with
        # placeholders ([tbl-0.html]) and moves actual HTML to a separate
        # "tables" field. For handwritten forms this caused Page 1 to drop
        # from 1084 chars to 65 chars. extract_header/footer similarly
        # strips content from the main body.
        #
        # The default (no table_format) keeps tables inline as markdown
        # pipe tables.
        # Do NOT set these parameters.
    }
    if force_base64_images and not include_images:
        body["image_limit"] = 0

    # Circuit breaker + retry with backoff on transient errors. The breaker
    # check is delegated to call_sync() below so the state machine is
    # actually driven (see _ocr_with_retry). We do NOT fast-fail on
    # is_open here to avoid a TOCTOU race; call_sync handles it internally.
    breaker = cfg.active_breaker()

    try:
        result = _ocr_with_retry(body, cfg, filename, breaker)
    except CircuitOpenError:
        # Breaker is open - the service is down. Surface as transient so
        # the caller retries via backoff, giving the
        # breaker's reset_timeout time to close.
        logger.warning("mistral_ocr_breaker_open", filename=filename)
        if uploaded_file_id:
            _delete_file(uploaded_file_id, cfg)
        _record_ocr_unavailable("circuit_breaker_open")
        raise OCRTransientError(f"Mistral OCR circuit breaker is open for {filename}") from None
    except OCRCredentialError:
        # Reason and alert are already recorded in _ocr_with_retry. Propagate
        # rather than returning None: a caller that gets None cannot tell "we
        # are not paying for this service" from "this document is unreadable",
        # and every caller downstream then blames the document.
        raise
    except Exception:
        _record_ocr_unavailable("unexpected_error")
        return None
    finally:
        # Always clean up uploaded files from Mistral's servers
        if uploaded_file_id:
            _delete_file(uploaded_file_id, cfg)

    if result is None:
        return None

    return _parse_ocr_response(result, filename, include_images)


def _ocr_with_retry(body: dict, cfg: MistralOcrConfig, filename: str, breaker: Any) -> dict | None:
    """Send OCR request with retry on transient errors (429 + 5xx + network).

    Drives the circuit breaker state machine via ``breaker.call_sync``
    so the breaker actually opens after repeated failures. Each retry
    is a separate protected call; the breaker counts each raised
    exception as a failure.

    Error classification:
      - 429 (rate limited): a THROTTLE, retried with Retry-After-aware
        backoff and NEVER counted as a breaker failure. It arrives as
        ``OCRRateLimitError`` because ``_post_json`` translates it at the
        transport boundary; see ``BREAKER_EXCLUDED_EXCEPTIONS``.
      - 401/402 (the caller's credential or billing): permanent, no retry, and NOT a
        breaker fault. Raised as ``OCRCredentialError`` rather than returned
        as None, and alerted on the first occurrence. See that class.
      - other 4xx (client error): permanent, no retry, return None.
      - 5xx / network: transient AND evidence the dependency is unhealthy,
        so these do go through the breaker's failure accounting. Retried
        with backoff, return None only after all attempts are exhausted.
      - CircuitOpenError: propagates to the caller so it can surface
        as OCRTransientError (the whole Mistral service is considered
        down until the breaker's reset_timeout elapses).

    Every path that returns None without the service having rendered a
    verdict on the document records why via ``_record_ocr_unavailable``, so
    a caller can tell an outage from an unreadable document.
    """
    for attempt in range(cfg.max_retries + 1):
        try:
            return breaker.call_sync(_post_json, OCR_ENDPOINT, body, cfg, 300)
        except CircuitOpenError:
            # Let this bubble up so mistral_ocr() can raise OCRTransientError.
            raise
        except OCRCredentialError as e:
            # Not retried on purpose: no number of attempts turns a rejected
            # key or an unpaid workspace into a valid one. Alert on the FIRST
            # one, because this is ours to fix and nobody else will notice.
            logger.error(
                "mistral_ocr_credential_rejected",
                filename=filename,
                status=e.status_code,
                detail=e.detail[:200],
            )
            report_error(
                cfg.report_error,
                "mistral_ocr_credential_rejected",
                e,
                status_code=e.status_code,
                detail=e.detail[:200],
            )
            _record_ocr_unavailable(f"http_{e.status_code}")
            raise
        except OCRRateLimitError as e:
            # Honor the server's Retry-After hint, capped so we don't block a
            # worker indefinitely on a hostile retry header.
            if attempt < cfg.max_retries:
                # Exponential backoff with full jitter + Retry-After override.
                base = _RETRY_BACKOFF_BASE * (2**attempt)
                jittered = random.uniform(base * 0.5, base * 1.5)
                if e.retry_after is not None and e.retry_after > 0:
                    wait = min(e.retry_after, cfg.retry_after_max_seconds)
                else:
                    wait = jittered
                logger.warning(
                    "mistral_ocr_rate_limited",
                    filename=filename,
                    status=429,
                    attempt=attempt,
                    wait_seconds=round(wait, 2),
                    retry_after_header=e.retry_after,
                )
                time.sleep(wait)
                continue
            logger.error(
                "mistral_ocr_rate_limit_retries_exhausted",
                filename=filename,
                retries=cfg.max_retries,
            )
            _record_ocr_unavailable("rate_limited")
            return None
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                error_body = ""
            # 429 can no longer reach here (translated in _post_json), so a
            # transient HTTPError is a 5xx and nothing else.
            is_transient = 500 <= e.code < 600
            if is_transient and attempt < cfg.max_retries:
                base = _RETRY_BACKOFF_BASE * (2**attempt)
                wait = random.uniform(base * 0.5, base * 1.5)
                logger.warning(
                    "mistral_ocr_transient_http_error",
                    filename=filename,
                    status=e.code,
                    attempt=attempt,
                    wait_seconds=round(wait, 2),
                )
                time.sleep(wait)
                continue
            logger.error(
                "mistral_ocr_http_error",
                filename=filename,
                status=e.code,
                transient=is_transient,
                error=error_body[:300],
            )
            _record_ocr_unavailable(f"http_{e.code}")
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < cfg.max_retries:
                wait = _RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "mistral_ocr_network_error_retrying",
                    filename=filename,
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(e)[:200],
                )
                time.sleep(wait)
                continue
            logger.error(
                "mistral_ocr_network_error_exhausted",
                filename=filename,
                error=str(e)[:200],
            )
            _record_ocr_unavailable("network_error")
            return None
        except Exception as e:
            logger.error("mistral_ocr_request_failed", filename=filename, error=str(e)[:200])
            _record_ocr_unavailable("request_failed")
            return None

    logger.error("mistral_ocr_all_retries_failed", filename=filename, retries=cfg.max_retries)
    _record_ocr_unavailable("retries_exhausted")
    return None


def _parse_ocr_response(result: dict, filename: str, include_images: bool) -> dict | None:
    """Parse Mistral OCR API response into structured output."""
    import html as html_mod

    ocr_pages = result.get("pages", [])
    if not ocr_pages:
        logger.warning("mistral_ocr_empty_result", filename=filename)
        return None

    pages = []
    full_text_parts = []

    for page in ocr_pages:
        page_text = page.get("markdown", "")

        # Decode HTML entities (Mistral OCR embeds &amp; &lt; etc.)
        page_text = html_mod.unescape(page_text)

        # If include_images is True, inline base64 images into the markdown.
        # Mistral returns images in page["images"] as:
        #   [{id, top_left_x, top_left_y, bottom_right_x, bottom_right_y, image_base64}]
        # PDF pages include page["dimensions"] = {dpi, height, width}.
        # Office docs have dimensions=null and bbox=null for all images.
        # We encode dimensions into alt text for downstream rendering:
        #   PDF:  ![img-0.jpeg 300x400 of 1806x3210](data:image/...)
        #   DOCX: ![img-0.jpeg 400x200](data:image/...)
        if include_images:
            page_dims = page.get("dimensions") or {}
            page_w = page_dims.get("width", 0)
            page_h = page_dims.get("height", 0)
            page_images = page.get("images", [])
            for img in page_images:
                img_id = img.get("id", "")
                img_b64 = img.get("image_base64", "")
                if img_id and img_b64:
                    br_x = img.get("bottom_right_x")
                    tl_x = img.get("top_left_x")

                    tl_y = img.get("top_left_y")

                    if br_x is not None and tl_x is not None:
                        img_w = br_x - tl_x
                        img_h = (img.get("bottom_right_y") or 0) - (tl_y or 0)
                    else:
                        img_w, img_h = _get_image_size_from_b64(img_b64)

                    if img_w > 0 and img_h > 0 and page_w > 0:
                        # Encode position: "img-0.jpeg 92x94@50,200 of 720x1018"
                        # @x,y = top-left corner in page coordinates
                        if tl_x is not None and tl_y is not None:
                            alt = (
                                f"{img_id} {img_w}x{img_h}"
                                f"@{int(tl_x)},{int(tl_y)}"
                                f" of {page_w}x{page_h}"
                            )
                        else:
                            alt = f"{img_id} {img_w}x{img_h} of {page_w}x{page_h}"
                    elif img_w > 0 and img_h > 0:
                        alt = f"{img_id} {img_w}x{img_h}"
                    else:
                        alt = img_id

                    page_text = page_text.replace(
                        f"![{img_id}]({img_id})",
                        f"![{alt}]({img_b64})",
                    )

        full_text_parts.append(page_text)
        pages.append(
            {
                "page_number": page.get("index", len(pages)) + 1,
                "text": page_text,
                "char_count": len(page_text),
            }
        )

    # Detect and remove bleedthrough pages from duplex scans.
    # Bleedthrough occurs when the back side of a scanned page picks up
    # faint reversed text from the front. Mistral OCR still extracts text
    # from these pages, producing garbled/duplicate content.
    #
    # Detection: a page is bleedthrough if its word set overlaps heavily
    # (>= 50% Jaccard) with an adjacent page AND its own text is short
    # relative to that neighbor (the bleedthrough is always fainter, so
    # OCR extracts less). This is conservative: two genuinely similar
    # pages of equal length won't be flagged.
    bleed_indices: set[int] = set()
    if len(full_text_parts) >= 3:
        _strip_img_re = re.compile(r"!\[[^\]]*\]\([^)]*\)")
        page_word_sets: list[set[str]] = []
        page_word_counts: list[int] = []
        for pt in full_text_parts:
            # Strip image refs so they don't inflate similarity
            clean = _strip_img_re.sub("", pt)
            words = set(re.findall(r"\w{3,}", clean.lower()))
            page_word_sets.append(words)
            page_word_counts.append(len(words))

        for i in range(1, len(full_text_parts)):
            curr_words = page_word_sets[i]
            prev_words = page_word_sets[i - 1]
            if not curr_words or not prev_words:
                continue
            # Jaccard similarity
            intersection = len(curr_words & prev_words)
            union = len(curr_words | prev_words)
            jaccard = intersection / union if union else 0
            # Flag as bleedthrough if:
            # 1. High word overlap with neighbor (>= 50%)
            # 2. Current page has fewer words (bleedthrough is fainter)
            # 3. Current page isn't the longer page (avoids false positives)
            if jaccard >= 0.50 and page_word_counts[i] < page_word_counts[i - 1] * 0.85:
                bleed_indices.add(i)

    if bleed_indices:
        logger.info(
            "bleedthrough_pages_detected",
            filename=filename,
            pages=[pages[i]["page_number"] for i in sorted(bleed_indices)],
            total_pages=len(pages),
        )
        full_text_parts = [pt for i, pt in enumerate(full_text_parts) if i not in bleed_indices]
        pages = [p for i, p in enumerate(pages) if i not in bleed_indices]

    full_text = "\n\n".join(full_text_parts)

    # Reject ONLY if every page is empty. Previously there was an absolute
    # 50-char floor on the whole document, which rejected legitimately
    # tiny valid inputs (single-page receipts, NOCs, short court orders,
    # dismissal notes). The per-page check correctly distinguishes a
    # fully blank scan (all pages empty → reject) from a valid short
    # document (some pages have content → accept).
    non_empty_pages = sum(1 for p in pages if p["text"].strip())
    if non_empty_pages == 0:
        logger.warning(
            "mistral_ocr_all_pages_empty",
            filename=filename,
            total_pages=len(pages),
        )
        return None

    logger.info(
        "mistral_ocr_parsed",
        filename=filename,
        total_pages=len(pages),
        non_empty_pages=non_empty_pages,
        total_chars=len(full_text.strip()),
    )
    return {
        "text": full_text,
        "pages": pages,
        "page_count": len(pages),
    }


# ---------------------------------------------------------------------------
# Inline image hoisting
# ---------------------------------------------------------------------------
#
# With ``include_image_base64`` the OCR response embeds each figure as a base64 data URL
# inside the markdown. Left in place they bloat the text, so each unique image is stored
# through the supplied ``ObjectStore`` and the reference is rewritten to the URL it returns.
# Without a store, the references are removed.

_OCR_IMG_DATA_URL_RE = re.compile(
    r"!\[([^\]]*)\]\((data:image/(?:jpeg|png|webp|gif);base64,([A-Za-z0-9+/=\s]+))\)"
)

# Bounded concurrency for parallel image uploads.
_OCR_IMAGE_UPLOAD_WORKERS = 10


def _derive_image_filename(alt_text: str, data_url: str, b64_data: str, fallback_idx: int) -> str:
    """Build a collision-free filename for an OCR image.

    The alt-derived base is sanitized and a short content hash is appended so two pages
    with matching alt text but different bytes cannot overwrite each other.
    """
    ext = "jpg"
    if "image/png" in data_url:
        ext = "png"
    elif "image/webp" in data_url:
        ext = "webp"
    elif "image/gif" in data_url:
        ext = "gif"

    base_raw = alt_text.replace(" ", "_") if alt_text else f"img_{fallback_idx}"
    base = re.sub(r"[^A-Za-z0-9_\-]", "", base_raw)
    if not base:
        base = f"img_{fallback_idx}"

    digest = hashlib.sha256(b64_data.encode("ascii", errors="ignore")).hexdigest()[:8]
    return f"{base}_{digest}.{ext}"


def _build_ocr_image_url_map(
    markdown_text: str,
    document_id: str,
    store: ObjectStore | None,
    key_prefix: str = "ocr-images",
) -> dict[str, str | None]:
    """Store every unique base64 image in ``markdown_text`` and map data URL to stored URL.

    ``None`` as a value means the upload failed, so the rewriter strips that reference
    rather than leaving broken base64. Without a store the map is empty.
    """
    matches = list(_OCR_IMG_DATA_URL_RE.finditer(markdown_text))
    if not matches:
        return {}
    if store is None:
        logger.info(
            "ocr_image_upload_skipped_no_store",
            document_id=document_id,
            image_count=len(matches),
        )
        return {}

    unique: dict[str, tuple[str, str]] = {}
    for idx, match in enumerate(matches):
        data_url = match.group(2)
        if data_url in unique:
            continue
        unique[data_url] = (match.group(1) or f"img-{idx}", match.group(3).strip())

    def _upload_one(
        data_url: str, alt_text: str, b64_data: str, fallback_idx: int
    ) -> tuple[str, str | None]:
        filename = _derive_image_filename(alt_text, data_url, b64_data, fallback_idx)
        try:
            image_bytes = base64.b64decode(b64_data)
        except (ValueError, TypeError) as e:
            logger.warning(
                "ocr_image_decode_failed",
                document_id=document_id,
                filename=filename,
                error=str(e)[:120],
            )
            return data_url, None

        subtype = filename.rsplit(".", 1)[-1]
        content_type = "image/jpeg" if subtype == "jpg" else f"image/{subtype}"
        key = f"{key_prefix}/{document_id}/{filename}"
        try:
            return data_url, store.put_object(key, image_bytes, content_type=content_type)
        except Exception:
            logger.warning(
                "ocr_image_upload_failed",
                document_id=document_id,
                filename=filename,
                exc_info=True,
            )
            return data_url, None

    url_map: dict[str, str | None] = {}
    upload_start = time.time()
    workers = min(_OCR_IMAGE_UPLOAD_WORKERS, len(unique))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-img-upload") as pool:
        futures = [
            pool.submit(_upload_one, data_url, alt_text, b64_data, idx)
            for idx, (data_url, (alt_text, b64_data)) in enumerate(unique.items())
        ]
        for fut in as_completed(futures):
            data_url, stored_url = fut.result()
            url_map[data_url] = stored_url

    uploaded = sum(1 for v in url_map.values() if v is not None)
    logger.info(
        "ocr_images_processed",
        document_id=document_id,
        total=len(matches),
        unique=len(unique),
        uploaded=uploaded,
        failed=len(url_map) - uploaded,
        duration_ms=round((time.time() - upload_start) * 1000, 1),
    )
    return url_map


def _rewrite_ocr_image_refs(markdown_text: str, url_map: dict[str, str | None]) -> str:
    """Rewrite base64 image refs using a precomputed ``url_map``.

    Refs whose data URL is missing from the map or maps to ``None`` are stripped.
    """
    if not url_map:
        return _OCR_IMG_DATA_URL_RE.sub("", markdown_text)

    def _replace(match: re.Match) -> str:
        alt_text = match.group(1) or ""
        stored_url = url_map.get(match.group(2))
        if stored_url is None:
            return ""
        return f"![{alt_text}]({stored_url})"

    return _OCR_IMG_DATA_URL_RE.sub(_replace, markdown_text)


class MistralOcrClient:
    """``OcrClient`` implementation backed by the Mistral OCR API."""

    def __init__(self, config: MistralOcrConfig) -> None:
        self.config = config

    def ocr(
        self, content: bytes, filename: str, *, include_images: bool = False
    ) -> OcrResult | None:
        result = mistral_ocr(content, filename, self.config, include_images=include_images)
        return result  # type: ignore[return-value]


class MistralOCRIngestor(BaseIngestor):
    """
    Ingestor that uses Mistral OCR API for documents and images.

    Supports PDF, DOCX, PPTX, XLSX, CSV, and common image formats.
    Used as fallback when PyMuPDF4LLM detects a scanned document,
    and as primary extractor for Office docs in the translation pipeline.
    """

    def __init__(
        self,
        client: OcrClient | None = None,
        *,
        object_store: ObjectStore | None = None,
        image_key_prefix: str = "ocr-images",
        model_name: str | None = None,
    ):
        self.client = client
        self.object_store = object_store
        self.image_key_prefix = image_key_prefix
        self.model_name = model_name or getattr(getattr(client, "config", None), "model", "unknown")
        if client is None:
            logger.warning("mistral_ocr_no_client")

    async def supports_format(self, filename: str) -> bool:
        """Mistral OCR supports PDFs, images, and Office docs."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        return ext in {
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "avif",
            "tiff",
            "tif",
            "docx",
            "doc",
            "pptx",
            "ppt",
            "xlsx",
            "csv",
        }

    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "document.pdf",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """Ingest a document or image using Mistral OCR."""
        start_time = time.time()

        if self.client is None:
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error="No OCR client configured",
                parser_name="mistral_ocr",
            )

        # Get file content
        if file_content is None and file_path:
            with open(file_path, "rb") as f:
                file_content = f.read()

        if not file_content:
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error="No file content provided",
                parser_name="mistral_ocr",
            )

        logger.info(
            "mistral_ocr_started",
            filename=filename,
            document_id=document_id,
            size_mb=round(len(file_content) / 1024 / 1024, 1),
        )

        # Ask Mistral to inline base64 image bytes in the markdown so we can
        # hoist them to R2 below. Without this flag, Mistral still emits
        # ``![img-N.jpeg](img-N.jpeg)`` ref placeholders but returns no image
        # data, leaving dangling tokens in the user-facing text (the bug this
        # flag exists to fix).
        # ``OcrClient.ocr`` is a blocking call, so run it in a worker thread and keep the
        # event loop free.
        result = await asyncio.to_thread(
            self.client.ocr, file_content, filename, include_images=True
        )

        processing_time = (time.time() - start_time) * 1000

        if not result:
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error="Mistral OCR returned no text",
                parser_name="mistral_ocr",
                processing_time_ms=processing_time,
            )

        # Hoist inline base64 images to the object store and rewrite
        # markdown refs to the auth-gated proxy URL. The full_text is the
        # concatenation of every page body, so a single scan finds every
        # unique image; we then rewrite full_text + per-page bodies using
        # the same map. Previously we called the upload function once per
        # text blob (1 + N pages), re-uploading every image N+1 times and
        # paying that latency sequentially.
        if document_id:
            url_map = _build_ocr_image_url_map(
                result["text"], document_id, self.object_store, self.image_key_prefix
            )
            result["text"] = _rewrite_ocr_image_refs(result["text"], url_map)
            for page in result["pages"]:
                page["text"] = _rewrite_ocr_image_refs(page["text"], url_map)
                page["char_count"] = len(page["text"])

        full_text = result["text"]
        pages = result["pages"]
        page_count = result["page_count"]

        # Build sections from pages (each page = one section)
        sections = [
            DocumentSection(
                title=f"Page {p['page_number']}",
                content=p["text"],
                level=0,
                page_number=p["page_number"],
            )
            for p in pages
            if p["text"].strip()
        ]

        logger.info(
            "mistral_ocr_completed",
            filename=filename,
            document_id=document_id,
            page_count=page_count,
            total_chars=len(full_text),
            sections=len(sections),
            processing_time_ms=round(processing_time, 1),
        )

        return IngestorResult(
            document_id=document_id or "",
            filename=filename,
            content=full_text,
            sections=sections,
            page_count=page_count,
            pages=[
                {
                    "page_number": p["page_number"],
                    "text": p["text"],
                    "char_count": p["char_count"],
                }
                for p in pages
            ],
            metadata={
                "parser": "mistral_ocr",
                "model": self.model_name,
                "ocr_used": True,
            },
            success=True,
            ocr_used=True,
            total_chars=len(full_text),
            parser_name="mistral_ocr",
            processing_time_ms=processing_time,
        )
