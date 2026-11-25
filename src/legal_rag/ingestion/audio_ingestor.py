# ruff: noqa: BLE001, ASYNC230, SIM115
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Audio file ingestor for the RAG pipeline.

Transcribes audio files (.mp3, .wav, .m4a) through an injected ``TranscriptionClient``,
producing text that can be chunked, embedded and searched.

Features:
- MP3, WAV, M4A format support
- Speech quality gate (rejects music and non-speech, with pause tolerance)
- Duration and word count metadata
- Temporary file management with cleanup
- Optional circuit breaker (``AsyncBreaker``) around the transcription call
- Retry with exponential backoff for transient errors (any exception that carries a
  ``status_code`` of 429, 500, 502 or 503)
- Transcription timeout, so a worker cannot hang on long audio

Usage:
    ingestor = AudioIngestor(my_transcription_client)
    result = await ingestor.ingest(file_content=audio_bytes, filename="recording.mp3")
"""

import asyncio
import math
import os
import random
import tempfile
import time
from pathlib import Path

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.base import BaseIngestor, DocumentSection, IngestorResult
from legal_rag.ingestion.extractors.base import estimate_pages_from_text
from legal_rag.ingestion.protocols import AsyncBreaker, NullBreaker, TranscriptionClient

logger = get_logger("legal_rag.ingestion.audio")

# Audio extensions this ingestor handles
AUDIO_EXTENSIONS = {"mp3", "wav", "m4a"}

# Default upload size cap. Many hosted speech APIs cap uploads near 25 MB; override per client.
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024

# Maximum audio duration we'll process (seconds)
# 25MB mp3 at 64kbps ≈ 50 min; cap at 60 min to bound cost and worker time
MAX_AUDIO_DURATION_SECONDS = 60 * 60  # 60 minutes

# the transcription service timeout: generous to handle long files, but prevents infinite hang
# ~1 minute of audio takes ~5-10s to transcribe; 60 min → ~10 min max
TRANSCRIPTION_TIMEOUT_SECONDS = 600  # 10 minutes

# Retry configuration for transient the transcription service errors
TRANSCRIPTION_MAX_RETRIES = 3  # Total attempts = 1 original + 3 retries
TRANSCRIPTION_RETRY_BASE_DELAY = 2.0  # Base delay in seconds (2s, 4s, 8s with jitter)
# HTTP status codes that warrant a retry
TRANSCRIPTION_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Speech quality gate thresholds (conservative for pause tolerance)
# Normal speech: ~12-15 chars/sec average
# Slow speech with long pauses (depositions, hearings): ~2-5 chars/sec average
# Very sparse speech (90%+ silence): ~1-2 chars/sec
# Music/non-speech hallucination: typically <1 char/sec
MIN_CHARS_PER_SECOND = 1.0  # Rejects music (<1 cps) while allowing very sparse speech (1+ cps)
MIN_TRANSCRIPTION_CHARS = 50  # Absolute minimum text to be useful for search
# Only apply quality gate for audio longer than this (short clips are unreliable)
QUALITY_GATE_MIN_DURATION = 10  # seconds


class AudioIngestor(BaseIngestor):
    """
    Ingestor for audio files using an injected transcription client.

    Transcribes audio to text for RAG indexing and search.
    Protected by an optional circuit breaker, a timeout and retries.

    Args:
        client: Any object satisfying ``TranscriptionClient``.
        breaker: Optional ``AsyncBreaker``. Defaults to pass-through.
        max_file_size_bytes: Reject larger uploads before calling the client.
        timeout_seconds: Per-document transcription timeout.
        max_retries: Retries after the first attempt for transient errors.
        cost_per_minute_usd: When set, an estimated cost is added to the metadata.
    """

    def __init__(
        self,
        client: TranscriptionClient,
        *,
        breaker: AsyncBreaker | None = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        timeout_seconds: float = TRANSCRIPTION_TIMEOUT_SECONDS,
        max_retries: int = TRANSCRIPTION_MAX_RETRIES,
        cost_per_minute_usd: float | None = None,
    ):
        self.client = client
        self.breaker: AsyncBreaker = breaker or NullBreaker()
        self.max_file_size_bytes = max_file_size_bytes
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cost_per_minute_usd = cost_per_minute_usd
        self.model_name = getattr(client, "model_name", "unknown")

    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "audio.mp3",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest an audio file by transcribing it to text.

        Args:
            file_path: Path to audio file
            file_content: Raw audio bytes
            filename: Original filename
            document_id: Document identifier
            **kwargs: Additional options (language hint, etc.)

        Returns:
            IngestorResult with transcribed text
        """
        start_time = time.time()
        doc_id = document_id or "unknown"
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        logger.debug(
            "audio_ingestion_started",
            filename=filename,
            document_id=doc_id,
            extension=ext,
        )

        # Get file content
        if file_content is None and file_path:
            with open(file_path, "rb") as f:
                file_content = f.read()

        if not file_content:
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error="No audio content provided",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Check file size against the configured cap
        if len(file_content) > self.max_file_size_bytes:
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error=(
                    f"Audio file too large ({len(file_content)} bytes). "
                    f"Maximum {self.max_file_size_bytes} bytes."
                ),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Map extension to proper suffix for temp file
        ext_map = {"mp3": ".mp3", "wav": ".wav", "m4a": ".m4a"}
        suffix = ext_map.get(ext, f".{ext}")

        temp_file = None
        try:
            # Write to a temp file (clients read audio from a path)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="audio_ingest_"
            )
            temp_path = Path(temp_file.name)
            temp_file.write(file_content)
            temp_file.close()

            # Transcribe (circuit breaker, timeout and retry protected)
            language = kwargs.get("language")
            result = await self._transcribe_with_protection(temp_path, language, doc_id)

            transcribed_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            duration_seconds = result.get("duration")

            if not transcribed_text:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error="Transcription produced no text (audio may be silent or unintelligible)",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Speech quality gate: detect music/non-speech
            # Conservative thresholds to allow speech with long pauses
            quality_result = self._check_speech_quality(
                text=transcribed_text,
                duration_seconds=duration_seconds,
                filename=filename,
                document_id=doc_id,
            )
            if not quality_result["pass"]:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error=quality_result["reason"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Duration limit check (after transcription since the service reports actual duration)
            if duration_seconds and duration_seconds > MAX_AUDIO_DURATION_SECONDS:
                logger.warning(
                    "audio_duration_exceeded",
                    document_id=doc_id,
                    duration_seconds=duration_seconds,
                    max_seconds=MAX_AUDIO_DURATION_SECONDS,
                )
                # Still process - we already paid for transcription.
                # The limit is enforced at upload time via file size.

            # Build sections from the transcribed text
            sections = [
                DocumentSection(
                    title="Audio Transcription",
                    content=transcribed_text,
                    level=1,
                    metadata={
                        "source": self.model_name,
                        "language": detected_language,
                        "duration_seconds": duration_seconds,
                    },
                )
            ]

            # Estimate page count: for audio, use duration-based estimation
            # ~250 words/page, ~150 words/min speech → ~1.67 min/page
            if duration_seconds and duration_seconds > 0:
                page_count = max(1, math.ceil(duration_seconds / 100))  # ~1.67 min/page
            else:
                page_count = estimate_pages_from_text(transcribed_text)

            processing_time_ms = (time.time() - start_time) * 1000

            # Calculate cost metadata
            estimated_cost = (
                (duration_seconds / 60) * self.cost_per_minute_usd
                if duration_seconds and self.cost_per_minute_usd
                else 0.0
            )

            chars_per_second = len(transcribed_text) / duration_seconds if duration_seconds else 0.0

            logger.debug(
                "audio_ingestion_completed",
                filename=filename,
                document_id=doc_id,
                text_length=len(transcribed_text),
                language=detected_language,
                duration_seconds=duration_seconds,
                page_count=page_count,
                chars_per_second=round(chars_per_second, 2),
                estimated_cost_usd=round(estimated_cost, 4),
                processing_time_ms=processing_time_ms,
            )

            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content=transcribed_text,
                sections=sections,
                document_type=self.detect_document_type(filename),
                page_count=page_count,
                metadata={
                    "audio_language": detected_language,
                    "audio_duration_seconds": duration_seconds,
                    "audio_format": ext,
                    "transcription_model": self.model_name,
                    "file_size_bytes": len(file_content),
                    "chars_per_second": round(chars_per_second, 2),
                    "estimated_cost_usd": round(estimated_cost, 4),
                    "speech_quality": quality_result.get("quality", "unknown"),
                },
                success=True,
                processing_time_ms=processing_time_ms,
                total_chars=len(transcribed_text),
                parser_name="audio_transcription",
            )

        except TimeoutError:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "audio_transcription_timeout",
                filename=filename,
                document_id=doc_id,
                timeout_seconds=self.timeout_seconds,
                processing_time_ms=processing_time_ms,
            )
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error=f"Audio transcription timed out after {self.timeout_seconds}s",
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "audio_ingestion_failed",
                filename=filename,
                document_id=doc_id,
                error=str(e),
                error_type=type(e).__name__,
                processing_time_ms=processing_time_ms,
            )
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error="Audio transcription failed",
                processing_time_ms=processing_time_ms,
            )

        finally:
            # Always clean up temp file
            if temp_file is not None:
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "audio_temp_cleanup_failed",
                        path=str(temp_path),
                        error=str(cleanup_err),
                    )

    async def _transcribe_with_protection(
        self, audio_path: Path, language: str | None, document_id: str
    ) -> dict:
        """
        Transcribe with retry, circuit breaker, and timeout protection.

        Layers (outer → inner):
        1. Circuit breaker - trips after repeated failures, prevents cascading
        2. Timeout - prevents worker hang on unexpectedly long transcriptions
        3. Retry with backoff - handles transient 429/5xx from the transcription service

        Retry sits inside the circuit breaker so each failed retry sequence
        counts as one breaker failure (not N failures for N retries).
        """

        async def _do_transcribe_with_retry():
            return await self._transcribe_with_retry(audio_path, language, document_id)

        # Circuit breaker wraps the timeout + retry protected call
        return await self.breaker.call(
            lambda: asyncio.wait_for(_do_transcribe_with_retry(), timeout=self.timeout_seconds)
        )

    async def _transcribe_with_retry(
        self, audio_path: Path, language: str | None, document_id: str
    ) -> dict:
        """
        Retry the transcription service calls with exponential backoff for transient errors.

        Retries on: 429 (rate limit), 500, 502, 503 (server errors).
        Does NOT retry on: 400 (bad request), 401 (auth), 413 (too large).
        """
        last_exception: Exception | None = None

        for attempt in range(1 + self.max_retries):
            try:
                return await self._transcribe(audio_path, language)
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code not in TRANSCRIPTION_RETRYABLE_STATUS_CODES:
                    raise  # Non-retryable error (auth, bad request, timeout, etc.)
                last_exception = e

                if attempt >= self.max_retries:
                    logger.error(
                        "transcription_retries_exhausted",
                        document_id=document_id,
                        attempts=attempt + 1,
                        status_code=status_code,
                    )
                    raise

                # Exponential backoff with jitter: 2s, 4s, 8s + random(0, 1s)
                delay = TRANSCRIPTION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "transcription_transient_error_retrying",
                    document_id=document_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    status_code=status_code,
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)

        # Should not reach here, but satisfy type checker
        if last_exception:
            raise last_exception
        raise RuntimeError("transcription retry loop exited unexpectedly")

    async def _transcribe(self, audio_path: Path, language: str | None = None) -> dict:
        """
        Call the transcription client.

        Args:
            audio_path: Path to audio file
            language: Optional language hint (ISO-639-1)

        Returns:
            Dict with text, language, duration
        """
        response = await self.client.transcribe(audio_path, language=language)
        return {
            "text": response.text,
            "language": response.language or language,
            "duration": response.duration_seconds,
        }

    def _check_speech_quality(
        self,
        text: str,
        duration_seconds: float | None,
        filename: str,
        document_id: str,
    ) -> dict:
        """
        Check if transcription looks like real speech vs music/noise.

        Uses chars_per_second ratio as a heuristic. Conservative thresholds
        to tolerate speech with long pauses (depositions, hearings, etc.).

        Returns:
            dict with 'pass' (bool), 'reason' (str), 'quality' (str)
        """
        text_len = len(text)

        # Absolute minimum: even with pauses, useful audio has some text
        if text_len < MIN_TRANSCRIPTION_CHARS:
            logger.warning(
                "audio_quality_gate_failed",
                document_id=document_id,
                filename=filename,
                reason="insufficient_text",
                text_length=text_len,
                min_required=MIN_TRANSCRIPTION_CHARS,
            )
            return {
                "pass": False,
                "reason": (
                    f"Transcription too short ({text_len} chars). "
                    "Audio may be silent, music, or unintelligible."
                ),
                "quality": "rejected",
            }

        # Duration-based quality check (only for audio long enough to be reliable)
        if duration_seconds and duration_seconds > QUALITY_GATE_MIN_DURATION:
            chars_per_second = text_len / duration_seconds

            if chars_per_second < MIN_CHARS_PER_SECOND:
                logger.warning(
                    "audio_quality_gate_failed",
                    document_id=document_id,
                    filename=filename,
                    reason="low_speech_density",
                    chars_per_second=round(chars_per_second, 3),
                    threshold=MIN_CHARS_PER_SECOND,
                    duration_seconds=round(duration_seconds, 1),
                    text_length=text_len,
                )
                return {
                    "pass": False,
                    "reason": (
                        f"Audio appears to be music or non-speech "
                        f"({round(chars_per_second, 2)} chars/sec over "
                        f"{round(duration_seconds)}s). "
                        "Only spoken audio can be indexed for search."
                    ),
                    "quality": "rejected",
                }

            # Classify quality level for metadata
            if chars_per_second >= 8.0:
                quality = "good"
            elif chars_per_second >= 3.0:
                quality = "fair"  # Speech with significant pauses
            elif chars_per_second >= 1.0:
                quality = "sparse"  # Very sparse speech, mostly silent
            else:
                quality = "low"  # Borderline - passed gate but barely

            if quality in ("sparse", "low"):
                logger.debug(
                    "audio_quality_gate_warning",
                    document_id=document_id,
                    filename=filename,
                    quality=quality,
                    chars_per_second=round(chars_per_second, 2),
                    duration_seconds=round(duration_seconds, 1),
                )

            return {"pass": True, "reason": "", "quality": quality}

        # Short audio or no duration info - trust the transcription
        return {"pass": True, "reason": "", "quality": "unknown"}

    async def supports_format(self, filename: str) -> bool:
        """Check if filename is a supported audio format."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        return ext in AUDIO_EXTENSIONS
