"""Test doubles for the ingestion protocols.

They are small and in memory: no network, no files outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.ingestion.protocols import OcrResult, TranscriptionResult


class FakeOcrClient:
    """Returns a fixed OCR result and records every call."""

    def __init__(self, text: str = "scanned text", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[str, bool]] = []

    def ocr(
        self, content: bytes, filename: str, *, include_images: bool = False
    ) -> OcrResult | None:
        self.calls.append((filename, include_images))
        if self.fail:
            raise RuntimeError("ocr down")
        page = {"page_number": 1, "text": self.text, "char_count": len(self.text)}
        return {"text": self.text, "pages": [page], "page_count": 1}


class FakeVisionClient:
    """Returns a fixed description for any image."""

    def __init__(self, reply: str = "a described image") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def describe_image(self, image: bytes, mime_type: str, *, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class FakeTranscriber:
    """Transcription client that fails a set number of times, then succeeds."""

    model_name = "fake-asr"

    def __init__(self, text: str, duration: float = 30.0, failures: int = 0) -> None:
        self.text = text
        self.duration = duration
        self.failures = failures
        self.calls = 0

    async def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        self.calls += 1
        if self.calls <= self.failures:
            error = RuntimeError("busy")
            error.status_code = 503  # type: ignore[attr-defined]
            raise error
        return TranscriptionResult(
            text=self.text, language=language or "en", duration_seconds=self.duration
        )


class FakeObjectStore:
    """Keeps stored objects in a dict and returns a fake URL."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    def put_object(self, key: str, data: bytes, *, content_type: str) -> str:
        self.objects[key] = (data, content_type)
        return f"https://objects.example/{key}"


@pytest.fixture
def ocr_client() -> FakeOcrClient:
    return FakeOcrClient()


@pytest.fixture
def vision_client() -> FakeVisionClient:
    return FakeVisionClient()


@pytest.fixture
def object_store() -> FakeObjectStore:
    return FakeObjectStore()
