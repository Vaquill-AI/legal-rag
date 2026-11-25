"""Tests for the ingestion component, using small in-memory fixtures."""

from __future__ import annotations

import base64
import doctest
import re
import subprocess
import sys
import threading
from email.message import EmailMessage
from pathlib import Path

import pytest

from legal_rag import ingestion
from legal_rag.ingestion import mistral_ocr as ocr_module
from legal_rag.ingestion import text_cleaner
from legal_rag.ingestion.audio_ingestor import AudioIngestor
from legal_rag.ingestion.email_file import EmailFileIngestor
from legal_rag.ingestion.extractors.base import estimate_pages_from_text, synthesize_pages_from_text
from legal_rag.ingestion.extractors.image import ImageExtractor
from legal_rag.ingestion.extractors.text import TextExtractor
from legal_rag.ingestion.index_config import (
    HybridCollectionConfig,
    PayloadIndexSpec,
    TextIndexOptions,
    missing_indexes,
)
from legal_rag.ingestion.mistral_ocr import (
    MistralOcrClient,
    MistralOcrConfig,
    MistralOCRIngestor,
    OCRPermanentError,
    mistral_ocr,
)
from legal_rag.ingestion.multi_format import MultiFormatIngestor
from legal_rag.ingestion.protocols import (
    CircuitOpenError,
    NullBreaker,
    ObjectStore,
    OcrClient,
    TranscriptionClient,
    VisionClient,
)
from legal_rag.ingestion.router import DocumentRouter, ParserType, route_and_parse_document
from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts, is_encoding_corrupted
from legal_rag.ingestion.text_quality import is_text_readable

from .conftest import (
    FakeObjectStore,
    FakeOcrClient,
    FakeTranscriber,
    FakeVisionClient,
)

PACKAGE_DIR = Path(ingestion.__file__).parent


# ---------------------------------------------------------------------------
# Package hygiene
# ---------------------------------------------------------------------------


def test_package_import_does_not_load_the_layout_parser() -> None:
    code = (
        "import sys, legal_rag.ingestion;"
        "bad = [m for m in ('legal_rag.ingestion.pymupdf4llm_ingestor', 'pymupdf', 'fitz',"
        " 'pymupdf4llm') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_package_docstring_only_lists_names_the_package_exports() -> None:
    listed = re.findall(r"^- (\w+):", ingestion.__doc__ or "", flags=re.MULTILINE)
    assert listed, "the package docstring should list its ingestors"
    assert set(listed) <= set(ingestion.__all__)
    for name in ("MultiFormatIngestor", "MistralOCRIngestor"):
        assert name not in ingestion.__all__
        assert f"``{name}``" in (ingestion.__doc__ or "")


def test_text_cleaner_docstring_examples_are_true() -> None:
    assert doctest.testmod(text_cleaner).failed == 0
    assert "stats['artifacts_removed']" not in (text_cleaner.__doc__ or "")
    cleaned, stats = clean_ocr_artifacts(
        "Text ⇒ picture [527 x 52] intentionally omitted ⇐ more", return_stats=True
    )
    assert cleaned == "Text more"
    assert stats.artifacts_removed == 1
    with pytest.raises(TypeError):
        stats["artifacts_removed"]  # type: ignore[index]


def test_sources_have_no_forbidden_references() -> None:
    forbidden = ("from app.", "import app.", "from scripts.", "os.environ", "getenv(", "—")
    for path in PACKAGE_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.name} contains {needle!r}"
        for tenancy in ("user_id", "organization_id", "matter_id"):
            assert tenancy not in text, f"{path.name} contains {tenancy}"


def test_layout_module_carries_licence_note() -> None:
    text = (PACKAGE_DIR / "pymupdf4llm_ingestor.py").read_text(encoding="utf-8")
    assert "LICENCE NOTICE" in text and "AGPL" in text


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeOcrClient(), OcrClient)
    assert isinstance(FakeVisionClient(), VisionClient)
    assert isinstance(FakeTranscriber("x"), TranscriptionClient)
    assert isinstance(FakeObjectStore(), ObjectStore)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def test_synthesized_pages_round_trip() -> None:
    text = "\n\n".join(f"paragraph {i} " + "word " * 200 for i in range(12))
    pages = synthesize_pages_from_text(text, chars_per_page=1500)
    assert len(pages) > 1
    assert "\n\n".join(p["text"] for p in pages) == text
    assert [p["page_number"] for p in pages] == list(range(1, len(pages) + 1))
    assert estimate_pages_from_text(text) >= 1


def test_text_quality_and_corruption_detection() -> None:
    assert is_text_readable("The court held that the statute applies to all parties here.")
    assert not is_text_readable("\x00\x01\x02\x03\x04\x05\x06\x07" * 20)
    assert not is_encoding_corrupted("Plain readable English sentence.")


def test_clean_ocr_artifacts_keeps_content() -> None:
    cleaned = clean_ocr_artifacts("Section 5.  The   parties agree.\n\n\n\nSection 6.")
    assert "Section 5" in cleaned and "Section 6" in cleaned


async def test_text_extractor_reads_plain_text() -> None:
    result = await TextExtractor().extract(b"Hello court.\n\nSecond paragraph.", "note.txt")
    assert result.success
    assert "Second paragraph" in (result.text_clean or result.text)


# ---------------------------------------------------------------------------
# Image extractor with injected clients
# ---------------------------------------------------------------------------


async def test_image_extractor_prefers_vision(
    ocr_client: FakeOcrClient, vision_client: FakeVisionClient
) -> None:
    extractor = ImageExtractor(ocr_client, vision_client)
    result = await extractor.extract(b"not-really-a-png", "scan.png")
    assert result.success
    assert "a described image" in result.text
    assert ocr_client.calls == []
    assert vision_client.prompts and "evidence" not in vision_client.prompts[0].lower()


async def test_image_extractor_falls_back_to_ocr(ocr_client: FakeOcrClient) -> None:
    extractor = ImageExtractor(ocr_client, FakeVisionClient(reply="NO_TEXT_FOUND"))
    result = await extractor.extract(b"not-really-a-png", "scan.png")
    assert "scanned text" in result.text
    assert ocr_client.calls == [("scan.png", False)]


async def test_image_extractor_without_clients_returns_no_text() -> None:
    result = await ImageExtractor().extract(b"not-really-a-png", "scan.png")
    assert not (result.text or "").strip()


async def test_image_without_any_client_is_a_failure_not_an_empty_success() -> None:
    result = await ImageExtractor().extract(b"not-really-a-png", "scan.png")
    assert result.success is False
    assert "OcrClient" in (result.error or "") and "VisionClient" in (result.error or "")

    ingested = await MultiFormatIngestor().ingest(
        file_content=b"not-really-a-png", filename="scan.png", document_id="img-1"
    )
    assert ingested.success is False
    assert ingested.content == ""
    assert "no OCR or vision client" in (ingested.error or "")


async def test_image_with_a_client_that_sees_nothing_is_still_a_blank_success() -> None:
    result = await ImageExtractor(vision_client=FakeVisionClient(reply="NO_TEXT_FOUND")).extract(
        b"not-really-a-png", "blank.png"
    )
    assert result.success is True
    assert not result.text


# ---------------------------------------------------------------------------
# OCR client, config and image hoisting
# ---------------------------------------------------------------------------


def _ocr_response(markdown: str) -> dict:
    return {"pages": [{"index": 0, "markdown": markdown, "images": []}]}


def test_mistral_ocr_uses_injected_config_and_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_post(url: str, body: dict, cfg: MistralOcrConfig, timeout: int = 300) -> dict:
        seen["model"] = body["model"]
        seen["key"] = cfg.api_key
        return _ocr_response("# Heading\n\nBody text.")

    monkeypatch.setattr(ocr_module, "_post_json", fake_post)
    cfg = MistralOcrConfig(api_key="test-key", model="ocr-test-model", max_retries=0)
    result = mistral_ocr(b"%PDF-1.4 tiny", "doc.pdf", cfg)
    assert result is not None
    assert "Body text." in result["text"]
    assert seen == {"model": "ocr-test-model", "key": "test-key"}


def test_mistral_ocr_open_breaker_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    class OpenBreaker:
        def call_sync(self, func, *args, **kwargs):
            raise CircuitOpenError("open")

    cfg = MistralOcrConfig(api_key="k", breaker=OpenBreaker(), max_retries=0)
    with pytest.raises(ocr_module.OCRTransientError):
        mistral_ocr(b"%PDF-1.4 tiny", "doc.pdf", cfg)
    assert ocr_module.take_ocr_unavailable_reason() == "circuit_breaker_open"


def test_mistral_ocr_rejects_mislabelled_office_file() -> None:
    cfg = MistralOcrConfig(api_key="k")
    with pytest.raises(OCRPermanentError):
        mistral_ocr(b"definitely not a zip", "memo.docx", cfg)


def test_mistral_ocr_client_is_an_ocr_client() -> None:
    assert isinstance(MistralOcrClient(MistralOcrConfig(api_key="k")), OcrClient)
    assert isinstance(NullBreaker(), object)


def _data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


async def test_ocr_ingestor_hoists_images_to_object_store(object_store: FakeObjectStore) -> None:
    markdown = f"Figure below.\n\n![fig one]({_data_url(b'png-bytes-1')})\n"
    client = FakeOcrClient(text=markdown)
    ingestor = MistralOCRIngestor(client, object_store=object_store, image_key_prefix="imgs")
    result = await ingestor.ingest(file_content=b"%PDF-1.4", filename="a.pdf", document_id="d1")
    assert result.success
    assert "base64" not in result.content
    assert "https://objects.example/imgs/d1/" in result.content
    assert len(object_store.objects) == 1


async def test_ocr_ingestor_strips_images_without_store() -> None:
    markdown = f"Text ![x]({_data_url(b'abc')}) more"
    ingestor = MistralOCRIngestor(FakeOcrClient(text=markdown))
    result = await ingestor.ingest(file_content=b"%PDF-1.4", filename="a.pdf", document_id="d2")
    assert result.success and "base64" not in result.content


async def test_ocr_ingestor_runs_the_blocking_client_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    seen: list[int] = []

    class ThreadRecordingClient(FakeOcrClient):
        def ocr(self, content: bytes, filename: str, *, include_images: bool = False):
            seen.append(threading.get_ident())
            return super().ocr(content, filename, include_images=include_images)

    result = await MistralOCRIngestor(ThreadRecordingClient()).ingest(
        file_content=b"%PDF-1.4", filename="a.pdf", document_id="t1"
    )
    assert result.success
    assert seen and seen[0] != loop_thread


async def test_ocr_ingestor_without_client_reports_error() -> None:
    result = await MistralOCRIngestor().ingest(file_content=b"%PDF", filename="a.pdf")
    assert not result.success and "OCR client" in (result.error or "")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

SPEECH = "The witness stated that the meeting took place on the second floor of the building. " * 2


async def test_audio_ingestor_transcribes_with_fake_client() -> None:
    ingestor = AudioIngestor(FakeTranscriber(SPEECH), cost_per_minute_usd=0.01)
    result = await ingestor.ingest(file_content=b"RIFFfake", filename="clip.wav", document_id="a1")
    assert result.success
    assert result.content.startswith("The witness")
    assert result.metadata["transcription_model"] == "fake-asr"
    assert result.metadata["estimated_cost_usd"] > 0


async def test_audio_ingestor_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("legal_rag.ingestion.audio_ingestor.asyncio.sleep", no_sleep)
    client = FakeTranscriber(SPEECH, failures=2)
    result = await AudioIngestor(client).ingest(file_content=b"abc", filename="x.mp3")
    assert result.success and client.calls == 3


async def test_audio_ingestor_rejects_oversized_and_music_like_audio() -> None:
    big = await AudioIngestor(FakeTranscriber(SPEECH), max_file_size_bytes=2).ingest(
        file_content=b"abcdef", filename="x.mp3"
    )
    assert not big.success and "too large" in (big.error or "")
    sparse = await AudioIngestor(FakeTranscriber("la " * 30, duration=600.0)).ingest(
        file_content=b"abc", filename="x.mp3"
    )
    assert not sparse.success


# ---------------------------------------------------------------------------
# Email, multi-format and router
# ---------------------------------------------------------------------------


def _eml_bytes() -> bytes:
    msg = EmailMessage()
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg["Subject"] = "Filing deadline"
    msg["Date"] = "Mon, 02 Mar 2026 10:00:00 +0000"
    msg.set_content("Please confirm the filing deadline of Friday.")
    return msg.as_bytes()


async def test_email_ingestor_parses_eml() -> None:
    result = await EmailFileIngestor().ingest(file_content=_eml_bytes(), filename="m.eml")
    assert result.success
    assert "filing deadline" in result.content.lower()


async def test_multi_format_handles_text_and_reports_missing_handlers() -> None:
    ingestor = MultiFormatIngestor()
    ok = await ingestor.ingest(file_content=b"Plain body text.", filename="a.txt", document_id="t")
    assert ok.success and "Plain body text." in ok.content
    pdf = await ingestor.ingest(file_content=b"%PDF-1.4", filename="a.pdf", document_id="p")
    assert not pdf.success and "PDF ingestor" in (pdf.error or "")
    audio = await ingestor.ingest(file_content=b"abc", filename="a.mp3", document_id="s")
    assert not audio.success and "audio" in (audio.error or "").lower()


async def test_multi_format_uses_injected_audio_ingestor() -> None:
    ingestor = MultiFormatIngestor(audio_ingestor=AudioIngestor(FakeTranscriber(SPEECH)))
    result = await ingestor.ingest(file_content=b"abc", filename="a.mp3", document_id="s")
    assert result.success


async def test_router_routes_text_through_default_multi_format() -> None:
    routed = await route_and_parse_document(file_content=b"Some text body.", filename="n.txt")
    assert routed.parser_used == ParserType.MULTI_FORMAT
    assert routed.ingestor_result.success


async def test_router_prefers_registered_ocr_parser_as_pdf_fallback() -> None:
    plain = DocumentRouter()
    assert plain._get_fallback_parser(ParserType.PYMUPDF4LLM, "pdf") == ParserType.MULTI_FORMAT
    with_ocr = DocumentRouter(
        parser_factories={ParserType.OCR: lambda: MistralOCRIngestor(FakeOcrClient())}
    )
    assert with_ocr._get_fallback_parser(ParserType.PYMUPDF4LLM, "pdf") == ParserType.OCR
    with pytest.raises(LookupError):
        plain._get_parser(ParserType.OCR)


async def test_router_office_routing_flag() -> None:
    assert DocumentRouter().get_format_routing()["docx"] == ParserType.MULTI_FORMAT
    layout = DocumentRouter(office_via_layout_parser=True).get_format_routing()
    assert layout["docx"] == ParserType.PYMUPDF4LLM


# ---------------------------------------------------------------------------
# Index configuration and the optional layout module
# ---------------------------------------------------------------------------


def test_index_config_is_plain_data_without_names() -> None:
    config = HybridCollectionConfig(
        dense_dimension=1024,
        payload_indexes=(
            PayloadIndexSpec("scope", is_scope_key=True),
            PayloadIndexSpec("body", kind="text", text=TextIndexOptions(min_token_len=3)),
            PayloadIndexSpec("year", kind="integer"),
        ),
    )
    kwargs = config.to_create_kwargs()
    assert "collection_name" not in kwargs
    assert kwargs["vectors_config"]["dense"]["size"] == 1024
    assert kwargs["vectors_config"]["dense"]["quantization_config"]["scalar"]["type"] == "int8"
    assert kwargs["sparse_vectors_config"]["sparse"]["modifier"] == "idf"
    indexes = config.index_kwargs()
    assert indexes[0]["field_schema"]["is_tenant"] is True
    assert indexes[1]["field_schema"]["min_token_len"] == 3
    assert [s.field for s in missing_indexes(config, ["scope"])] == ["body", "year"]


def test_index_config_validation() -> None:
    with pytest.raises(ValueError):
        HybridCollectionConfig(dense_dimension=0)
    with pytest.raises(ValueError):
        HybridCollectionConfig(dense_dimension=8, replication_factor=1, write_consistency_factor=2)


async def test_layout_module_imports_lazily_and_takes_injected_config() -> None:
    from legal_rag.ingestion.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor, PyMuPDFConfig

    ingestor = PyMuPDF4LLMIngestor(PyMuPDFConfig(ocr_dpi=150))
    assert ingestor.config.ocr_dpi == 150
    assert isinstance(await ingestor.supports_format("a.pdf"), bool)
    assert not await ingestor.supports_format("a.mp3")
