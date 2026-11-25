# ruff: noqa: BLE001, ASYNC230
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Multi-Format Ingestor - Universal Document Ingestor Facade.

Intelligently routes document ingestion to specialized handlers:
- PDF → injected PDF ingestor
- Office → OfficeExtractor (DOCX, XLSX, PPTX)
- Text → TextExtractor (TXT, HTML, CSV, JSON, XML)
- Images → ImageExtractor (OCR and vision-model fallback)

This is the recommended default ingestor for user uploads.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.audio_ingestor import AUDIO_EXTENSIONS, AudioIngestor
from legal_rag.ingestion.base import (
    BaseIngestor,
    IngestorResult,
)
from legal_rag.ingestion.extractors.image import ImageExtractor
from legal_rag.ingestion.extractors.office import OfficeExtractor
from legal_rag.ingestion.extractors.text import TextExtractor

if TYPE_CHECKING:
    from legal_rag.ingestion.extractors.base import ExtractedContent

logger = get_logger("legal_rag.ingestion.multi_format")


class MultiFormatIngestor(BaseIngestor):
    """
    Universal document ingestor that handles 25+ file formats.

    Routing logic:
    - PDF → the injected PDF ingestor
    - Office → OfficeExtractor (DOCX, XLSX, PPTX, DOC, XLS, PPT)
    - Text → TextExtractor (TXT, HTML, RTF, CSV, JSON, XML, Markdown)
    - Images → ImageExtractor (PNG, JPEG, TIFF, BMP, GIF, WebP via OCR and vision-model fallback)

    Features:
    - Intelligent format detection
    - Automatic fallback for failed extractions
    - Unified IngestorResult output
    """

    # All supported extensions
    SUPPORTED_FORMATS = (
        # PDF
        {"pdf"}
        # Office
        | {"docx", "dotx", "doc", "xlsx", "xlsm", "xltx", "xltm", "xls", "pptx", "ppt"}
        # Text
        | {"txt", "html", "htm", "rtf", "md", "markdown", "csv", "json", "xml"}
        # Images (heic/heif = iPhone default, decoded via pillow-heif)
        | {"png", "jpeg", "jpg", "tiff", "tif", "bmp", "gif", "webp", "heic", "heif"}
        # Audio (transcribed to text)
        | AUDIO_EXTENSIONS
    )

    def __init__(
        self,
        *,
        pdf_ingestor: BaseIngestor | None = None,
        audio_ingestor: AudioIngestor | None = None,
        office_extractor: OfficeExtractor | None = None,
        text_extractor: TextExtractor | None = None,
        image_extractor: ImageExtractor | None = None,
    ):
        """Initialize the multi-format ingestor with specialized handlers.

        Every handler is injectable. Defaults are built without any external service, so
        OCR, vision, conversion and audio stay off until they are supplied.

        Args:
            pdf_ingestor: Handler for PDFs, for example the optional layout parser.
            audio_ingestor: Handler for .mp3, .wav and .m4a files.
            office_extractor: Office extractor, with its OCR, vision and converter clients.
            text_extractor: Text-format extractor.
            image_extractor: Image extractor, with its OCR and vision clients.
        """
        self.pdf_ingestor: BaseIngestor | None = pdf_ingestor
        self.office_extractor = office_extractor or OfficeExtractor()
        self.text_extractor = text_extractor or TextExtractor()
        self.image_extractor = image_extractor or ImageExtractor()
        self.audio_ingestor: AudioIngestor | None = audio_ingestor

    async def supports_format(self, filename: str) -> bool:
        """Check if this ingestor supports the given file format."""
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in self.SUPPORTED_FORMATS

    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "document",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest a document using the appropriate specialized handler.

        Args:
            file_path: Path to the document file
            file_content: Raw file bytes (alternative to file_path)
            filename: Original filename (required for format detection)
            document_id: Unique document identifier
            **kwargs: Format-specific options passed to handlers

        Returns:
            IngestorResult with extracted content and metadata
        """
        start_time = time.time()
        doc_id = document_id or "unknown"
        ext = self._get_extension(filename)

        # Load file content if only path provided
        if file_content is None and file_path is not None:
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
            except Exception as e:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error=f"Failed to read file: {e!s}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

        if file_content is None:
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error="No file content provided",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        logger.debug(
            "multi_format_routing",
            filename=filename,
            ext=ext,
            size=len(file_content),
        )

        # Route to appropriate handler
        try:
            if ext == "pdf":
                return await self._ingest_pdf(
                    file_path, file_content, filename, doc_id, start_time, **kwargs
                )

            elif ext in AUDIO_EXTENSIONS:
                # Audio files are transcribed to text by the injected audio ingestor.
                if self.audio_ingestor is None:
                    return IngestorResult(
                        document_id=doc_id,
                        filename=filename,
                        content="",
                        success=False,
                        error="No audio ingestor configured",
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
                return await self.audio_ingestor.ingest(
                    file_path=file_path,
                    file_content=file_content,
                    filename=filename,
                    document_id=doc_id,
                    **kwargs,
                )

            elif self.office_extractor.supports(filename):
                return await self._ingest_with_extractor(
                    self.office_extractor,
                    file_content,
                    filename,
                    doc_id,
                    start_time,
                    **kwargs,
                )

            elif self.text_extractor.supports(filename):
                return await self._ingest_with_extractor(
                    self.text_extractor,
                    file_content,
                    filename,
                    doc_id,
                    start_time,
                    **kwargs,
                )

            elif self.image_extractor.supports(filename):
                return await self._ingest_with_extractor(
                    self.image_extractor,
                    file_content,
                    filename,
                    doc_id,
                    start_time,
                    **kwargs,
                )

            else:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error=f"Unsupported file format: {ext}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            logger.exception(
                "multi_format_ingestion_failed",
                filename=filename,
                ext=ext,
                error=str(e),
            )
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error=f"Ingestion failed: {e!s}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _ingest_pdf(
        self,
        file_path: str | None,
        file_content: bytes,
        filename: str,
        document_id: str,
        start_time: float,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest a PDF with the injected PDF ingestor.
        """
        if self.pdf_ingestor is not None:
            result = await self.pdf_ingestor.ingest(
                file_path=file_path,
                file_content=file_content,
                filename=filename,
                document_id=document_id,
                **kwargs,
            )

            logger.debug(
                "pdf_ingested",
                document_id=document_id,
                chars=result.total_chars,
                ocr_used=result.ocr_used,
                success=result.success,
            )
            return result

        # No PDF handler available
        return IngestorResult(
            document_id=document_id,
            filename=filename,
            content="",
            success=False,
            error="No PDF ingestor configured. Pass pdf_ingestor to MultiFormatIngestor.",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    async def _ingest_with_extractor(
        self,
        extractor: Any,
        file_content: bytes,
        filename: str,
        document_id: str,
        start_time: float,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest using an extractor and convert to IngestorResult.
        """
        # Extract content
        extracted: ExtractedContent = await extractor.extract(
            file_content=file_content,
            filename=filename,
            **kwargs,
        )

        if not extracted.success:
            return IngestorResult(
                document_id=document_id,
                filename=filename,
                content="",
                success=False,
                error=extracted.error,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Detect document type
        doc_type = self.detect_document_type(filename)

        # Build metadata
        metadata: dict[str, Any] = extracted.metadata.copy() if extracted.metadata else {}
        if extracted.tables:
            metadata["tables"] = extracted.tables
        if extracted.sheets:
            metadata["sheets"] = extracted.sheets
        if extracted.slides:
            metadata["slides"] = extracted.slides
        if extracted.ocr_confidence is not None:
            metadata["ocr_confidence"] = extracted.ocr_confidence

        processing_time = (time.time() - start_time) * 1000

        # ✅ Use cleaned text as primary content (extractors already apply cleaning)
        content_text = extracted.text_clean or extracted.text

        logger.debug(
            "document_ingested",
            document_id=document_id,
            filename=filename,
            doc_type=doc_type.value,
            chars=len(content_text),
            original_chars=len(extracted.text),
            processing_time_ms=processing_time,
        )

        return IngestorResult(
            document_id=document_id,
            filename=filename,
            content=content_text,
            text_clean=extracted.text_clean,
            document_type=doc_type,
            page_count=extracted.page_count,
            pages=extracted.pages,
            metadata=metadata,
            total_chars=len(content_text),
            parser_name=f"multi_format_{extractor.__class__.__name__.lower()}",
            success=True,
            processing_time_ms=processing_time,
        )

    def _get_extension(self, filename: str) -> str:
        """Get lowercase file extension without dot."""
        return filename.lower().split(".")[-1] if "." in filename else ""
