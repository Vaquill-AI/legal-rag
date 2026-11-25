"""
Abstract base class for document ingestors.
Provides interface for extracting text from various document formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from legal_rag.ingestion._log import get_logger

logger = get_logger("legal_rag.ingestion.base")


class DocumentType(str, Enum):
    """Supported document types."""

    # PDF
    PDF = "pdf"

    # Office Documents
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"

    # Text/Data formats
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "md"
    RTF = "rtf"
    CSV = "csv"
    JSON = "json"
    XML = "xml"

    # Images (OCR)
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"

    # OpenDocument
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"

    # eBook
    EPUB = "epub"

    # Email formats
    EML = "eml"
    MSG = "msg"

    UNKNOWN = "unknown"


@dataclass
class DocumentSection:
    """Represents a section within a document (legal-aware)."""

    title: str | None = None
    content: str = ""
    level: int = 0  # Heading level (0 = no heading)
    section_number: str | None = None  # e.g., "1.2.3", "Article 5"
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestorResult:
    """Result from document ingestion."""

    document_id: str
    filename: str
    content: str  # Full extracted text
    sections: list[DocumentSection] = field(default_factory=list)
    document_type: DocumentType = DocumentType.UNKNOWN
    page_count: int = 1  # Default to 1 (every document is at least 1 page for quota counting)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legal document specific
    title: str | None = None
    date: str | None = None
    parties: list[str] = field(default_factory=list)
    jurisdiction: str | None = None

    # Processing info
    success: bool = True
    error: str | None = None
    processing_time_ms: float = 0.0

    # ============================================
    # PyMuPDF4LLM Specific Fields (Jan 2026)
    # ============================================
    # Box-class data from GNN layout analysis
    # Each box: {page_number, box_class, text}
    # Note: No bbox coordinates stored - use char positions for PDF highlighting
    boxes: list[dict[str, Any]] | None = None

    # Per-page data with char positions
    # Each page: {page_number, text, text_clean, char_count, word_count}
    pages: list[dict[str, Any]] | None = None

    # Clean text without page headers/footers (better for embeddings)
    text_clean: str | None = None

    # Box class distribution stats (e.g., {"title": 5, "text": 120})
    box_class_counts: dict[str, int] | None = None

    # Whether OCR was used during extraction
    ocr_used: bool = False

    # True for PyMuPDF4LLM - supports char_start/char_end positioning
    supports_positions: bool = False

    # Total characters extracted (for quality validation)
    total_chars: int = 0

    # Parser identification
    parser_name: str | None = None

    # Legacy font detection (non-Unicode fonts that extract as garbage)
    has_legacy_fonts: bool = False
    legacy_fonts_found: list[str] = field(default_factory=list)

    # OCR artifact cleaning metrics
    artifacts_removed: int = 0
    cleaning_applied: bool = False


class BaseIngestor(ABC):
    """
    Abstract base class for document ingestors.

    Implementations should handle:
    - Text extraction from documents
    - Section detection (headings, clauses)
    - Metadata extraction
    - Legal document structure awareness
    """

    @abstractmethod
    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "document",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest a document and extract its content.

        Args:
            file_path: Path to the document file
            file_content: Raw file bytes (alternative to file_path)
            filename: Original filename
            document_id: Unique document identifier
            **kwargs: Provider-specific options

        Returns:
            IngestorResult with extracted content and metadata
        """

    @abstractmethod
    async def supports_format(self, filename: str) -> bool:
        """
        Check if this ingestor supports the given file format.

        Args:
            filename: Filename to check

        Returns:
            True if format is supported
        """

    def detect_document_type(self, filename: str) -> DocumentType:
        """Detect document type from filename."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        type_map = {
            # PDF
            "pdf": DocumentType.PDF,
            # Office - Word
            "docx": DocumentType.DOCX,
            "doc": DocumentType.DOC,
            "dotx": DocumentType.DOCX,  # Template
            # Office - Excel
            "xlsx": DocumentType.XLSX,
            "xls": DocumentType.XLS,
            "xlsm": DocumentType.XLSX,  # Macro-enabled
            "xltx": DocumentType.XLSX,  # Template
            "xltm": DocumentType.XLSX,  # Macro template
            # Office - PowerPoint
            "pptx": DocumentType.PPTX,
            "ppt": DocumentType.PPT,
            # Text/Data
            "txt": DocumentType.TXT,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
            "md": DocumentType.MARKDOWN,
            "markdown": DocumentType.MARKDOWN,
            "rtf": DocumentType.RTF,
            "csv": DocumentType.CSV,
            "json": DocumentType.JSON,
            "xml": DocumentType.XML,
            # Images
            "png": DocumentType.PNG,
            "jpeg": DocumentType.JPEG,
            "jpg": DocumentType.JPG,
            "gif": DocumentType.GIF,
            "bmp": DocumentType.BMP,
            "tiff": DocumentType.TIFF,
            "tif": DocumentType.TIFF,
            "webp": DocumentType.WEBP,
            # OpenDocument
            "odt": DocumentType.ODT,
            "ods": DocumentType.ODS,
            "odp": DocumentType.ODP,
            # eBook
            "epub": DocumentType.EPUB,
        }

        doc_type = type_map.get(ext, DocumentType.UNKNOWN)
        logger.debug(
            "document_type_detected",
            filename=filename,
            extension=ext,
            document_type=doc_type.value,
        )
        return doc_type
