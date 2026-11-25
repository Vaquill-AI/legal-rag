# ruff: noqa: RUF012
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Base Extractor - Abstract base class for format-specific text extractors.

Extractors are simpler than Ingestors - they only extract text content
without full IngestorResult metadata. Used by MultiFormatIngestor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Standard page estimation: ~3000 characters per page for legal documents.
# Based on US Letter/A4 with 12pt font, 1" margins (2500-3500 chars range).
# Used across ALL non-PDF extractors for consistent quota enforcement.
CHARS_PER_PAGE: int = 3000


def synthesize_pages_from_text(
    text: str,
    chars_per_page: int = CHARS_PER_PAGE,
) -> list[dict[str, Any]]:
    """
    Synthesize virtual pages from text for formats without physical pages.

    Splits at paragraph boundaries ("\\n\\n") every ~chars_per_page characters.
    Critical invariant: "\\n\\n".join(p["text"] for p in result) == text

    Args:
        text: Full document text (already cleaned)
        chars_per_page: Target characters per virtual page (default: 3000)

    Returns:
        List of page dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    if not text:
        return [{"page_number": 1, "text": ""}]

    paragraphs = text.split("\n\n")

    pages: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_length = 0
    page_number = 1

    for para in paragraphs:
        para_len = len(para)

        # If adding this paragraph exceeds limit and we already have content,
        # close current page. Exception: empty page must take at least one paragraph.
        if current_parts and (current_length + 2 + para_len) > chars_per_page:
            pages.append({"page_number": page_number, "text": "\n\n".join(current_parts)})
            page_number += 1
            current_parts = []
            current_length = 0

        current_parts.append(para)
        current_length += para_len + (2 if len(current_parts) > 1 else 0)

    # Flush remaining
    if current_parts:
        pages.append({"page_number": page_number, "text": "\n\n".join(current_parts)})

    return pages or [{"page_number": 1, "text": text}]


def estimate_pages_from_text(text: str) -> int:
    """
    Estimate page count from extracted text content.

    Uses character count as the universal metric since "page" is a rendering
    artifact that doesn't exist in flow-based formats (DOCX, HTML, CSV, etc.).

    This is critical for quota enforcement - underestimating pages allows
    users to bypass page-based limits (e.g., a 200-page DOCX counted as 1 page).

    Args:
        text: Extracted text content

    Returns:
        Estimated page count (minimum 1)
    """
    if not text:
        return 1
    return max(1, len(text) // CHARS_PER_PAGE)


@dataclass
class ExtractedContent:
    """Result from text extraction."""

    text: str  # Extracted text content
    text_clean: str | None = None  # Clean text (optional)
    page_count: int = 1  # Number of pages (if applicable)
    metadata: dict[str, Any] = field(default_factory=dict)  # Format-specific metadata
    success: bool = True
    error: str | None = None

    # For tabular data
    tables: list[dict[str, Any]] | None = None  # Extracted tables
    sheets: list[str] | None = None  # Sheet names (Excel)

    # For presentations
    slides: list[dict[str, Any]] | None = None  # Slide content

    # For images
    ocr_confidence: float | None = None  # OCR confidence score

    # Per-page text data for chunker page range calculation
    # Each page: {"page_number": N, "text": "page content"}
    # Invariant: "\n\n".join(p["text"] for p in pages) == (text_clean or text)
    pages: list[dict[str, Any]] | None = None


class BaseExtractor(ABC):
    """
    Abstract base class for format-specific extractors.

    Each extractor handles a set of related file formats and
    returns ExtractedContent with the extracted text.
    """

    # Set of supported file extensions (without dot, lowercase)
    SUPPORTED: set[str] = set()

    @abstractmethod
    async def extract(
        self,
        file_content: bytes,
        filename: str,
        **kwargs,
    ) -> ExtractedContent:
        """
        Extract text from file content.

        Args:
            file_content: Raw file bytes
            filename: Original filename (for extension detection)
            **kwargs: Format-specific options

        Returns:
            ExtractedContent with extracted text
        """

    def supports(self, filename: str) -> bool:
        """Check if this extractor supports the given filename."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        return ext in self.SUPPORTED

    def get_extension(self, filename: str) -> str:
        """Get lowercase file extension without dot."""
        return filename.lower().split(".")[-1] if "." in filename else ""
