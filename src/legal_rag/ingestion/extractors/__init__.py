"""
Multi-Format Extractors Package.

Provides text extraction from various document formats:
- Office: DOCX, DOC, XLSX, XLS, PPTX, PPT
- Text: TXT, HTML, RTF, Markdown, CSV, JSON, XML
- Images: PNG, JPEG, TIFF, BMP, GIF, WebP (OCR via Mistral OCR + vision-LLM fallback)
- OpenDocument: ODT, ODS, ODP
- eBook: EPUB
"""

from legal_rag.ingestion.extractors.base import BaseExtractor, ExtractedContent

__all__ = [
    "BaseExtractor",
    "ExtractedContent",
]
