"""
Ingestion: document text extraction.

Parsing backends and a router:

- EmailFileIngestor: email files (.eml, .msg)
- DocumentRouter: parser selection with fallback
- Not imported here, import from the submodule: ``legal_rag.ingestion.multi_format``
  (``MultiFormatIngestor``, handler for 25+ formats) and ``legal_rag.ingestion.mistral_ocr``
  (``MistralOCRIngestor``, OCR through an injected ``OcrClient``). Importing
  ``MultiFormatIngestor`` loads the optional Office, image and spreadsheet libraries that are
  installed.
- Optional, never imported here: ``legal_rag.ingestion.pymupdf4llm_ingestor`` (PDF layout
  extraction; read the licence note in that module first)

OCR, transcription, vision, conversion, storage and circuit breaking are injected through
the protocols in ``legal_rag.ingestion.protocols``.

Usage:
    from legal_rag.ingestion import DocumentRouter

    router = DocumentRouter()
    result = await router.route_and_parse(file_content=data, filename="notes.docx")
"""

from legal_rag.ingestion.base import (
    BaseIngestor,
    DocumentSection,
    DocumentType,
    IngestorResult,
)
from legal_rag.ingestion.email_file import EmailFileIngestor
from legal_rag.ingestion.router import (
    DocumentCharacteristic,
    DocumentRouter,
    ParserType,
    RoutingDecision,
    RoutingResult,
    route_and_parse_document,
)

__all__ = [
    # Base classes
    "BaseIngestor",
    "DocumentCharacteristic",
    # Intelligent router
    "DocumentRouter",
    "DocumentSection",
    "DocumentType",
    # Email file parser
    "EmailFileIngestor",
    "IngestorResult",
    "ParserType",
    "RoutingDecision",
    "RoutingResult",
    "route_and_parse_document",
]
