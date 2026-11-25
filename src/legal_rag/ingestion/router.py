# ruff: noqa: BLE001, ASYNC230
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Document router.
Routes documents to a parser based on document characteristics.

Features:
- Automatic document analysis and classification
- Parser selection based on document characteristics
- Support for forced parser selection
- Fallback to secondary parsers on failure
- Hybrid routing for PDFs that mix text pages and scanned pages

Parsers are injected: pass ready-made ingestors (or factories) to ``DocumentRouter``.
The OCR parser is only used when one is registered, and handwriting OCR is only used when a
``VisionClient`` is supplied.

Parser priority:
- PDF: layout parser (primary), OCR parser when registered, then MultiFormat
- Office: MultiFormat, or the layout parser when ``office_via_layout_parser`` is set
- Text: MultiFormat (direct parsing)
- Images: MultiFormat (OCR and vision-model fallback)
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.base import BaseIngestor, IngestorResult
from legal_rag.ingestion.protocols import VisionClient
from legal_rag.ingestion.text_quality import (
    TextQualityVerdict,
    assess_extraction_quality,
)

logger = get_logger("legal_rag.ingestion.router")

DEFAULT_PAGE_OCR_PROMPT = (
    "Extract ALL text from this document page. Preserve paragraph structure. Output tables "
    "as markdown pipe tables. Read both printed and handwritten text. For faint or "
    "hard-to-read text, give your best guess followed by [?]. Output only the extracted "
    "text, no commentary."
)


def unreadable_error_message(verdict: TextQualityVerdict, ocr_attempted: bool) -> str:
    """Actionable error for a document whose text could not be recovered.

    Reached only after the parser chain (including OCR, when available) still
    produced unusable text, so the guidance is tailored to the failure mode.
    """
    ocr_note = (
        " We also tried reading it as an image (OCR) without success." if ocr_attempted else ""
    )
    if verdict in (TextQualityVerdict.GLYPH_NOISE, TextQualityVerdict.SYMBOL_MAPPED):
        return (
            "This PDF has no extractable text layer. It was exported in a way "
            "that stores the words as graphics with no character information "
            "(common with 'Print to PDF' via PostScript/Distiller), so the text "
            f"cannot be read.{ocr_note} Please re-export a text-based PDF "
            "(for example 'Save as PDF' from your word processor, or run "
            "'Recognize Text' / OCR in your PDF tool) and upload again."
        )
    if verdict == TextQualityVerdict.MOJIBAKE:
        return (
            "The text in this document appears to use an unsupported character "
            f"encoding, so it came through garbled.{ocr_note} Please re-save it "
            "as UTF-8 or as a PDF and upload again."
        )
    if verdict == TextQualityVerdict.DRM_SCRAMBLED:
        return (
            "This document appears to be copy-protected (DRM) or font-encrypted, "
            f"so its text cannot be extracted.{ocr_note} Please upload an "
            "unprotected copy."
        )
    return (
        "Could not extract readable text from this document. It may be "
        f"copy-protected, use custom font encoding, or contain only images.{ocr_note} "
        "Try re-uploading a plain-text or unprotected version."
    )


_unreadable_error_message = unreadable_error_message
"""Old private name, kept so existing imports keep working."""


def _is_text_readable(text: str) -> bool:
    """Check if extracted text is usable human-readable content.

    Delegates to :func:`legal_rag.ingestion.text_quality.assess_extraction_quality`,
    which catches every known "looks-extracted-but-is-junk" mode:

    - Glyph-index noise: Type3 / CID / custom-encoded fonts with no ToUnicode
      map. The "Print to PDF via PostScript/Distiller" artifact extracts as
      control bytes, so the printable ratio collapses. These bytes often
      contain whitespace (0x09/0x0a/0x20) by coincidence, which is why the
      older space-ratio / word-length checks alone let them through.
    - Mojibake: wrong charset, dense U+FFFD replacement characters.
    - Symbol mapping: glyphs in the Unicode Private Use Area.
    - DRM / font-encryption: alphanumeric streams with no whitespace and
      absurdly long "words" (some commercial legal databases).

    Returning False routes the document to OCR (recovery); if OCR also fails or
    is unavailable, the caller surfaces an actionable error.
    """
    from legal_rag.ingestion.text_quality import is_text_readable

    return is_text_readable(text)


class DocumentCharacteristic(str, Enum):
    """Document characteristics that influence parser selection."""

    BORN_DIGITAL = "born_digital"  # Native digital PDF/DOCX
    SCANNED = "scanned"  # Scanned document
    HAS_TABLES = "has_tables"  # Contains tables
    HAS_IMAGES = "has_images"  # Contains images/figures
    HANDWRITTEN = "handwritten"  # Contains handwritten text
    POOR_QUALITY = "poor_quality"  # Low quality/degraded
    FORM = "form"  # Structured form
    LEGAL = "legal"  # Legal document
    MULTILINGUAL = "multilingual"  # Multiple languages
    LEGACY_FONTS = "legacy_fonts"  # Non-Unicode fonts (garbage extraction)
    MIXED = "mixed"  # Mixed PDF: some pages text-rich, some scanned


class ParserType(str, Enum):
    """Available parser types."""

    PYMUPDF4LLM = "pymupdf4llm"  # Primary for PDFs - GNN layout analysis
    MULTI_FORMAT = "multi_format"  # Universal handler - 25+ formats
    EMAIL_FILE = "email_file"  # Email files (.eml, .msg)
    OCR = "ocr"  # Injected OCR parser for scanned PDFs


# Parsers whose output should be vetted by the DRM/font-encryption readability
# gate. PDF and OCR pipelines can produce visually-valid-but-semantically-junk
# output (encrypted glyphs, garbled OCR). EMAIL_FILE and MULTI_FORMAT operate
# on structured formats that decode their own content, so the gate would
# misfire on legitimately extracted text.
_READABILITY_GATED_PARSERS: frozenset[ParserType] = frozenset(
    {ParserType.PYMUPDF4LLM, ParserType.OCR}
)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    primary_parser: ParserType
    fallback_parser: ParserType | None = None
    parser_config: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    detected_characteristics: list[DocumentCharacteristic] = field(default_factory=list)
    ocr_analysis: Any = None  # OCRAnalysis from pre-flight scan (if PDF)


@dataclass
class RoutingResult:
    """Result from document routing and parsing."""

    ingestor_result: IngestorResult
    parser_used: ParserType
    fallback_used: bool = False
    routing_decision: RoutingDecision | None = None
    total_time_ms: float = 0.0


class DocumentRouter:
    """
    Intelligent document router that selects optimal parser.

    Routing Strategy:
    1. Analyze document characteristics (file type, size, heuristics)
    2. Select parser based on characteristics:
       - PDF: PyMuPDF4LLM (primary), MultiFormat (fallback)
       - Office docs: MultiFormat
       - Text/Data: MultiFormat
       - Images: MultiFormat (OCR + vision-LLM fallback)
    3. Attempt parsing with fallback on failure
    """

    # Parser capabilities matrix
    PARSER_CAPABILITIES: ClassVar[dict[ParserType, dict[str, Any]]] = {
        ParserType.PYMUPDF4LLM: {
            "formats": {
                "pdf",
                "xps",
                "epub",
                # Office formats (when PyMuPDF Pro is unlocked)
                "doc",
                "docx",
                "xls",
                "xlsx",
                "ppt",
                "pptx",
                "hwp",
                "hwpx",
            },
            "strengths": [
                DocumentCharacteristic.BORN_DIGITAL,
                DocumentCharacteristic.HAS_TABLES,
                DocumentCharacteristic.LEGAL,
            ],
            "weaknesses": [
                DocumentCharacteristic.SCANNED,
                DocumentCharacteristic.HANDWRITTEN,
                DocumentCharacteristic.LEGACY_FONTS,
            ],
            "cost_per_page": 0.0,  # Free
            "speed": "fast",
            "has_layout": True,  # GNN-based layout analysis
        },
        ParserType.MULTI_FORMAT: {
            "formats": {
                "pdf",
                "docx",
                "doc",
                "dotx",
                "xlsx",
                "xlsm",
                "xltx",
                "xltm",
                "xls",
                "pptx",
                "ppt",
                "txt",
                "html",
                "htm",
                "rtf",
                "md",
                "markdown",
                "csv",
                "json",
                "xml",
                "png",
                "jpeg",
                "jpg",
                "tiff",
                "tif",
                "bmp",
                "gif",
                "webp",
                "epub",
            },
            "strengths": [
                DocumentCharacteristic.BORN_DIGITAL,
                DocumentCharacteristic.HAS_TABLES,
            ],
            "weaknesses": [
                DocumentCharacteristic.HANDWRITTEN,
                DocumentCharacteristic.POOR_QUALITY,
            ],
            "cost_per_page": 0.0,  # Free (uses local libraries)
            "speed": "fast",
        },
        ParserType.EMAIL_FILE: {
            "formats": {"eml", "msg"},
            "strengths": [
                DocumentCharacteristic.LEGAL,  # metadata preservation
            ],
            "weaknesses": [],
            "cost_per_page": 0.0,  # Free (local parsing)
            "speed": "fast",
        },
        ParserType.OCR: {
            "formats": {"pdf"},
            "strengths": [
                DocumentCharacteristic.SCANNED,
                DocumentCharacteristic.HANDWRITTEN,
                DocumentCharacteristic.POOR_QUALITY,
                DocumentCharacteristic.HAS_TABLES,
                DocumentCharacteristic.LEGAL,
                DocumentCharacteristic.MULTILINGUAL,
            ],
            "weaknesses": [],
            "cost_per_page": 0.002,  # $2/1K pages
            "speed": "moderate",  # API call, network latency
        },
    }

    @staticmethod
    def _get_format_routing(office_via_layout_parser: bool = False) -> dict[str, ParserType]:
        """
        Get the format routing map.

        Office formats go to the layout parser when ``office_via_layout_parser`` is set,
        otherwise to MultiFormat.
        """
        # Office formats a layout parser may be able to handle
        office_formats_layout = {
            "doc",
            "docx",
            "xls",
            "xlsx",
            "ppt",
            "pptx",
            "hwp",
            "hwpx",
        }

        # Office variant formats that MultiFormat handles better
        office_variants_multiformat = {
            "dotx",  # Word templates
            "xlsm",  # Excel macros
            "xltx",  # Excel templates
            "xltm",  # Excel template macros
        }

        # Base routing for non-Office formats
        routing = {
            # PDF & eBooks - PyMuPDF4LLM primary
            "pdf": ParserType.PYMUPDF4LLM,
            "xps": ParserType.PYMUPDF4LLM,
            "epub": ParserType.PYMUPDF4LLM,
            # Text/Data formats - MultiFormat
            "txt": ParserType.MULTI_FORMAT,
            "html": ParserType.MULTI_FORMAT,
            "htm": ParserType.MULTI_FORMAT,
            "rtf": ParserType.MULTI_FORMAT,
            "md": ParserType.MULTI_FORMAT,
            "markdown": ParserType.MULTI_FORMAT,
            "csv": ParserType.MULTI_FORMAT,
            "json": ParserType.MULTI_FORMAT,
            "xml": ParserType.MULTI_FORMAT,
            # Images - MultiFormat (OCR + vision-LLM fallback)
            "png": ParserType.MULTI_FORMAT,
            "jpeg": ParserType.MULTI_FORMAT,
            "jpg": ParserType.MULTI_FORMAT,
            "tiff": ParserType.MULTI_FORMAT,
            "tif": ParserType.MULTI_FORMAT,
            "bmp": ParserType.MULTI_FORMAT,
            "gif": ParserType.MULTI_FORMAT,
            "webp": ParserType.MULTI_FORMAT,
            # Email files - EmailFile
            "eml": ParserType.EMAIL_FILE,
            "msg": ParserType.EMAIL_FILE,
        }

        # Route Office formats
        if office_via_layout_parser:
            # Use PyMuPDF4LLM for Office formats (superior extraction)
            for fmt in office_formats_layout:
                routing[fmt] = ParserType.PYMUPDF4LLM
            # Office variants still go to MultiFormat
            for fmt in office_variants_multiformat:
                routing[fmt] = ParserType.MULTI_FORMAT
            logger.debug(
                "office_routing_layout_parser",
                message="Routing Office formats to the layout parser",
                formats=list(office_formats_layout),
            )
        else:
            # Use MultiFormat for all Office formats
            for fmt in office_formats_layout | office_variants_multiformat:
                routing[fmt] = ParserType.MULTI_FORMAT
            logger.debug(
                "office_routing_multiformat",
                message="Routing Office formats to MultiFormat",
                formats=list(office_formats_layout),
            )

        return routing

    def get_format_routing(self) -> dict[str, ParserType]:
        """Get the format routing map for this router."""
        return self._get_format_routing(self.office_via_layout_parser)

    def _get_fallback_parser(self, primary: ParserType, file_ext: str) -> ParserType | None:
        """
        Get fallback parser based on primary parser and file type.

        For PDFs:
        - PyMuPDF4LLM → OCR (if an OCR parser is registered) → MultiFormat
        - OCR handles scanned docs, handwriting, and tables natively

        For Office formats:
        - PyMuPDF4LLM → MultiFormat
        """
        if primary == ParserType.PYMUPDF4LLM:
            # For PDFs, prefer OCR over MultiFormat
            if file_ext == "pdf" and self._has_parser(ParserType.OCR):
                return ParserType.OCR
            return ParserType.MULTI_FORMAT

        elif primary == ParserType.OCR:
            return ParserType.MULTI_FORMAT  # Final fallback if the OCR provider fails

        elif primary == ParserType.MULTI_FORMAT:
            return None  # No further fallback

        elif primary == ParserType.EMAIL_FILE:
            return None  # No fallback for email files

        return None

    CONFIDENCE_THRESHOLD = 0.75  # Below this, try fallback

    def __init__(
        self,
        parsers: Mapping[ParserType, BaseIngestor] | None = None,
        *,
        parser_factories: Mapping[ParserType, Callable[[], BaseIngestor]] | None = None,
        vision_client: VisionClient | None = None,
        vision_prompt: str = DEFAULT_PAGE_OCR_PROMPT,
        office_via_layout_parser: bool = False,
    ):
        """Initialize the router.

        Args:
            parsers: Ready-made ingestors by parser type.
            parser_factories: Lazy constructors used when a parser is first needed.
                MultiFormat and email parsers have no-argument defaults when absent.
                The OCR parser has no default: register one to enable OCR routing.
                The layout parser (an optional module with its own licence note) must be
                registered or importable on demand.
            vision_client: Enables handwriting OCR for scanned pages.
            vision_prompt: Instruction sent with each rendered page.
            office_via_layout_parser: Route Office formats to the layout parser.
        """
        self._parsers: dict[ParserType, BaseIngestor] = dict(parsers or {})
        self._factories: dict[ParserType, Callable[[], BaseIngestor]] = dict(parser_factories or {})
        self.vision_client = vision_client
        self.vision_prompt = vision_prompt
        self.office_via_layout_parser = office_via_layout_parser

    def _has_parser(self, parser_type: ParserType) -> bool:
        """Whether a parser of this type is registered (instance or factory)."""
        return parser_type in self._parsers or parser_type in self._factories

    def _get_parser(self, parser_type: ParserType) -> BaseIngestor:
        """Get or create the parser instance for ``parser_type``."""
        if parser_type in self._parsers:
            return self._parsers[parser_type]
        if parser_type in self._factories:
            self._parsers[parser_type] = self._factories[parser_type]()
        elif parser_type == ParserType.PYMUPDF4LLM:
            # Optional module: imported only when this parser is actually requested.
            from legal_rag.ingestion.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor

            self._parsers[parser_type] = PyMuPDF4LLMIngestor()
        elif parser_type == ParserType.MULTI_FORMAT:
            from legal_rag.ingestion.multi_format import MultiFormatIngestor

            self._parsers[parser_type] = MultiFormatIngestor()
        elif parser_type == ParserType.EMAIL_FILE:
            from legal_rag.ingestion.email_file import EmailFileIngestor

            self._parsers[parser_type] = EmailFileIngestor()
        else:
            raise LookupError(f"No parser registered for {parser_type.value}")
        return self._parsers[parser_type]

    async def _vision_ocr_pdf(
        self, file_content: bytes, filename: str, document_id: str, max_pages: int = 20
    ) -> IngestorResult:
        """Run page-by-page vision OCR with the injected client."""
        if self.vision_client is None:
            raise ValueError("No vision client configured for handwriting OCR")
        return await vision_ocr_pdf(
            self.vision_client,
            file_content,
            filename,
            document_id,
            max_pages=max_pages,
            prompt=self.vision_prompt,
        )

    async def route_and_parse(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "document",
        document_id: str | None = None,
        force_parser: ParserType | None = None,
        characteristics: list[DocumentCharacteristic] | None = None,
        **kwargs,
    ) -> RoutingResult:
        """
        Route document to optimal parser and parse it.

        Args:
            file_path: Path to document file
            file_content: Raw file bytes
            filename: Original filename
            document_id: Unique document identifier
            force_parser: Force specific parser (skip routing)
            characteristics: Pre-detected document characteristics
            **kwargs: Additional parser options

        Returns:
            RoutingResult with parsed content and routing info
        """
        start_time = time.time()

        # Get file content if needed
        if file_content is None and file_path:
            with open(file_path, "rb") as f:
                file_content = f.read()

        # Make routing decision
        if force_parser:
            decision = RoutingDecision(
                primary_parser=force_parser,
                confidence=1.0,
                reasoning="Forced parser selection",
            )
        else:
            decision = await self._make_routing_decision(
                file_content=file_content,
                filename=filename,
                characteristics=characteristics,
                **kwargs,
            )

        # Get file size for logging
        file_size_bytes = len(file_content) if file_content else 0

        logger.debug(
            "ingestion_routing_started",
            filename=filename,
            file_size_bytes=file_size_bytes,
            primary_parser=decision.primary_parser.value,
            fallback_parser=(decision.fallback_parser.value if decision.fallback_parser else None),
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            characteristics=[c.value for c in decision.detected_characteristics],
        )

        # Hybrid routing for mixed PDFs (some pages text, some scanned).
        # Split PDF by page type, route each range to the right parser, merge.
        if (
            decision.ocr_analysis
            and decision.ocr_analysis.strategy == "mixed"
            and DocumentCharacteristic.MIXED in decision.detected_characteristics
            and file_content
        ):
            try:
                hybrid_result = await self._hybrid_route_and_parse(
                    file_content=file_content,
                    filename=filename,
                    document_id=document_id or "",
                    ocr_analysis=decision.ocr_analysis,
                    **kwargs,
                )
                total_time = (time.time() - start_time) * 1000
                hybrid_result.total_time_ms = total_time
                hybrid_result.routing_decision = decision
                return hybrid_result
            except Exception as hybrid_err:
                logger.warning(
                    "hybrid_routing_fallback",
                    filename=filename,
                    error=str(hybrid_err),
                    reason="Hybrid routing failed, falling back to primary parser.",
                )
                # Fall through to normal primary parser path

        # Vision OCR for fully scanned handwritten documents.
        # When user toggled "handwritten" and the doc is all scanned,
        # use the vision client (capped at 20 pages) instead of the OCR parser.
        handwritten = kwargs.pop("handwritten", False)
        if (
            handwritten
            and self.vision_client is not None
            and file_content
            and decision.primary_parser == ParserType.OCR
            and DocumentCharacteristic.SCANNED in decision.detected_characteristics
        ):
            try:
                vision_result = await self._vision_ocr_pdf(
                    file_content, filename, document_id or "", max_pages=20
                )
                if vision_result.success and vision_result.total_chars > 0:
                    total_time = (time.time() - start_time) * 1000
                    return RoutingResult(
                        ingestor_result=vision_result,
                        parser_used=ParserType.OCR,  # Closest enum
                        routing_decision=decision,
                        total_time_ms=total_time,
                    )
                logger.warning(
                    "vision_ocr_empty_falling_back",
                    filename=filename,
                    chars=vision_result.total_chars,
                )
            except Exception as vision_err:
                logger.warning(
                    "vision_ocr_failed_falling_back",
                    filename=filename,
                    error=str(vision_err),
                )
            # Fall through to OCR

        # Try primary parser
        result = None
        fallback_used = False
        parser_chain = [decision.primary_parser]

        try:
            parser = self._get_parser(decision.primary_parser)
            result = await parser.ingest(
                file_path=file_path,
                file_content=file_content,
                filename=filename,
                document_id=document_id,
                **{**decision.parser_config, **kwargs},
            )

            # Check if we should try fallback
            # Cases: extraction failed, low text content, legacy fonts detected
            should_fallback = False
            fallback_reason = ""

            if not result.success:
                should_fallback = True
                fallback_reason = f"Primary parser failed: {result.error}"
            elif result.has_legacy_fonts:
                should_fallback = True
                fallback_reason = f"Legacy fonts detected: {result.legacy_fonts_found}"
            elif (
                result.total_chars < 100
                and file_size_bytes > 10000
                and decision.primary_parser == ParserType.PYMUPDF4LLM
            ):
                should_fallback = True
                fallback_reason = "Low text content detected (possibly scanned)"
            elif (
                result.total_chars >= 100
                and decision.primary_parser == ParserType.PYMUPDF4LLM
                and not _is_text_readable(result.content)
            ):
                should_fallback = True
                fallback_reason = (
                    "Extracted text is not readable (likely DRM-protected or "
                    "font-encrypted PDF). Routing to OCR for visual extraction."
                )
                logger.warning(
                    "ingestion_drm_detected",
                    filename=filename,
                    total_chars=result.total_chars,
                    sample=result.content[:100] if result.content else "",
                )
            elif (
                result.ocr_used
                and decision.primary_parser == ParserType.PYMUPDF4LLM
                and decision.fallback_parser == ParserType.OCR
            ):
                # PyMuPDF flagged the page as scanned and could not extract
                # text from the embedded text layer. OCR produces far
                # superior results on scanned docs (96.6% table accuracy,
                # 88.9% handwriting per OCR 3 benchmarks).
                should_fallback = True
                fallback_reason = (
                    "PyMuPDF detected scanned content with no text layer. Routing to OCR."
                )

            if should_fallback and decision.fallback_parser:
                logger.warning(
                    "ingestion_fallback_triggered",
                    filename=filename,
                    from_parser=decision.primary_parser.value,
                    to_parser=decision.fallback_parser.value,
                    reason=fallback_reason,
                )

                fallback_parser = self._get_parser(decision.fallback_parser)
                result = await fallback_parser.ingest(
                    file_path=file_path,
                    file_content=file_content,
                    filename=filename,
                    document_id=document_id,
                    **kwargs,
                )
                fallback_used = True
                parser_chain.append(decision.fallback_parser)

        except Exception as e:
            logger.error(
                "ingestion_primary_parser_error",
                filename=filename,
                parser=decision.primary_parser.value,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Try fallback on exception
            if decision.fallback_parser:
                try:
                    # If primary failed due to timeout (SoftTimeLimitExceeded inside
                    # pymupdf4llm OCR), skip OCR in the fallback to avoid hitting
                    # the hard time limit. When force_ocr=False, the ImageExtractor
                    # falls back to the vision LLM (no Tesseract anywhere).
                    fallback_kwargs = {**kwargs}
                    is_timeout = "TimeLimitExceeded" in type(
                        e
                    ).__name__ or "TimeLimitExceeded" in str(e)
                    if is_timeout:
                        fallback_kwargs["force_ocr"] = False
                        logger.warning(
                            "ingestion_fallback_skip_ocr",
                            filename=filename,
                            reason="Primary parser timed out on OCR - skipping OCR in fallback",
                        )

                    logger.debug(
                        "ingestion_fallback_on_error",
                        filename=filename,
                        from_parser=decision.primary_parser.value,
                        to_parser=decision.fallback_parser.value,
                    )

                    fallback_parser = self._get_parser(decision.fallback_parser)
                    result = await fallback_parser.ingest(
                        file_path=file_path,
                        file_content=file_content,
                        filename=filename,
                        document_id=document_id,
                        **fallback_kwargs,
                    )
                    fallback_used = True
                    parser_chain.append(decision.fallback_parser)

                except Exception as fallback_error:
                    logger.error(
                        "ingestion_fallback_also_failed",
                        filename=filename,
                        parser=decision.fallback_parser.value,
                        error=str(fallback_error),
                    )
                    # Re-raise original error
                    raise e from fallback_error
            else:
                raise

        total_time = (time.time() - start_time) * 1000
        final_parser = parser_chain[-1]

        # Final readability gate: after ALL parsers (primary + fallback) have run,
        # check that the final content is human-readable. This catches cases where
        # the DRM check triggered a OCR fallback that also failed or returned
        # garbage, or any other parser chain that produces unusable content.
        # Scoped to PDF-like parsers only - the heuristic (space ratio < 2%,
        # avg word length > 30) produces false positives on legitimately decoded
        # email content (long URLs, surviving base64 data: URIs, forwarded thread
        # noise). Structured parsers like EmailFileIngestor have already RFC-decoded
        # their input; there is no DRM/font-encryption failure mode to catch here.
        final_quality = (
            assess_extraction_quality(result.content)
            if (result and result.success and result.total_chars >= 100)
            else None
        )
        if (
            result
            and result.success
            and result.total_chars >= 100
            and final_parser in _READABILITY_GATED_PARSERS
            and final_quality is not None
            and not final_quality.is_readable
        ):
            logger.error(
                "ingestion_final_readability_failed",
                filename=filename,
                parsers_attempted=[p.value for p in parser_chain],
                total_chars=result.total_chars,
                verdict=final_quality.verdict.value,
                reason=final_quality.reason,
                sample=result.content[:100] if result.content else "",
            )
            result = IngestorResult(
                document_id=result.document_id,
                filename=result.filename,
                content="",
                success=False,
                error=unreadable_error_message(final_quality.verdict, fallback_used),
                parser_name=result.parser_name,
                processing_time_ms=result.processing_time_ms,
            )

        logger.debug(
            "ingestion_routing_completed",
            filename=filename,
            parser_used=final_parser.value,
            fallback_used=fallback_used,
            parsers_attempted=[p.value for p in parser_chain],
            success=result.success if result else False,
            content_length=len(result.content) if result else 0,
            section_count=len(result.sections) if result and result.sections else 0,
            total_chars=result.total_chars if result else 0,
            ocr_used=result.ocr_used if result else False,
            duration_ms=total_time,
        )

        return RoutingResult(
            ingestor_result=result,
            parser_used=final_parser,
            fallback_used=fallback_used,
            routing_decision=decision,
            total_time_ms=total_time,
        )

    async def _make_routing_decision(
        self,
        file_content: bytes | None,
        filename: str,
        characteristics: list[DocumentCharacteristic] | None = None,
        **kwargs,
    ) -> RoutingDecision:
        """Make routing decision based on document analysis."""

        # Detect document type
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        # Detect characteristics
        detected_chars = list(characteristics or [])
        if file_content:
            detected_chars.extend(await self._analyze_document(file_content, ext))

        # Remove duplicates
        detected_chars = list(set(detected_chars))

        # Get primary parser from dynamic format routing
        format_routing = self.get_format_routing()
        primary = format_routing.get(ext, ParserType.MULTI_FORMAT)
        fallback = self._get_fallback_parser(primary, ext)

        # For PDFs: if OCR is available, run a fast pre-flight scan check.
        # Smart pre-flight OCR analysis for PDFs.
        # Samples pages spread across the document to avoid being fooled by
        # blank cover pages. Decides between PyMuPDF (free), OCR,
        # or hybrid routing (Phase 2: split by page type).
        ocr_analysis = None
        if (
            ext == "pdf"
            and primary == ParserType.PYMUPDF4LLM
            and fallback == ParserType.OCR
            and file_content
        ):
            from legal_rag.ingestion.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor

            ingestor = PyMuPDF4LLMIngestor()
            ocr_analysis = ingestor._analyze_ocr_needs_from_bytes(file_content)

            if ocr_analysis.needs_ocr:
                logger.info(
                    "routing_scanned_pdf_to_ocr",
                    filename=filename,
                    reason="Pre-flight smart sampling detected scanned PDF.",
                    scanned_ratio=ocr_analysis.scanned_ratio,
                    strategy=ocr_analysis.strategy,
                    total_pages=ocr_analysis.total_pages,
                    sampled_count=len(ocr_analysis.sampled_pages),
                )
                primary = ParserType.OCR
                fallback = ParserType.MULTI_FORMAT
                detected_chars.append(DocumentCharacteristic.SCANNED)
            elif ocr_analysis.strategy == "mixed":
                logger.info(
                    "routing_mixed_pdf_hybrid",
                    filename=filename,
                    reason="Pre-flight detected mixed PDF (some text, some scanned pages).",
                    scanned_ratio=ocr_analysis.scanned_ratio,
                    total_pages=ocr_analysis.total_pages,
                    sampled_count=len(ocr_analysis.sampled_pages),
                )
                detected_chars.append(DocumentCharacteristic.MIXED)

        # Build parser config
        parser_config = self._build_parser_config(primary, detected_chars, **kwargs)

        # Calculate confidence
        confidence = self._calculate_confidence(primary, detected_chars, ext)

        # Build reasoning
        reasons = []
        if detected_chars:
            reasons.append(f"Detected: {', '.join(c.value for c in detected_chars)}")
        reasons.append(f"Format: {ext}")
        reasons.append(f"Parser: {primary.value}")
        if fallback:
            reasons.append(f"Fallback: {fallback.value}")

        return RoutingDecision(
            primary_parser=primary,
            fallback_parser=fallback,
            parser_config=parser_config,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            detected_characteristics=detected_chars,
            ocr_analysis=ocr_analysis,
        )

    def _calculate_confidence(
        self,
        parser: ParserType,
        characteristics: list[DocumentCharacteristic],
        ext: str,
    ) -> float:
        """Calculate confidence score for parser selection."""
        base_confidence = 0.80

        capabilities = self.PARSER_CAPABILITIES.get(parser, {})
        strengths = capabilities.get("strengths", [])
        weaknesses = capabilities.get("weaknesses", [])

        # Boost for matching strengths
        for char in characteristics:
            if char in strengths:
                base_confidence += 0.05

        # Penalty for matching weaknesses
        for char in characteristics:
            if char in weaknesses:
                base_confidence -= 0.10

        # Format match boost
        formats = capabilities.get("formats", set())
        if ext in formats:
            base_confidence += 0.05

        return max(0.1, min(1.0, base_confidence))

    async def _analyze_document(
        self, file_content: bytes, ext: str
    ) -> list[DocumentCharacteristic]:
        """Analyze document to detect characteristics."""
        characteristics = []

        if ext == "pdf":
            characteristics.extend(self._analyze_pdf(file_content))
        elif ext in ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"):
            # Images might be scanned documents or photos
            characteristics.append(DocumentCharacteristic.HAS_IMAGES)
            # Check image size - large images often scanned docs
            if len(file_content) > 500000:  # > 500KB
                characteristics.append(DocumentCharacteristic.SCANNED)

        # Check file size for quality hints
        size_mb = len(file_content) / (1024 * 1024)
        if ext == "pdf" and size_mb > 10:
            # Large PDFs often have images/scans
            characteristics.append(DocumentCharacteristic.HAS_IMAGES)

        return characteristics

    def _analyze_pdf(self, file_content: bytes) -> list[DocumentCharacteristic]:
        """Analyze PDF to detect characteristics."""
        characteristics = []

        try:
            # Quick header analysis
            header = file_content[:2000].decode("latin-1", errors="ignore")

            # Check for text content (born digital indicator)
            if "/Type /Page" in header and "/Contents" in header:
                characteristics.append(DocumentCharacteristic.BORN_DIGITAL)

            # Check for images (might be scanned)
            if "/Image" in header or "/XObject" in header:
                characteristics.append(DocumentCharacteristic.HAS_IMAGES)

            # Check for forms
            if "/AcroForm" in header or "/Widget" in header:
                characteristics.append(DocumentCharacteristic.FORM)

            # Check for OCR'd text (usually has specific markers)
            full_content = file_content[:10000].decode("latin-1", errors="ignore")
            if "ocr" in full_content.lower() or "/JBIG2" in full_content:
                characteristics.append(DocumentCharacteristic.SCANNED)

            # Check for non-standard encodings (legacy fonts)
            # This is a heuristic - Type1 fonts or WinAnsi encoding might cause issues
            if "/Type1" in full_content or "/WinAnsiEncoding" in full_content:
                # Not necessarily bad, but flag for awareness
                pass

        except Exception as e:
            logger.warning("pdf_analysis_error", error=str(e))

        return characteristics

    def _build_parser_config(
        self,
        parser_type: ParserType,
        characteristics: list[DocumentCharacteristic],
        **kwargs,
    ) -> dict[str, Any]:
        """Build parser-specific configuration."""
        config = {}
        # Note: No bbox config - use char positions for search-based highlighting
        return config

    async def _hybrid_route_and_parse(
        self,
        file_content: bytes,
        filename: str,
        document_id: str,
        ocr_analysis: Any,
        **kwargs,
    ) -> RoutingResult:
        """
        Hybrid routing for mixed PDFs: split by page type, route each range
        to the appropriate parser (PyMuPDF for text, OCR for scanned),
        then merge results into a single IngestorResult.

        Falls back to full OCR if any sub-range parse fails.
        """
        from legal_rag.ingestion.pymupdf4llm_ingestor import group_page_ranges

        ranges = group_page_ranges(ocr_analysis)
        if not ranges:
            raise ValueError("No page ranges from OCR analysis")

        # If all ranges are the same type, skip splitting
        range_types = {r[2] for r in ranges}
        if len(range_types) == 1:
            parser_type = ParserType.PYMUPDF4LLM if "text" in range_types else ParserType.OCR
            parser = self._get_parser(parser_type)
            result = await parser.ingest(
                file_content=file_content,
                filename=filename,
                document_id=document_id,
                **kwargs,
            )
            return RoutingResult(
                ingestor_result=result,
                parser_used=parser_type,
            )

        logger.info(
            "hybrid_routing_splitting",
            filename=filename,
            ranges=[(s, e, t) for s, e, t in ranges],
            total_ranges=len(ranges),
        )

        # Filter out truly blank OCR ranges to avoid wasting OCR cost.
        # A range is "truly blank" if ALL its sampled pages had 0 chars and no
        # images (blank covers, placeholder pages). No point OCR-ing them.
        filtered_ranges = _filter_blank_ocr_ranges(file_content, ranges, ocr_analysis)
        if not filtered_ranges:
            # All ranges were blank. Return empty result.
            from legal_rag.ingestion.base import IngestorResult as _IR

            return RoutingResult(
                ingestor_result=_IR(
                    document_id=document_id,
                    filename=filename,
                    content="",
                    success=True,
                    page_count=ocr_analysis.total_pages,
                    parser_name="hybrid_all_blank",
                ),
                parser_used=ParserType.PYMUPDF4LLM,
            )

        # Split PDF into sub-PDFs per range
        sub_pdfs = _split_pdf_by_ranges(file_content, filtered_ranges)

        # Parse each sub-PDF with the appropriate parser
        sub_results: list[tuple[int, int, IngestorResult]] = []

        handwritten = kwargs.pop("handwritten", False)

        for start, end, range_type, sub_pdf_bytes in sub_pdfs:
            if range_type == "text":
                parser_type = ParserType.PYMUPDF4LLM
            else:
                parser_type = ParserType.OCR
            parser = self._get_parser(parser_type)

            # When handwritten=True, use Vision OCR for scanned pages
            # Vision OCR is better at reading handwritten text (capped at 20 pages)
            use_vision = handwritten and range_type == "ocr" and self.vision_client is not None

            logger.debug(
                "hybrid_parsing_range",
                filename=filename,
                start_page=start,
                end_page=end,
                parser="vision_ocr" if use_vision else parser_type.value,
                range_type=range_type,
                handwritten=handwritten,
            )

            if use_vision:
                result = await self._vision_ocr_pdf(
                    sub_pdf_bytes, filename, document_id or "", max_pages=20
                )
            else:
                result = await parser.ingest(
                    file_content=sub_pdf_bytes,
                    filename=filename,
                    document_id=document_id,
                    **kwargs,
                )

            if not result.success:
                logger.warning(
                    "hybrid_range_failed",
                    filename=filename,
                    start_page=start,
                    end_page=end,
                    parser=parser_type.value,
                    error=result.error,
                )
                # Fall back to full OCR for the entire document
                fallback_parser = self._get_parser(ParserType.OCR)
                fallback_result = await fallback_parser.ingest(
                    file_content=file_content,
                    filename=filename,
                    document_id=document_id,
                    **kwargs,
                )
                return RoutingResult(
                    ingestor_result=fallback_result,
                    parser_used=ParserType.OCR,
                    fallback_used=True,
                )

            sub_results.append((start, end, result))

        # Merge all sub-results into a single IngestorResult
        merged = _merge_ingestor_results(sub_results, document_id, filename)

        logger.info(
            "hybrid_routing_completed",
            filename=filename,
            ranges_parsed=len(sub_results),
            total_chars=merged.total_chars,
            page_count=merged.page_count,
            ocr_used=merged.ocr_used,
        )

        return RoutingResult(
            ingestor_result=merged,
            parser_used=ParserType.PYMUPDF4LLM,  # Primary parser type
        )

    async def supports_format(self, filename: str) -> bool:
        """Check if this router supports the given file format."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        format_routing = self.get_format_routing()
        return ext in format_routing

    async def close(self):
        """Close all parser connections."""
        import asyncio

        for parser in self._parsers.values():
            close_method = getattr(parser, "close", None)
            if close_method is not None and callable(close_method):
                result = close_method()
                # Await if the method returns a coroutine
                if asyncio.iscoroutine(result):
                    await result
        self._parsers.clear()


# ============================================
# Hybrid Routing Utilities
# ============================================

# Vision OCR: default cap on pages processed per document
_VISION_OCR_MAX_PAGES = 20


async def vision_ocr_pdf(
    client: VisionClient,
    file_content: bytes,
    filename: str,
    document_id: str,
    *,
    max_pages: int = _VISION_OCR_MAX_PAGES,
    prompt: str = DEFAULT_PAGE_OCR_PROMPT,
    dpi: int = 300,
    max_parallel: int = 3,
) -> IngestorResult:
    """
    OCR a PDF page by page with a vision model, for handwritten documents.

    Renders each page as a PNG (requires PyMuPDF) and sends it to ``client``.
    Capped at ``max_pages`` to bound cost. Pages that fail come back empty.
    """
    import fitz  # optional dependency, imported only when this function runs

    from legal_rag.ingestion.base import DocumentSection

    doc = fitz.open(stream=file_content, filetype="pdf")
    total_pages = doc.page_count
    pages_to_process = min(total_pages, max_pages)

    if pages_to_process < total_pages:
        logger.info(
            "vision_ocr_capped",
            filename=filename,
            total_pages=total_pages,
            processing=pages_to_process,
        )

    sem = asyncio.Semaphore(max(1, max_parallel))

    async def _ocr_page(page_idx: int) -> tuple[int, str]:
        async with sem:
            try:
                png_bytes = await asyncio.to_thread(
                    lambda: doc[page_idx].get_pixmap(dpi=dpi).tobytes("png")
                )
                page_text = await client.describe_image(png_bytes, "image/png", prompt=prompt)
                # Truncate extremely long responses (hallucination guard)
                return page_idx, (page_text or "")[:20000]
            except Exception as e:
                logger.warning(
                    "vision_ocr_page_failed", filename=filename, page=page_idx, error=str(e)
                )
                return page_idx, ""

    try:
        results = await asyncio.gather(*(_ocr_page(i) for i in range(pages_to_process)))
    finally:
        doc.close()

    all_pages = [
        {"page_number": idx + 1, "text": text, "char_count": len(text)}
        for idx, text in sorted(results)
    ]
    full_text = "\n\n".join(p["text"] for p in all_pages if p["text"].strip())

    sections = [
        DocumentSection(
            title=f"Page {p['page_number']}",
            content=p["text"],
            level=0,
            page_number=p["page_number"],
        )
        for p in all_pages
        if p["text"].strip()
    ]

    return IngestorResult(
        document_id=document_id,
        filename=filename,
        content=full_text,
        sections=sections,
        page_count=total_pages,
        pages=all_pages,
        metadata={"parser": "vision_ocr", "handwritten": True},
        success=True,
        ocr_used=True,
        total_chars=len(full_text),
        parser_name="vision_ocr",
    )


def _filter_blank_ocr_ranges(
    file_content: bytes,
    ranges: list[tuple[int, int, str]],
    ocr_analysis: Any,
) -> list[tuple[int, int, str]]:
    """
    Remove truly blank OCR ranges that would waste OCR cost.

    A page is "truly blank" if it has 0 extractable chars AND no meaningful
    images (no scanned content to OCR). Common examples: blank covers,
    separator pages, "This page intentionally left blank" pages.

    For OCR ranges where ALL sampled pages are blank (0 chars), we check
    for images using PyMuPDF. If no images found, the range is dropped.
    Text ranges are never filtered.
    """
    import fitz

    if not ocr_analysis or not ocr_analysis.sampled_pages:
        return ranges

    # Build a set of page indices that are truly blank (0 chars in sample)
    blank_pages = {
        p.page_index
        for p in ocr_analysis.sampled_pages
        if p.char_count == 0 and not p.has_replacement_chars
    }

    filtered: list[tuple[int, int, str]] = []
    doc = fitz.open(stream=file_content, filetype="pdf")

    for start, end, range_type in ranges:
        if range_type == "text":
            filtered.append((start, end, range_type))
            continue

        # For OCR ranges, check if ALL pages in the range that we sampled are blank
        sampled_in_range = [p for p in ocr_analysis.sampled_pages if start <= p.page_index < end]

        # If no samples fall in this range, keep it (can't tell)
        if not sampled_in_range:
            filtered.append((start, end, range_type))
            continue

        all_blank = all(p.page_index in blank_pages for p in sampled_in_range)
        if not all_blank:
            # Some sampled pages have content, keep the range for OCR
            filtered.append((start, end, range_type))
            continue

        # All sampled pages in this range are blank text-wise.
        # Check if any page has images (scanned content worth OCR-ing).
        has_images = False
        for page_idx in range(start, min(end, start + 5)):  # Check first 5 pages max
            page = doc[page_idx]
            image_list = page.get_images(full=False)
            if image_list:
                has_images = True
                break

        if has_images:
            # Pages have images but no text. Likely scanned content. Keep for OCR.
            filtered.append((start, end, range_type))
        else:
            # Truly blank pages. No text, no images. Skip entirely.
            logger.info(
                "hybrid_skipping_blank_range",
                start_page=start,
                end_page=end,
                page_count=end - start,
                reason="All sampled pages blank with no images. Skipping OCR.",
            )

    doc.close()
    return filtered


def _split_pdf_by_ranges(
    file_content: bytes,
    ranges: list[tuple[int, int, str]],
) -> list[tuple[int, int, str, bytes]]:
    """
    Split a PDF into sub-PDFs based on page ranges.

    Uses PyMuPDF's insert_pdf() for fast in-memory splitting.
    Splits at the same page offsets that the OCR client uses for large PDFs.

    Args:
        file_content: Full PDF bytes
        ranges: List of (start_page, end_page_exclusive, range_type)

    Returns:
        List of (start_page, end_page_exclusive, range_type, sub_pdf_bytes)
    """
    import fitz

    src = fitz.open(stream=file_content, filetype="pdf")
    result = []

    for start, end, range_type in ranges:
        sub_doc = fitz.open()
        sub_doc.insert_pdf(src, from_page=start, to_page=end - 1)
        sub_bytes = sub_doc.tobytes(deflate=True)
        sub_doc.close()
        result.append((start, end, range_type, sub_bytes))

    src.close()
    return result


def _merge_ingestor_results(
    results: list[tuple[int, int, IngestorResult]],
    document_id: str,
    filename: str,
) -> IngestorResult:
    """
    Merge multiple IngestorResults from hybrid routing into a single result.

    Page numbers in sections and pages are offset by each range's start page.
    Page offsets match the split performed by ``_split_pdf_by_ranges``.

    Args:
        results: List of (start_page, end_page_exclusive, IngestorResult), sorted by start_page
        document_id: Document ID
        filename: Original filename
    """
    from legal_rag.ingestion.base import DocumentSection, DocumentType

    # Sort by start page
    sorted_results = sorted(results, key=lambda r: r[0])

    all_content_parts: list[str] = []
    all_text_clean_parts: list[str] = []
    all_sections: list[DocumentSection] = []
    all_pages: list[dict[str, Any]] = []
    all_boxes: list[dict[str, Any]] = []
    merged_box_counts: dict[str, int] = {}
    total_page_count = 0
    total_chars = 0
    total_time_ms = 0.0
    any_ocr_used = False
    any_has_legacy_fonts = False
    all_legacy_fonts: list[str] = []
    total_artifacts_removed = 0
    any_cleaning_applied = False
    all_support_positions = True
    has_text_clean = False

    for start_page, _end_page, sub_result in sorted_results:
        # Content
        if sub_result.content:
            all_content_parts.append(sub_result.content)

        # Text clean
        if sub_result.text_clean is not None:
            has_text_clean = True
            all_text_clean_parts.append(sub_result.text_clean)

        # Sections with page number offset
        for section in sub_result.sections or []:
            offset_section = DocumentSection(
                title=section.title,
                content=section.content,
                level=section.level,
                section_number=section.section_number,
                page_number=(section.page_number or 0) + start_page,
                metadata=section.metadata,
            )
            all_sections.append(offset_section)

        # Pages with page number offset
        for page in sub_result.pages or []:
            offset_page = {**page}
            offset_page["page_number"] = page.get("page_number", 0) + start_page
            all_pages.append(offset_page)

        # Boxes with page number offset
        for box in sub_result.boxes or []:
            offset_box = {**box}
            if "page_number" in offset_box:
                offset_box["page_number"] = box["page_number"] + start_page
            all_boxes.append(offset_box)

        # Box class counts (merge by summing)
        for cls, count in (sub_result.box_class_counts or {}).items():
            merged_box_counts[cls] = merged_box_counts.get(cls, 0) + count

        # Scalars
        total_page_count += sub_result.page_count or 0
        total_chars += sub_result.total_chars or 0
        total_time_ms += sub_result.processing_time_ms or 0.0

        # Flags
        if sub_result.ocr_used:
            any_ocr_used = True
        if sub_result.has_legacy_fonts:
            any_has_legacy_fonts = True
        all_legacy_fonts.extend(sub_result.legacy_fonts_found or [])
        total_artifacts_removed += sub_result.artifacts_removed or 0
        if sub_result.cleaning_applied:
            any_cleaning_applied = True
        if not sub_result.supports_positions:
            all_support_positions = False

    return IngestorResult(
        document_id=document_id,
        filename=filename,
        content="\n\n".join(all_content_parts),
        text_clean="\n\n".join(all_text_clean_parts) if has_text_clean else None,
        sections=all_sections,
        document_type=DocumentType.PDF,
        page_count=total_page_count,
        metadata={
            "parser": "hybrid_pymupdf4llm_ocr",
            "hybrid_routing": True,
            "ranges": [(s, e, r.parser_name) for s, e, r in sorted_results],
        },
        boxes=all_boxes if all_boxes else None,
        pages=all_pages if all_pages else None,
        box_class_counts=merged_box_counts if merged_box_counts else None,
        ocr_used=any_ocr_used,
        supports_positions=all_support_positions,
        total_chars=total_chars,
        parser_name="hybrid_pymupdf4llm_ocr",
        has_legacy_fonts=any_has_legacy_fonts,
        legacy_fonts_found=all_legacy_fonts,
        artifacts_removed=total_artifacts_removed,
        cleaning_applied=any_cleaning_applied,
        success=True,
        processing_time_ms=total_time_ms,
    )


async def route_and_parse_document(
    file_path: str | None = None,
    file_content: bytes | None = None,
    filename: str = "document",
    document_id: str | None = None,
    *,
    router: DocumentRouter | None = None,
    **kwargs,
) -> RoutingResult:
    """
    Convenience function to route and parse a document.

    Args:
        file_path: Path to document file
        file_content: Raw file bytes
        filename: Original filename
        document_id: Unique document identifier
        router: Router to use. A default router (no OCR, no vision) is built when omitted.
        **kwargs: Additional options

    Returns:
        RoutingResult with parsed content
    """
    active = router or DocumentRouter()
    return await active.route_and_parse(
        file_path=file_path,
        file_content=file_content,
        filename=filename,
        document_id=document_id,
        **kwargs,
    )
