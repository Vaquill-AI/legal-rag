# ruff: noqa: BLE001, S112, RUF012, PERF102
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Office Document Extractor - DOCX, DOC, XLSX, XLS, PPTX, PPT.

Uses:
- python-docx for DOCX
- openpyxl for XLSX
- python-pptx for PPTX
- xlrd for legacy XLS (read-only)
- Basic CFB parsing for DOC/PPT (limited support)
- An injected OCR client for embedded images in DOCX

OCR strategy for DOCX:
Embedded images in DOCX are OCR'd through the injected ``OcrClient`` (no Tesseract anywhere
in the app). Tiny decorative images (signatures, separator lines, small
icons) are filtered out by a dimension threshold before any API call is
made. The remaining images are OCR'd concurrently via asyncio.gather so a
docx with N images costs roughly the wall time of the slowest image, not
the sum of all images. Each OCR call runs in a worker thread so the
event loop stays responsive.

Security Note:
OCR output is sanitized to reduce prompt injection risk when extracted text is passed to
language models. Control characters are removed and text is length-limited.
"""

import asyncio
import dataclasses
import io
import re
import time
import unicodedata
from typing import Any

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedContent,
    estimate_pages_from_text,
    synthesize_pages_from_text,
)
from legal_rag.ingestion.protocols import DocumentConverter, OcrClient, VisionClient
from legal_rag.ingestion.text_cleaner import clean_format_specific_markers, clean_ocr_artifacts

logger = get_logger("legal_rag.ingestion.extractors.office")

# Heading that precedes a slide's speaker notes in the extracted text.
PPTX_NOTES_LABEL = "Speaker notes:"

DEFAULT_IMAGE_PROMPT = (
    "Extract ALL text content from this image exactly as it appears. "
    "Preserve formatting, line breaks, and structure. "
    "If the image contains no readable text, respond with exactly: NO_TEXT_FOUND"
)

# Security: Maximum OCR text length per image to prevent resource exhaustion
OCR_MAX_CHARS_PER_IMAGE = 10000

# Security: Pattern to detect potential prompt injection attempts
# Matches common LLM instruction patterns that shouldn't appear in OCR'd images
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"disregard\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"forget\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you\s+are",
    r"pretend\s+you\s+are",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
]

# Compile patterns for efficiency
_INJECTION_REGEX = re.compile(
    "|".join(PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def sanitize_ocr_text(text: str, source_info: str = "") -> tuple[str, bool]:
    """
    Sanitize OCR-extracted text to prevent prompt injection attacks.

    This function:
    1. Removes control characters (except newlines/tabs)
    2. Normalizes Unicode to prevent homograph attacks
    3. Limits text length to prevent resource exhaustion
    4. Detects and flags potential prompt injection patterns

    Args:
        text: Raw OCR text to sanitize
        source_info: Optional source identifier for logging

    Returns:
        Tuple of (sanitized_text, injection_detected)
        - sanitized_text: Cleaned text safe for embedding/RAG
        - injection_detected: True if potential injection patterns found
    """
    if not text:
        return "", False

    # Step 1: Normalize Unicode (NFC normalization to canonical form)
    # This prevents homograph attacks using lookalike characters
    normalized = unicodedata.normalize("NFC", text)

    # Step 2: Remove control characters except \n (newline) and \t (tab)
    # Control characters can be used to manipulate text rendering/parsing
    sanitized_chars = []
    for char in normalized:
        category = unicodedata.category(char)
        # Keep: letters, numbers, punctuation, symbols, separators (spaces)
        # Also keep newlines (\n) and tabs (\t) for formatting
        if category.startswith(("L", "N", "P", "S", "Z")) or char in "\n\t":
            sanitized_chars.append(char)
        elif category == "Cc":  # Control characters
            # Replace with space to preserve word boundaries
            sanitized_chars.append(" ")

    sanitized = "".join(sanitized_chars)

    # Step 3: Collapse multiple whitespace into single space (except newlines)
    sanitized = re.sub(r"[^\S\n]+", " ", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)  # Max 2 consecutive newlines
    sanitized = sanitized.strip()

    # Step 4: Check for prompt injection patterns
    injection_detected = bool(_INJECTION_REGEX.search(sanitized))

    if injection_detected:
        logger.warning(
            "potential_prompt_injection_in_ocr",
            source=source_info,
            text_preview=sanitized[:200],
            message="Potential prompt injection pattern detected in OCR text",
        )

    # Step 5: Truncate to maximum length
    original_length = len(sanitized)
    if original_length > OCR_MAX_CHARS_PER_IMAGE:
        sanitized = sanitized[:OCR_MAX_CHARS_PER_IMAGE]
        logger.debug(
            "ocr_text_truncated",
            source=source_info,
            original_length=original_length,
            truncated_to=OCR_MAX_CHARS_PER_IMAGE,
        )

    return sanitized, injection_detected


def _looks_like_readable_text(text: str) -> bool:
    """Heuristic: does this extracted text look like real document prose?

    Rejects the noise LibreOffice emits when handed bytes it cannot truly
    parse (corrupt / wrong-format input): it exits 0 and writes a .docx full
    of binary misread as text. An ``isalpha`` ratio does NOT separate that
    from prose (random latin-1 bytes are ~20% "letters", and binary often
    lands in CJK codepoints, so a CJK allowance backfires too). Word spacing
    does: real prose in space-delimited languages runs ~12-18% whitespace, while binary
    noise sits well under 1%.
    """
    stripped = text.strip()
    if len(stripped) < 50:
        return False
    whitespace = sum(1 for ch in stripped if ch.isspace())
    return (whitespace / len(stripped)) >= 0.05


# Check available libraries
DOCX_AVAILABLE = False
XLSX_AVAILABLE = False
PPTX_AVAILABLE = False
XLS_AVAILABLE = False
OCR_AVAILABLE = False

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    logger.info("python_docx_not_installed", message="DOCX support disabled")

try:
    from openpyxl import load_workbook

    XLSX_AVAILABLE = True
except ImportError:
    logger.info("openpyxl_not_installed", message="XLSX support disabled")

try:
    from pptx import Presentation

    PPTX_AVAILABLE = True
except ImportError:
    logger.info("python_pptx_not_installed", message="PPTX support disabled")

try:
    import xlrd

    XLS_AVAILABLE = True
except ImportError:
    logger.info("xlrd_not_installed", message="XLS support disabled")

try:
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    logger.info(
        "ocr_not_available",
        message="OCR support disabled (install pillow); the OCR client is used for embedded images",
    )

# Embedded-image OCR thresholds. Images smaller than these are treated as
# decorative (signature lines, separator dots, tiny icons) and skipped so we
# don't spend an OCR call on something that has no readable text.
_DOCX_OCR_MIN_BYTES = 8 * 1024  # 8 KB
_DOCX_OCR_MIN_DIM = 100  # px on either axis
# Concurrency cap on parallel OCR calls per docx. Each call is a
# network round-trip; running more than a handful in parallel
# saturates connections and the rate limit without improving wall time.
_DOCX_OCR_PARALLELISM = 4


class OfficeExtractor(BaseExtractor):
    """
    Extract text from Microsoft Office documents.

    Supported formats:
    - DOCX, DOTX (Word 2007+) - full support
    - DOC (Word 97-2003) - limited support via basic parsing
    - XLSX, XLSM, XLTX (Excel 2007+) - full support
    - XLS (Excel 97-2003) - read-only via xlrd
    - PPTX (PowerPoint 2007+) - full support
    - PPT (PowerPoint 97-2003) - limited support

    Args:
        ocr_client: ``OcrClient`` for images embedded in DOCX files.
        vision_client: ``VisionClient`` tried when OCR returns nothing.
        converter: ``DocumentConverter`` that turns legacy .doc into .docx.
        vision_prompt: Instruction sent with each image to the vision client.
    """

    def __init__(
        self,
        ocr_client: OcrClient | None = None,
        vision_client: VisionClient | None = None,
        converter: DocumentConverter | None = None,
        *,
        vision_prompt: str = DEFAULT_IMAGE_PROMPT,
    ) -> None:
        self.ocr_client = ocr_client
        self.vision_client = vision_client
        self.converter = converter
        self.vision_prompt = vision_prompt

    SUPPORTED = {
        "docx",
        "dotx",  # Word
        "doc",  # Legacy Word (limited)
        "xlsx",
        "xlsm",
        "xltx",
        "xltm",  # Excel
        "xls",  # Legacy Excel
        "pptx",  # PowerPoint
        "ppt",  # Legacy PowerPoint (limited)
    }

    async def extract(
        self,
        file_content: bytes,
        filename: str,
        **kwargs,
    ) -> ExtractedContent:
        """Extract text from Office document."""
        ext = self.get_extension(filename)

        try:
            # Word documents
            if ext in ("docx", "dotx"):
                return await self._extract_docx(file_content, filename)
            elif ext == "doc":
                return await self._extract_doc(file_content, filename)

            # Excel documents
            elif ext in ("xlsx", "xlsm", "xltx", "xltm"):
                return await self._extract_xlsx(file_content, filename)
            elif ext == "xls":
                return await self._extract_xls(file_content, filename)

            # PowerPoint documents
            elif ext == "pptx":
                return await self._extract_pptx(file_content, filename)
            elif ext == "ppt":
                return await self._extract_ppt(file_content, filename)

            else:
                return ExtractedContent(
                    text="",
                    success=False,
                    error=f"Unsupported Office format: {ext}",
                )

        except Exception as e:
            logger.exception("office_extraction_failed", filename=filename, ext=ext, error=str(e))
            return ExtractedContent(
                text="",
                success=False,
                error=f"Office extraction failed: {e!s}",
            )

    async def _extract_docx(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from DOCX document with embedded image OCR."""
        if not DOCX_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="python-docx not installed - run: pip install python-docx",
            )

        doc = DocxDocument(io.BytesIO(file_content))

        # Extract paragraphs
        paragraphs: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # Extract tables
        tables: list[dict[str, Any]] = []
        for table_idx, table in enumerate(doc.tables):
            table_data: list[list[str]] = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)

            if table_data:
                tables.append(
                    {
                        "index": table_idx,
                        "rows": len(table_data),
                        "cols": len(table_data[0]) if table_data else 0,
                        "data": table_data,
                    }
                )

                # Add table to text
                for row in table_data:
                    paragraphs.append(" | ".join(row))

        # Extract and OCR embedded images via the OCR client (parallel).
        # Decorative images (small icons, signature lines) are filtered out
        # before any API call so we don't waste latency on images without
        # readable text.
        image_texts, images_processed, images_failed, total_ocr_chars = await self._ocr_docx_images(
            doc, filename
        )

        # Combine all extracted text
        all_text_parts = paragraphs + image_texts
        full_text = "\n\n".join(all_text_parts)

        # ✅ Apply OCR artifact cleaning
        cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)

        # Get metadata
        metadata: dict[str, Any] = {}
        if doc.core_properties:
            props = doc.core_properties
            if props.title:
                metadata["title"] = props.title
            if props.author:
                metadata["author"] = props.author
            if props.created:
                metadata["created"] = str(props.created)
            if props.modified:
                metadata["modified"] = str(props.modified)

        # Add image extraction metadata
        if images_processed > 0:
            metadata["images_processed"] = images_processed
            metadata["images_failed"] = images_failed
            metadata["ocr_chars"] = total_ocr_chars

        logger.debug(
            "docx_extracted",
            filename=filename,
            paragraphs=len(paragraphs),
            tables=len(tables),
            images_processed=images_processed,
            total_chars=len(full_text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=full_text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(full_text),
            metadata=metadata,
            tables=tables if tables else None,
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    async def _ocr_docx_images(self, doc: Any, filename: str) -> tuple[list[str], int, int, int]:
        """OCR every meaningful embedded image in a DOCX via the OCR client.

        Returns (image_texts, images_processed, images_failed, total_ocr_chars).

        Strategy:
          1. Walk doc.part.rels, filter to image relationships.
          2. Skip images that are too small to contain readable text
             (saves a real round-trip per decorative signature/separator).
          3. Run remaining images through OCR concurrently, capped
             at _DOCX_OCR_PARALLELISM to avoid saturating the rate limit.
          4. Each OCR call runs in a worker thread because the OCR client
             is sync (urllib3 under the hood).
        """
        if not OCR_AVAILABLE:
            image_count = sum(1 for rel in doc.part.rels.values() if "image" in rel.reltype)
            if image_count > 0:
                logger.info(
                    "docx_images_skipped_pil_missing",
                    filename=filename,
                    image_count=image_count,
                    message="Install pillow for image size detection",
                )
            return [], 0, 0, 0

        if self.ocr_client is None and self.vision_client is None:
            logger.info(
                "docx_ocr_skipped_no_client",
                filename=filename,
                message="No OCR or vision client configured; embedded images will not be OCR'd",
            )
            return [], 0, 0, 0

        # Stage 1: gather + filter candidates so we don't pay for tiny decoratives
        candidates: list[tuple[int, bytes, str, tuple[int, int]]] = []
        skipped_decorative = 0
        next_idx = 0
        for _rel_id, rel in doc.part.rels.items():
            if "image" not in rel.reltype:
                continue
            next_idx += 1
            try:
                image_bytes = rel.target_part.blob
                if len(image_bytes) < _DOCX_OCR_MIN_BYTES:
                    skipped_decorative += 1
                    continue
                pil_img = Image.open(io.BytesIO(image_bytes))
                w, h = pil_img.size
                if w < _DOCX_OCR_MIN_DIM and h < _DOCX_OCR_MIN_DIM:
                    skipped_decorative += 1
                    continue
                # Pick an OCR-supported extension from the PIL format
                fmt = (pil_img.format or "PNG").upper()
                ext = {
                    "JPEG": ".jpg",
                    "JPG": ".jpg",
                    "PNG": ".png",
                    "TIFF": ".tiff",
                    "AVIF": ".avif",
                }.get(fmt)
                if ext is None:
                    # Re-encode unsupported formats (GIF, BMP, WEBP) to PNG so
                    # the OCR client can ingest them. Cheap relative to the OCR call.
                    buf = io.BytesIO()
                    pil_img.convert("RGB").save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                    ext = ".png"
                candidates.append((next_idx, image_bytes, ext, (w, h)))
            except Exception as e:
                logger.warning(
                    "docx_image_candidate_failed",
                    filename=filename,
                    image_index=next_idx,
                    error=str(e),
                )
                continue

        if not candidates:
            if next_idx > 0:
                logger.debug(
                    "docx_ocr_no_candidates",
                    filename=filename,
                    total_images=next_idx,
                    skipped_decorative=skipped_decorative,
                )
            return [], 0, 0, 0

        # Stage 2: parallel OCR. Bounded by a semaphore so we never have more
        # than _DOCX_OCR_PARALLELISM in-flight OCR calls per docx.
        sem = asyncio.Semaphore(_DOCX_OCR_PARALLELISM)

        async def _ocr_one(idx: int, blob: bytes, ext: str, dims: tuple[int, int]):
            async with sem:
                t0 = time.monotonic()
                img_filename = f"{filename}_image_{idx}{ext}"
                text = ""
                vendor = "ocr"
                # Primary: the OCR client.
                try:
                    result = await asyncio.to_thread(self._ocr_blob, blob, img_filename)
                    text = (result or {}).get("text", "").strip() if result else ""
                except Exception as e:
                    logger.warning(
                        "docx_image_ocr_failed",
                        filename=filename,
                        image_index=idx,
                        error=str(e),
                        dims=dims,
                    )
                # Fallback: vision model. Triggers on an OCR outage OR when
                # OCR returned no text (e.g. unusual handwriting).
                if not text:
                    try:
                        fallback_text = await self._vision_llm_fallback_for_image(
                            blob, img_filename
                        )
                    except Exception as e:
                        logger.warning(
                            "docx_image_ocr_vision_fallback_errored",
                            filename=filename,
                            image_index=idx,
                            error=str(e),
                        )
                        fallback_text = None
                    if fallback_text:
                        text = fallback_text
                        vendor = "vision_llm"

                duration_ms = round((time.monotonic() - t0) * 1000)
                if not text:
                    logger.info(
                        "docx_image_ocr_no_text",
                        filename=filename,
                        image_index=idx,
                        dims=dims,
                        duration_ms=duration_ms,
                    )
                    return idx, None
                logger.info(
                    "docx_image_ocr_complete",
                    filename=filename,
                    image_index=idx,
                    dims=dims,
                    bytes=len(blob),
                    chars=len(text),
                    duration_ms=duration_ms,
                    vendor=vendor,
                )
                return idx, text

        ocr_tasks = [_ocr_one(idx, blob, ext, dims) for idx, blob, ext, dims in candidates]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=False)

        # Stage 3: sanitize and assemble. Sort by original document order.
        image_texts: list[str] = []
        images_processed = 0
        images_failed = 0
        total_ocr_chars = 0
        for idx, raw_text in sorted(ocr_results, key=lambda r: r[0]):
            if raw_text is None:
                images_failed += 1
                continue
            images_processed += 1
            if not raw_text:
                continue
            sanitized_text, injection_detected = sanitize_ocr_text(
                raw_text, source_info=f"{filename}:image_{idx}"
            )
            if not sanitized_text:
                continue
            marker = f"[Image {idx} OCR Text" + (" - SANITIZED]" if injection_detected else "]")
            image_texts.append(f"\n{marker}\n{sanitized_text}")
            total_ocr_chars += len(sanitized_text)

        logger.info(
            "docx_ocr_summary",
            filename=filename,
            total_images=next_idx,
            skipped_decorative=skipped_decorative,
            ocr_attempted=len(candidates),
            images_processed=images_processed,
            images_failed=images_failed,
            total_ocr_chars=total_ocr_chars,
        )
        return image_texts, images_processed, images_failed, total_ocr_chars

    def _ocr_blob(self, blob: bytes, filename: str) -> dict | None:
        """Blocking OCR call used from a worker thread. None when no client is set."""
        if self.ocr_client is None:
            return None
        return self.ocr_client.ocr(blob, filename, include_images=False)  # type: ignore[return-value]

    async def _vision_llm_fallback_for_image(self, image_bytes: bytes, filename: str) -> str | None:
        """OCR an image through the vision client when OCR returns no text or is unavailable.

        Returns the text or ``None`` if no client is set or nothing usable came back.
        """
        if self.vision_client is None:
            return None
        try:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
            mime_map = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "tiff": "image/tiff",
                "tif": "image/tiff",
                "gif": "image/gif",
                "webp": "image/webp",
                "bmp": "image/bmp",
                "avif": "image/avif",
            }
            raw = await self.vision_client.describe_image(
                image_bytes, mime_map.get(ext, "image/png"), prompt=self.vision_prompt
            )
            result_text = (raw or "").strip()
            if not result_text or result_text == "NO_TEXT_FOUND":
                return None
            return result_text
        except Exception as e:
            logger.warning("docx_image_vision_fallback_failed", filename=filename, error=str(e))
            return None

    async def _convert_doc_to_docx(self, file_content: bytes, filename: str) -> bytes | None:
        """Convert legacy binary .doc to .docx through the injected ``DocumentConverter``.

        Returns the converted bytes, or None when no converter is configured or the
        conversion fails, so the caller can fall back to a best-effort raw extraction.
        """
        if self.converter is None:
            return None
        try:
            docx_bytes = await self.converter.to_docx(file_content, filename)
            if docx_bytes is None:
                return None
            logger.info(
                "doc_converted_to_docx",
                filename=filename,
                input_bytes=len(file_content),
                output_bytes=len(docx_bytes),
            )
            return docx_bytes
        except Exception as e:
            logger.warning("doc_conversion_failed", filename=filename, error=str(e))
            return None

    async def _extract_doc(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from legacy binary DOC (Word 97-2003, OLE2 / CFB).

        Primary path: convert the .doc to .docx with the injected converter, then run the
        python-docx extractor (including embedded-image OCR). The olefile byte-scrape below
        only reads the raw WordDocument stream and mis-handles the piece-table layout, so
        it is kept solely as a last-resort fallback when no converter is available or the
        conversion fails.
        """
        # Primary: LibreOffice .doc -> .docx, then the full DOCX pipeline.
        converted = await self._convert_doc_to_docx(file_content, filename)
        if converted is not None:
            result = await self._extract_docx(converted, filename)
            # LibreOffice exits 0 and emits a .docx even when handed bytes it
            # cannot actually parse (corrupt / wrong-format / not-a-real-doc),
            # producing a document full of non-text noise. Gate on a quality
            # check so we never accept that junk as a successful extraction.
            if result.success and _looks_like_readable_text(result.text or ""):
                return dataclasses.replace(
                    result,
                    metadata={
                        **(result.metadata or {}),
                        "converted_from_doc": True,
                        "doc_conversion": "converter",
                    },
                )
            logger.warning(
                "doc_docx_extract_unusable_after_conversion",
                filename=filename,
                chars=len((result.text or "").strip()),
            )

        # Fallback: best-effort olefile scrape (unreliable; partial text only).
        # Use olefile to parse the DOC OLE2 container directly.
        try:
            import olefile

            if not olefile.isOleFile(file_content):
                return ExtractedContent(
                    text="",
                    success=False,
                    error="Not a valid DOC (OLE2) file. Consider converting to DOCX.",
                )

            ole = olefile.OleFileIO(io.BytesIO(file_content))
            text = ""

            # Extract text from WordDocument stream (UTF-16LE encoded in DOC)
            if ole.exists("WordDocument"):
                # DOC stores text in Word Document stream - try UTF-16LE first
                raw = ole.openstream("WordDocument").read()
                try:
                    text = raw.decode("utf-16-le", errors="ignore")
                except Exception:
                    text = raw.decode("utf-8", errors="ignore")

            # Fallback: try all text-bearing streams
            if len(text.strip()) < 50:
                for stream_name in ole.listdir():
                    try:
                        raw = ole.openstream(stream_name).read()
                        decoded = raw.decode("utf-16-le", errors="ignore")
                        if len(decoded.strip()) > len(text.strip()):
                            text = decoded
                    except Exception:
                        continue

            ole.close()

            # Clean control characters but preserve Unicode (Hindi, Arabic, CJK)
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Quality check: ratio of printable chars to file size
            # A valid DOC should produce meaningful text relative to its size
            printable_ratio = len(text) / max(len(file_content), 1)
            if len(text) < 50 or (len(file_content) > 10240 and printable_ratio < 0.01):
                return ExtractedContent(
                    text="",
                    success=False,
                    error="DOC format has limited support. Consider converting to DOCX.",
                )

            # ✅ Apply OCR artifact cleaning
            from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

            cleaned_text = clean_ocr_artifacts(text, preserve_structure=False)

            # DOC extraction via olefile is unreliable - may extract only partial text.
            # Use the higher of char-based estimate or file-size-based minimum
            # (~10KB per page for binary DOC) to prevent severe undercounting.
            char_estimate = estimate_pages_from_text(text)
            size_estimate = max(1, len(file_content) // 10_000)
            page_count = max(char_estimate, size_estimate)

            return ExtractedContent(
                text=text,
                text_clean=cleaned_text,
                page_count=page_count,
                metadata={"format_note": "DOC extraction via olefile"},
                pages=synthesize_pages_from_text(cleaned_text),
                success=True,
            )

        except Exception as e:
            return ExtractedContent(
                text="",
                success=False,
                error=f"DOC extraction failed: {e!s}. Consider converting to DOCX.",
            )

    async def _extract_xlsx(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from XLSX spreadsheet."""
        if not XLSX_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="openpyxl not installed - run: pip install openpyxl",
            )

        workbook = load_workbook(io.BytesIO(file_content), data_only=True)

        all_text_parts: list[str] = []
        sheets: list[str] = workbook.sheetnames
        tables: list[dict[str, Any]] = []
        per_sheet_cleaned: list[str] = []  # Per-sheet cleaned text for pages

        for sheet_name in sheets:
            sheet = workbook[sheet_name]
            all_text_parts.append(f"## Sheet: {sheet_name}")

            sheet_data: list[list[str]] = []
            sheet_row_texts: list[str] = []  # Row texts for this sheet only
            for row in sheet.iter_rows():
                row_values = []
                for cell in row:
                    value = cell.value
                    if value is not None:
                        row_values.append(str(value))
                    else:
                        row_values.append("")

                # Only add non-empty rows
                if any(v.strip() for v in row_values):
                    sheet_data.append(row_values)
                    row_text = " | ".join(v for v in row_values if v.strip())
                    all_text_parts.append(row_text)
                    sheet_row_texts.append(row_text)

            if sheet_data:
                tables.append(
                    {
                        "sheet": sheet_name,
                        "rows": len(sheet_data),
                        "cols": len(sheet_data[0]) if sheet_data else 0,
                        "data": sheet_data,
                    }
                )

            # Clean individual sheet text (without ## Sheet: marker)
            sheet_clean = "\n".join(sheet_row_texts)
            sheet_clean = clean_ocr_artifacts(sheet_clean, preserve_structure=True)
            per_sheet_cleaned.append(sheet_clean)

        full_text = "\n".join(all_text_parts)

        # Build pages from per-sheet cleaned text (each sheet = 1 page)
        pages: list[dict[str, Any]] = [
            {"page_number": idx + 1, "text": sheet_text}
            for idx, sheet_text in enumerate(per_sheet_cleaned)
            if sheet_text.strip()
        ]

        # Reconstruct cleaned_text from pages to guarantee invariant
        if pages:
            cleaned_text = "\n\n".join(p["text"] for p in pages)
        else:
            cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)
            cleaned_text = clean_format_specific_markers(cleaned_text, "xlsx")
            pages = [{"page_number": 1, "text": cleaned_text}]

        logger.debug(
            "xlsx_extracted",
            filename=filename,
            sheets=len(sheets),
            tables=len(tables),
            chars=len(full_text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=full_text,
            text_clean=cleaned_text,
            page_count=max(len(sheets), estimate_pages_from_text(full_text)),
            metadata={"sheets": sheets},
            tables=tables,
            sheets=sheets,
            pages=pages,
            success=True,
        )

    async def _extract_xls(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from legacy XLS spreadsheet."""
        if not XLS_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="xlrd not installed - run: pip install xlrd",
            )

        workbook = xlrd.open_workbook(file_contents=file_content)

        all_text_parts: list[str] = []
        sheets: list[str] = workbook.sheet_names()
        tables: list[dict[str, Any]] = []
        per_sheet_cleaned: list[str] = []  # Per-sheet cleaned text for pages

        for sheet_name in sheets:
            sheet = workbook.sheet_by_name(sheet_name)
            all_text_parts.append(f"## Sheet: {sheet_name}")

            sheet_data: list[list[str]] = []
            sheet_row_texts: list[str] = []  # Row texts for this sheet only
            for row_idx in range(sheet.nrows):
                row_values = [str(cell.value) for cell in sheet.row(row_idx)]

                if any(v.strip() for v in row_values):
                    sheet_data.append(row_values)
                    row_text = " | ".join(v for v in row_values if v.strip())
                    all_text_parts.append(row_text)
                    sheet_row_texts.append(row_text)

            if sheet_data:
                tables.append(
                    {
                        "sheet": sheet_name,
                        "rows": len(sheet_data),
                        "cols": len(sheet_data[0]) if sheet_data else 0,
                    }
                )

            # Clean individual sheet text (without ## Sheet: marker)
            sheet_clean = "\n".join(sheet_row_texts)
            sheet_clean = clean_ocr_artifacts(sheet_clean, preserve_structure=True)
            per_sheet_cleaned.append(sheet_clean)

        full_text = "\n".join(all_text_parts)

        # Build pages from per-sheet cleaned text (each sheet = 1 page)
        pages: list[dict[str, Any]] = [
            {"page_number": idx + 1, "text": sheet_text}
            for idx, sheet_text in enumerate(per_sheet_cleaned)
            if sheet_text.strip()
        ]

        # Reconstruct cleaned_text from pages to guarantee invariant
        if pages:
            cleaned_text = "\n\n".join(p["text"] for p in pages)
        else:
            cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)
            cleaned_text = clean_format_specific_markers(cleaned_text, "xls")
            pages = [{"page_number": 1, "text": cleaned_text}]

        logger.debug(
            "xls_extracted",
            filename=filename,
            sheets=len(sheets),
            chars=len(full_text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=full_text,
            text_clean=cleaned_text,
            page_count=max(len(sheets), estimate_pages_from_text(full_text)),
            metadata={"sheets": sheets},
            tables=tables,
            sheets=sheets,
            pages=pages,
            success=True,
        )

    async def _extract_pptx(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from PPTX presentation."""
        if not PPTX_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="python-pptx not installed - run: pip install python-pptx",
            )

        prs = Presentation(io.BytesIO(file_content))

        all_text_parts: list[str] = []
        slides: list[dict[str, Any]] = []
        per_slide_cleaned: list[str] = []  # Per-slide cleaned text for pages

        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_text_parts: list[str] = []
            all_text_parts.append(f"## Slide {slide_idx}")

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text = shape.text.strip()
                    slide_text_parts.append(text)
                    all_text_parts.append(text)

                # Extract text from tables
                if hasattr(shape, "table"):
                    table = shape.table
                    for row in table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_text_parts.append(row_text)
                            all_text_parts.append(row_text)

            slide_entry: dict[str, Any] = {
                "index": slide_idx,
                "text": " ".join(slide_text_parts),
            }

            # Speaker notes live on a separate notes slide. Keep them under a label so a
            # reader can tell them from the slide text. Check ``has_notes_slide`` first:
            # reading ``notes_slide`` on a slide without one creates it.
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                notes = notes_frame.text.strip() if notes_frame is not None else ""
                if notes:
                    labelled_notes = f"{PPTX_NOTES_LABEL}\n{notes}"
                    slide_entry["notes"] = notes
                    slide_text_parts.append(labelled_notes)
                    all_text_parts.append(labelled_notes)

            slides.append(slide_entry)

            # Clean individual slide text (without ## Slide marker)
            slide_clean = "\n\n".join(slide_text_parts)
            slide_clean = clean_ocr_artifacts(slide_clean, preserve_structure=True)
            per_slide_cleaned.append(slide_clean)

        full_text = "\n\n".join(all_text_parts)

        # Build pages from per-slide cleaned text (each slide = 1 page)
        pages: list[dict[str, Any]] = [
            {"page_number": idx + 1, "text": slide_text}
            for idx, slide_text in enumerate(per_slide_cleaned)
            if slide_text.strip()
        ]

        # Reconstruct cleaned_text from pages to guarantee invariant:
        # "\n\n".join(p["text"] for p in pages) == cleaned_text
        if pages:
            cleaned_text = "\n\n".join(p["text"] for p in pages)
        else:
            cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)
            cleaned_text = clean_format_specific_markers(cleaned_text, "pptx")
            pages = [{"page_number": 1, "text": cleaned_text}]

        logger.debug(
            "pptx_extracted",
            filename=filename,
            slides=len(slides),
            chars=len(full_text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=full_text,
            text_clean=cleaned_text,
            page_count=len(slides),
            metadata={"slides_count": len(slides)},
            slides=slides,
            pages=pages,
            success=True,
        )

    async def _extract_ppt(self, file_content: bytes, filename: str) -> ExtractedContent:
        """
        Extract text from legacy PPT format.

        Limited support - basic text extraction only.
        """
        try:
            import olefile

            if not olefile.isOleFile(file_content):
                return ExtractedContent(
                    text="",
                    success=False,
                    error="Not a valid PPT (OLE2) file. Consider converting to PPTX.",
                )

            ole = olefile.OleFileIO(io.BytesIO(file_content))
            texts: list[str] = []

            # PPT stores text in "PowerPoint Document" stream as UTF-16LE
            for stream_name in ole.listdir():
                try:
                    raw = ole.openstream(stream_name).read()
                    # Try UTF-16LE (standard for PPT text records)
                    decoded = raw.decode("utf-16-le", errors="ignore")
                    # Clean control chars but preserve Unicode
                    decoded = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", decoded)
                    cleaned = decoded.strip()
                    if len(cleaned) > 20:
                        texts.append(cleaned)
                except Exception:
                    continue

            ole.close()

            text = "\n".join(texts)
            text = re.sub(r"\s+", " ", text).strip()

            # Quality check: meaningful text relative to file size
            printable_ratio = len(text) / max(len(file_content), 1)
            if len(text) < 50 or (len(file_content) > 10240 and printable_ratio < 0.01):
                return ExtractedContent(
                    text="",
                    success=False,
                    error="PPT format has limited support. Consider converting to PPTX.",
                )

            # ✅ Apply OCR artifact cleaning
            from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

            cleaned_text = clean_ocr_artifacts(text, preserve_structure=False)

            # PPT extraction via olefile is unreliable - may extract only partial text.
            # Use the higher of char-based estimate or file-size-based minimum
            # (~15KB per slide for binary PPT) to prevent severe undercounting.
            char_estimate = estimate_pages_from_text(text)
            size_estimate = max(1, len(file_content) // 15_000)
            page_count = max(char_estimate, size_estimate)

            return ExtractedContent(
                text=text,
                text_clean=cleaned_text,
                page_count=page_count,
                metadata={"format_note": "PPT extraction via olefile"},
                pages=synthesize_pages_from_text(cleaned_text),
                success=True,
            )

        except Exception as e:
            return ExtractedContent(
                text="",
                success=False,
                error=f"PPT extraction failed: {e!s}. Consider converting to PPTX.",
            )
