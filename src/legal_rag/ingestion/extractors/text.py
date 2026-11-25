# ruff: noqa: RUF012
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Text Format Extractor - TXT, HTML, RTF, Markdown, CSV, JSON, XML.

Uses:
- beautifulsoup4 for HTML (with sanitization)
- striprtf for RTF
- defusedxml for XML (XXE protection)
- Built-in csv/json modules
"""

import csv
import io
import json
from typing import Any

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedContent,
    estimate_pages_from_text,
    synthesize_pages_from_text,
)

logger = get_logger("legal_rag.ingestion.extractors.text")

# Check available libraries
HTML_AVAILABLE = False
RTF_AVAILABLE = False
XML_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    HTML_AVAILABLE = True
except ImportError:
    logger.info("beautifulsoup4_not_installed", message="HTML support limited")

try:
    from striprtf.striprtf import rtf_to_text

    RTF_AVAILABLE = True
except ImportError:
    logger.info("striprtf_not_installed", message="RTF support disabled")

try:
    import defusedxml.ElementTree as ET

    XML_AVAILABLE = True
except ImportError:
    logger.info("defusedxml_not_installed", message="XML support disabled (XXE risk)")


class TextExtractor(BaseExtractor):
    """
    Extract text from text-based document formats.

    Supported formats:
    - TXT: Plain text (UTF-8 with fallback encodings)
    - HTML/HTM: Web pages (sanitized, text only)
    - RTF: Rich Text Format
    - MD/Markdown: Markdown files (preserved as-is)
    - CSV: Comma-separated values (converted to text)
    - JSON: JSON data (flattened to text)
    - XML: XML documents (text content only, XXE protected)
    """

    SUPPORTED = {
        "txt",
        "html",
        "htm",
        "rtf",
        "md",
        "markdown",
        "csv",
        "json",
        "xml",
    }

    # Encodings to try for text files.
    # Order matters: utf-16 must come AFTER latin-1/cp1252 because utf-16 can
    # successfully decode arbitrary byte sequences (treating them as 2-byte code
    # points) which produces garbage output. Latin-1 and cp1252 only decode valid
    # single-byte characters and are more discriminating.
    ENCODINGS = ["utf-8", "latin-1", "cp1252", "utf-16", "ascii"]

    async def extract(
        self,
        file_content: bytes,
        filename: str,
        **kwargs,
    ) -> ExtractedContent:
        """Extract text from text-based document."""
        ext = self.get_extension(filename)

        try:
            if ext == "txt":
                return await self._extract_txt(file_content, filename)
            elif ext in ("html", "htm"):
                return await self._extract_html(file_content, filename)
            elif ext == "rtf":
                return await self._extract_rtf(file_content, filename)
            elif ext in ("md", "markdown"):
                return await self._extract_markdown(file_content, filename)
            elif ext == "csv":
                return await self._extract_csv(file_content, filename)
            elif ext == "json":
                return await self._extract_json(file_content, filename)
            elif ext == "xml":
                return await self._extract_xml(file_content, filename)
            else:
                return ExtractedContent(
                    text="",
                    success=False,
                    error=f"Unsupported text format: {ext}",
                )

        except Exception as e:
            logger.exception("text_extraction_failed", filename=filename, ext=ext, error=str(e))
            return ExtractedContent(
                text="",
                success=False,
                error=f"Text extraction failed: {e!s}",
            )

    def _decode_with_fallback(self, content: bytes) -> str:
        """Decode bytes trying multiple encodings.

        UTF-16 BOM detection is checked first because the utf-16 codec can
        successfully decode arbitrary byte sequences (treating them as 2-byte
        code points), which would produce garbage for latin-1 or cp1252 files.
        By checking the BOM explicitly, we only use utf-16 when the file was
        genuinely encoded that way.
        """
        # Check for UTF-16 BOM before trying other encodings
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return content.decode("utf-16")
            except (UnicodeDecodeError, LookupError):
                pass

        for encoding in self.ENCODINGS:
            # Skip utf-16 here since we already handled it via BOM detection above
            if encoding == "utf-16":
                continue
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        # Last resort: decode with replacement characters
        return content.decode("utf-8", errors="replace")

    async def _extract_txt(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from plain text file."""
        text = self._decode_with_fallback(file_content)

        # Clean up common issues
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # ✅ Apply OCR artifact cleaning (minimal for TXT, but normalizes Unicode)
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        # Count lines for metadata
        lines = text.split("\n")
        # Use standard character-based page estimate (consistent with all other extractors)
        page_count = estimate_pages_from_text(text)

        logger.debug(
            "txt_extracted",
            filename=filename,
            lines=len(lines),
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=page_count,
            metadata={"lines": len(lines)},
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    async def _extract_html(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from HTML document."""
        html_text = self._decode_with_fallback(file_content)

        if HTML_AVAILABLE:
            soup = BeautifulSoup(html_text, "html.parser")

            # Extract title before <head> is removed below
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Remove script and style elements
            for element in soup(["script", "style", "head", "meta", "link"]):
                element.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            # ✅ Apply OCR artifact cleaning
            from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

            cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

            metadata: dict[str, Any] = {}
            if title:
                metadata["title"] = title

            logger.debug(
                "html_extracted",
                filename=filename,
                title=title,
                chars=len(text),
                cleaned_chars=len(cleaned_text),
            )

            return ExtractedContent(
                text=text,
                text_clean=cleaned_text,
                page_count=estimate_pages_from_text(text),
                metadata=metadata,
                pages=synthesize_pages_from_text(cleaned_text),
                success=True,
            )
        else:
            # Fallback: basic HTML tag removal
            import re

            text = re.sub(r"<[^>]+>", " ", html_text)
            text = re.sub(r"\s+", " ", text).strip()

            # ✅ Apply OCR artifact cleaning
            from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

            cleaned_text = clean_ocr_artifacts(text, preserve_structure=False)

            return ExtractedContent(
                text=text,
                text_clean=cleaned_text,
                page_count=estimate_pages_from_text(text),
                metadata={"note": "Basic HTML extraction (beautifulsoup4 not installed)"},
                pages=synthesize_pages_from_text(cleaned_text),
                success=True,
            )

    async def _extract_rtf(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from RTF document."""
        if not RTF_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="striprtf not installed - run: pip install striprtf",
            )

        rtf_text = self._decode_with_fallback(file_content)
        text = rtf_to_text(rtf_text)

        # ✅ Apply OCR artifact cleaning
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        logger.debug(
            "rtf_extracted",
            filename=filename,
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(text),
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    async def _extract_markdown(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from Markdown file (preserve formatting)."""
        text = self._decode_with_fallback(file_content)

        # Markdown is already text, just clean up
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Count sections by headers
        import re

        headers = re.findall(r"^#+\s+.+$", text, re.MULTILINE)

        # ✅ Apply OCR artifact cleaning (minimal for markdown)
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        logger.debug(
            "markdown_extracted",
            filename=filename,
            headers=len(headers),
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(text),
            metadata={"headers_count": len(headers)},
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    async def _extract_csv(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from CSV file."""
        csv_text = self._decode_with_fallback(file_content)

        # Detect delimiter
        try:
            dialect = csv.Sniffer().sniff(csv_text[:1024])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

        # Parse CSV
        reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
        rows = list(reader)

        # Convert to text
        text_parts: list[str] = []
        tables: list[dict[str, Any]] = []

        if rows:
            # First row might be headers
            headers = rows[0] if rows else []

            for row in rows:
                # Skip empty rows
                if any(cell.strip() for cell in row):
                    text_parts.append(" | ".join(cell.strip() for cell in row))

            tables.append(
                {
                    "headers": headers,
                    "rows": len(rows),
                    "cols": len(headers),
                }
            )

        text = "\n".join(text_parts)

        # ✅ Apply OCR artifact cleaning
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        logger.debug(
            "csv_extracted",
            filename=filename,
            rows=len(rows),
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(text),
            metadata={"rows": len(rows), "delimiter": delimiter},
            tables=tables,
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    async def _extract_json(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from JSON file."""
        json_text = self._decode_with_fallback(file_content)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            return ExtractedContent(
                text="",
                success=False,
                error=f"Invalid JSON: {e!s}",
            )

        # Flatten JSON to text
        text_parts: list[str] = []
        self._flatten_json(data, text_parts)
        text = "\n".join(text_parts)

        # ✅ Apply OCR artifact cleaning
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        logger.debug(
            "json_extracted",
            filename=filename,
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(text),
            metadata={"json_type": type(data).__name__},
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    def _flatten_json(
        self,
        data: Any,
        text_parts: list[str],
        prefix: str = "",
    ) -> None:
        """Recursively flatten JSON to text."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_json(value, text_parts, new_prefix)
                else:
                    text_parts.append(f"{new_prefix}: {value}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    self._flatten_json(item, text_parts, new_prefix)
                else:
                    text_parts.append(f"{new_prefix}: {item}")

        else:
            text_parts.append(f"{prefix}: {data}" if prefix else str(data))

    async def _extract_xml(self, file_content: bytes, filename: str) -> ExtractedContent:
        """Extract text from XML document (XXE protected)."""
        if not XML_AVAILABLE:
            return ExtractedContent(
                text="",
                success=False,
                error="defusedxml not installed - run: pip install defusedxml",
            )

        xml_text = self._decode_with_fallback(file_content)

        try:
            # Use defusedxml to prevent XXE attacks
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            return ExtractedContent(
                text="",
                success=False,
                error=f"Invalid XML: {e!s}",
            )

        # Extract text from all elements
        text_parts: list[str] = []
        self._extract_xml_text(root, text_parts)
        text = "\n".join(text_parts)

        # ✅ Apply OCR artifact cleaning
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(text, preserve_structure=True)

        logger.debug(
            "xml_extracted",
            filename=filename,
            root_tag=root.tag,
            chars=len(text),
            cleaned_chars=len(cleaned_text),
        )

        return ExtractedContent(
            text=text,
            text_clean=cleaned_text,
            page_count=estimate_pages_from_text(text),
            metadata={"root_tag": root.tag},
            pages=synthesize_pages_from_text(cleaned_text),
            success=True,
        )

    def _extract_xml_text(self, element: Any, text_parts: list[str]) -> None:
        """Recursively extract text from XML elements."""
        # Get element text
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())

        # Process children
        for child in element:
            self._extract_xml_text(child, text_parts)

        # Get tail text
        if element.tail and element.tail.strip():
            text_parts.append(element.tail.strip())
