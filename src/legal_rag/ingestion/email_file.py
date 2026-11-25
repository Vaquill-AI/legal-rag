# ruff: noqa: BLE001, RUF012
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
Email File Ingestor for RAG Pipeline.

Parses email files (.eml, .msg) into RAG-compatible format for legal document processing.

Features:
- EML file parsing (RFC 822 format) using Python's built-in email library
- MSG file parsing (Microsoft Outlook) using extract-msg library
- Full metadata extraction (From, To, CC, BCC, Subject, Date, Message-ID)
- Thread/conversation detection via In-Reply-To and References headers
- HTML to text conversion with BeautifulSoup fallback
- Attachment listing for legal preservation (content not extracted here)
- E-discovery metadata preservation

Output Format:
Same structure as other ingestors for pipeline compatibility:
- IngestorResult with text, sections, and email-specific metadata

Installation:
    pip install extract-msg  # For MSG files (optional)
    pip install beautifulsoup4  # For better HTML parsing (optional)
"""

import asyncio
import contextlib
import email
import email.message
import email.policy
import html
import re
import time
from datetime import datetime
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.base import BaseIngestor, DocumentSection, DocumentType, IngestorResult
from legal_rag.ingestion.extractors.base import estimate_pages_from_text, synthesize_pages_from_text

logger = get_logger("legal_rag.ingestion.email_file")

# Try to import extract-msg for MSG files
try:
    import extract_msg

    MSG_AVAILABLE = True
except ImportError:
    MSG_AVAILABLE = False
    logger.info("extract_msg_not_available", hint="pip install extract-msg for MSG support")

# Try to import beautifulsoup for HTML parsing
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.info("beautifulsoup4_not_available", hint="pip install beautifulsoup4 for better HTML")


# Inline image / embedded resource URI schemes that survive tag stripping as
# raw base64 text. Forwarded HTML emails (Outlook, Gmail) embed logos,
# signatures, and screenshots as `data:image/...;base64,<blob>` directly in
# the markup. Stripping these before text extraction prevents thousands of
# base64 chars from polluting the indexable content and tripping downstream
# readability heuristics.
_DATA_URI_HREF_RE = re.compile(r'\s(?:src|href)\s*=\s*"(?:data|cid):[^"]*"', re.IGNORECASE)
_DATA_URI_HREF_SQ_RE = re.compile(r"\s(?:src|href)\s*=\s*'(?:data|cid):[^']*'", re.IGNORECASE)
_BARE_DATA_URI_RE = re.compile(
    r"data:[a-z0-9.+-]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE
)


def _html_to_text(html_content: str) -> str:
    """
    Convert HTML email content to plain text.

    Uses BeautifulSoup if available, falls back to regex-based stripping.
    Inline `data:` and `cid:` resources are stripped before extraction to
    keep base64 blobs out of the indexable text.
    """
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Drop inline base64 / cid resources entirely. These contribute no
        # retrieval value and produce dense unbroken character runs.
        for tag in soup.find_all(["img", "source", "video", "audio", "iframe"]):
            src = tag.get("src", "")
            if isinstance(src, str) and src.lower().startswith(("data:", "cid:")):
                tag.decompose()
        for tag in soup.find_all("a"):
            href = tag.get("href", "")
            if isinstance(href, str) and href.lower().startswith(("data:", "cid:")):
                tag.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Belt-and-braces: strip any bare data: URI that slipped through
        # (e.g., when present in CDATA, comments, or non-standard attributes).
        text = _BARE_DATA_URI_RE.sub("", text)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    else:
        # Basic HTML stripping without BeautifulSoup
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip data:/cid: src/href attributes before generic tag removal so
        # the base64 payload doesn't get exposed as text content.
        text = _DATA_URI_HREF_RE.sub("", text)
        text = _DATA_URI_HREF_SQ_RE.sub("", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = _BARE_DATA_URI_RE.sub("", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def _parse_email_address(addr_string: str) -> dict[str, str]:
    """Parse email address into name and email components."""
    name, email_addr = parseaddr(addr_string)
    return {"name": name, "email": email_addr, "full": addr_string}


def _parse_email_addresses(addr_string: str) -> list[dict[str, str]]:
    """Parse multiple email addresses from a header."""
    if not addr_string:
        return []

    addresses = getaddresses([addr_string])
    return [
        {"name": name, "email": addr, "full": f"{name} <{addr}>" if name else addr}
        for name, addr in addresses
        if addr
    ]


def _get_email_body(msg: email.message.Message) -> dict[str, str]:
    """
    Extract email body, preferring plain text over HTML.

    Returns both plain and HTML versions if available.
    """
    plain_body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload and isinstance(payload, bytes):
                        plain_body = payload.decode(charset, errors="replace")
                except Exception:
                    plain_body = str(part.get_payload())

            elif content_type == "text/html":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload and isinstance(payload, bytes):
                        html_body = payload.decode(charset, errors="replace")
                except Exception:
                    html_body = str(part.get_payload())
    else:
        content_type = msg.get_content_type()
        try:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload and isinstance(payload, bytes):
                body = payload.decode(charset, errors="replace")
            else:
                body = str(msg.get_payload())

            if content_type == "text/html":
                html_body = body
            else:
                plain_body = body
        except Exception:
            plain_body = str(msg.get_payload())

    return {"plain": plain_body, "html": html_body}


def _extract_attachments_info(msg: email.message.Message) -> list[dict[str, Any]]:
    """Extract attachment metadata without downloading content."""
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                filename = part.get_filename()
                content_type = part.get_content_type()

                # Get size
                payload = part.get_payload(decode=True)
                size = len(payload) if payload else 0

                attachments.append(
                    {
                        "filename": filename or "unnamed",
                        "content_type": content_type,
                        "size_bytes": size,
                    }
                )

    return attachments


class EmailFileIngestor(BaseIngestor):
    """
    Email file ingestor for .eml and .msg files.

    Designed for document collections that include email.
    Preserves full email metadata for legal compliance and thread detection.
    """

    SUPPORTED_EXTENSIONS = {"eml", "msg"}

    def __init__(self, preserve_attachments_list: bool = True):
        """
        Initialize email file ingestor.

        Args:
            preserve_attachments_list: Include attachment metadata in output
        """
        self.preserve_attachments_list = preserve_attachments_list

    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "email",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest email file and extract content.

        Args:
            file_path: Path to email file
            file_content: Raw file bytes
            filename: Original filename
            document_id: Unique document identifier
            **kwargs: Additional options

        Returns:
            IngestorResult with extracted email content and metadata
        """
        start_time = time.time()

        # Determine file extension
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        # Read content if file_path provided
        if file_content is None and file_path:
            # Use asyncio to avoid blocking on file I/O
            loop = asyncio.get_event_loop()
            file_content = await loop.run_in_executor(None, self._read_file, file_path)

        if file_content is None:
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error="No file content provided",
                parser_name="email_file_ingestor",
            )

        try:
            if ext == "eml":
                result = await self._parse_eml(file_content, filename, document_id)
            elif ext == "msg":
                result = await self._parse_msg(file_content, filename, document_id)
            else:
                return IngestorResult(
                    document_id=document_id or "",
                    filename=filename,
                    content="",
                    success=False,
                    error=f"Unsupported email format: {ext}. Supported: .eml, .msg",
                    parser_name="email_file_ingestor",
                )

            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error(
                "email_ingest_failed",
                filename=filename,
                extension=ext,
                error=str(e),
                error_type=type(e).__name__,
            )
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error=f"Failed to parse email: {e!s}",
                parser_name="email_file_ingestor",
            )

    def _read_file(self, file_path: str) -> bytes:
        """Read file synchronously (called via executor)."""
        with open(file_path, "rb") as f:
            return f.read()

    async def _parse_eml(
        self, file_content: bytes, filename: str, document_id: str | None
    ) -> IngestorResult:
        """Parse EML file into IngestorResult."""
        # Parse email message
        msg = email.message_from_bytes(file_content, policy=email.policy.default)

        # Extract headers
        subject = msg.get("Subject", "")
        from_addr = msg.get("From", "")
        to_addr = msg.get("To", "")
        cc_addr = msg.get("Cc", "")
        bcc_addr = msg.get("Bcc", "")
        date_str = msg.get("Date", "")
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        references = msg.get("References", "")

        # Parse date
        date_parsed: datetime | None = None
        if date_str:
            with contextlib.suppress(Exception):
                date_parsed = parsedate_to_datetime(date_str)

        # Get body
        body = _get_email_body(msg)

        # Prefer plain text, fall back to converted HTML
        if body["plain"]:
            text_content = body["plain"]
        elif body["html"]:
            text_content = _html_to_text(body["html"])
        else:
            text_content = ""

        # Get attachments
        attachments = _extract_attachments_info(msg) if self.preserve_attachments_list else []

        # Build structured text for RAG
        email_text_parts = []

        # Header section
        email_text_parts.append(f"Subject: {subject}")
        email_text_parts.append(f"From: {from_addr}")
        email_text_parts.append(f"To: {to_addr}")
        if cc_addr:
            email_text_parts.append(f"Cc: {cc_addr}")
        if date_str:
            email_text_parts.append(f"Date: {date_str}")
        email_text_parts.append("")
        email_text_parts.append("--- Email Body ---")
        email_text_parts.append("")
        email_text_parts.append(text_content)

        if attachments:
            email_text_parts.append("")
            email_text_parts.append("--- Attachments ---")
            for att in attachments:
                email_text_parts.append(
                    f"- {att['filename']} ({att['content_type']}, {att['size_bytes']} bytes)"
                )

        full_text = "\n".join(email_text_parts)

        # Build sections
        sections = [
            DocumentSection(
                title="Email Headers",
                content="\n".join(email_text_parts[:6]),
                level=1,
                metadata={"section_type": "headers"},
            ),
            DocumentSection(
                title="Email Body",
                content=text_content,
                level=1,
                metadata={"section_type": "body"},
            ),
        ]

        if attachments:
            sections.append(
                DocumentSection(
                    title="Attachments",
                    content="\n".join(
                        f"- {att['filename']} ({att['content_type']})" for att in attachments
                    ),
                    level=1,
                    metadata={"section_type": "attachments", "attachment_count": len(attachments)},
                )
            )

        # Build email-specific metadata
        email_metadata = {
            "file_type": "eml",
            "subject": subject,
            "from": _parse_email_address(from_addr),
            "to": _parse_email_addresses(to_addr),
            "cc": _parse_email_addresses(cc_addr),
            "bcc": _parse_email_addresses(bcc_addr),
            "date": date_parsed.isoformat() if date_parsed else date_str,
            "date_parsed": date_parsed,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "references": references.split() if references else [],
            "has_html": bool(body["html"]),
            "has_plain": bool(body["plain"]),
            "attachment_count": len(attachments),
            "attachments": attachments,
        }

        # ✅ Apply OCR artifact cleaning to email content
        from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

        cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)

        logger.debug(
            "eml_parsed",
            filename=filename,
            subject=subject[:50] if subject else "(no subject)",
            from_addr=from_addr,
            attachment_count=len(attachments),
            content_length=len(full_text),
            cleaned_length=len(cleaned_text),
        )

        return IngestorResult(
            document_id=document_id or "",
            filename=filename,
            content=cleaned_text,  # ✅ Cleaned email content
            text_clean=cleaned_text,
            sections=sections,
            document_type=DocumentType.UNKNOWN,  # Will add EML type below
            page_count=estimate_pages_from_text(cleaned_text),
            pages=synthesize_pages_from_text(cleaned_text),
            metadata=email_metadata,
            title=f"Email: {subject}" if subject else "Email: (No Subject)",
            date=date_parsed.isoformat() if date_parsed else None,
            success=True,
            total_chars=len(cleaned_text),
            parser_name="email_file_ingestor",
            supports_positions=False,
        )

    async def _parse_msg(
        self, file_content: bytes, filename: str, document_id: str | None
    ) -> IngestorResult:
        """Parse MSG file (Outlook format) into IngestorResult."""
        if not MSG_AVAILABLE:
            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content="",
                success=False,
                error="extract-msg not installed. Run: pip install extract-msg",
                parser_name="email_file_ingestor",
            )

        # Write to temp file since extract-msg needs a file path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            msg = extract_msg.Message(tmp_path)

            # Extract fields
            subject = msg.subject or ""
            sender = msg.sender or ""
            to = msg.to or ""
            cc = msg.cc or ""
            date = msg.date
            body = msg.body or ""

            # Get attachments
            attachments = []
            if self.preserve_attachments_list:
                for att in msg.attachments:
                    attachments.append(
                        {
                            "filename": att.longFilename or att.shortFilename or "unnamed",
                            "content_type": getattr(att, "mimetype", None)
                            or "application/octet-stream",
                            "size_bytes": len(att.data) if att.data else 0,
                        }
                    )

            # Build structured text
            email_text_parts = []
            email_text_parts.append(f"Subject: {subject}")
            email_text_parts.append(f"From: {sender}")
            email_text_parts.append(f"To: {to}")
            if cc:
                email_text_parts.append(f"Cc: {cc}")
            if date:
                email_text_parts.append(f"Date: {date}")
            email_text_parts.append("")
            email_text_parts.append("--- Email Body ---")
            email_text_parts.append("")
            email_text_parts.append(body)

            if attachments:
                email_text_parts.append("")
                email_text_parts.append("--- Attachments ---")
                for att in attachments:
                    email_text_parts.append(
                        f"- {att['filename']} ({att['content_type']}, {att['size_bytes']} bytes)"
                    )

            full_text = "\n".join(email_text_parts)

            # Build sections
            sections = [
                DocumentSection(
                    title="Email Headers",
                    content="\n".join(email_text_parts[:6]),
                    level=1,
                    metadata={"section_type": "headers"},
                ),
                DocumentSection(
                    title="Email Body",
                    content=body,
                    level=1,
                    metadata={"section_type": "body"},
                ),
            ]

            if attachments:
                sections.append(
                    DocumentSection(
                        title="Attachments",
                        content="\n".join(
                            f"- {att['filename']} ({att['content_type']})" for att in attachments
                        ),
                        level=1,
                        metadata={
                            "section_type": "attachments",
                            "attachment_count": len(attachments),
                        },
                    )
                )

            # Parse date if available
            date_parsed: datetime | None = None
            if date:
                if isinstance(date, datetime):
                    date_parsed = date
                else:
                    with contextlib.suppress(Exception):
                        date_parsed = parsedate_to_datetime(str(date))

            # Build metadata
            email_metadata = {
                "file_type": "msg",
                "subject": subject,
                "from": {"email": sender, "name": "", "full": sender},
                "to": [{"email": to, "name": "", "full": to}] if to else [],
                "cc": [{"email": cc, "name": "", "full": cc}] if cc else [],
                "date": str(date) if date else "",
                "date_parsed": date_parsed,
                "attachment_count": len(attachments),
                "attachments": attachments,
            }

            msg.close()

            # ✅ Apply OCR artifact cleaning to MSG content
            from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

            cleaned_text = clean_ocr_artifacts(full_text, preserve_structure=True)

            logger.debug(
                "msg_parsed",
                filename=filename,
                subject=subject[:50] if subject else "(no subject)",
                sender=sender,
                attachment_count=len(attachments),
                content_length=len(full_text),
                cleaned_length=len(cleaned_text),
            )

            return IngestorResult(
                document_id=document_id or "",
                filename=filename,
                content=cleaned_text,  # ✅ Cleaned MSG content
                text_clean=cleaned_text,
                sections=sections,
                document_type=DocumentType.UNKNOWN,  # Will add MSG type below
                page_count=estimate_pages_from_text(cleaned_text),
                pages=synthesize_pages_from_text(cleaned_text),
                metadata=email_metadata,
                title=f"Email: {subject}" if subject else "Email: (No Subject)",
                date=date_parsed.isoformat() if date_parsed else None,
                success=True,
                total_chars=len(cleaned_text),
                parser_name="email_file_ingestor",
                supports_positions=False,
            )

        except Exception as e:
            logger.error(
                "msg_parse_failed",
                filename=filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Cleanup temp file
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink()

    async def supports_format(self, filename: str) -> bool:
        """Check if this ingestor supports the given file format."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        return ext in self.SUPPORTED_EXTENSIONS
