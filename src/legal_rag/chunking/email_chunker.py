"""
Email-specific chunking strategy for RAG pipeline.

Key Differences from Document Chunking:
1. Preserves email headers with each chunk for retrieval context
2. Respects conversation structure (don't split mid-reply)
3. Handles quoted text (> lines) specially
4. Optionally removes/separates email signatures
5. Maintains thread context across chunks
6. Produces proper Chunk objects with position tracking

Best Practices from Research:
- Chunk by conversation messages, not arbitrary tokens
- Preserve sender/subject/date in chunk metadata
- Handle email-specific features (quotes, signatures, forwarded messages)
- Link reply chains for thread-aware retrieval

References:
- https://medium.com/@jojokirby/rag-on-email-data-a-general-guide-based-on-my-professional-experience-bb7f55b11412
- https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
"""

import re
import time
from typing import Any

from legal_rag.chunking._logging import get_logger
from legal_rag.chunking.base import BaseChunker, Chunk, ChunkerResult
from legal_rag.chunking.tokens import count_tokens as _count_tokens

logger = get_logger("email")

# Email section types
EMAIL_SECTION_HEADER = "email_header"
EMAIL_SECTION_BODY = "email_body"
EMAIL_SECTION_QUOTED = "email_quoted"
EMAIL_SECTION_FORWARDED = "email_forwarded"
EMAIL_SECTION_SIGNATURE = "email_signature"
EMAIL_SECTION_ATTACHMENT_LIST = "email_attachments"

# Common signature separators
SIGNATURE_SEPARATORS = {"--", "---", "____", "\u2014", "\u2501\u2501\u2501"}

# Forward markers
FORWARD_MARKERS = [
    "---------- forwarded message",
    "----original message",
    "begin forwarded message",
    "----- forwarded message -----",
    "forwarded by",
]

# Reply markers (lines starting with these indicate quoted text)
REPLY_PATTERN = re.compile(r"^>+\s?", re.MULTILINE)

# Common "On <date>, <person> wrote:" patterns
REPLY_HEADER_PATTERN = re.compile(
    r"^On .+? wrote:\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class EmailChunker(BaseChunker):
    """
    Email-specific semantic chunker conforming to BaseChunker interface.

    Chunks emails by preserving conversation structure and email-specific features.
    Produces proper Chunk/ChunkerResult objects compatible with the pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 30,
        preserve_signatures: bool = False,
        preserve_quoted_text: bool = True,
        header_boost: bool = True,
    ):
        """
        Initialize email chunker.

        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum tokens to keep a chunk
            preserve_signatures: Keep email signatures in chunks
            preserve_quoted_text: Keep quoted text (> lines) in chunks
            header_boost: Prepend email header context to each chunk for BM25 boost
        """
        super().__init__(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size
        )
        self.preserve_signatures = preserve_signatures
        self.preserve_quoted_text = preserve_quoted_text
        self.header_boost = header_boost

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using centralized tiktoken counter."""
        if not text or not text.strip():
            return 0
        return _count_tokens(text)

    async def chunk(
        self,
        text: str,
        document_id: str,
        sections: list[Any] | None = None,
        document_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> ChunkerResult:
        """
        Chunk email content with email-specific handling.

        Algorithm:
        1. Extract email header for contextual prepending
        2. Parse email into logical sections (body, quotes, forwards, signatures)
        3. Filter sections based on config (signatures, quoted text)
        4. Chunk each section with proper token-based splitting and overlap
        5. Prepend email header context to each chunk for BM25 retrieval boost
        6. Link chunks with prev/next references

        Args:
            text: Full email text (from EmailFileIngestor)
            document_id: Document identifier
            sections: Optional section info from ingestor (DocumentSection list)
            document_metadata: Email metadata dict (subject, from, to, etc.)
            **kwargs: Additional options

        Returns:
            ChunkerResult with email-aware chunks
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return ChunkerResult(
                document_id=document_id,
                chunks=[],
                total_chunks=0,
                total_tokens=0,
                processing_time_ms=0.0,
            )

        metadata = document_metadata or {}

        # Step 1: Build compact email header for context injection
        email_header = self._build_email_header(metadata)
        header_tokens = self.count_tokens(email_header) if email_header else 0

        # Effective chunk size after reserving space for header
        effective_chunk_size = (
            self.chunk_size - header_tokens if self.header_boost else self.chunk_size
        )

        # Step 2: Parse email into logical sections
        parsed_sections = self._parse_email_sections(text)

        # Step 3: Filter sections based on config
        if not self.preserve_signatures:
            parsed_sections = [s for s in parsed_sections if s["type"] != EMAIL_SECTION_SIGNATURE]

        if not self.preserve_quoted_text:
            parsed_sections = [
                s
                for s in parsed_sections
                if s["type"] not in (EMAIL_SECTION_QUOTED, EMAIL_SECTION_FORWARDED)
            ]

        # Step 4: Chunk each section with proper token-based splitting
        all_chunks: list[Chunk] = []
        char_offset = 0  # Track position in original text

        for section in parsed_sections:
            section_content = section["content"]
            section_type = section["type"]

            if not section_content.strip():
                char_offset += len(section_content)
                continue

            # Find actual position of this section in original text
            section_start = text.find(section_content[:50], char_offset)
            if section_start == -1:
                section_start = char_offset

            section_tokens = self.count_tokens(section_content)

            if section_tokens <= effective_chunk_size:
                # Section fits in one chunk, no splitting needed
                chunk_text = section_content.strip()
                if self.header_boost and email_header:
                    chunk_text = f"{email_header}\n\n{chunk_text}"

                chunk = Chunk(
                    content=chunk_text,
                    document_id=document_id,
                    chunk_index=len(all_chunks),
                    start_char=section_start,
                    end_char=section_start + len(section_content),
                    section_type=section_type,
                    token_count=self.count_tokens(chunk_text),
                    metadata={
                        **metadata,
                        "section_type": section_type,
                        "email_chunker": True,
                    },
                )
                if chunk.token_count >= self.min_chunk_size:
                    all_chunks.append(chunk)
            else:
                # Section too large, split with overlap
                sub_chunks = self._split_section_with_overlap(
                    section_content=section_content,
                    section_type=section_type,
                    section_start_char=section_start,
                    document_id=document_id,
                    email_header=email_header if self.header_boost else "",
                    effective_chunk_size=effective_chunk_size,
                    metadata=metadata,
                    start_index=len(all_chunks),
                )
                all_chunks.extend(sub_chunks)

            char_offset = section_start + len(section_content)

        # Step 5: Set total_chunks and link with prev/next references
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(all_chunks)

        all_chunks = self.link_chunks(all_chunks)

        # Calculate stats
        total_count = len(all_chunks)
        total_tokens = sum(c.token_count for c in all_chunks)
        token_counts = [c.token_count for c in all_chunks] if all_chunks else [0]
        avg_size = total_tokens / total_count if total_count > 0 else 0.0
        min_size = min(token_counts)
        max_size = max(token_counts)
        processing_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "email_chunking_completed",
            document_id=document_id,
            chunk_count=total_count,
            total_tokens=total_tokens,
            avg_chunk_size=round(avg_size, 2),
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            duration_ms=round(processing_time, 2),
            sections_parsed=len(parsed_sections),
            subject=metadata.get("subject", "")[:50],
        )

        return ChunkerResult(
            document_id=document_id,
            chunks=all_chunks,
            total_chunks=total_count,
            total_tokens=total_tokens,
            processing_time_ms=processing_time,
            avg_chunk_size=avg_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
        )

    def _build_email_header(self, metadata: dict) -> str:
        """
        Build compact email header for context injection into each chunk.

        This header is prepended to every chunk so hybrid search (BM25 + dense)
        can match on sender/subject/date even when the chunk body doesn't contain them.
        """
        parts = []

        if metadata.get("from"):
            sender = metadata["from"]
            if isinstance(sender, dict):
                sender = sender.get("address", sender.get("name", str(sender)))
            parts.append(f"From: {sender}")

        if metadata.get("subject"):
            parts.append(f"Subject: {metadata['subject']}")

        if metadata.get("to"):
            recipients = metadata["to"]
            if isinstance(recipients, list):
                to_str = ", ".join(
                    r.get("address", str(r)) if isinstance(r, dict) else str(r)
                    for r in recipients[:3]  # Limit to 3 recipients to save tokens
                )
                if len(recipients) > 3:
                    to_str += f" (+{len(recipients) - 3} more)"
            else:
                to_str = str(recipients)
            parts.append(f"To: {to_str}")

        if metadata.get("date"):
            parts.append(f"Date: {metadata['date']}")

        if metadata.get("thread_id"):
            parts.append(f"Thread: {metadata['thread_id']}")

        return "\n".join(parts)

    def _parse_email_sections(self, text: str) -> list[dict]:
        """
        Parse email text into logical sections with type annotations.

        Detects:
        - Original message body
        - Quoted replies (lines starting with >)
        - Forwarded content (---------- Forwarded message)
        - Signatures (--, ---, ____)
        - Attachment listings (--- Attachments ---)

        Returns list of dicts with 'type' and 'content' keys.
        """
        sections: list[dict] = []
        current_type = EMAIL_SECTION_BODY
        current_lines: list[str] = []

        lines = text.splitlines(keepends=True)

        for line in lines:
            stripped = line.strip()

            # Check for attachment listing section
            if stripped == "--- Attachments ---":
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_ATTACHMENT_LIST
                current_lines.append(line)
                continue

            # Check for email body marker (from EmailFileIngestor)
            if stripped == "--- Email Body ---":
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_BODY
                continue

            # Check for signature separator
            if stripped in SIGNATURE_SEPARATORS:
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_SIGNATURE
                current_lines.append(line)
                continue

            # Check for forward markers
            if any(marker in stripped.lower() for marker in FORWARD_MARKERS):
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_FORWARDED
                current_lines.append(line)
                continue

            # Check for "On ... wrote:" reply header
            if REPLY_HEADER_PATTERN.match(stripped):
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_QUOTED
                current_lines.append(line)
                continue

            # Check for quoted text (> prefix)
            is_quote = stripped.startswith(">")
            if is_quote and current_type != EMAIL_SECTION_QUOTED:
                if current_lines:
                    sections.append({"type": current_type, "content": "".join(current_lines)})
                    current_lines = []
                current_type = EMAIL_SECTION_QUOTED
            elif not is_quote and current_type == EMAIL_SECTION_QUOTED and stripped:
                # End of quoted block, switch back to body
                sections.append({"type": current_type, "content": "".join(current_lines)})
                current_lines = []
                current_type = EMAIL_SECTION_BODY

            current_lines.append(line)

        # Flush remaining lines
        if current_lines:
            content = "".join(current_lines)
            if content.strip():
                sections.append({"type": current_type, "content": content})

        return sections

    def _split_section_with_overlap(
        self,
        section_content: str,
        section_type: str,
        section_start_char: int,
        document_id: str,
        email_header: str,
        effective_chunk_size: int,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        """
        Split a large section into token-bounded chunks with true overlap.

        Uses sentence-level splitting to avoid breaking mid-sentence,
        with token counting for accurate size control.
        """
        chunks: list[Chunk] = []

        # Split into sentences (preserve newlines as potential boundaries)
        sentences = self._split_into_sentences(section_content)

        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > effective_chunk_size:
                # Flush current buffer
                if current_sentences:
                    chunk = self._create_chunk(
                        sentences=current_sentences,
                        section_type=section_type,
                        section_content=section_content,
                        section_start_char=section_start_char,
                        document_id=document_id,
                        email_header=email_header,
                        metadata=metadata,
                        index=start_index + len(chunks),
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_sentences = []
                    current_tokens = 0

                # Force-split the oversized sentence by words
                words = sentence.split()
                word_buffer: list[str] = []
                word_tokens = 0
                for word in words:
                    wt = self.count_tokens(word + " ")
                    if word_tokens + wt > effective_chunk_size and word_buffer:
                        forced_text = " ".join(word_buffer)
                        chunk = self._create_chunk(
                            sentences=[forced_text],
                            section_type=section_type,
                            section_content=section_content,
                            section_start_char=section_start_char,
                            document_id=document_id,
                            email_header=email_header,
                            metadata=metadata,
                            index=start_index + len(chunks),
                        )
                        if chunk:
                            chunks.append(chunk)
                        word_buffer = []
                        word_tokens = 0
                    word_buffer.append(word)
                    word_tokens += wt
                if word_buffer:
                    current_sentences = [" ".join(word_buffer)]
                    current_tokens = word_tokens
                continue

            if current_tokens + sentence_tokens > effective_chunk_size and current_sentences:
                # Flush current chunk
                chunk = self._create_chunk(
                    sentences=current_sentences,
                    section_type=section_type,
                    section_content=section_content,
                    section_start_char=section_start_char,
                    document_id=document_id,
                    email_header=email_header,
                    metadata=metadata,
                    index=start_index + len(chunks),
                )
                if chunk:
                    chunks.append(chunk)

                # Start new chunk with overlap from end of previous
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self.chunk_overlap
                )
                current_sentences = [*overlap_sentences, sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Flush remaining
        if current_sentences:
            chunk = self._create_chunk(
                sentences=current_sentences,
                section_type=section_type,
                section_content=section_content,
                section_start_char=section_start_char,
                document_id=document_id,
                email_header=email_header,
                metadata=metadata,
                index=start_index + len(chunks),
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        sentences: list[str],
        section_type: str,
        section_content: str,
        section_start_char: int,
        document_id: str,
        email_header: str,
        metadata: dict,
        index: int,
    ) -> Chunk | None:
        """Create a Chunk object from a list of sentences with proper position tracking."""
        text = " ".join(sentences).strip()
        if not text:
            return None

        # Find position in section content
        start_pos = section_content.find(text[:40])
        if start_pos == -1:
            start_pos = 0
        end_pos = start_pos + len(text)

        # Build final content with header
        final_content = f"{email_header}\n\n{text}" if email_header else text
        token_count = self.count_tokens(final_content)

        if token_count < self.min_chunk_size:
            return None

        return Chunk(
            content=final_content,
            document_id=document_id,
            chunk_index=index,
            start_char=section_start_char + start_pos,
            end_char=section_start_char + end_pos,
            section_type=section_type,
            token_count=token_count,
            metadata={
                **metadata,
                "section_type": section_type,
                "email_chunker": True,
            },
        )

    def _get_overlap_sentences(self, sentences: list[str], overlap_tokens: int) -> list[str]:
        """Get sentences from the end that fit within the overlap token budget."""
        overlap: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            st = self.count_tokens(sentence)
            if tokens + st > overlap_tokens:
                break
            overlap.insert(0, sentence)
            tokens += st
        return overlap

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentence-level units preserving email structure.

        Handles:
        - Standard sentence endings (. ! ?)
        - Paragraph breaks (double newlines)
        - Email-style line breaks
        """
        # First split on paragraph breaks
        paragraphs = re.split(r"\n\s*\n", text)

        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Split paragraph into sentences
            # Use regex to split on sentence boundaries while preserving delimiters
            parts = re.split(r"(?<=[.!?])\s+", para)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)

        return sentences


def get_email_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    preserve_signatures: bool = False,
    preserve_quoted_text: bool = True,
) -> EmailChunker:
    """Factory function to create an EmailChunker with sensible defaults."""
    return EmailChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_signatures=preserve_signatures,
        preserve_quoted_text=preserve_quoted_text,
    )
