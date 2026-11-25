"""
Hybrid Recursive Legal Chunker with Universal Document Type Support.

Combines recursive character splitting with legal section awareness for:
- TRUE overlap support (not simulated via neighbor accumulation)
- Guaranteed clean splits via fallback hierarchy
- Legal section metadata preservation
- Multi-document-type support (judgments, contracts, statutes, arbitration, etc.)
- Research-backed chunking parameters

Architecture:
    Document -> DocumentTypeClassifier.classify() -> DocumentType
                                                          |
             -> UniversalSectionDetector.detect() -> [SectionBoundary...]
                                                          |
             RecursiveLegalChunker.chunk() <-- For each section:
                  |-- Recursive split with legal-aware separators
                  |-- Apply overlap between chunks
                  |-- Map to LegalChunk with section metadata
                  +-- Link neighbors (prev/next)
                                                          |
                                                    ChunkerResult

Supported document types:
- JUDGMENT: Court judgments (US, UK, other common-law courts)
- CONTRACT: Commercial contracts, agreements, NDAs
- STATUTE: Acts, rules, regulations, ordinances
- ARBITRATION: Awards, claims, procedural documents
- CORPORATE: MOA, AOA, resolutions, minutes
- MNA: M&A documents, SPAs, APAs
- REGULATORY: Agency filings, notices and compliance documents
- IP: Patents, trademarks
- REAL_ESTATE: Deeds, leases
- ESTATE: Wills, trusts, POA
- PLEADING: Complaints, petitions
- CIRCULAR: Agency circulars, bulletins and notices

Research basis:
- Chroma Research (2024): RecursiveCharacterTextSplitter optimal at 200-400 tokens
- arxiv:2410.13070: Semantic chunking computational cost not justified
- This chunker: Best of both - recursive splitting + legal section awareness

Cost: $0.00 per document (no API calls during chunking)
"""

import asyncio
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import uuid4

from legal_rag.chunking._logging import get_logger
from legal_rag.chunking.base import BaseChunker, Chunk, ChunkerResult
from legal_rag.chunking.config import get_chunking_config
from legal_rag.chunking.document_types import (
    DocumentType,
    LegalSectionType,
    UniversalSectionBoundary,
    get_section_priority,
    get_universal_section_detector,
)
from legal_rag.chunking.legal_sections import (
    LegalSection,
    SectionBoundary,
    get_legal_section_detector,
)
from legal_rag.chunking.legal_sections import (
    get_section_priority as get_legacy_section_priority,
)
from legal_rag.chunking.tokens import count_tokens as _count_tokens
from legal_rag.chunking.tokens import get_encoding

logger = get_logger("recursive_legal")


# =========================================================================
# Per-section parallel chunking
# =========================================================================
# Section chunking is CPU-bound (regex splits plus token encodes). Each section is
# an independent unit of work: `_chunk_universal_section` and `_chunk_legacy_section`
# only read immutable `self` state and never write back. Sections are fanned out
# across OS threads with `asyncio.gather + asyncio.to_thread` because tiktoken's
# Rust core releases the GIL inside `.encode()`, so threads get real parallelism
# on token counting, and threads avoid serializing the parsed text between processes.
#
# Set `max_concurrency=1` on the chunker to force sequential processing. The code
# path is identical (same per-section function, same boundary order); a limit of 1
# only turns the semaphore into a mutex.
DEFAULT_CHUNK_SECTIONS_MAX_CONCURRENCY = 8


@dataclass
class LegalChunk(Chunk):
    """
    Chunk with legal section metadata.

    Extends base Chunk with:
    - section: LegalSection enum (legacy detector)
    - section_type_universal: LegalSectionType enum (universal, ~50 section types)
    - document_type: DocumentType enum (JUDGMENT, CONTRACT, STATUTE, etc.)
    - section_confidence: Detection confidence score (0.0-1.0)
    - section_priority: Priority score for retrieval boosting (0-100)
    - contextual_header: Case/document metadata header for BM25 boost
    - has_true_overlap: Whether this chunk has true overlap with neighbors
    - hierarchy_level: Section nesting level (0=top, 1=subsection, etc.)
    """

    # Legacy field (for backward compatibility)
    section: LegalSection = LegalSection.UNKNOWN
    # Universal section type (supports ~50 section types across all document categories)
    section_type_universal: LegalSectionType = LegalSectionType.UNKNOWN
    # Document type classification
    document_type: DocumentType = DocumentType.UNKNOWN
    section_confidence: float = 0.0
    section_priority: int = 30  # 0-100 score from get_section_priority()
    contextual_header: str = ""  # Document metadata for BM25
    has_true_overlap: bool = True  # This chunker supports true overlap
    hierarchy_level: int = 0  # Section nesting level


class RecursiveLegalChunker(BaseChunker):
    """
    Hybrid chunker: Recursive splitting + Universal legal section awareness + TRUE overlap.

    Key advantages over semchunk-based approach:
    1. TRUE OVERLAP: Actual token overlap between chunks (not simulated)
    2. GUARANTEED SPLITS: Fallback hierarchy ensures clean splits
    3. UNIVERSAL LEGAL AWARENESS: Supports ~50 section types across 12 document categories
    4. ZERO COST: No API calls during chunking
    5. AUTOMATIC DOCUMENT TYPE DETECTION: Routes to optimal section patterns

    Supported document types:
    - JUDGMENT: Court judgments (US, UK, other common-law courts)
    - CONTRACT: Commercial contracts, agreements, NDAs
    - STATUTE: Acts, rules, regulations
    - ARBITRATION: Awards, claims, procedural documents
    - CORPORATE: MOA, AOA, resolutions, minutes
    - MNA: M&A documents, SPAs, APAs
    - REGULATORY: Agency filings, notices and compliance documents
    - IP: Patents, trademarks
    - REAL_ESTATE: Deeds, leases
    - ESTATE: Wills, trusts, POA
    - PLEADING: Complaints, petitions
    - CIRCULAR: Agency circulars, bulletins and notices

    Recursive splitting fallback hierarchy (legal-aware):
    1. Legal section markers: "\\n\\nJUDGMENT", "\\n\\nARTICLE", etc.
    2. Double newlines: "\\n\\n" (paragraph boundaries)
    3. Single newlines: "\\n" (line boundaries)
    4. Sentence endings: ". " (sentence boundaries)
    5. Spaces: " " (word boundaries)
    6. Characters: "" (last resort)

    Example:
        chunker = RecursiveLegalChunker()
        result = await chunker.chunk(text, document_id)

        for chunk in result.chunks:
            print(f"DocType: {chunk.document_type.value}")
            print(f"Section: {chunk.section_type_universal.value}")
            print(f"Priority: {chunk.section_priority}")
    """

    # Universal legal-aware separators (tried in order)
    # These preserve legal document structure during recursive splitting
    LEGAL_SEPARATORS: ClassVar[list[str]] = [
        # Judgment section markers
        "\n\nJUDGMENT",
        "\n\nORDER",
        "\n\nHEADNOTE",
        "\n\nOPERATIVE PART",
        # Contract section markers
        "\n\nARTICLE",
        "\n\nWHEREAS",
        "\n\nNOW, THEREFORE",
        "\n\nIN WITNESS WHEREOF",
        # Statute section markers
        "\n\nCHAPTER",
        "\n\nPART",
        "\n\nSection",
        # Arbitration section markers
        "\n\nAWARD",
        "\n\nTRIBUNAL",
        # Corporate section markers
        "\n\nRESOLVED",
        # Standard structural separators
        "\n\n",  # Paragraph boundaries
        "\n",  # Line boundaries
        ". ",  # Sentence boundaries
        ", ",  # Clause boundaries
        " ",  # Word boundaries
        "",  # Character-level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
        separators: list[str] | None = None,
        use_universal_detector: bool = True,
        max_concurrency: int = DEFAULT_CHUNK_SECTIONS_MAX_CONCURRENCY,
    ):
        """
        Initialize hybrid recursive legal chunker with universal document type support.

        Args:
            chunk_size: Target chunk size in tokens (default: from config, 512)
            chunk_overlap: TRUE overlap between chunks in tokens (default: from config, 100)
            min_chunk_size: Minimum chunk size to keep (default: from config, 50)
            separators: Custom separators list (default: LEGAL_SEPARATORS)
            use_universal_detector: Use UniversalSectionDetector (default: True)
                                   Set to False for legacy judgment-only detection
        """
        config = get_chunking_config()

        super().__init__(
            chunk_size=chunk_size or config.chunk_size,
            chunk_overlap=config.chunk_overlap if chunk_overlap is None else chunk_overlap,
            min_chunk_size=config.min_chunk_size if min_chunk_size is None else min_chunk_size,
        )

        self.config = config
        self.use_universal_detector = use_universal_detector

        # Initialize appropriate detector
        if use_universal_detector:
            self._universal_detector = get_universal_section_detector()
            self._legacy_detector = None
        else:
            self._universal_detector = None
            self._legacy_detector = get_legal_section_detector()

        self.separators = separators or self.LEGAL_SEPARATORS
        self.max_concurrency = max(1, max_concurrency)

        logger.info(
            "recursive_legal_chunker_initialized",
            component="chunker",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            overlap_percentage=round(self.chunk_overlap / self.chunk_size * 100, 1),
            min_chunk_size=self.min_chunk_size,
            separator_count=len(self.separators),
            detector_type="universal" if use_universal_detector else "legacy",
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using centralized tiktoken counter."""
        if not text or not text.strip():
            return 0
        return _count_tokens(text)

    def _split_text_recursive(
        self,
        text: str,
        separators: list[str],
        _depth: int = 0,
    ) -> list[str]:
        """
        Recursively split text using separator hierarchy with TRUE overlap.

        This is the core recursive splitting algorithm:
        1. Try first separator in list
        2. If chunks still too big, recurse with next separator
        3. Apply overlap ONLY at the top level (depth=0) to prevent
           cascading duplication from nested overlap application

        Args:
            text: Text to split
            separators: List of separators to try (in order)
            _depth: Recursion depth (internal, do not set)

        Returns:
            List of text chunks with overlap applied (at top level only)
        """
        if not text.strip():
            return []

        # Base case: text fits in chunk_size
        if self.count_tokens(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        final_chunks = []
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else [""]

        # Split by current separator (character-level is last resort)
        splits = text.split(separator) if separator else list(text)

        # A heading marker such as "\n\nSection" belongs to the START of the piece it
        # introduces. Any other separator (whitespace, ". ", ", ") ends the piece before it.
        # The separator is removed as an exact prefix or suffix, never as a character set.
        marker_leads = any(ch.isalnum() for ch in separator)
        lead = separator[: len(separator) - len(separator.lstrip())] if marker_leads else ""
        # Only the whitespace of a sentence or clause separator is trimmed: the ". " that ends a
        # sentence keeps its period.
        trail = "" if marker_leads else separator[len(separator.rstrip()) :]

        def clean(chunk: str) -> str:
            return chunk.removeprefix(lead).removesuffix(trail)

        # Process splits and merge small ones
        current_chunk = ""
        for index, split in enumerate(splits):
            # Add separator back (except for character splits)
            if not separator:
                piece = split
            elif marker_leads:
                piece = separator + split if index > 0 else split
            else:
                # The separator sits between two splits, so the last split has none after it.
                piece = split + separator if index < len(splits) - 1 else split

            test_chunk = current_chunk + piece
            test_tokens = self.count_tokens(test_chunk)

            if test_tokens <= self.chunk_size:
                # Piece fits, add to current chunk
                current_chunk = test_chunk
            else:
                # Piece doesn't fit
                if current_chunk:
                    # Save current chunk
                    if self.count_tokens(current_chunk) > self.chunk_size:
                        # Current chunk still too big, recurse with finer separator
                        sub_chunks = self._split_text_recursive(
                            clean(current_chunk),
                            remaining_separators,
                            _depth + 1,
                        )
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(clean(current_chunk))

                # Start new chunk with current piece
                if self.count_tokens(piece) > self.chunk_size:
                    # Single piece too big, recurse
                    sub_chunks = self._split_text_recursive(
                        clean(piece),
                        remaining_separators,
                        _depth + 1,
                    )
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        # Don't forget last chunk
        if current_chunk.strip():
            if self.count_tokens(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text_recursive(
                    clean(current_chunk),
                    remaining_separators,
                    _depth + 1,
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(clean(current_chunk))

        # Apply overlap ONLY at top level to prevent cascading duplication.
        # Recursive calls return clean splits; overlap is added once at the end.
        if _depth == 0 and self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._apply_overlap(final_chunks)

        return final_chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply TRUE overlap between chunks.

        This creates actual token overlap by prepending content from
        the previous chunk to the current chunk.

        Args:
            chunks: List of text chunks

        Returns:
            Chunks with overlap applied
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]  # First chunk has no previous overlap

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap content from end of previous chunk
            overlap_text = self._get_overlap_text(prev_chunk)

            if overlap_text:
                # Prepend overlap to current chunk
                overlapped_chunk = overlap_text + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)

        return overlapped_chunks

    def _get_overlap_text(self, text: str) -> str:
        """
        Extract overlap text from end of a chunk.

        Extracts approximately chunk_overlap tokens from the end,
        trying to break at sentence or word boundaries.

        Args:
            text: Text to extract overlap from

        Returns:
            Overlap text (approximately chunk_overlap tokens)
        """
        if not text or self.chunk_overlap <= 0:
            return ""

        # Tokenize the text
        encoder = get_encoding()
        tokens = encoder.encode(text)

        if len(tokens) <= self.chunk_overlap:
            return (
                ""  # Chunk too small for meaningful overlap, skip to prevent cascading duplication
            )

        # Get last chunk_overlap tokens
        overlap_tokens = tokens[-self.chunk_overlap :]
        overlap_text = encoder.decode(overlap_tokens)

        # Try to break at sentence boundary for cleaner overlap
        sentence_match = re.search(r"[.!?]\s+", overlap_text)
        if sentence_match:
            # Start from after the sentence boundary
            overlap_text = overlap_text[sentence_match.end() :]

        return overlap_text.strip() + " " if overlap_text.strip() else ""

    def _compute_chunk_char_offsets(
        self,
        chunk_texts: list[str],
        section_start_char: int,
    ) -> list[tuple[int, int]]:
        """Compute (char_start, char_end) for each chunk in source content coordinates.

        ``_split_text_recursive`` applies TRUE overlap by prepending the
        previous chunk's tail to each subsequent chunk. The naive
        bookkeeping pattern ``char_offset += len(chunk_text)`` therefore
        double-counts those overlap bytes and drifts linearly: a 43-page
        document chunked at 512 tokens / 100-overlap accumulated ~8 KB
        of drift across 40 chunks, pushing later chunks' offsets past
        the document's end and silently mapping them all to page 1.

        For each chunk we subtract the length of the overlap that was
        prepended from the previous chunk's tail, so ``char_end`` of
        chunk N equals ``char_start`` of chunk N+1, exactly the
        invariant ``get_page_range_from_char_offsets`` relies on.
        """
        offsets: list[tuple[int, int]] = []
        cursor = section_start_char
        for i, raw in enumerate(chunk_texts):
            stripped = raw.strip()
            if not stripped:
                offsets.append((cursor, cursor))
                continue

            # ``_apply_overlap`` prepends ``_get_overlap_text(chunks[i-1])``
            # to ``chunks[i]``. That prefix duplicates the previous chunk's
            # tail and must not be counted as new content.
            overlap_len = 0
            if i > 0 and chunk_texts[i - 1]:
                overlap_text = self._get_overlap_text(chunk_texts[i - 1])
                overlap_len = len(overlap_text)
                # Defensive: never let the overlap exceed the chunk itself.
                overlap_len = min(overlap_len, len(stripped))

            new_content_len = len(stripped) - overlap_len
            char_start = cursor
            char_end = cursor + new_content_len
            offsets.append((char_start, char_end))
            cursor = char_end

        return offsets

    def build_contextual_header(
        self,
        metadata: dict[str, Any],
        section_type: LegalSectionType | LegalSection,
        document_type: DocumentType = DocumentType.UNKNOWN,
        max_title_length: int = 100,
    ) -> str:
        """
        Build contextual header for a chunk.

        Provides 35-40% BM25 recall improvement by adding document metadata.

        Format varies by document type:
        - Judgments: Case: {title} [{citation}] ({year}) / Court: {court}
        - Contracts: Agreement: {title} / Parties: {parties} / Date: {date}
        - Statutes: Act: {title} / Section: {section}
        - Arbitration: Arbitration: {title} / Institution: {institution}
        - etc.

        Args:
            metadata: Document metadata dict
            section_type: Detected section type (LegalSectionType or legacy LegalSection)
            document_type: Document type classification
            max_title_length: Maximum title length before truncation

        Returns:
            Contextual header string
        """
        title = metadata.get("title", "Untitled")
        if len(title) > max_title_length:
            title = title[:max_title_length] + "..."

        # Get section type value (handle both enum types)
        section_value = section_type.value if hasattr(section_type, "value") else str(section_type)

        # Build document-type-specific header
        if document_type in (DocumentType.JUDGMENT, DocumentType.UNKNOWN):
            # Legacy judgment format
            citation = metadata.get("citation", "")
            year = metadata.get("year", 0)
            court = metadata.get("court", "")

            header_parts = [f"Case: {title}"]
            if citation:
                header_parts.append(f"[{citation}]")
            if year and year > 0:
                header_parts.append(f"({year})")
            header = " ".join(header_parts)

            if court:
                header += f"\nCourt: {court}"

        elif document_type in (DocumentType.CONTRACT, DocumentType.MNA):
            # Contract/M&A format
            parties = metadata.get("parties", "")
            date = metadata.get("date", metadata.get("execution_date", ""))

            header = f"Agreement: {title}"
            if parties:
                header += f"\nParties: {parties}"
            if date:
                header += f"\nDate: {date}"

        elif document_type == DocumentType.STATUTE:
            # Statute format
            act_number = metadata.get("act_number", "")
            year = metadata.get("year", 0)

            header = f"Act: {title}"
            if act_number:
                header += f" [{act_number}]"
            if year and year > 0:
                header += f" ({year})"

        elif document_type == DocumentType.ARBITRATION:
            # Arbitration format
            institution = metadata.get("institution", "")
            case_no = metadata.get("case_number", metadata.get("reference", ""))

            header = f"Arbitration: {title}"
            if institution:
                header += f"\nInstitution: {institution}"
            if case_no:
                header += f"\nCase No: {case_no}"

        elif document_type == DocumentType.CORPORATE:
            # Corporate format
            company = metadata.get("company", metadata.get("organization", ""))
            date = metadata.get("date", "")

            header = f"Corporate Document: {title}"
            if company:
                header += f"\nCompany: {company}"
            if date:
                header += f"\nDate: {date}"

        elif document_type == DocumentType.IP:
            # IP format
            app_number = metadata.get("application_number", metadata.get("patent_number", ""))
            filing_date = metadata.get("filing_date", "")

            header = f"IP Document: {title}"
            if app_number:
                header += f"\nApplication No: {app_number}"
            if filing_date:
                header += f"\nFiled: {filing_date}"

        elif document_type == DocumentType.REAL_ESTATE:
            # Real estate format
            property_desc = metadata.get("property", metadata.get("property_description", ""))

            header = f"Property Document: {title}"
            if property_desc:
                header += f"\nProperty: {property_desc[:80]}..."

        elif document_type == DocumentType.ESTATE:
            # Estate format
            testator = metadata.get("testator", metadata.get("settlor", ""))

            header = f"Estate Document: {title}"
            if testator:
                header += f"\nTestator/Settlor: {testator}"

        else:
            # Generic format for PLEADING, CIRCULAR, REGULATORY, UNKNOWN
            header = f"Document: {title}"

        # Only include section type when actually detected (not noise)
        if section_value.upper() != "UNKNOWN":
            section_header = f"Section: {section_value.upper()}"
            return f"{header}\n{section_header}\n\n"
        return f"{header}\n\n"

    def get_page_range_from_char_offsets(
        self,
        pages: list[dict[str, Any]],
        char_start: int,
        char_end: int,
    ) -> tuple[int, int]:
        """Calculate page range from character offsets for PDF highlighting."""
        if not pages:
            return (1, 1)

        page_start, page_end = None, None
        running_offset = 0

        for i, page in enumerate(pages):
            page_num = page.get("page_number", i + 1)
            page_text = page.get("text", "")
            page_end_offset = running_offset + len(page_text)

            if char_end > running_offset and char_start < page_end_offset:
                if page_start is None:
                    page_start = page_num
                page_end = page_num

            running_offset = page_end_offset + 2  # +2 for "\n\n" between pages

        return (page_start or 1, page_end or 1)

    async def chunk(
        self,
        text: str,
        document_id: str,
        sections: list[Any] | None = None,
        boxes: list[dict[str, Any]] | None = None,
        pages: list[dict[str, Any]] | None = None,
        document_metadata: dict[str, Any] | None = None,
        document_type: DocumentType | None = None,
        **kwargs,
    ) -> ChunkerResult:
        """
        Split document into chunks with TRUE overlap and universal legal section awareness.

        Algorithm:
        1. Classify document type (if not provided)
        2. Detect section boundaries using type-specific patterns
        3. For each section, use recursive splitting with TRUE overlap
        4. Add contextual headers for BM25 boost (format varies by doc type)
        5. Tag each chunk with section metadata and priority
        6. Link chunks with prev/next references

        Args:
            text: Full document text
            document_id: Document identifier
            sections: Ignored - we detect our own sections
            boxes: Ignored (recursive chunker doesn't need box-aware preprocessing)
            pages: Page data for page range calculation
            document_metadata: Document metadata for contextual headers
            document_type: If known, skip classification. Otherwise auto-detect.
            **kwargs: Additional options

        Returns:
            ChunkerResult with LegalChunk objects (with true overlap)
        """
        start_time = time.time()

        if not text or not text.strip():
            return ChunkerResult(
                document_id=document_id,
                chunks=[],
                total_chunks=0,
                total_tokens=0,
                processing_time_ms=0,
            )

        # Step 1: Detect section boundaries (using universal or legacy detector)
        detected_doc_type = document_type or DocumentType.UNKNOWN

        if self.use_universal_detector and self._universal_detector:
            # Use universal detector (supports all document types)
            universal_boundaries = self._universal_detector.detect(text, document_type)
            detected_doc_type = (
                universal_boundaries[0].document_type
                if universal_boundaries
                else DocumentType.UNKNOWN
            )

            logger.info(
                "universal_section_detection_completed",
                document_id=document_id,
                document_type=detected_doc_type.value,
                section_count=len(universal_boundaries),
                sections=[s.section_type.value for s in universal_boundaries],
            )

            # Step 2: Chunk each section using recursive splitting with overlap.
            # Per-section work is CPU-bound but GIL-releasing (tiktoken Rust
            # core), so fan out across threads. Output is byte-identical to
            # the sequential version because (a) we preserve boundary order
            # in the result and (b) chunk_index is recomputed via prefix-sum
            # after all sections return.
            metadata = document_metadata or {}
            pages_data = pages or []
            all_chunks = await self._chunk_sections_parallel(
                boundaries=universal_boundaries,
                per_section_fn=lambda boundary: self._chunk_universal_section(
                    section=boundary,
                    document_id=document_id,
                    start_index=0,  # placeholder; rewritten by prefix-sum below
                    metadata=metadata,
                    pages=pages_data,
                ),
                document_id=document_id,
                detector="universal",
            )

        else:
            # Use the legacy judgment-only detector
            if self._legacy_detector is None:
                self._legacy_detector = get_legal_section_detector()
            legacy_boundaries = self._legacy_detector.detect(text)

            logger.info(
                "legacy_section_detection_completed",
                document_id=document_id,
                section_count=len(legacy_boundaries),
                sections=[s.section.value for s in legacy_boundaries],
            )

            # Step 2: Chunk each section using recursive splitting with overlap
            # (see note above on parallelization strategy).
            metadata = document_metadata or {}
            pages_data = pages or []
            all_chunks = await self._chunk_sections_parallel(
                boundaries=legacy_boundaries,
                per_section_fn=lambda boundary: self._chunk_legacy_section(
                    section=boundary,
                    document_id=document_id,
                    start_index=0,  # placeholder; rewritten by prefix-sum below
                    metadata=metadata,
                    pages=pages_data,
                ),
                document_id=document_id,
                detector="legacy",
            )

        # Step 3: Update total_chunks for all chunks
        total_count = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total_count

        # Step 4: Link chunks with neighbors
        all_chunks = self.link_chunks(all_chunks)

        # Calculate stats
        token_counts = [c.token_count for c in all_chunks]
        total_tokens = sum(token_counts)
        processing_time = (time.time() - start_time) * 1000

        avg_chunk_size = total_tokens / total_count if total_count > 0 else 0
        min_size = min(token_counts) if token_counts else 0
        max_size = max(token_counts) if token_counts else 0

        logger.info(
            "chunking_completed",
            strategy="recursive_legal_universal"
            if self.use_universal_detector
            else "recursive_legal_legacy",
            document_id=document_id,
            document_type=detected_doc_type.value,
            chunk_count=total_count,
            total_tokens=total_tokens,
            avg_chunk_size=round(avg_chunk_size, 2),
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            duration_ms=round(processing_time, 2),
            overlap_tokens=self.chunk_overlap,
            has_true_overlap=True,
        )

        return ChunkerResult(
            document_id=document_id,
            chunks=all_chunks,
            total_chunks=total_count,
            total_tokens=total_tokens,
            processing_time_ms=processing_time,
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
        )

    async def _chunk_sections_parallel(
        self,
        *,
        boundaries: list[Any],
        per_section_fn: Callable[[Any], list[LegalChunk]],
        document_id: str,
        detector: str,
    ) -> list[LegalChunk]:
        """Run per-section chunking across threads, preserve boundary order.

        Each boundary is processed by `per_section_fn`, which must be a
        closure over the boundary-agnostic arguments (document_id,
        metadata, pages). The function must call one of the
        `_chunk_{universal,legacy}_section` methods with
        ``start_index=0``; this helper rewrites the final chunk_index
        values via prefix-sum after all sections return so that the
        output is byte-identical to a sequential run.

        Concurrency
        -----------
        Up to ``self.max_concurrency`` threads at once. Construct the chunker
        with ``max_concurrency=1`` to force sequential behaviour (for example,
        when debugging a correctness regression).

        Warm-up
        -------
        The tiktoken encoder is lazily initialised inside `count_tokens`.
        We force initialisation on the main event-loop thread before
        fanning out so all worker threads see an already-cached encoder
        and never race on the global token-counter singleton.
        """
        if not boundaries:
            return []

        # Warm up the tiktoken encoder on the main thread so worker
        # threads never race on the global counter's lazy init.
        self.count_tokens("warmup")

        t0 = time.time()
        max_workers = min(self.max_concurrency, len(boundaries), os.cpu_count() or 2)
        semaphore = asyncio.Semaphore(max_workers)

        async def _run_one(boundary: Any) -> list[LegalChunk]:
            async with semaphore:
                return await asyncio.to_thread(per_section_fn, boundary)

        # gather preserves the order of the input boundaries in the
        # returned list, so the result is equivalent to sequential
        # iteration for the purposes of prefix-sum.
        per_section_results: list[list[LegalChunk]] = await asyncio.gather(
            *(_run_one(b) for b in boundaries)
        )

        # Prefix-sum fix-up: each section's local chunk_index starts at
        # 0 in the parallel path. Walk in boundary order and rewrite.
        all_chunks: list[LegalChunk] = []
        running_index = 0
        for section_chunks in per_section_results:
            for local_idx, chunk in enumerate(section_chunks):
                chunk.chunk_index = running_index + local_idx
            running_index += len(section_chunks)
            all_chunks.extend(section_chunks)

        logger.info(
            "chunk_sections_parallel_complete",
            component="chunker",
            document_id=document_id,
            detector=detector,
            section_count=len(boundaries),
            chunk_count=len(all_chunks),
            max_workers=max_workers,
            duration_ms=round((time.time() - t0) * 1000, 2),
        )

        return all_chunks

    def _chunk_universal_section(
        self,
        section: UniversalSectionBoundary,
        document_id: str,
        start_index: int,
        metadata: dict[str, Any] | None = None,
        pages: list[dict[str, Any]] | None = None,
    ) -> list[LegalChunk]:
        """
        Chunk a universal section using recursive splitting with TRUE overlap.

        Args:
            section: Universal section boundary to chunk
            document_id: Document identifier
            start_index: Starting chunk index
            metadata: Document metadata for contextual headers
            pages: Page data for page range calculation

        Returns:
            List of LegalChunk objects with true overlap
        """
        content = section.content.strip()
        if not content:
            return []

        # Build contextual header for BM25 boost (format varies by doc type)
        contextual_header = ""
        if metadata:
            contextual_header = self.build_contextual_header(
                metadata, section.section_type, section.document_type
            )

        # Get section priority for retrieval boosting
        section_priority = get_section_priority(section.section_type)

        # Map universal section type to legacy LegalSection (for backward compatibility)
        legacy_section = self._map_to_legacy_section(section.section_type)

        # Use recursive splitting with TRUE overlap
        chunk_texts = self._split_text_recursive(content, self.separators)

        # Compute per-chunk source-text offsets up-front. The overlap-aware
        # helper accounts for the prefix that ``_apply_overlap`` adds, so
        # ``char_end`` of one chunk equals ``char_start`` of the next and
        # the page-range mapper sees offsets that actually fall inside the
        # document (rather than drifting past its end after ~10 chunks).
        chunk_offsets = self._compute_chunk_char_offsets(chunk_texts, section.start_char)

        chunks: list[LegalChunk] = []

        for i, raw_chunk_text in enumerate(chunk_texts):
            chunk_text = raw_chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self.count_tokens(chunk_text)
            char_start, char_end = chunk_offsets[i]

            # Skip chunks below minimum size (unless it's the only chunk).
            # The cursor has already advanced inside ``_compute_chunk_char_offsets``,
            # so we just drop this entry from the output.
            if token_count < self.min_chunk_size and len(chunk_texts) > 1:
                continue

            # Prepend contextual header
            chunk_content_with_header = (
                contextual_header + chunk_text if contextual_header else chunk_text
            )

            # Calculate page range if pages available
            page_start, page_end = 1, 1
            if pages:
                page_start, page_end = self.get_page_range_from_char_offsets(
                    pages, char_start, char_end
                )

            chunk = LegalChunk(
                id=str(uuid4()),
                content=chunk_content_with_header,
                chunk_index=start_index + len(chunks),
                start_char=char_start,
                end_char=char_end,
                document_id=document_id,
                section_title=section.title,
                token_count=self.count_tokens(chunk_content_with_header),
                # Legacy field
                section=legacy_section,
                # Universal fields
                section_type_universal=section.section_type,
                document_type=section.document_type,
                section_confidence=section.confidence,
                section_type=section.section_type.value,
                section_priority=section_priority,
                contextual_header=contextual_header,
                page_start=page_start,
                page_end=page_end,
                has_true_overlap=True,
                hierarchy_level=section.hierarchy_level,
            )
            chunks.append(chunk)

        logger.debug(
            "recursive_split_complete",
            document_type=section.document_type.value,
            section=section.section_type.value,
            input_tokens=self.count_tokens(content),
            output_chunks=len(chunks),
            overlap_applied=self.chunk_overlap > 0,
        )

        return chunks

    def _chunk_legacy_section(
        self,
        section: SectionBoundary,
        document_id: str,
        start_index: int,
        metadata: dict[str, Any] | None = None,
        pages: list[dict[str, Any]] | None = None,
    ) -> list[LegalChunk]:
        """
        Chunk a legacy section using recursive splitting with TRUE overlap.

        For the legacy judgment-only detector.

        Args:
            section: Legacy section boundary to chunk
            document_id: Document identifier
            start_index: Starting chunk index
            metadata: Document metadata for contextual headers
            pages: Page data for page range calculation

        Returns:
            List of LegalChunk objects with true overlap
        """
        content = section.content.strip()
        if not content:
            return []

        # Build contextual header for BM25 boost
        contextual_header = ""
        if metadata:
            contextual_header = self.build_contextual_header(
                metadata, section.section, DocumentType.JUDGMENT
            )

        # Map legacy section to universal type
        universal_section_type = self._map_to_universal_section(section.section)

        # Get section priority for retrieval boosting (use legacy priority map)
        section_priority = get_legacy_section_priority(section.section)

        # Use recursive splitting with TRUE overlap
        chunk_texts = self._split_text_recursive(content, self.separators)

        # Per-chunk offsets that account for prepended overlap; see
        # ``_compute_chunk_char_offsets`` for the rationale.
        chunk_offsets = self._compute_chunk_char_offsets(chunk_texts, section.start_char)

        chunks: list[LegalChunk] = []

        for i, raw_chunk_text in enumerate(chunk_texts):
            chunk_text = raw_chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self.count_tokens(chunk_text)
            char_start, char_end = chunk_offsets[i]

            # Skip chunks below minimum size (unless it's the only chunk)
            if token_count < self.min_chunk_size and len(chunk_texts) > 1:
                continue

            # Prepend contextual header
            chunk_content_with_header = (
                contextual_header + chunk_text if contextual_header else chunk_text
            )

            # Calculate page range if pages available
            page_start, page_end = 1, 1
            if pages:
                page_start, page_end = self.get_page_range_from_char_offsets(
                    pages, char_start, char_end
                )

            chunk = LegalChunk(
                id=str(uuid4()),
                content=chunk_content_with_header,
                chunk_index=start_index + len(chunks),
                start_char=char_start,
                end_char=char_end,
                document_id=document_id,
                section_title=section.title,
                token_count=self.count_tokens(chunk_content_with_header),
                section=section.section,
                section_type_universal=universal_section_type,
                document_type=DocumentType.JUDGMENT,
                section_confidence=section.confidence,
                section_type=section.section.value,
                section_priority=section_priority,
                contextual_header=contextual_header,
                page_start=page_start,
                page_end=page_end,
                has_true_overlap=True,
                hierarchy_level=0,
            )
            chunks.append(chunk)

        logger.debug(
            "legacy_recursive_split_complete",
            section=section.section.value,
            input_tokens=self.count_tokens(content),
            output_chunks=len(chunks),
            overlap_applied=self.chunk_overlap > 0,
        )

        return chunks

    def _map_to_legacy_section(self, section_type: LegalSectionType) -> LegalSection:
        """Map universal section type to legacy LegalSection for backward compatibility."""
        mapping = {
            LegalSectionType.HEADNOTE: LegalSection.HEADNOTE,
            LegalSectionType.JUDGMENT: LegalSection.JUDGMENT,
            LegalSectionType.ORDER: LegalSection.ORDER,
            LegalSectionType.OPERATIVE_PART: LegalSection.OPERATIVE_PART,
        }
        return mapping.get(section_type, LegalSection.UNKNOWN)

    def _map_to_universal_section(self, legacy_section: LegalSection) -> LegalSectionType:
        """Map legacy LegalSection to universal section type."""
        mapping = {
            LegalSection.HEADNOTE: LegalSectionType.HEADNOTE,
            LegalSection.JUDGMENT: LegalSectionType.JUDGMENT,
            LegalSection.ORDER: LegalSectionType.ORDER,
            LegalSection.OPERATIVE_PART: LegalSectionType.OPERATIVE_PART,
            LegalSection.UNKNOWN: LegalSectionType.UNKNOWN,
        }
        return mapping.get(legacy_section, LegalSectionType.UNKNOWN)

    # ----------------------------------------------------------------
    # Backward compatibility methods
    # ----------------------------------------------------------------

    def detect_sections(self, text: str) -> list[SectionBoundary]:
        """
        Detect section boundaries in legal document using legacy detector.

        For universal detection, use detect_sections_universal() instead.
        """
        if self._legacy_detector is None:
            self._legacy_detector = get_legal_section_detector()
        return self._legacy_detector.detect(text)

    def detect_sections_universal(
        self,
        text: str,
        document_type: DocumentType | None = None,
    ) -> list[UniversalSectionBoundary]:
        """
        Detect section boundaries using universal detector.

        Supports all document types (contracts, statutes, arbitration, etc.).

        Args:
            text: Document text
            document_type: If known, skip classification

        Returns:
            List of UniversalSectionBoundary objects
        """
        if self._universal_detector is None:
            self._universal_detector = get_universal_section_detector()
        return self._universal_detector.detect(text, document_type)

    async def chunk_document(
        self,
        text: str,
        document_id: str = "",
        metadata: dict[str, Any] | None = None,
        document_type: DocumentType | None = None,
    ) -> list[LegalChunk]:
        """Chunk document while preserving section boundaries."""
        if not text or not text.strip():
            return []

        result = await self.chunk(
            text=text,
            document_id=document_id,
            document_metadata=metadata,
            document_type=document_type,
        )
        return [chunk for chunk in result.chunks if isinstance(chunk, LegalChunk)]


# Singleton instance
_chunker: RecursiveLegalChunker | None = None


def get_recursive_legal_chunker() -> RecursiveLegalChunker:
    """Get recursive legal chunker singleton."""
    global _chunker
    if _chunker is None:
        _chunker = RecursiveLegalChunker()
    return _chunker
