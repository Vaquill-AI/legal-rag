"""
Centralized OCR artifact sanitization for all document formats.

Removes OCR metadata and parser artifacts that pollute embeddings:
- Image placeholders from PDF parsers (PyMuPDF4LLM markdown conversion)
- OCR confidence markers and metadata
- Unicode artifacts (replacement chars, zero-width spaces)
- Page markers and separators
- Format-specific markers (Excel sheets, PowerPoint slides)

Architecture:
- Stores ONLY cleaned text in Qdrant (no dual storage)
- PDF highlighting uses char_start/char_end + page_number
- PyMuPDF search_for() finds text positions in original PDF

Usage:
    from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

    # Basic cleaning
    cleaned = clean_ocr_artifacts(raw_text)

    # With statistics
    cleaned, stats = clean_ocr_artifacts(raw_text, return_stats=True)
    print(f"Removed {stats.artifacts_removed} artifacts")

References:
- PyMuPDF search_for(): https://pymupdf.readthedocs.io/en/latest/page.html
- Hypothesis fuzzy anchoring: https://github.com/hypothesis/pdf.js-hypothes.is
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

# ============================================
# Compiled Regex Patterns (for performance)
# ============================================

# Image placeholders from PyMuPDF4LLM markdown conversion
# Example: "picture [527 x 52] intentionally omitted" (with or without arrows/×)
PICTURE_PLACEHOLDER: Final = re.compile(
    r"[⇒←→]?\s*picture\s*\[\s*\d+\s*[×xX]\s*\d+\s*\].*?(?:intentionally\s+)?omitted\s*[⇐←→]?",
    re.IGNORECASE,
)

# OCR confidence markers from various OCR engines
# Matches: "[OCR confidence: 85%]", "[ocr confidence: 90]"
OCR_CONFIDENCE: Final = re.compile(
    r"\[(?:OCR|ocr)\s+confidence:\s*\d+%?\]",
    re.IGNORECASE,
)

# Page markers and separators
# Matches: "--- Page 1 ---", "=== Page 2 ===", "*** Page 3 ***"
PAGE_MARKERS: Final = re.compile(
    r"^(?:---|===|\*\*\*)\s*Page\s+\d+\s*(?:---|===|\*\*\*)$",
    re.MULTILINE | re.IGNORECASE,
)

# Running page-number footers/headers that PDF/DOCX extraction leaves inline
# in the body, e.g. "2 | Page", "2 | P a g e" (letter-spaced tracked-out
# footers), "Page 2", "Page | 2", "Page 2 of 10", "- 2 -".
#
# Matched ONLY as a WHOLE line (anchored, MULTILINE) so body prose that
# mentions "page 2 of the report" is never touched. The page word tolerates the
# per-letter spacing that tracked-out Word/PDF footers produce ("P a g e").
_PAGE_WORD: Final = r"p\s*a\s*g\s*e"
RUNNING_PAGE_FOOTER: Final = re.compile(
    r"^[ \t]*(?:"
    # "2 | Page", "2 | P a g e", "2 | Page of 10"
    rf"\d+[ \t]*\|[ \t]*{_PAGE_WORD}(?:[ \t]+of[ \t]+\d+)?"
    # "Page 2", "Page | 2", "Page 2 of 10"
    rf"|{_PAGE_WORD}[ \t]*\|?[ \t]*\d+(?:[ \t]+of[ \t]+\d+)?"
    # "- 2 -" bare centered page number
    r"|-[ \t]*\d+[ \t]*-"
    r")[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Image OCR markers from Office extractor
# Matches: "[Image 1 OCR Text]", "[Image 5 OCR Text - SANITIZED]"
# Note: Removes marker but keeps the OCR text that follows
IMAGE_OCR_MARKERS: Final = re.compile(
    r"\[Image\s+\d+\s+OCR\s+Text(?:\s*-\s*SANITIZED)?\]\s*",
    re.IGNORECASE,
)

# OCR page separator
# Matches: "<<<" on its own line (common OCR page separator)
OCR_PAGE_SEP: Final = re.compile(r"^<<<\s*$", re.MULTILINE)

# Uncertain text markers from OCR engines (example: uncertain tags)
UNCERTAIN_MARKERS: Final = re.compile(
    r"<uncertain>(.*?)</uncertain>",
    re.IGNORECASE | re.DOTALL,
)

# ============================================
# Unicode Artifacts to Remove
# ============================================

UNICODE_ARTIFACTS: Final[frozenset[str]] = frozenset(
    {
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u200e",  # Left-to-right mark
        "\u200f",  # Right-to-left mark
        "\ufeff",  # Byte order mark (BOM)
        chr(0xFFFD),  # Unicode replacement character �
    }
)


# ============================================
# Cleaning Statistics
# ============================================


@dataclass
class CleaningStats:
    """Statistics from text cleaning operation."""

    artifacts_removed: int
    original_length: int
    cleaned_length: int
    reduction_pct: float
    blocks_deduplicated: int = 0

    def __str__(self) -> str:
        dedup_msg = (
            f", {self.blocks_deduplicated} blocks deduplicated" if self.blocks_deduplicated else ""
        )
        return (
            f"Cleaned {self.artifacts_removed} artifacts "
            f"({self.reduction_pct:.1f}% reduction, "
            f"{self.original_length} → {self.cleaned_length} chars{dedup_msg})"
        )


# ============================================
# Main Cleaning Functions
# ============================================


def clean_ocr_artifacts(
    text: str,
    preserve_structure: bool = True,
    return_stats: bool = False,
) -> str | tuple[str, CleaningStats]:
    """
    Clean OCR artifacts and metadata from extracted text.

    This function removes parser artifacts that pollute embeddings while
    preserving the actual document content. PDF highlighting still works
    because we use char_start/char_end + page_number to search the original
    PDF with PyMuPDF's search_for().

    Args:
        text: Raw extracted text with OCR artifacts
        preserve_structure: Keep paragraph breaks (True) or collapse all whitespace (False)
        return_stats: Return cleaning statistics along with cleaned text

    Returns:
        Cleaned text string, or tuple of (cleaned_text, stats) if return_stats=True

    Example:
        >>> raw = "Text ⇒ picture [527 x 52] intentionally omitted ⇐ more"
        >>> clean_ocr_artifacts(raw)
        'Text more'

        >>> cleaned, stats = clean_ocr_artifacts(raw, return_stats=True)
        >>> print(stats)
        Cleaned 1 artifacts (83.3% reduction, 54 → 9 chars)

    Only the arrow characters ``⇒``, ``⇐``, ``←`` and ``→`` around a placeholder are
    stripped with it. The ASCII forms ``=>`` and ``<=`` are ordinary text and stay.
    """
    if not text:
        if return_stats:
            return "", CleaningStats(0, 0, 0, 0.0)
        return ""

    original_length = len(text)
    artifacts_count = 0

    # Step 1: Remove image placeholders (most common artifact)
    text, count = PICTURE_PLACEHOLDER.subn("", text)
    artifacts_count += count

    # Step 2: Remove OCR confidence markers
    text, count = OCR_CONFIDENCE.subn("", text)
    artifacts_count += count

    # Step 3: Remove page markers
    text, count = PAGE_MARKERS.subn("", text)
    artifacts_count += count

    # Step 3b: Remove running page-number footers/headers ("2 | Page", "Page 2
    # of 10", etc.) that extraction leaves inline between paragraphs.
    text, count = RUNNING_PAGE_FOOTER.subn("", text)
    artifacts_count += count

    # Step 4: Remove OCR page separators
    text, count = OCR_PAGE_SEP.subn("", text)
    artifacts_count += count

    # Step 5: Remove image OCR markers (keep the OCR text itself)
    text, count = IMAGE_OCR_MARKERS.subn("", text)
    artifacts_count += count

    # Step 6: Remove uncertain text markers (keep the text, remove tags)
    # Replace <uncertain>text</uncertain> with just text
    text = UNCERTAIN_MARKERS.sub(r"\1", text)

    # Step 7: Remove Unicode artifacts
    for artifact in UNICODE_ARTIFACTS:
        if artifact in text:
            text = text.replace(artifact, "")
            artifacts_count += 1

    # Step 8: Normalize Unicode (NFC - canonical composition)
    # This ensures consistent representation of accented characters
    text = unicodedata.normalize("NFC", text)

    # Step 9: Clean whitespace
    if preserve_structure:
        # Collapse horizontal whitespace but keep paragraph breaks
        text = re.sub(r"[^\S\n]+", " ", text)  # Multiple spaces → single space
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    else:
        # Aggressive whitespace collapse (for non-structured text)
        text = re.sub(r"\s+", " ", text)

    # Step 10: Strip leading/trailing whitespace
    text = text.strip()

    # Step 11: Deduplicate repeated text blocks from OCR double-extraction
    # PyMuPDF can extract the same text multiple times from overlapping PDF
    # text layers (common in scanned and OCR-processed documents).
    # Must run AFTER whitespace normalization so identical blocks match exactly.
    text, blocks_deduped = deduplicate_repeated_text(text)

    cleaned_length = len(text)

    if return_stats:
        reduction_pct = (
            ((original_length - cleaned_length) / original_length * 100)
            if original_length > 0
            else 0.0
        )
        stats = CleaningStats(
            artifacts_removed=artifacts_count,
            original_length=original_length,
            cleaned_length=cleaned_length,
            reduction_pct=reduction_pct,
            blocks_deduplicated=blocks_deduped,
        )
        return text, stats

    return text


# ============================================
# OCR Duplicate Text Detection & Removal
# ============================================

# Minimum character length for a block to be considered for dedup.
# Blocks shorter than this are left alone (short headers, formatting can repeat legitimately).
_DEDUP_MIN_BLOCK_LEN: Final[int] = 100

# Minimum consecutive repetitions before collapsing.
# 3 = extremely conservative - no legitimate document repeats 100+ chars 3x in a row.
_DEDUP_MIN_REPEATS: Final[int] = 3

# Maximum paragraph length for inline dedup (Phase 3).
# OCR double-extraction produces repeated paragraphs of 100-5000 chars.
# Paragraphs above this limit are skipped to prevent O(S*N) blowup where
# S = sentence boundary count and N = paragraph length (e.g., flattened JSON
# with no \n\n breaks creates a single 500KB+ "paragraph" → 600s timeout).
_INLINE_DEDUP_MAX_PARA_LEN: Final[int] = 50_000


def deduplicate_repeated_text(
    text: str, min_block_len: int = _DEDUP_MIN_BLOCK_LEN
) -> tuple[str, int]:
    """Remove consecutive duplicate text caused by PyMuPDF OCR double-extraction.

    When PyMuPDF extracts text from PDFs with overlapping text layers (common in
    scanned documents), the same text block can appear 2-15+ times.
    This wastes embedding dimensions and context budget on redundant content.

    Three-phase approach (each phase is O(n)):
      Phase 1: Paragraph dedup - consecutive identical paragraphs (\\n\\n separated)
      Phase 2: Line dedup - consecutive identical lines (\\n separated)
      Phase 3: Inline block dedup - repeated blocks within a continuous text stream
               (handles the case where PyMuPDF concatenates overlapping text objects
               with no separator between them)

    Safety constraints:
      - Only collapses blocks >= min_block_len characters (default 100)
      - Only collapses when 3+ consecutive repetitions are found
      - Exact string match only (no fuzzy matching)
      - Empty/whitespace-only segments are never collapsed

    Args:
        text: Cleaned text (should be called after artifact removal)
        min_block_len: Minimum characters for a block to be dedup-eligible

    Returns:
        Tuple of (deduplicated_text, count_of_removed_blocks)
    """
    if len(text) < min_block_len * _DEDUP_MIN_REPEATS:
        return text, 0

    total_removed = 0

    # Phase 1: Paragraph-level consecutive dedup
    text, removed = _dedup_consecutive_segments(text, sep="\n\n", min_len=min_block_len)
    total_removed += removed

    # Phase 2: Line-level consecutive dedup
    text, removed = _dedup_consecutive_segments(text, sep="\n", min_len=min_block_len)
    total_removed += removed

    # Phase 3: Inline block dedup (for text with no separators between repetitions)
    # Only run on paragraphs long enough to contain 3+ repetitions
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for para in paragraphs:
        if len(para) >= min_block_len * _DEDUP_MIN_REPEATS:
            cleaned, removed = _collapse_inline_repeats(para, min_block_len)
            total_removed += removed
            cleaned_paragraphs.append(cleaned)
        else:
            cleaned_paragraphs.append(para)
    text = "\n\n".join(cleaned_paragraphs)

    return text, total_removed


def _dedup_consecutive_segments(
    text: str, sep: str, min_len: int = _DEDUP_MIN_BLOCK_LEN
) -> tuple[str, int]:
    """Remove consecutive identical segments split by a separator.

    O(n) - single pass through the segment list.
    """
    segments = text.split(sep)
    if len(segments) < _DEDUP_MIN_REPEATS:
        return text, 0

    result: list[str] = []
    removed = 0
    prev_stripped: str | None = None
    consecutive_count = 1

    for segment in segments:
        stripped = segment.strip()
        if stripped and stripped == prev_stripped and len(stripped) >= min_len:
            consecutive_count += 1
            if consecutive_count > _DEDUP_MIN_REPEATS:
                # Already past threshold, keep dropping
                removed += 1
                continue
            elif consecutive_count == _DEDUP_MIN_REPEATS:
                # Just hit threshold - retroactively drop the 2nd copy we already kept
                # Find and remove the last occurrence of this segment from result
                for i in range(len(result) - 1, -1, -1):
                    if result[i].strip() == stripped:
                        result.pop(i)
                        removed += 1
                        break
                removed += 1  # Drop current one too
                continue
            else:
                # consecutive_count == 2, not yet at threshold - keep it for now
                result.append(segment)
                prev_stripped = stripped
                continue
        else:
            consecutive_count = 1

        result.append(segment)
        prev_stripped = stripped if stripped else prev_stripped

    return sep.join(result), removed


def _collapse_inline_repeats(
    text: str, min_block_len: int = _DEDUP_MIN_BLOCK_LEN
) -> tuple[str, int]:
    """Detect and collapse repeated blocks within a continuous text stream.

    Handles the case where PyMuPDF extracts overlapping text objects on the same
    page with no newlines between them (e.g., same paragraph appears 15x inline).

    Algorithm:
      1. Identify candidate start positions (beginning of text + sentence boundaries)
      2. At each position, try increasing block sizes starting from min_block_len
      3. If a block repeats 3+ times consecutively, collapse to 1 occurrence
      4. Move past the collapsed region and continue

    Efficient because:
      - Only checks at sentence-like boundaries (not every character position)
      - String comparison short-circuits on first differing character
      - Exits early once a repeating pattern is found at each position
    """
    if len(text) < min_block_len * _DEDUP_MIN_REPEATS:
        return text, 0

    # Guard: skip paragraphs too large for the O(S*N) scan.
    # Real OCR duplication lives in paragraphs of a few thousand chars.
    if len(text) > _INLINE_DEDUP_MAX_PARA_LEN:
        return text, 0

    total_removed = 0

    # Find candidate start positions: position 0 + after each ". " followed by uppercase
    # False sentence boundaries (e.g., "Rs. 15000") are harmless - they just add extra
    # positions that won't match any repeating block.
    starts = [0]
    for match in re.finditer(r"\. [A-Z]", text):
        starts.append(match.start() + 2)

    # Process from leftmost position to rightmost
    for start in starts:
        if start >= len(text):
            continue

        remaining = len(text) - start
        if remaining < min_block_len * _DEDUP_MIN_REPEATS:
            continue

        max_block = remaining // _DEDUP_MIN_REPEATS

        for block_len in range(min_block_len, max_block + 1):
            block = text[start : start + block_len]

            # Quick fail: check if next occurrence matches
            next_pos = start + block_len
            if text[next_pos : next_pos + block_len] != block:
                continue

            # Count forward consecutive repetitions
            forward_count = 2
            pos = next_pos + block_len
            while text[pos : pos + block_len] == block:
                forward_count += 1
                pos += block_len

            # Extend backward: the repetition might have started before `start`
            # (e.g., "HEADNOTE: [block][block][block]..." - sentence boundary points
            # to the 2nd block, but the 1st block is also identical)
            true_start = start
            backward_count = 0
            while true_start >= block_len and text[true_start - block_len : true_start] == block:
                true_start -= block_len
                backward_count += 1

            total_count = forward_count + backward_count

            if total_count >= _DEDUP_MIN_REPEATS:
                # Collapse: keep one occurrence at true_start, remove the rest
                text = text[: true_start + block_len] + text[pos:]
                total_removed += total_count - 1
                # Recurse in case there are more repetitions elsewhere
                # (text has changed, positions are invalidated)
                more_text, more_removed = _collapse_inline_repeats(text, min_block_len)
                return more_text, total_removed + more_removed

    return text, total_removed


# ============================================
# Encoding Corruption / Gibberish Detection
# ============================================

_LATIN1_SUPPLEMENT: Final = re.compile(r"[\u00c0-\u00ff]")

# Only match repeated chars in corruption-indicative ranges.
# Excludes: Devanagari (U+0900-097F), ellipsis (U+2026), CJK,
# smart quotes, em-dashes - these repeat legitimately in legal text.
_REPEATED_CORRUPT: Final = re.compile(r"([\u0080-\u00ff\u25a0-\u25ff\ufffd])\1{2,}")

_CONSECUTIVE_LATIN1: Final = re.compile(r"[\u00c0-\u00ff]{3,}")
_SYMBOL_ARTIFACTS: Final = re.compile(
    r"[\u0080-\u009f"  # C1 controls
    r"\u25a0-\u25ff"  # Geometric Shapes
    r"]"
)
_REPLACEMENT_CHAR: Final = re.compile(r"\ufffd")


def corruption_score(text: str) -> float:
    """Return a continuous 0.0-1.0 corruption severity score.

    Six signals, weighted so no single signal alone reaches
    the 0.5 deletion threshold (requires corroboration):
    1. Repeated corruption chars (Latin-1/C1/geometric/FFFD repeated 3+)
    2. Latin-1 Supplement density
    3. Consecutive Latin-1 chars (3+ in a row)
    4. Question-mark density + encoding artifacts
    5. Extreme question-mark density alone (>25%)
    6. Unicode replacement char (U+FFFD) count
    """
    if not text:
        return 0.0

    sample = text[:2000]
    length = len(sample)
    if length < 10:
        return 0.0

    score = 0.0

    # Signal 1: repeated corruption chars (NOT Devanagari/ellipsis)
    repeated_runs = _REPEATED_CORRUPT.findall(sample)
    score += min(len(repeated_runs) * 0.15, 0.45)

    # Signal 2: Latin-1 Supplement density
    latin1_count = len(_LATIN1_SUPPLEMENT.findall(sample))
    latin1_ratio = latin1_count / length
    score += min(latin1_ratio * 2.5, 1.0)

    # Signal 3: consecutive Latin-1 chars (3+)
    consecutive_chars = sum(len(m) for m in _CONSECUTIVE_LATIN1.findall(sample))
    score += min((consecutive_chars / length) * 3.0, 0.6)

    # Signal 4: question marks + encoding artifacts
    symbol_count = len(_SYMBOL_ARTIFACTS.findall(sample))
    has_artifacts = latin1_count > 0 or symbol_count > 0
    question_count = sample.count("?")
    question_ratio = question_count / length

    if has_artifacts:
        score += min(question_ratio * 2.0, 0.5)
        score += min(symbol_count / length * 3.0, 0.3)

    # Signal 5: extreme question-mark density (>25%)
    if question_ratio > 0.25:
        score += 0.4

    # Signal 6: Unicode replacement chars (U+FFFD) - always corruption
    fffd_count = len(_REPLACEMENT_CHAR.findall(sample))
    if fffd_count > 0:
        score += min(fffd_count * 0.05, 0.4)

    return round(min(score, 1.0), 3)


def is_encoding_corrupted(text: str, threshold: float = 0.15) -> bool:
    """Check if text contains encoding corruption / gibberish.

    Thin boolean wrapper around corruption_score().
    Default threshold 0.15 is tuned for query-time filtering (aggressive).
    Use higher thresholds (0.5+) for permanent deletion decisions.
    """
    return corruption_score(text) >= threshold


def clean_format_specific_markers(text: str, format_type: str) -> str:
    """
    Remove format-specific markers added by extractors.

    Args:
        text: Raw extracted text
        format_type: File extension (xlsx, pptx, etc.)

    Returns:
        Cleaned text with format markers removed

    Example:
        >>> text = "## Sheet: Sheet1\\nData\\n## Sheet: Sheet2"
        >>> clean_format_specific_markers(text, "xlsx")
        'Data'
    """
    if format_type in ("xlsx", "xls", "xlsm", "xltx", "xltm"):
        # Remove "## Sheet: SheetName" markers from Excel files
        text = re.sub(r"^##\s+Sheet:\s+.+$", "", text, flags=re.MULTILINE)

    elif format_type in ("pptx", "ppt"):
        # Remove "## Slide N" markers from PowerPoint files
        text = re.sub(r"^##\s+Slide\s+\d+$", "", text, flags=re.MULTILINE)

    # Clean up any resulting multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ============================================
# Convenience Functions
# ============================================


def should_clean_text(filename: str) -> bool:
    """
    Determine if text from this file type should be cleaned.

    Args:
        filename: Original filename with extension

    Returns:
        True if cleaning should be applied, False otherwise

    Note:
        We clean all formats, but this function exists for future
        fine-tuning (e.g., skip cleaning for plain text files).
    """
    # Currently clean all formats
    # Future: Could skip cleaning for .txt files or other formats
    return True


def get_cleaning_report(stats: CleaningStats) -> dict[str, int | float]:
    """
    Convert CleaningStats to a dictionary for logging/metrics.

    Args:
        stats: CleaningStats object from clean_ocr_artifacts()

    Returns:
        Dictionary with cleaning metrics

    Example:
        >>> _, stats = clean_ocr_artifacts("Some text", return_stats=True)
        >>> report = get_cleaning_report(stats)
        >>> report["artifacts_removed"]
        0
    """
    return {
        "artifacts_removed": stats.artifacts_removed,
        "original_chars": stats.original_length,
        "cleaned_chars": stats.cleaned_length,
        "reduction_pct": round(stats.reduction_pct, 2),
        "blocks_deduplicated": stats.blocks_deduplicated,
    }
