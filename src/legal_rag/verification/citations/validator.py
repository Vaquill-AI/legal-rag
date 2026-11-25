"""
Cross-Reference Citation Validator - New Verification Layer.

This module adds a critical verification layer that catches citation-level
hallucinations that other verification methods miss:

1. **Citation Index Validation**: Verifies [N] references map to actual sources
2. **Quote-Source Matching**: Checks quoted text exists in cited source
3. **Section Reference Validation**: Verifies "Section X" claims against document structure
4. **Span-Level Attribution**: Tracks exact character ranges in sources

Why This Layer is Critical:
- LLM verification checks semantic accuracy but not citation accuracy
- Fuzzy matching catches content similarity but misses citation fabrication
- This layer prevents the common "phantom citation" hallucination pattern

Research Background:
- "Attributed QA" (Google, 2023): Citation accuracy is independent of content accuracy
- "FActScore" (AI2, 2023): Fine-grained fact verification at atomic level
- Legal AI requires citation precision - wrong citation is as bad as wrong fact

Performance:
- ~5-10ms per citation (regex + text search)
- 0% LLM cost (fully deterministic)
- Runs in parallel with other verification methods

Integration Point:
- Called BEFORE LLM verification as first-pass filter
- If citation validation fails, skip expensive verification
- Results feed into calibrated confidence calculation
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from legal_rag.verification import _fuzz as fuzz
from legal_rag.verification._log import get_logger

logger = get_logger(__name__)


# ============================================
# Types
# ============================================


class CitationValidationStatus(str, Enum):
    """Status of citation validation."""

    VALID = "valid"  # Citation exists and content matches
    PARTIAL = "partial"  # Citation exists but content partially matches
    INVALID_INDEX = "invalid_index"  # Citation [N] references non-existent source
    QUOTE_NOT_FOUND = "quote_not_found"  # Quoted text not in cited source
    SECTION_NOT_FOUND = "section_not_found"  # Referenced section doesn't exist
    FABRICATED = "fabricated"  # Citation appears to be fabricated


@dataclass
class CitationSpan:
    """Tracks exact span in source that supports a citation."""

    source_index: int
    start_char: int
    end_char: int
    matched_text: str
    confidence: float = 1.0


@dataclass
class CitationValidationResult:
    """Result of validating a single citation."""

    citation_ref: str  # e.g., "[1]", "[2,3]", "Section 4.2"
    status: CitationValidationStatus
    confidence: float  # 0.0-1.0
    source_indices: list[int] = field(default_factory=list)  # Valid source indices
    invalid_indices: list[int] = field(default_factory=list)  # Invalid source indices
    span: CitationSpan | None = None  # Exact span if found
    explanation: str = ""
    validation_time_ms: float = 0.0


@dataclass
class CrossReferenceValidationResult:
    """Overall cross-reference validation result."""

    is_valid: bool
    overall_score: float  # 0.0-1.0, percentage of valid citations
    total_citations: int
    valid_citations: int
    invalid_citations: int
    fabricated_citations: int
    citation_results: list[CitationValidationResult] = field(default_factory=list)
    validation_time_ms: float = 0.0

    # Additional aggregated stats
    quote_validations: int = 0  # Number of quotes validated against sources
    section_validations: int = 0  # Number of section references validated
    invalid_indices: list[int] = field(default_factory=list)  # All invalid citation indices

    # Property aliases for compatibility
    @property
    def citation_count(self) -> int:
        """Alias for total_citations."""
        return self.total_citations

    @property
    def overall_confidence(self) -> float:
        """Alias for overall_score."""
        return self.overall_score


# ============================================
# Regex Patterns
# ============================================

# Standard citation references [1], [2], [1,2], [1-3]
CITATION_REF_PATTERN = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")

# Quoted text patterns
QUOTE_PATTERN = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{15,})["\u201c\u201d]')

# Section/Clause references
SECTION_REF_PATTERN = re.compile(
    r"(?:Section|Clause|Article|Paragraph|Rule|Regulation)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)",
    re.IGNORECASE,
)

# "According to [N]" patterns that make explicit source claims
ACCORDING_TO_PATTERN = re.compile(
    r"(?:According to|As per|As stated in|Per|Based on)\s+\[(\d+)\]",
    re.IGNORECASE,
)

# Quote + citation patterns: "some text" [1]
QUOTE_WITH_CITATION_PATTERN = re.compile(r'["\u201c]([^"\u201c\u201d]{10,})["\u201d]\s*\[(\d+)\]')


# ============================================
# Citation Validator
# ============================================


class CrossReferenceCitationValidator:
    """
    Validates citation accuracy against source documents.

    This validator catches citation-level hallucinations that semantic
    verification misses:
    - Citations to non-existent sources
    - Quotes not found in cited sources
    - Section references that don't exist
    - Fabricated legal citations

    Usage:
        validator = CrossReferenceCitationValidator()
        result = validator.validate(
            response_text="According to [1], the contract states...",
            sources=[{"text": "...", ...}]
        )

        if not result.is_valid:
            print(f"Found {result.invalid_citations} invalid citations")
    """

    def __init__(
        self,
        quote_similarity_threshold: float = 0.85,
        min_quote_length: int = 15,
        section_fuzzy_match: bool = True,
    ):
        """
        Initialize validator.

        Args:
            quote_similarity_threshold: Min similarity for quote matching (0-1)
            min_quote_length: Minimum quote length to validate
            section_fuzzy_match: Allow fuzzy section number matching
        """
        self.quote_similarity_threshold = quote_similarity_threshold
        self.min_quote_length = min_quote_length
        self.section_fuzzy_match = section_fuzzy_match

    def validate(
        self,
        response_text: str,
        sources: list[dict[str, Any]],
    ) -> CrossReferenceValidationResult:
        """
        Validate all citations in response against sources.

        Args:
            response_text: AI response text with citations
            sources: List of source chunks with 'text' field

        Returns:
            CrossReferenceValidationResult with detailed validation
        """
        start_time = time.perf_counter()

        logger.info(
            "cross_reference_validation_started",
            response_length=len(response_text),
            source_count=len(sources),
        )

        citation_results: list[CitationValidationResult] = []
        max_source_index = len(sources)

        # Step 1: Validate [N] citation references
        index_results = self._validate_citation_indices(response_text, max_source_index)
        citation_results.extend(index_results)

        # Step 2: Validate quote + citation combinations
        quote_results = self._validate_quotes_with_citations(response_text, sources)
        citation_results.extend(quote_results)

        # Step 3: Validate section references
        section_results = self._validate_section_references(response_text, sources)
        citation_results.extend(section_results)

        # Step 4: Calculate overall metrics
        valid_count = sum(
            1
            for r in citation_results
            if r.status in (CitationValidationStatus.VALID, CitationValidationStatus.PARTIAL)
        )
        invalid_count = sum(
            1
            for r in citation_results
            if r.status
            in (
                CitationValidationStatus.INVALID_INDEX,
                CitationValidationStatus.QUOTE_NOT_FOUND,
                # SECTION_NOT_FOUND is intentionally excluded: in legal RAG, section
                # references like "Section 1983" often cite statutory law that is
                # not in the uploaded documents. Absence of the section heading in
                # source chunks is not evidence of fabrication.
            )
        )
        fabricated_count = sum(
            1 for r in citation_results if r.status == CitationValidationStatus.FABRICATED
        )

        # Collect all invalid indices across results
        all_invalid_indices: list[int] = []
        for r in citation_results:
            all_invalid_indices.extend(r.invalid_indices)
        # Deduplicate
        all_invalid_indices = list(set(all_invalid_indices))

        total = len(citation_results)
        overall_score = valid_count / total if total > 0 else 1.0
        is_valid = invalid_count == 0 and fabricated_count == 0

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = CrossReferenceValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=invalid_count,
            fabricated_citations=fabricated_count,
            citation_results=citation_results,
            validation_time_ms=elapsed_ms,
            quote_validations=len(quote_results),
            section_validations=len(section_results),
            invalid_indices=all_invalid_indices,
        )

        logger.info(
            "cross_reference_validation_completed",
            is_valid=is_valid,
            overall_score=f"{overall_score:.1%}",
            total=total,
            valid=valid_count,
            invalid=invalid_count,
            fabricated=fabricated_count,
            time_ms=f"{elapsed_ms:.2f}",
        )

        return result

    def _validate_citation_indices(
        self,
        response_text: str,
        max_index: int,
    ) -> list[CitationValidationResult]:
        """
        Validate that [N] citations reference existing sources.

        Args:
            response_text: Response text with citations
            max_index: Maximum valid source index (len(sources))

        Returns:
            List of validation results for each citation reference
        """
        results = []
        seen_refs: set[str] = set()

        for match in CITATION_REF_PATTERN.finditer(response_text):
            start_time = time.perf_counter()
            citation_ref = match.group(0)  # e.g., "[1]"
            ref_content = match.group(1)  # e.g., "1" or "1,2" or "1-3"

            # Skip duplicates
            if citation_ref in seen_refs:
                continue
            seen_refs.add(citation_ref)

            # Parse citation numbers
            citation_nums = self._parse_citation_numbers(ref_content)

            valid_indices = []
            invalid_indices = []

            for num in citation_nums:
                # Citations are 1-indexed, sources are 0-indexed
                if 1 <= num <= max_index:
                    valid_indices.append(num)
                else:
                    invalid_indices.append(num)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if invalid_indices:
                status = CitationValidationStatus.INVALID_INDEX
                confidence = len(valid_indices) / len(citation_nums) if citation_nums else 0.0
                explanation = (
                    f"Citation(s) {invalid_indices} exceed available sources (max: {max_index})"
                )
            else:
                status = CitationValidationStatus.VALID
                confidence = 1.0
                explanation = f"All citation indices valid: {valid_indices}"

            results.append(
                CitationValidationResult(
                    citation_ref=citation_ref,
                    status=status,
                    confidence=confidence,
                    source_indices=valid_indices,
                    invalid_indices=invalid_indices,
                    explanation=explanation,
                    validation_time_ms=elapsed_ms,
                )
            )

        return results

    def _validate_quotes_with_citations(
        self,
        response_text: str,
        sources: list[dict[str, Any]],
    ) -> list[CitationValidationResult]:
        """
        Validate that quoted text exists in the cited source.

        Catches the common hallucination pattern:
        "Quote that doesn't exist" [1]

        Args:
            response_text: Response with quotes and citations
            sources: Source documents

        Returns:
            Validation results for quote+citation pairs
        """
        results = []

        for match in QUOTE_WITH_CITATION_PATTERN.finditer(response_text):
            start_time = time.perf_counter()
            quote_text = match.group(1)
            citation_num = int(match.group(2))

            # Skip short quotes
            if len(quote_text) < self.min_quote_length:
                continue

            # Get cited source (1-indexed to 0-indexed)
            source_idx = citation_num - 1
            if source_idx < 0 or source_idx >= len(sources):
                results.append(
                    CitationValidationResult(
                        citation_ref=f'"{quote_text[:30]}..." [{citation_num}]',
                        status=CitationValidationStatus.INVALID_INDEX,
                        confidence=0.0,
                        invalid_indices=[citation_num],
                        explanation=f"Citation [{citation_num}] references non-existent source",
                        validation_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                )
                continue

            source_text = sources[source_idx].get("text", "")

            # Check for exact match
            span = self._find_quote_in_source(quote_text, source_text, source_idx)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if span:
                results.append(
                    CitationValidationResult(
                        citation_ref=f'"{quote_text[:30]}..." [{citation_num}]',
                        status=CitationValidationStatus.VALID,
                        confidence=span.confidence,
                        source_indices=[citation_num],
                        span=span,
                        explanation=f"Quote found in source [{citation_num}] at chars {span.start_char}-{span.end_char}",
                        validation_time_ms=elapsed_ms,
                    )
                )
            else:
                results.append(
                    CitationValidationResult(
                        citation_ref=f'"{quote_text[:30]}..." [{citation_num}]',
                        status=CitationValidationStatus.QUOTE_NOT_FOUND,
                        confidence=0.0,
                        source_indices=[citation_num],
                        explanation=f"Quote not found in source [{citation_num}] - possible fabrication",
                        validation_time_ms=elapsed_ms,
                    )
                )

        return results

    def _validate_section_references(
        self,
        response_text: str,
        sources: list[dict[str, Any]],
    ) -> list[CitationValidationResult]:
        """
        Validate section/clause references against source documents.

        Catches claims like "Section 4.2 states..." when Section 4.2
        doesn't exist in any source.

        Args:
            response_text: Response with section references
            sources: Source documents

        Returns:
            Validation results for section references
        """
        results = []
        seen_sections: set[str] = set()

        for match in SECTION_REF_PATTERN.finditer(response_text):
            start_time = time.perf_counter()
            full_ref = match.group(0)  # e.g., "Section 4.2"
            section_num = match.group(1)  # e.g., "4.2"

            # Skip duplicates
            if full_ref.lower() in seen_sections:
                continue
            seen_sections.add(full_ref.lower())

            # Search for section in sources
            found, source_idx, span = self._find_section_in_sources(section_num, sources)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if found:
                # The span carries the confidence, because only
                # `_find_section_in_sources` knows whether the number was
                # unique across the panel. Hardcoding 1.0 here is what let an
                # ambiguous match report as certain.
                confidence = span.confidence if span else 1.0
                results.append(
                    CitationValidationResult(
                        citation_ref=full_ref,
                        status=CitationValidationStatus.VALID,
                        confidence=confidence,
                        source_indices=[source_idx + 1] if source_idx >= 0 else [],
                        span=span,
                        explanation=(
                            f"{full_ref} found in sources"
                            if confidence >= 0.5
                            else f"{full_ref} appears in more than one source; "
                            "which one the response meant is undetermined"
                        ),
                        validation_time_ms=elapsed_ms,
                    )
                )
            else:
                # Section not found - could be fabricated
                results.append(
                    CitationValidationResult(
                        citation_ref=full_ref,
                        status=CitationValidationStatus.SECTION_NOT_FOUND,
                        confidence=0.0,
                        explanation=f"{full_ref} not found in any source document",
                        validation_time_ms=elapsed_ms,
                    )
                )

        return results

    def _parse_citation_numbers(self, ref_content: str) -> list[int]:
        """
        Parse citation numbers from reference content.

        Handles:
        - Single: "1" → [1]
        - Comma: "1,2" → [1, 2]
        - Range: "1-3" → [1, 2, 3]
        """
        numbers = []

        if "-" in ref_content:
            # Range
            parts = ref_content.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    numbers = list(range(start, end + 1))
                except ValueError:
                    pass
        elif "," in ref_content:
            # Comma-separated
            for part in ref_content.split(","):
                try:
                    numbers.append(int(part.strip()))
                except ValueError:
                    pass
        else:
            # Single number
            try:
                numbers = [int(ref_content.strip())]
            except ValueError:
                pass

        return numbers

    def _find_quote_in_source(
        self,
        quote: str,
        source_text: str,
        source_idx: int,
    ) -> CitationSpan | None:
        """
        Find quote in source text with fuzzy matching.

        Returns CitationSpan if found, None otherwise.
        """
        quote_lower = quote.lower().strip()
        source_lower = source_text.lower()

        # Try exact match first
        idx = source_lower.find(quote_lower)
        if idx != -1:
            return CitationSpan(
                source_index=source_idx,
                start_char=idx,
                end_char=idx + len(quote),
                matched_text=source_text[idx : idx + len(quote)],
                confidence=1.0,
            )

        # Try fuzzy matching on sentences
        sentences = re.split(r"[.!?]+", source_text)
        best_match: tuple[int, str, float] | None = None

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Calculate similarity
            similarity = fuzz.ratio(quote_lower, sentence.lower()) / 100.0

            if similarity >= self.quote_similarity_threshold and (
                best_match is None or similarity > best_match[2]
            ):
                # Find position in original text
                start = source_text.find(sentence)
                if start != -1:
                    best_match = (start, sentence, similarity)

        if best_match:
            start, matched, confidence = best_match
            return CitationSpan(
                source_index=source_idx,
                start_char=start,
                end_char=start + len(matched),
                matched_text=matched,
                confidence=confidence,
            )

        return None

    def _find_section_in_sources(
        self,
        section_num: str,
        sources: list[dict[str, Any]],
    ) -> tuple[bool, int, CitationSpan | None]:
        """
        Find section reference in sources, one source at a time.

        The concatenated-sources parameter this used to take is GONE, not just
        unused: keeping it invited the next reader to search it again, and it
        cost a full join of every source on every call.

        Returns:
            (found, source_index, span)
        """
        # Handle variations: "4.2", "4.2(a)", etc.
        escaped_num = re.escape(section_num)
        pattern = re.compile(
            rf"(?:Section|Clause|Article|Paragraph)\s*{escaped_num}\b",
            re.IGNORECASE,
        )

        # Search each source SEPARATELY, never the concatenation.
        #
        # This used to match against `all_source_text`, the sources joined with
        # a space, and then work out which source the offset landed in. Two
        # things were wrong with that. The small one: a match can straddle the
        # join, so a source ending in "Section" and the next beginning with
        # "4.2" manufactured a reference neither document contains. The large
        # one is the semantics -- a hit anywhere in the concatenation was
        # reported VALID at confidence 1.0, so a response asserting "Section 4.2
        # of the Lease provides..." was confirmed by an unrelated NDA in the
        # same panel that happens to have its own Section 4.2. Same number,
        # different document, and the validator said valid.
        #
        # Per-source matching cannot straddle a boundary, and the ambiguity is
        # now visible rather than resolved by whichever source sorted first.
        hits: list[tuple[int, re.Match[str]]] = []
        for idx, source in enumerate(sources):
            match = pattern.search(source.get("text", "") or "")
            if match:
                hits.append((idx, match))

        if not hits:
            return (False, -1, None)

        idx, match = hits[0]
        # Several documents carry the same section number, so which one the
        # response meant is not something this can tell. Confidence drops below
        # the 0.5 floor the callers treat as decisive rather than reporting a
        # coin flip as certainty.
        confidence = 1.0 if len(hits) == 1 else 0.4
        return (
            True,
            idx,
            CitationSpan(
                source_index=idx,
                start_char=match.start(),
                end_char=match.end(),
                matched_text=match.group(0),
                confidence=confidence,
            ),
        )


# ============================================
# Quick Check Function
# ============================================


def quick_citation_validation(
    response_text: str,
    source_count: int,
) -> tuple[bool, list[int]]:
    """
    Ultra-fast citation index validation.

    Use this as first-pass filter BEFORE expensive verification.
    ~0.5ms execution time.

    Args:
        response_text: AI response with citations
        source_count: Number of available sources

    Returns:
        (all_valid, list_of_invalid_indices)
    """
    invalid_indices: list[int] = []

    for match in CITATION_REF_PATTERN.finditer(response_text):
        ref_content = match.group(1)

        # Parse numbers
        if "-" in ref_content:
            parts = ref_content.split("-")
            try:
                nums = list(range(int(parts[0]), int(parts[1]) + 1))
            except ValueError:
                continue
        elif "," in ref_content:
            try:
                nums = [int(n.strip()) for n in ref_content.split(",")]
            except ValueError:
                continue
        else:
            try:
                nums = [int(ref_content)]
            except ValueError:
                continue

        # Check bounds (citations are 1-indexed)
        for num in nums:
            if (num < 1 or num > source_count) and num not in invalid_indices:
                invalid_indices.append(num)

    return (len(invalid_indices) == 0, invalid_indices)


# ============================================
# Inline citation cleanup (drop-before-display)
# ============================================


def _expand_ref_numbers(ref_content: str) -> list[int]:
    """Expand a citation ref body into its numbers.

    "2" -> [2]; "2,5" -> [2, 5]; "2-4" -> [2, 3, 4]. Returns [] if unparseable.
    """
    if "-" in ref_content:
        parts = ref_content.split("-")
        if len(parts) != 2:
            return []
        try:
            return list(range(int(parts[0].strip()), int(parts[1].strip()) + 1))
        except ValueError:
            return []
    if "," in ref_content:
        numbers: list[int] = []
        for part in ref_content.split(","):
            try:
                numbers.append(int(part.strip()))
            except ValueError:
                pass
        return numbers
    try:
        return [int(ref_content.strip())]
    except ValueError:
        return []


# Matches a citation marker plus an optional single preceding space. Removing a
# fully-fabricated marker takes its separating space with it, so no double space
# is created. This keeps cleanup LOCAL to each removed marker: we never run a
# document-wide whitespace pass, which would collapse intentional formatting
# elsewhere (code-block indentation, markdown hard-break double-spaces).
_CITATION_WITH_LEAD_SPACE = re.compile(r"( ?)(\[(\d+(?:[-,]\d+)*)\])")

# A citation that stood alone inside parentheses leaves "()" behind once removed.
_EMPTY_PARENS = re.compile(r"\(\s*\)")


def strip_fabricated_citations(
    response_text: str,
    source_count: int,
) -> tuple[str, list[int]]:
    """Remove citation markers that reference non-existent sources.

    A generated answer sometimes cites ``[N]`` where N exceeds the number of
    retrieved sources: a fabricated citation the reader must never see. This
    deterministically rewrites or removes those markers before display,
    mirroring the Document Matrix cell extractor which drops failed citations
    inline rather than surfacing a citation it cannot stand behind.

    Only out-of-range indices are treated as fabricated, so an in-range citation
    is never touched (zero false positives). Grouped refs are partially cleaned:
    ``[2,7]`` with 5 sources becomes ``[2]``; ``[7]`` alone is removed entirely.

    Args:
        response_text: Answer text with ``[N]`` / ``[N,M]`` / ``[N-M]`` markers.
        source_count: Number of available sources (valid indices are 1..source_count).

    Returns:
        ``(cleaned_text, removed_indices)``. ``removed_indices`` lists every
        fabricated citation number stripped, in order (may repeat across markers).
        When nothing is fabricated, the original text is returned unchanged.
    """
    if not response_text or source_count < 0:
        return response_text, []

    removed: list[int] = []

    def _rewrite(match: re.Match[str]) -> str:
        lead = match.group(1)  # optional single space preceding the marker
        nums = _expand_ref_numbers(match.group(3))
        if not nums:
            return match.group(0)  # unparseable, leave untouched
        valid = [n for n in nums if 1 <= n <= source_count]
        invalid = [n for n in nums if n < 1 or n > source_count]
        if not invalid:
            return match.group(0)  # every index in range, keep as-is (with its space)
        removed.extend(invalid)
        if valid:
            return lead + "[" + ",".join(str(n) for n in valid) + "]"
        # All indices fabricated: drop the marker. Consume its leading space too so
        # no double space is created and no document-wide whitespace pass is needed
        # -- UNLESS a word character immediately follows the marker, in which case
        # keep the space so the surrounding words are not merged
        # ("the tenant [9]must" -> "the tenant must", never "tenantmust").
        after = match.string[match.end() : match.end() + 1]
        if after and (after.isalnum() or after == "_"):
            return lead
        return ""

    cleaned = _CITATION_WITH_LEAD_SPACE.sub(_rewrite, response_text)

    if removed:
        # Only the local "()" left where a parenthesized citation was removed.
        # Deliberately NO document-wide whitespace normalization: that collapsed
        # intentional formatting (code-block indentation, hard-break spaces).
        cleaned = _EMPTY_PARENS.sub("", cleaned)

    return cleaned, removed


def citation_failure_ratio(
    response_text: str,
    source_count: int,
) -> tuple[int, int]:
    """Count citation markers and how many are entirely fabricated.

    Counts distinct ``[N]`` markers (not distinct indices), so a lone fabricated
    marker in a one-citation answer yields ``(1, 1)``, the "all citations fail"
    signal a corrective regeneration gate keys on. A marker is counted as
    fabricated only when every index it contains is out of range.

    Returns ``(total_markers, fabricated_markers)``.
    """
    total = 0
    fabricated = 0
    for match in CITATION_REF_PATTERN.finditer(response_text or ""):
        nums = _expand_ref_numbers(match.group(1))
        if not nums:
            continue
        total += 1
        if all(n < 1 or n > source_count for n in nums):
            fabricated += 1
    return total, fabricated


_BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>\s?")
_CALLOUT_RE = re.compile(r"^\s*>\s*\[!(?:NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]", re.IGNORECASE)
_QUOTE_MARKS_RE = re.compile(r"[‘’“”\"']")


def _normalize_for_quote_match(text: str) -> str:
    """Lowercase, drop [N] markers and quote marks, collapse whitespace."""
    text = CITATION_REF_PATTERN.sub(" ", text or "")
    text = _QUOTE_MARKS_RE.sub("", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def demote_ungrounded_blockquotes(
    response_text: str,
    sources: list[dict[str, Any]],
    *,
    min_quote_chars: int = 25,
    fuzzy_threshold: int = 88,
) -> tuple[str, int]:
    """Demote verbatim ``> "..."`` blockquotes whose text is NOT in any source.

    A markdown blockquote claims the text is a VERBATIM quotation from a source.
    When the model invents or paraphrases clause text and dresses it as a
    blockquote (the audit's "block-quotes a clause the persisted chunk does not
    contain"), that is a citation-grade fabrication: the reader sees exact words
    that were never retrieved. This deterministically checks each verbatim
    blockquote against the concatenated source text (normalized substring OR high
    fuzzy overlap) and, for any that does not match, strips the leading ``> `` so
    it renders as ordinary prose instead of an exact quote. It NEVER touches
    callout boxes (``> [!NOTE]`` etc.), and skips blocks shorter than
    ``min_quote_chars`` to avoid false positives on trivial fragments.

    Deterministic, no LLM. Returns ``(cleaned_text, demoted_block_count)``.
    """
    if not response_text or not sources:
        return response_text, 0

    haystack = _normalize_for_quote_match(" ".join((s.get("text") or "") for s in sources))
    if not haystack:
        return response_text, 0

    lines = response_text.split("\n")
    n = len(lines)
    out: list[str] = []
    demoted = 0
    i = 0
    while i < n:
        if not _BLOCKQUOTE_LINE_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        # Collect the whole consecutive `>` block.
        is_callout = bool(_CALLOUT_RE.match(lines[i]))
        block: list[str] = []
        while i < n and _BLOCKQUOTE_LINE_RE.match(lines[i]):
            block.append(lines[i])
            i += 1

        # Callouts are context/warnings, never verbatim quotes -- leave untouched.
        if is_callout:
            out.extend(block)
            continue

        block_text = " ".join(_BLOCKQUOTE_LINE_RE.sub("", bl) for bl in block)
        norm = _normalize_for_quote_match(block_text)
        grounded = (
            len(norm) < min_quote_chars  # too short to adjudicate -> leave alone
            or norm in haystack
            or fuzz.partial_ratio(norm, haystack) >= fuzzy_threshold
        )
        if grounded:
            out.extend(block)
        else:
            # Demote: drop the `>` prefix so it is no longer a verbatim source quote.
            out.extend(_BLOCKQUOTE_LINE_RE.sub("", bl) for bl in block)
            demoted += 1

    return "\n".join(out), demoted


# ============================================
# Module-level singleton
# ============================================

_citation_validator: CrossReferenceCitationValidator | None = None


def get_citation_validator() -> CrossReferenceCitationValidator:
    """Get or create the global citation validator instance."""
    global _citation_validator
    if _citation_validator is None:
        _citation_validator = CrossReferenceCitationValidator()
    return _citation_validator
