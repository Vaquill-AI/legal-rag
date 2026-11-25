"""
Temporal precedent tracking.

Tracks the validity of legal precedents over time so that cited cases can be
checked for whether they are still good law.

Key features:
- Overruled case detection
- Distinguished case tracking
- Statute amendment awareness
- Citation freshness scoring
- Temporal validation (prevents impossible relationships such as an earlier
  case overruling a later one)

Citation patterns and the seed list of known overrulings are data, kept in
``data/precedent_data.json`` so they can be extended without code changes.

Background: "Do LLMs Truly Understand When a Precedent Is Overruled?"
(Zhang et al., 2025) reports that language models identify overruling with
limited accuracy and produce temporally impossible relationships, which is why
this module validates dates in code.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from legal_rag.chunking._logging import get_logger

logger = get_logger("precedent")

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "precedent_data.json"


@lru_cache(maxsize=1)
def _load_data() -> dict[str, Any]:
    with _DATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=16)
def _compiled_year_patterns(groups: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile year patterns for the requested groups, generic patterns last."""
    table = _load_data()["year_patterns"]
    ordered = [g for g in groups if g != "generic"] + ["generic"]
    compiled: list[re.Pattern[str]] = []
    for name in ordered:
        if name not in table:
            raise KeyError(f"Unknown citation group: {name!r}")
        compiled.extend(re.compile(pattern, re.IGNORECASE) for pattern, _note in table[name])
    return tuple(compiled)


def available_citation_groups() -> list[str]:
    """Names of the citation pattern groups in the data file."""
    return sorted(_load_data()["year_patterns"])


def extract_year_from_citation(citation: str, groups: Sequence[str] | None = None) -> int | None:
    """
    Extract the year from a legal citation.

    Patterns live in ``data/precedent_data.json``, grouped by citation family
    (``us``, ``uk_commonwealth``, ``india``) plus a ``generic`` group that is
    always tried last. Within the selected groups, patterns are ordered from
    most specific to least specific so the first valid year found wins.

    Args:
        citation: The citation string to extract the year from.
        groups: Citation families to use, in priority order. Defaults to the
            file's ``default_groups`` (US first, then UK and Commonwealth).

    Returns:
        Year as an integer, or None if none is found.
    """
    names = tuple(_load_data()["default_groups"] if groups is None else groups)
    for pattern in _compiled_year_patterns(names):
        match = pattern.search(citation)
        if match:
            year = int(match.group(1))
            # 1600: early English common law reports. 2100: future buffer.
            if 1600 <= year <= 2100:
                return year
    return None


class PrecedentStatus(str, Enum):
    """Status of a legal precedent."""

    GOOD_LAW = "good_law"  # Valid and citable
    OVERRULED = "overruled"  # No longer valid law
    DISTINGUISHED = "distinguished"  # Limited applicability
    QUESTIONED = "questioned"  # Validity under question
    SUPERSEDED = "superseded"  # Replaced by statute/ruling
    AFFIRMED = "affirmed"  # Confirmed by higher court
    PARTIALLY_OVERRULED = "partial"  # Some holdings overruled
    UNKNOWN = "unknown"  # Status not determined


@dataclass
class TemporalValidationResult:
    """
    Result of temporal validation for an overruling relationship.

    Validates that the overruling case is temporally posterior to the overruled case.
    Based on research showing LLMs create impossible relationships like 1914 case
    overruling a 1925 case (Zhang et al., 2025).
    """

    is_valid: bool
    original_year: int | None
    overruling_year: int | None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "original_year": self.original_year,
            "overruling_year": self.overruling_year,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


@dataclass
class PrecedentHistory:
    """
    History entry for a precedent change.

    Tracks what changed, when, and by which case/statute.
    """

    change_date: date | None
    change_type: PrecedentStatus
    changed_by_citation: str
    changed_by_court: str
    description: str
    affected_holdings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_date": self.change_date.isoformat() if self.change_date else None,
            "change_type": self.change_type.value,
            "changed_by_citation": self.changed_by_citation,
            "changed_by_court": self.changed_by_court,
            "description": self.description,
            "affected_holdings": self.affected_holdings,
        }


@dataclass
class PrecedentValidity:
    """
    Validity assessment for a legal precedent.

    Attributes:
        citation: The case citation
        status: Current validity status
        is_valid: Whether the case is still good law
        validity_score: Score from 0-1 (1 = fully valid)
        decision_date: Date of the original decision
        last_checked: When validity was last verified
        history: List of changes to validity
        warnings: Any warnings about citing this case
        related_citations: Citations that affect this case
    """

    citation: str
    status: PrecedentStatus
    is_valid: bool
    validity_score: float
    decision_date: date | None = None
    last_checked: datetime | None = None
    history: list[PrecedentHistory] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    related_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation": self.citation,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "validity_score": round(self.validity_score, 3),
            "decision_date": (self.decision_date.isoformat() if self.decision_date else None),
            "last_checked": (self.last_checked.isoformat() if self.last_checked else None),
            "history": [h.to_dict() for h in self.history],
            "warnings": self.warnings,
            "related_citations": self.related_citations,
        }


class PrecedentTracker:
    """
    Tracks temporal validity of legal precedents.

    Maintains knowledge of overruled cases, distinguishments, and
    statutory changes to ensure cited precedents are still valid.

    Example:
        tracker = PrecedentTracker()

        # Check if a citation is still good law
        validity = await tracker.check_validity("163 U.S. 537")

        if not validity.is_valid:
            print(f"Warning: {validity.citation} was {validity.status.value}")
            print(f"Overruled by: {validity.related_citations}")
    """

    # Patterns indicating case status in text
    OVERRULED_PATTERNS: ClassVar[list[str]] = [
        r"overruled\s+(?:in|by)\s+",
        r"no\s+longer\s+good\s+law",
        r"expressly\s+overruled",
        r"impliedly\s+overruled",
        r"overruled\s+the\s+decision",
    ]

    DISTINGUISHED_PATTERNS: ClassVar[list[str]] = [
        r"distinguished\s+(?:in|by|from)\s+",
        r"distinguishing\s+",
        r"does\s+not\s+apply\s+to",
        r"limited\s+to\s+its\s+facts",
    ]

    AFFIRMED_PATTERNS: ClassVar[list[str]] = [
        r"affirmed\s+(?:in|by)\s+",
        r"approved\s+(?:in|by)\s+",
        r"followed\s+(?:in|by)\s+",
        r"upheld\s+(?:in|by)\s+",
    ]

    def __init__(
        self,
        known_overruled: Mapping[str, tuple] | None = None,
        citation_groups: Sequence[str] | None = None,
    ):
        """
        Initialize precedent tracker.

        Args:
            known_overruled: Mapping of citation to ``(status, overruled_by, date,
                description)``. Defaults to the seed list in the data file.
            citation_groups: Citation families used for year extraction.
        """
        self._cache: dict[str, PrecedentValidity] = {}
        self._citation_groups = tuple(citation_groups) if citation_groups is not None else None
        self.known_overruled_cases: dict[str, tuple] = {}
        if known_overruled is None:
            for row in _load_data()["known_overruled"]:
                key = self._normalize_citation(row["citation"])
                self.known_overruled_cases[key] = (
                    PrecedentStatus(row["status"]),
                    row["overruled_by"],
                    date.fromisoformat(row["date"]),
                    row["description"],
                )
        else:
            for citation, entry in known_overruled.items():
                self.known_overruled_cases[self._normalize_citation(citation)] = entry

    async def check_validity(
        self,
        citation: str,
        context: str | None = None,
    ) -> PrecedentValidity:
        """
        Check the validity of a legal precedent.

        Args:
            citation: The citation to check
            context: Optional context text that might contain status info

        Returns:
            PrecedentValidity with current status
        """
        # Normalize citation
        normalized = self._normalize_citation(citation)

        # Check cache first
        if normalized in self._cache:
            cached = self._cache[normalized]
            # Invalidate cache after 24 hours
            if cached.last_checked and (datetime.now() - cached.last_checked).days < 1:
                return cached

        # Check known overruled cases
        if normalized in self.known_overruled_cases:
            status, overruled_by, change_date, description = self.known_overruled_cases[normalized]

            validity = PrecedentValidity(
                citation=citation,
                status=status,
                is_valid=False,
                validity_score=0.2,
                last_checked=datetime.now(),
                history=[
                    PrecedentHistory(
                        change_date=change_date,
                        change_type=status,
                        changed_by_citation=overruled_by,
                        changed_by_court="Supreme Court",
                        description=description,
                    )
                ],
                warnings=[f"This case has been {status.value} by {overruled_by}"],
                related_citations=[overruled_by],
            )

            self._cache[normalized] = validity
            return validity

        # Check context for status indicators
        status = PrecedentStatus.GOOD_LAW
        validity_score = 1.0
        warnings = []
        related = []

        if context:
            detected_status, related_citations = self._detect_status_from_context(citation, context)
            if detected_status != PrecedentStatus.GOOD_LAW:
                status = detected_status
                related = related_citations

                if detected_status == PrecedentStatus.OVERRULED:
                    validity_score = 0.1
                    warnings.append("Context indicates this case may be overruled")
                elif detected_status == PrecedentStatus.DISTINGUISHED:
                    validity_score = 0.6
                    warnings.append("Context indicates this case has been distinguished")
                elif detected_status == PrecedentStatus.QUESTIONED:
                    validity_score = 0.7
                    warnings.append("Context indicates this case has been questioned")

        # Calculate recency penalty
        year = self._extract_year(citation)
        if year:
            current_year = datetime.now().year
            age = current_year - year
            if age > 50:
                validity_score *= 0.8
                warnings.append(f"Case is {age} years old - verify current applicability")
            elif age > 30:
                validity_score *= 0.9

        validity = PrecedentValidity(
            citation=citation,
            status=status,
            is_valid=status in (PrecedentStatus.GOOD_LAW, PrecedentStatus.AFFIRMED),
            validity_score=validity_score,
            decision_date=date(year, 1, 1) if year else None,
            last_checked=datetime.now(),
            warnings=warnings,
            related_citations=related,
        )

        self._cache[normalized] = validity

        logger.info(
            "precedent_validity_checked",
            citation=citation,
            status=status.value,
            is_valid=validity.is_valid,
            validity_score=validity_score,
        )

        return validity

    async def check_multiple(
        self,
        citations: list[str],
        context: str | None = None,
    ) -> dict[str, PrecedentValidity]:
        """
        Check validity of multiple citations.

        Args:
            citations: List of citations to check
            context: Optional context text

        Returns:
            Dictionary mapping citations to their validity
        """
        results = {}
        for citation in citations:
            results[citation] = await self.check_validity(citation, context)
        return results

    def get_validity_summary(
        self,
        validities: list[PrecedentValidity],
    ) -> dict[str, Any]:
        """
        Generate summary of validity checks.

        Args:
            validities: List of validity results

        Returns:
            Summary statistics
        """
        total = len(validities)
        valid_count = sum(1 for v in validities if v.is_valid)
        invalid_count = total - valid_count
        avg_score = sum(v.validity_score for v in validities) / max(total, 1)

        by_status = {}
        for v in validities:
            by_status[v.status.value] = by_status.get(v.status.value, 0) + 1

        all_warnings = []
        for v in validities:
            all_warnings.extend(v.warnings)

        return {
            "total_checked": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "average_validity_score": round(avg_score, 3),
            "by_status": by_status,
            "warnings": all_warnings,
            "has_invalid_citations": invalid_count > 0,
        }

    def _normalize_citation(self, citation: str) -> str:
        """Normalize citation for consistent lookup."""
        # Remove extra whitespace
        normalized = " ".join(citation.split())
        # Uppercase for consistency
        normalized = normalized.upper()
        # Collapse dotted abbreviations: S.C.R. → SCR, I.N.S.C. → INSC
        normalized = re.sub(
            r"(?<![A-Z])([A-Z])(?:\.([A-Z]))+\.?",
            lambda m: m.group(0).replace(".", ""),
            normalized,
        )
        return normalized

    def _citation_regexes(self) -> list[re.Pattern[str]]:
        """Compiled citation regexes for this tracker's citation groups."""
        table = _load_data()["citation_patterns"]
        names = self._citation_groups or tuple(_load_data()["default_groups"])
        return [re.compile(p, re.IGNORECASE) for name in names for p in table.get(name, [])]

    def _extract_year(self, citation: str) -> int | None:
        """Extract year from citation."""
        return extract_year_from_citation(citation, self._citation_groups)

    def validate_temporal_relationship(
        self,
        original_citation: str,
        overruling_citation: str,
        overruling_date: date | None = None,
    ) -> TemporalValidationResult:
        """
        Validate that an overruling relationship is temporally possible.

        A case can only be overruled by a LATER decision. This prevents
        impossible relationships like "1914 case overruled 1925 case".

        Research shows LLMs create these impossible relationships ~27% of the time
        (Zhang et al., 2025 - "Do LLMs Truly Understand When a Precedent Is Overruled?").

        Args:
            original_citation: The case being overruled (must be earlier)
            overruling_citation: The case doing the overruling (must be later)
            overruling_date: Optional explicit date of overruling decision

        Returns:
            TemporalValidationResult with validation status and details
        """
        original_year = self._extract_year(original_citation)
        overruling_year = self._extract_year(overruling_citation)

        # If explicit date provided, use that year
        if overruling_date:
            overruling_year = overruling_date.year

        warnings: list[str] = []

        # Case 1: Cannot extract either year
        if original_year is None and overruling_year is None:
            return TemporalValidationResult(
                is_valid=True,  # Allow but warn
                original_year=None,
                overruling_year=None,
                warnings=[
                    "Could not extract years from either citation - temporal validation skipped"
                ],
            )

        # Case 2: Can only extract one year
        if original_year is None:
            return TemporalValidationResult(
                is_valid=True,
                original_year=None,
                overruling_year=overruling_year,
                warnings=[f"Could not extract year from original citation: {original_citation}"],
            )

        if overruling_year is None:
            return TemporalValidationResult(
                is_valid=True,
                original_year=original_year,
                overruling_year=None,
                warnings=[
                    f"Could not extract year from overruling citation: {overruling_citation}"
                ],
            )

        # Case 3: Temporal impossibility - overruling case is EARLIER than original
        if overruling_year < original_year:
            error_msg = (
                f"Temporal impossibility: {overruling_citation} ({overruling_year}) "
                f"cannot overrule {original_citation} ({original_year}) - "
                f"a case cannot be overruled by an earlier decision"
            )
            logger.warning(
                "temporal_validation_failed",
                original_citation=original_citation,
                original_year=original_year,
                overruling_citation=overruling_citation,
                overruling_year=overruling_year,
                error="overruling_case_predates_original",
            )
            return TemporalValidationResult(
                is_valid=False,
                original_year=original_year,
                overruling_year=overruling_year,
                error_message=error_msg,
            )

        # Case 4: Same year - valid but add warning
        if overruling_year == original_year:
            warnings.append(
                f"Both cases from same year ({original_year}) - "
                "ensure overruling decision was actually later in the year"
            )

        # Case 5: Suspiciously quick overruling (same or next year)
        year_gap = overruling_year - original_year
        if 0 < year_gap <= 1:
            warnings.append(
                f"Unusually quick overruling ({year_gap} year gap) - "
                "verify this is a true overruling and not a distinction"
            )

        # Case 6: Very old original case with modern overruling
        current_year = datetime.now().year
        original_age = current_year - original_year
        if original_age > 100:
            warnings.append(
                f"Original case is {original_age} years old - "
                "verify overruling is still the current status"
            )

        return TemporalValidationResult(
            is_valid=True,
            original_year=original_year,
            overruling_year=overruling_year,
            warnings=warnings,
        )

    def _detect_status_from_context(
        self,
        citation: str,
        context: str,
    ) -> tuple:
        """
        Detect precedent status from context text.

        Returns:
            Tuple of (status, related_citations)
        """
        context_lower = context.lower()
        citation_lower = citation.lower()

        # Check for overruled patterns near the citation
        for pattern in self.OVERRULED_PATTERNS:
            regex = (
                f"({re.escape(citation_lower)}.*{pattern}|{pattern}.*{re.escape(citation_lower)})"
            )
            if re.search(regex, context_lower, re.IGNORECASE):
                # Try to extract the overruling case
                related = self._extract_related_citations(context, citation)
                return PrecedentStatus.OVERRULED, related

        # Check for distinguished patterns
        for pattern in self.DISTINGUISHED_PATTERNS:
            regex = (
                f"({re.escape(citation_lower)}.*{pattern}|{pattern}.*{re.escape(citation_lower)})"
            )
            if re.search(regex, context_lower, re.IGNORECASE):
                related = self._extract_related_citations(context, citation)
                return PrecedentStatus.DISTINGUISHED, related

        # Check for affirmed patterns
        for pattern in self.AFFIRMED_PATTERNS:
            regex = (
                f"({re.escape(citation_lower)}.*{pattern}|{pattern}.*{re.escape(citation_lower)})"
            )
            if re.search(regex, context_lower, re.IGNORECASE):
                related = self._extract_related_citations(context, citation)
                return PrecedentStatus.AFFIRMED, related

        return PrecedentStatus.GOOD_LAW, []

    def _extract_related_citations(
        self,
        context: str,
        exclude_citation: str,
    ) -> list[str]:
        """Extract related citations from context, excluding the original."""
        citations = []

        for pattern in self._citation_regexes():
            for match in pattern.finditer(context):
                cit = match.group(0)
                if cit.lower() != exclude_citation.lower():
                    citations.append(cit)

        return citations[:5]  # Limit to 5 related citations

    async def extract_citations(self, text: str) -> list[str]:
        """
        Extract legal citations from text.

        Recognizes the citation forms listed under ``citation_patterns`` in the
        data file for the tracker's citation groups.

        Args:
            text: Text to extract citations from

        Returns:
            List of unique citations found
        """
        citations: set[str] = set()

        for pattern in self._citation_regexes():
            for match in pattern.finditer(text):
                citations.add(match.group(0).strip())

        return list(citations)

    def register_overruling(
        self,
        original_citation: str,
        overruling_citation: str,
        overruling_date: date,
        court: str,
        description: str,
        skip_temporal_validation: bool = False,
    ) -> TemporalValidationResult:
        """
        Register a case as overruled with temporal validation.

        This can be called to update the known overruled cases database.
        In production, this would update a persistent store.

        Validates that the overruling relationship is temporally possible
        (overruling case must be later than overruled case).

        Args:
            original_citation: The citation being overruled
            overruling_citation: The citation that overruled it
            overruling_date: Date of overruling
            court: Court that overruled
            description: Description of overruling
            skip_temporal_validation: Skip validation (use with caution)

        Returns:
            TemporalValidationResult with validation status

        Raises:
            ValueError: If temporal validation fails and skip_temporal_validation is False
        """
        # Validate temporal relationship first
        validation = self.validate_temporal_relationship(
            original_citation=original_citation,
            overruling_citation=overruling_citation,
            overruling_date=overruling_date,
        )

        if not validation.is_valid and not skip_temporal_validation:
            logger.error(
                "overruling_registration_rejected",
                original=original_citation,
                overruling=overruling_citation,
                reason=validation.error_message,
            )
            raise ValueError(validation.error_message)

        # Log warnings if any
        for warning in validation.warnings:
            logger.warning(
                "overruling_registration_warning",
                original=original_citation,
                overruling=overruling_citation,
                warning=warning,
            )

        normalized = self._normalize_citation(original_citation)

        self.known_overruled_cases[normalized] = (
            PrecedentStatus.OVERRULED,
            overruling_citation,
            overruling_date,
            description,
        )

        # Clear cache for this citation
        if normalized in self._cache:
            del self._cache[normalized]

        logger.info(
            "overruling_registered",
            original=original_citation,
            overruling=overruling_citation,
            court=court,
            original_year=validation.original_year,
            overruling_year=validation.overruling_year,
        )

        return validation

    def clear_cache(self):
        """Clear the validity cache."""
        self._cache.clear()


# Singleton instance
_default_tracker: PrecedentTracker | None = None


def get_precedent_tracker() -> PrecedentTracker:
    """Get or create default precedent tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = PrecedentTracker()
    return _default_tracker
