"""
Legal section detection for court judgments.

Extracts section boundaries from legal documents using pattern matching:
- HEADNOTE: Case summary, parties, judges, citation
- JUDGMENT: Court reasoning and analysis
- ORDER: Final court decision and directions
- OPERATIVE_PART: Actionable directives

This module is used as a pre-processing layer before semantic chunking
to preserve legal document structure and enable section-specific retrieval.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from legal_rag.chunking._logging import get_logger

logger = get_logger("legal_sections")


class LegalSection(Enum):
    """Judgment section types with priority scoring."""

    HEADNOTE = "headnote"
    JUDGMENT = "judgment"
    ORDER = "order"
    OPERATIVE_PART = "operative_part"
    UNKNOWN = "unknown"


# Section priority scores for retrieval boosting (0-100)
# Higher scores = more important for legal analysis
SECTION_PRIORITY_MAP = {
    LegalSection.OPERATIVE_PART: 100,  # Most important - final binding decision
    LegalSection.JUDGMENT: 90,  # Main legal reasoning and analysis
    LegalSection.ORDER: 85,  # Court order and directions
    LegalSection.HEADNOTE: 60,  # Case summary and metadata
    LegalSection.UNKNOWN: 30,  # Unclassified content
}


def get_section_priority(section: LegalSection) -> int:
    """
    Get priority score for a legal section (0-100).

    Higher scores indicate more importance for legal analysis and retrieval.
    Use these scores for:
    - Retrieval boosting in a vector store
    - Ranking search results
    - Filtering by section importance

    Args:
        section: LegalSection enum value

    Returns:
        Priority score (0-100)
    """
    return SECTION_PRIORITY_MAP.get(section, 30)


@dataclass
class SectionBoundary:
    """Detected section boundary in legal document."""

    section: LegalSection
    start_char: int
    end_char: int
    content: str
    confidence: float  # 0.0-1.0 confidence in section detection
    title: str | None = None  # Optional section title


class LegalSectionDetector:
    """
    Detects legal document section boundaries using pattern matching.

    Designed for court judgments with a conventional structure:
    - Appellate and supreme court opinions
    - Trial court orders
    - Tribunal decisions

    Example:
        detector = LegalSectionDetector()
        sections = detector.detect("IN THE SUPREME COURT OF THE UNITED STATES...")

        for section in sections:
            print(f"{section.section.value}: {len(section.content)} chars")
    """

    # Section detection patterns - comprehensive patterns for court judgments
    SECTION_PATTERNS: ClassVar[dict[LegalSection, list[re.Pattern[str]]]] = {
        LegalSection.HEADNOTE: [
            # Case header patterns
            re.compile(r"^HEADNOTE\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^IN THE (?:SUPREME COURT|HIGH COURT)", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^(?:CIVIL|CRIMINAL) APPEAL NO\.", re.MULTILINE | re.IGNORECASE),
            # Parties section
            re.compile(r"^BETWEEN\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^APPELLANT(?:\(S\))?\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^PETITIONER(?:\(S\))?\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^VERSUS", re.MULTILINE | re.IGNORECASE),
            # Bench line
            re.compile(r"^BEFORE\s*[:\-]", re.MULTILINE | re.IGNORECASE),
        ],
        LegalSection.JUDGMENT: [
            # Judgment header
            re.compile(r"^JUDGMENT\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^J\s*U\s*D\s*G\s*M\s*E\s*N\s*T", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^J\s*U\s*D\s*G\s*E\s*M\s*E\s*N\s*T", re.MULTILINE | re.IGNORECASE),
            # Judgment attribution
            re.compile(r"^(?:delivered by|per)\s+(?:Justice|J\.)", re.MULTILINE | re.IGNORECASE),
            re.compile(
                r"^(?:Justice|J\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[:\-]",
                re.MULTILINE | re.IGNORECASE,
            ),
            # Judgment opening
            re.compile(
                r"^\d+\.\s+(?:This|The present|The)\s+(?:appeal|petition|case|matter)",
                re.MULTILINE | re.IGNORECASE,
            ),
        ],
        LegalSection.ORDER: [
            # Order header
            re.compile(r"^ORDER\s*[:\-]", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^O\s*R\s*D\s*E\s*R", re.MULTILINE | re.IGNORECASE),
            # Order conclusion phrases
            re.compile(r"^For the foregoing reasons", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^In view of the above", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^In the light of the above", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^Having considered the submissions", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^In the result", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^For the reasons stated above", re.MULTILINE | re.IGNORECASE),
        ],
        LegalSection.OPERATIVE_PART: [
            # Dispositive phrases
            re.compile(
                r"^(?:The appeal is|Appeals are)\s+(?:allowed|dismissed|partly allowed)",
                re.MULTILINE | re.IGNORECASE,
            ),
            re.compile(
                r"^(?:The petition is|Petitions are)\s+(?:allowed|dismissed|disposed)",
                re.MULTILINE | re.IGNORECASE,
            ),
            re.compile(
                r"^(?:We|The Court)\s+(?:direct|order|hold)",
                re.MULTILINE | re.IGNORECASE,
            ),
            re.compile(r"^Consequently,", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^Accordingly,", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^(?:It is hereby ordered|We order) that", re.MULTILINE | re.IGNORECASE),
        ],
    }

    def detect(self, text: str) -> list[SectionBoundary]:
        """
        Detect section boundaries in legal document.

        Args:
            text: Full document text

        Returns:
            List of detected section boundaries with confidence scores
        """
        if not text or not text.strip():
            return []

        # Track all pattern matches with their positions
        matches = []

        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "section": section_type,
                            "start": match.start(),
                            "end": match.end(),
                            "matched_text": match.group(),
                            "pattern": pattern.pattern,
                        }
                    )

        if not matches:
            # No sections detected - treat entire document as UNKNOWN
            logger.debug("section_detection_no_matches", text_length=len(text))
            return [
                SectionBoundary(
                    section=LegalSection.UNKNOWN,
                    start_char=0,
                    end_char=len(text),
                    content=text,
                    confidence=0.5,
                )
            ]

        # Sort matches by position
        matches.sort(key=lambda x: x["start"])

        # Build section boundaries
        boundaries = self._build_boundaries(matches, text)

        logger.info(
            "section_detection_completed",
            section_count=len(boundaries),
            section_types=[b.section.value for b in boundaries],
            avg_confidence=round(
                sum(b.confidence for b in boundaries) / len(boundaries) if boundaries else 0,
                3,
            ),
        )

        return boundaries

    def _build_boundaries(
        self,
        matches: list[dict],
        text: str,
    ) -> list[SectionBoundary]:
        """Build section boundaries from pattern matches."""
        boundaries = []
        current_section = None
        section_start = 0
        section_matches = []

        for i, match in enumerate(matches):
            # Start new section if:
            # 1. First match
            # 2. Different section type than current
            # 3. Significant gap from previous match (>200 chars)
            is_new_section = (
                current_section is None
                or match["section"] != current_section
                or (i > 0 and match["start"] - matches[i - 1]["end"] > 200)
            )

            if is_new_section:
                # Save previous section if exists
                if current_section is not None and section_matches:
                    confidence = self._calculate_confidence(
                        section_matches, text[section_start : match["start"]]
                    )

                    boundaries.append(
                        SectionBoundary(
                            section=current_section,
                            start_char=section_start,
                            end_char=match["start"],
                            content=text[section_start : match["start"]],
                            confidence=confidence,
                            title=section_matches[0]["matched_text"].strip(),
                        )
                    )

                # Start new section
                current_section = match["section"]
                section_start = match["start"]
                section_matches = [match]
            else:
                section_matches.append(match)

        # Don't forget the last section
        if current_section is not None and section_matches:
            confidence = self._calculate_confidence(section_matches, text[section_start:])

            boundaries.append(
                SectionBoundary(
                    section=current_section,
                    start_char=section_start,
                    end_char=len(text),
                    content=text[section_start:],
                    confidence=confidence,
                    title=section_matches[0]["matched_text"].strip(),
                )
            )

        return boundaries

    def _calculate_confidence(self, section_matches: list[dict], content: str) -> float:
        """Calculate confidence score for section detection."""
        if not section_matches:
            return 0.5

        # Base confidence from number of matches
        match_confidence = min(len(section_matches) * 0.3, 0.6)

        # Bonus for header position (first 500 chars)
        first_match_pos = section_matches[0]["start"]
        position_confidence = 0.2 if first_match_pos < 500 else 0.1

        # Bonus for reasonable content length
        content_len = len(content.strip())
        length_confidence = 0.2 if 100 < content_len < 50000 else 0.1

        total_confidence = match_confidence + position_confidence + length_confidence
        return min(total_confidence, 1.0)


# Singleton instance
_detector: LegalSectionDetector | None = None


def get_legal_section_detector() -> LegalSectionDetector:
    """Get legal section detector singleton."""
    global _detector
    if _detector is None:
        _detector = LegalSectionDetector()
    return _detector
