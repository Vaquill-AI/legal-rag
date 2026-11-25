"""
Universal Document Type Detection and Section Patterns.

Supports multiple legal document categories:
- JUDGMENT: Court judgments (US, UK, other common-law courts)
- CONTRACT: Commercial contracts, agreements, NDAs
- STATUTE: Acts, rules, regulations, ordinances
- ARBITRATION: Awards, claims, procedural documents
- CORPORATE: Articles, bylaws, resolutions, minutes
- REGULATORY: Agency filings, notices and compliance documents
- IP: Patents, trademarks, licensing
- REAL_ESTATE: Deeds, leases, mortgages
- ESTATE: Wills, trusts, powers of attorney

Architecture:
    Text → DocumentTypeClassifier.classify() → DocumentType
                                                    ↓
    Text → UniversalSectionDetector.detect(doc_type) → [SectionBoundary...]

Cost: $0.00 per document (pattern-based, no API calls)
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from legal_rag.chunking._logging import get_logger

logger = get_logger("document_types")


class DocumentType(Enum):
    """
    Legal document type classification.

    Each type has associated section patterns for optimal chunking.
    """

    # Litigation
    JUDGMENT = "judgment"  # Court judgments, orders
    PLEADING = "pleading"  # Complaints, petitions, answers

    # Transactional
    CONTRACT = "contract"  # Agreements, NDAs, SLAs
    MNA = "mna"  # M&A documents, SPAs, APAs

    # Legislative
    STATUTE = "statute"  # Acts, rules, regulations
    CIRCULAR = "circular"  # Agency circulars, bulletins, notices

    # Dispute Resolution
    ARBITRATION = "arbitration"  # Awards, claims, procedural docs

    # Corporate
    CORPORATE = "corporate"  # Articles, bylaws, resolutions, minutes

    # Regulatory
    REGULATORY = "regulatory"  # Agency filings, compliance docs

    # Intellectual Property
    IP = "ip"  # Patents, trademarks, IP agreements

    # Property
    REAL_ESTATE = "real_estate"  # Deeds, leases, mortgages

    # Estate Planning
    ESTATE = "estate"  # Wills, trusts, POA

    # Fallback
    UNKNOWN = "unknown"


_PATTERN_FILE = Path(__file__).parent / "data" / "jurisdiction_patterns.json"
_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


@lru_cache(maxsize=1)
def _load_pattern_file() -> dict[str, Any]:
    with _PATTERN_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def available_pattern_packs() -> list[str]:
    """Names of the jurisdiction pattern packs shipped in the data file."""
    return sorted(_load_pattern_file()["packs"])


def load_pattern_packs(
    jurisdictions: Sequence[str] | None = None,
) -> dict["DocumentType", list[tuple[re.Pattern[str], float]]]:
    """Compile the classification patterns of the requested packs.

    Args:
        jurisdictions: Pack names. ``None`` selects the file's ``default_packs``.

    Raises:
        KeyError: If a requested pack does not exist.
    """
    data = _load_pattern_file()
    names = list(data["default_packs"] if jurisdictions is None else jurisdictions)
    compiled: dict[DocumentType, list[tuple[re.Pattern[str], float]]] = {}
    for name in names:
        if name not in data["packs"]:
            raise KeyError(f"Unknown jurisdiction pattern pack: {name!r}")
        for type_name, rows in data["packs"][name]["classification"].items():
            doc_type = DocumentType(type_name)
            for pattern, weight, flags in rows:
                flag_bits = 0
                for ch in flags:
                    flag_bits |= _FLAGS[ch]
                compiled.setdefault(doc_type, []).append((re.compile(pattern, flag_bits), weight))
    return compiled


class LegalSectionType(Enum):
    """
    Universal section types across all document categories.

    Normalized section types for consistent retrieval and UI display.
    """

    # Judgment sections
    HEADNOTE = "headnote"
    JUDGMENT = "judgment"
    ORDER = "order"
    OPERATIVE_PART = "operative_part"
    DISSENT = "dissent"
    CONCURRENCE = "concurrence"

    # Contract sections
    RECITALS = "recitals"
    DEFINITIONS = "definitions"
    TERMS = "terms"
    REPRESENTATIONS = "representations"
    WARRANTIES = "warranties"
    COVENANTS = "covenants"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    DISPUTE_RESOLUTION = "dispute_resolution"
    MISCELLANEOUS = "miscellaneous"
    SCHEDULES = "schedules"
    EXHIBITS = "exhibits"

    # Statute sections
    PREAMBLE = "preamble"
    CHAPTER = "chapter"
    PART = "part"
    SECTION = "section"
    SUBSECTION = "subsection"
    PROVISO = "proviso"
    EXPLANATION = "explanation"
    SCHEDULE = "schedule"

    # Arbitration sections
    TRIBUNAL = "tribunal"
    PROCEDURAL_HISTORY = "procedural_history"
    FACTS = "facts"
    CLAIMS = "claims"
    ANALYSIS = "analysis"
    AWARD = "award"
    COSTS = "costs"

    # Corporate sections
    ARTICLES = "articles"
    RESOLUTIONS = "resolutions"
    ATTENDANCE = "attendance"
    AGENDA = "agenda"
    MINUTES = "minutes"

    # IP sections
    ABSTRACT = "abstract"
    SPECIFICATION = "specification"
    DRAWINGS = "drawings"
    PRIOR_ART = "prior_art"

    # Real Estate sections
    PARTIES = "parties"
    PROPERTY = "property"
    CONSIDERATION = "consideration"
    ENCUMBRANCES = "encumbrances"

    # Estate sections
    DECLARATION = "declaration"
    BEQUESTS = "bequests"
    EXECUTOR = "executor"
    RESIDUARY = "residuary"
    POWERS = "powers"

    # Universal
    UNKNOWN = "unknown"


# Section priority scores by type (0-100)
# Higher = more important for retrieval
SECTION_PRIORITY_MAP: dict[LegalSectionType, int] = {
    # Judgment - highest priority to operative/order
    LegalSectionType.OPERATIVE_PART: 100,
    LegalSectionType.JUDGMENT: 90,
    LegalSectionType.ORDER: 85,
    LegalSectionType.DISSENT: 75,
    LegalSectionType.CONCURRENCE: 70,
    LegalSectionType.HEADNOTE: 60,
    # Contract - highest priority to substantive terms
    LegalSectionType.INDEMNIFICATION: 95,
    LegalSectionType.REPRESENTATIONS: 90,
    LegalSectionType.WARRANTIES: 90,
    LegalSectionType.COVENANTS: 85,
    LegalSectionType.TERMINATION: 85,
    LegalSectionType.TERMS: 80,
    LegalSectionType.DISPUTE_RESOLUTION: 75,
    LegalSectionType.DEFINITIONS: 70,
    LegalSectionType.RECITALS: 50,
    LegalSectionType.MISCELLANEOUS: 40,
    LegalSectionType.SCHEDULES: 60,
    LegalSectionType.EXHIBITS: 60,
    # Statute - highest priority to main sections
    LegalSectionType.SECTION: 90,
    LegalSectionType.SUBSECTION: 85,
    LegalSectionType.PROVISO: 80,
    LegalSectionType.EXPLANATION: 75,
    LegalSectionType.CHAPTER: 70,
    LegalSectionType.PART: 65,
    LegalSectionType.PREAMBLE: 50,
    LegalSectionType.SCHEDULE: 60,
    # Arbitration - highest priority to award/analysis
    LegalSectionType.AWARD: 100,
    LegalSectionType.ANALYSIS: 90,
    LegalSectionType.CLAIMS: 85,
    LegalSectionType.FACTS: 75,
    LegalSectionType.COSTS: 70,
    LegalSectionType.PROCEDURAL_HISTORY: 60,
    LegalSectionType.TRIBUNAL: 50,
    # Corporate
    LegalSectionType.RESOLUTIONS: 90,
    LegalSectionType.ARTICLES: 80,
    LegalSectionType.MINUTES: 75,
    LegalSectionType.AGENDA: 60,
    LegalSectionType.ATTENDANCE: 40,
    # IP
    LegalSectionType.CLAIMS: 95,  # Patent claims are critical
    LegalSectionType.SPECIFICATION: 85,
    LegalSectionType.ABSTRACT: 70,
    LegalSectionType.DRAWINGS: 50,
    LegalSectionType.PRIOR_ART: 75,
    # Real Estate
    LegalSectionType.PROPERTY: 90,
    LegalSectionType.CONSIDERATION: 85,
    LegalSectionType.ENCUMBRANCES: 80,
    LegalSectionType.PARTIES: 60,
    # Estate
    LegalSectionType.BEQUESTS: 95,
    LegalSectionType.EXECUTOR: 85,
    LegalSectionType.RESIDUARY: 80,
    LegalSectionType.POWERS: 85,
    LegalSectionType.DECLARATION: 60,
    # Universal
    LegalSectionType.UNKNOWN: 30,
}


def get_section_priority(section: LegalSectionType) -> int:
    """Get priority score for a section type (0-100)."""
    return SECTION_PRIORITY_MAP.get(section, 30)


@dataclass
class UniversalSectionBoundary:
    """
    Detected section boundary with universal type information.

    Extends SectionBoundary with document type context.
    """

    section_type: LegalSectionType
    document_type: DocumentType
    start_char: int
    end_char: int
    content: str
    confidence: float  # 0.0-1.0
    title: str | None = None
    hierarchy_level: int = 0  # 0=top level, 1=subsection, etc.
    parent_section: str | None = None  # For nested sections

    @property
    def priority(self) -> int:
        """Get priority score for retrieval boosting."""
        return get_section_priority(self.section_type)


class DocumentTypeClassifier:
    """
    Classifies legal documents by type using pattern matching.

    Fast, zero-cost classification to route to appropriate section detector.
    """

    # Classification patterns by document type
    # Higher score = stronger match
    CLASSIFICATION_PATTERNS: ClassVar[dict[DocumentType, list[tuple[re.Pattern[str], float]]]] = {
        DocumentType.JUDGMENT: [
            # Generic court headings (jurisdiction-specific forms live in the pattern packs)
            (re.compile(r"IN THE (?:SUPREME COURT|HIGH COURT)", re.IGNORECASE), 0.9),
            (re.compile(r"^JUDGMENT\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0.9),
            (re.compile(r"^ORDER\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0.6),
            # US courts
            (
                re.compile(r"UNITED STATES (?:DISTRICT|CIRCUIT|BANKRUPTCY) COURT", re.IGNORECASE),
                0.9,
            ),
            (re.compile(r"SUPREME COURT OF THE UNITED STATES", re.IGNORECASE), 0.95),
            (re.compile(r"Case No\.?\s*\d+[:-]cv[:-]\d+", re.IGNORECASE), 0.7),
            (re.compile(r"OPINION (?:AND ORDER|OF THE COURT)", re.IGNORECASE), 0.85),
            # UK courts
            (re.compile(r"IN THE (?:HIGH COURT OF JUSTICE|COURT OF APPEAL)", re.IGNORECASE), 0.95),
            (re.compile(r"QUEEN'S BENCH|KING'S BENCH|CHANCERY DIVISION", re.IGNORECASE), 0.9),
            (re.compile(r"MR\s+JUSTICE|MRS?\s+JUSTICE", re.IGNORECASE), 0.85),
            (re.compile(r"\[20\d{2}\] (?:UKSC|EWCA|EWHC)", re.IGNORECASE), 0.85),
        ],
        DocumentType.CONTRACT: [
            # Specific contract types - highly distinctive (weight 0.95+)
            (
                re.compile(
                    r"NON-?DISCLOSURE AGREEMENT|\bNDA\b|CONFIDENTIALITY AGREEMENT", re.IGNORECASE
                ),
                0.98,
            ),
            (re.compile(r"SERVICE (?:LEVEL )?AGREEMENT|\bSLA\b", re.IGNORECASE), 0.95),
            (re.compile(r"EMPLOYMENT (?:AGREEMENT|CONTRACT)", re.IGNORECASE), 0.95),
            (re.compile(r"CONSULTING AGREEMENT", re.IGNORECASE), 0.95),
            (re.compile(r"MASTER (?:SERVICE )?AGREEMENT|\bMSA\b", re.IGNORECASE), 0.95),
            (re.compile(r"LICENSE AGREEMENT", re.IGNORECASE), 0.9),
            # Agreement headers - moderately distinctive (reduced weights)
            (
                re.compile(
                    r"(?:THIS )?(?:AGREEMENT|CONTRACT) (?:IS )?(?:MADE|ENTERED)", re.IGNORECASE
                ),
                0.75,
            ),
            (
                re.compile(r"^WHEREAS[,:]", re.MULTILINE | re.IGNORECASE),
                0.65,
            ),  # Reduced - appears in many docs
            (
                re.compile(r"NOW,?\s*THEREFORE", re.IGNORECASE),
                0.65,
            ),  # Reduced - appears in many docs
            (
                re.compile(r"IN WITNESS WHEREOF", re.IGNORECASE),
                0.6,
            ),  # Reduced - appears in many docs
            # Contract sections - lower weight as these appear in M&A too
            (
                re.compile(r"^ARTICLE\s+[IVXLC\d]+[\.\s:\-]", re.MULTILINE | re.IGNORECASE),
                0.55,
            ),  # Reduced
            (
                re.compile(r"REPRESENTATIONS AND WARRANTIES", re.IGNORECASE),
                0.7,
            ),  # Reduced - common in M&A
            (re.compile(r"INDEMNIFICATION", re.IGNORECASE), 0.55),  # Reduced - common in M&A
            (re.compile(r"GOVERNING LAW", re.IGNORECASE), 0.5),
            (re.compile(r"TERM AND TERMINATION", re.IGNORECASE), 0.7),
        ],
        DocumentType.MNA: [
            # M&A specific - highly distinctive (weight 0.95+)
            (re.compile(r"SHARE PURCHASE AGREEMENT|SPA", re.IGNORECASE), 0.98),
            (re.compile(r"ASSET PURCHASE AGREEMENT|APA", re.IGNORECASE), 0.98),
            (re.compile(r"MERGER AGREEMENT", re.IGNORECASE), 0.98),
            (re.compile(r"STOCK PURCHASE AGREEMENT", re.IGNORECASE), 0.98),
            (re.compile(r"ACQUISITION AGREEMENT", re.IGNORECASE), 0.95),
            # M&A specific terms
            (re.compile(r"DISCLOSURE SCHEDULE", re.IGNORECASE), 0.9),
            (re.compile(r"REPRESENTATIONS.{0,30}(?:SELLER|TARGET|COMPANY)", re.IGNORECASE), 0.9),
            (re.compile(r"CONDITIONS (?:TO )?(?:PRECEDENT|CLOSING)", re.IGNORECASE), 0.9),
            (re.compile(r"PURCHASE PRICE ADJUSTMENT", re.IGNORECASE), 0.9),
            (re.compile(r"BREAK(?:-?UP)? FEE", re.IGNORECASE), 0.9),
            # M&A moderately distinctive
            (re.compile(r"TERM SHEET", re.IGNORECASE), 0.8),
            (re.compile(r"LETTER OF INTENT|LOI", re.IGNORECASE), 0.8),
            (re.compile(r"ESCROW", re.IGNORECASE), 0.7),
            (re.compile(r"EARNOUT", re.IGNORECASE), 0.85),
            (re.compile(r"WORKING CAPITAL ADJUSTMENT", re.IGNORECASE), 0.85),
        ],
        DocumentType.STATUTE: [
            # Act title patterns
            (re.compile(r"THE\s+[\w\s]+ACT,?\s*(?:19|20)\d{2}", re.IGNORECASE), 0.95),
            (re.compile(r"(?:AN )?ACT (?:TO |OF |FOR )", re.IGNORECASE), 0.85),
            # Enactment clauses
            (re.compile(r"BE IT ENACTED BY", re.IGNORECASE), 0.95),
            (re.compile(r"ENACTED BY (?:PARLIAMENT|LEGISLATURE)", re.IGNORECASE), 0.95),
            # Structure patterns
            (re.compile(r"^CHAPTER\s+[IVXLC\d]+", re.MULTILINE | re.IGNORECASE), 0.8),
            (re.compile(r"CHAPTER\s+[IVXLC\d]+\s*[-\u2013\u2014:]\s*\w+", re.IGNORECASE), 0.85),
            (re.compile(r"^PART\s+[IVXLC\d]+", re.MULTILINE | re.IGNORECASE), 0.7),
            (re.compile(r"^Section\s+\d+[\.\s:\-]", re.MULTILINE | re.IGNORECASE), 0.75),
            (
                re.compile(
                    r"^\d+\.\s+(?:Short title|Definitions|Commencement)",
                    re.MULTILINE | re.IGNORECASE,
                ),
                0.8,
            ),
            (re.compile(r"^SCHEDULE\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0.7),
            (re.compile(r"PRELIMINARY", re.IGNORECASE), 0.6),
            (re.compile(r"Provided that", re.IGNORECASE), 0.5),
            (re.compile(r"Explanation[\.\s:\-]", re.IGNORECASE), 0.5),
            # Rules/Regulations
            (re.compile(r"(?:THE|THESE) (?:\w+ )+RULES,?\s*\d{4}", re.IGNORECASE), 0.9),
            (re.compile(r"(?:THE|THESE) (?:\w+ )+REGULATIONS,?\s*\d{4}", re.IGNORECASE), 0.9),
        ],
        DocumentType.ARBITRATION: [
            # Arbitration awards
            (re.compile(r"ARBITR(?:AL|ATION)\s+(?:AWARD|TRIBUNAL)", re.IGNORECASE), 0.95),
            (re.compile(r"PARTIAL (?:FINAL )?AWARD", re.IGNORECASE), 0.9),
            (re.compile(r"FINAL AWARD", re.IGNORECASE), 0.9),
            (re.compile(r"\bAWARD\b.*(?:CLAIMANT|RESPONDENT)", re.IGNORECASE), 0.85),
            # Arbitration institutions
            (re.compile(r"(?:ICC|LCIA|SIAC|HKIAC|ICDR|AAA|UNCITRAL)", re.IGNORECASE), 0.7),
            (re.compile(r"INTERNATIONAL (?:CHAMBER OF COMMERCE|ARBITRATION)", re.IGNORECASE), 0.8),
            (re.compile(r"LONDON COURT OF INTERNATIONAL ARBITRATION", re.IGNORECASE), 0.85),
            (re.compile(r"SINGAPORE INTERNATIONAL ARBITRATION CENTRE", re.IGNORECASE), 0.85),
            # Arbitration structure
            (re.compile(r"(?:^|\n)(?:THE )?TRIBUNAL", re.IGNORECASE), 0.75),
            (re.compile(r"CLAIMANT(?:\(S\))?(?:\s*:|.*(?:SEEKS|CLAIMS))", re.IGNORECASE), 0.8),
            (re.compile(r"RESPONDENT(?:\(S\))?\s*:", re.IGNORECASE), 0.75),
            (re.compile(r"PROCEDURAL HISTORY", re.IGNORECASE), 0.8),
            (re.compile(r"STATEMENT OF (?:CLAIM|DEFENCE)", re.IGNORECASE), 0.85),
            (re.compile(r"SEAT OF (?:THE )?ARBITRATION", re.IGNORECASE), 0.8),
        ],
        DocumentType.CORPORATE: [
            # Formation documents - highly distinctive (weight 0.95+)
            (re.compile(r"MEMORANDUM OF ASSOCIATION|MOA", re.IGNORECASE), 0.98),
            (re.compile(r"ARTICLES OF ASSOCIATION|AOA", re.IGNORECASE), 0.98),
            (re.compile(r"CERTIFICATE OF INCORPORATION", re.IGNORECASE), 0.95),
            (re.compile(r"ARTICLES OF INCORPORATION", re.IGNORECASE), 0.95),
            (re.compile(r"BY-?LAWS", re.IGNORECASE), 0.9),
            (re.compile(r"OPERATING AGREEMENT", re.IGNORECASE), 0.9),
            # Governance - highly distinctive
            (re.compile(r"BOARD (?:OF DIRECTORS )?RESOLUTION", re.IGNORECASE), 0.95),
            (re.compile(r"MINUTES OF (?:THE )?(?:BOARD )?MEETING", re.IGNORECASE), 0.95),
            (re.compile(r"ANNUAL GENERAL MEETING|AGM", re.IGNORECASE), 0.9),
            (re.compile(r"EXTRAORDINARY GENERAL MEETING|EGM", re.IGNORECASE), 0.9),
            (re.compile(r"^RESOLVED\s*(?:THAT|,)", re.MULTILINE | re.IGNORECASE), 0.9),
            (re.compile(r"FURTHER RESOLVED", re.IGNORECASE), 0.9),
            # Corporate governance terms
            (re.compile(r"SHAREHOLDER(?:S)?(?:'S)? AGREEMENT", re.IGNORECASE), 0.85),
            (re.compile(r"QUORUM", re.IGNORECASE), 0.7),
            (re.compile(r"VOTE OF THANKS", re.IGNORECASE), 0.8),
            (re.compile(r"MEMBERS? PRESENT", re.IGNORECASE), 0.75),
        ],
        DocumentType.REGULATORY: [
            # Disclosure rules
            (re.compile(r"DISCLOSURE.*?REGULATION", re.IGNORECASE), 0.8),
            # US regulatory
            (re.compile(r"FORM (?:10-K|10-Q|8-K|S-1|DEF 14A)", re.IGNORECASE), 0.95),
            (re.compile(r"SECURITIES AND EXCHANGE COMMISSION|SEC FILING", re.IGNORECASE), 0.7),
            (re.compile(r"ANNUAL REPORT PURSUANT TO SECTION 13", re.IGNORECASE), 0.9),
            # General regulatory
            (re.compile(r"COMPLIANCE (?:CERTIFICATE|REPORT)", re.IGNORECASE), 0.8),
            (re.compile(r"REGULATORY FILING", re.IGNORECASE), 0.75),
            (
                re.compile(r"(?:BSE|NSE)\s*(?:CODE|SYMBOL)", re.IGNORECASE),
                0.7,
            ),  # Stock exchange codes
        ],
        DocumentType.IP: [
            # Patents
            (
                re.compile(
                    r"(?:PATENT|UTILITY MODEL) (?:APPLICATION|SPECIFICATION)", re.IGNORECASE
                ),
                0.95,
            ),
            (re.compile(r"CLAIMS?\s*[:\.]?\s*(?:\n|$)", re.IGNORECASE), 0.7),
            (re.compile(r"FIELD OF (?:THE )?INVENTION", re.IGNORECASE), 0.9),
            (re.compile(r"BACKGROUND OF (?:THE )?INVENTION", re.IGNORECASE), 0.9),
            (re.compile(r"DETAILED DESCRIPTION", re.IGNORECASE), 0.8),
            (re.compile(r"(?:PATENT|APPLICATION) NO\.?\s*[A-Z]?\d+", re.IGNORECASE), 0.75),
            # Trademarks
            (re.compile(r"TRADEMARK (?:APPLICATION|REGISTRATION)", re.IGNORECASE), 0.95),
            (re.compile(r"(?:NICE|VIENNA) CLASSIFICATION", re.IGNORECASE), 0.85),
            # IP Agreements
            (
                re.compile(r"(?:IP|INTELLECTUAL PROPERTY) (?:ASSIGNMENT|LICENSE)", re.IGNORECASE),
                0.9,
            ),
            (re.compile(r"TECHNOLOGY (?:TRANSFER|LICENSE) AGREEMENT", re.IGNORECASE), 0.9),
        ],
        DocumentType.REAL_ESTATE: [
            # Deeds - highly distinctive (weight 0.98)
            (re.compile(r"(?:SALE|CONVEYANCE|GIFT|MORTGAGE) DEED", re.IGNORECASE), 0.98),
            (re.compile(r"DEED OF (?:SALE|CONVEYANCE|ASSIGNMENT|TRANSFER)", re.IGNORECASE), 0.98),
            (re.compile(r"QUITCLAIM DEED", re.IGNORECASE), 0.98),
            (re.compile(r"WARRANTY DEED", re.IGNORECASE), 0.98),
            (re.compile(r"LEASE DEED", re.IGNORECASE), 0.98),
            (re.compile(r"LEAVE AND LICENSE AGREEMENT", re.IGNORECASE), 0.98),
            # Real estate specific
            (re.compile(r"(?:ALL THAT|BEING) (?:PIECE|PARCEL) OF LAND", re.IGNORECASE), 0.95),
            (re.compile(r"SCHEDULE OF PROPERTY", re.IGNORECASE), 0.9),
            (re.compile(r"DEMISED PREMISES", re.IGNORECASE), 0.9),
            (re.compile(r"SQUARE (?:YARDS|METERS|FEET)", re.IGNORECASE), 0.85),
            (re.compile(r"ENCUMBRANCES?", re.IGNORECASE), 0.85),
            (re.compile(r"SALE CONSIDERATION", re.IGNORECASE), 0.9),
            # Property boundaries
            (
                re.compile(
                    r"(?:ON THE )?(?:NORTH|SOUTH|EAST|WEST)\s*[:\-]?\s*(?:BY|BOUNDED)",
                    re.IGNORECASE,
                ),
                0.9,
            ),
            (re.compile(r"(?:RENT|LEASE) AGREEMENT", re.IGNORECASE), 0.85),
            # RERA
            (re.compile(r"RERA (?:REGISTRATION|COMPLIANCE)", re.IGNORECASE), 0.95),
        ],
        DocumentType.ESTATE: [
            # Wills
            (re.compile(r"LAST WILL AND TESTAMENT", re.IGNORECASE), 0.95),
            (re.compile(r"THIS IS (?:MY|THE) (?:LAST )?WILL", re.IGNORECASE), 0.95),
            (re.compile(r"I,?\s+\w+(?:\s+\w+)*,?\s+(?:BEING|DO HEREBY)", re.IGNORECASE), 0.6),
            (re.compile(r"REVOKE ALL (?:PREVIOUS|FORMER) WILLS", re.IGNORECASE), 0.9),
            (re.compile(r"(?:APPOINT|NOMINATE).{0,30}EXECUTOR", re.IGNORECASE), 0.85),
            (re.compile(r"RESIDUARY (?:ESTATE|CLAUSE)", re.IGNORECASE), 0.8),
            # Trusts
            (re.compile(r"DECLARATION OF TRUST", re.IGNORECASE), 0.95),
            (re.compile(r"TRUST (?:DEED|AGREEMENT)", re.IGNORECASE), 0.95),
            (re.compile(r"SETTLOR|TRUSTOR", re.IGNORECASE), 0.8),
            (re.compile(r"BENEFICIAR(?:Y|IES)", re.IGNORECASE), 0.7),
            # POA
            (re.compile(r"POWER OF ATTORNEY|POA", re.IGNORECASE), 0.95),
            (
                re.compile(r"(?:GENERAL|SPECIAL|LIMITED|DURABLE) POWER OF ATTORNEY", re.IGNORECASE),
                0.95,
            ),
            (re.compile(r"ATTORNEY-?IN-?FACT", re.IGNORECASE), 0.85),
        ],
        DocumentType.PLEADING: [
            # Pleading headers - highly distinctive (weight 0.98)
            (re.compile(r"^PLAINT\s*$", re.MULTILINE | re.IGNORECASE), 0.98),
            (re.compile(r"COMPLAINT FOR", re.IGNORECASE), 0.95),
            (re.compile(r"CIVIL SUIT NO\.", re.IGNORECASE), 0.95),
            (re.compile(r"PRAYER FOR RELIEF", re.IGNORECASE), 0.95),
            (re.compile(r"WHEREFORE,?\s+(?:THE )?(?:PLAINTIFF|PETITIONER)", re.IGNORECASE), 0.95),
            (re.compile(r"RELIEF SOUGHT", re.IGNORECASE), 0.9),
            # Cause of action patterns
            (re.compile(r"(?:FIRST|SECOND|THIRD) CAUSE OF ACTION", re.IGNORECASE), 0.95),
            (re.compile(r"CAUSE OF ACTION", re.IGNORECASE), 0.9),
            # Pleading structure
            (re.compile(r"STATEMENT OF FACTS", re.IGNORECASE), 0.8),
            (re.compile(r"GENERAL ALLEGATIONS?", re.IGNORECASE), 0.85),
            (re.compile(r"FACTUAL (?:BACKGROUND|ALLEGATIONS)", re.IGNORECASE), 0.85),
            # Answers/Defenses
            (re.compile(r"ANSWER (?:TO|AND) (?:COMPLAINT|COUNTERCLAIM)", re.IGNORECASE), 0.95),
            (re.compile(r"AFFIRMATIVE DEFENSES?", re.IGNORECASE), 0.9),
            (re.compile(r"WRITTEN STATEMENT", re.IGNORECASE), 0.9),
            # Motions
            (re.compile(r"MOTION (?:TO|FOR) (?:DISMISS|SUMMARY JUDGMENT)", re.IGNORECASE), 0.95),
            (re.compile(r"MEMORANDUM (?:OF LAW )?IN SUPPORT", re.IGNORECASE), 0.9),
            (re.compile(r"VERIFICATION", re.IGNORECASE), 0.7),
        ],
        DocumentType.CIRCULAR: [
            # Circular headers - highly distinctive
            (re.compile(r"CIRCULAR NO\.?", re.IGNORECASE), 0.95),
            (re.compile(r"MASTER (?:CIRCULAR|DIRECTION)", re.IGNORECASE), 0.98),
            # Circular structure
            (re.compile(r"(?:DEPARTMENT OF|DOR\.)", re.IGNORECASE), 0.8),
            (re.compile(r"CENTRAL OFFICE", re.IGNORECASE), 0.75),
            (re.compile(r"(?:ALL )?(?:SCHEDULED )?COMMERCIAL BANKS", re.IGNORECASE), 0.9),
            (re.compile(r"PLEASE ACKNOWLEDGE RECEIPT", re.IGNORECASE), 0.85),
        ],
    }

    def __init__(self, jurisdictions: Sequence[str] | None = None) -> None:
        """
        Args:
            jurisdictions: Names of jurisdiction pattern packs to add on top of the
                built-in patterns. Defaults to the packs listed under
                ``default_packs`` in ``data/jurisdiction_patterns.json``.
        """
        self._patterns: dict[DocumentType, list[tuple[re.Pattern[str], float]]] = {
            doc_type: list(patterns) for doc_type, patterns in self.CLASSIFICATION_PATTERNS.items()
        }
        for doc_type, extra in load_pattern_packs(jurisdictions).items():
            self._patterns.setdefault(doc_type, []).extend(extra)

    def classify(self, text: str, threshold: float = 0.5) -> tuple[DocumentType, float]:
        """
        Classify document type based on pattern matching.

        Uses a scoring system that:
        1. Gives higher weight to more specific patterns
        2. Uses raw score sum as tie-breaker
        3. Considers number of high-confidence matches (weight >= 0.9)

        Args:
            text: Full document text (first 5000 chars used for efficiency)
            threshold: Minimum confidence to return non-UNKNOWN type

        Returns:
            Tuple of (DocumentType, confidence_score)
        """
        if not text or not text.strip():
            return DocumentType.UNKNOWN, 0.0

        # Use first 5000 chars for classification (efficiency)
        sample = text[:5000]

        # Scores track tuple of (normalized_score, raw_sum, high_confidence_count)
        scores: dict[DocumentType, tuple[float, float, int]] = {}

        for doc_type, patterns in self._patterns.items():
            type_score = 0.0
            matches = 0
            high_confidence_count = 0  # Patterns with weight >= 0.9

            for pattern, weight in patterns:
                if pattern.search(sample):
                    type_score += weight
                    matches += 1
                    if weight >= 0.9:
                        high_confidence_count += 1

            # Normalize by number of patterns (max 3 matches considered)
            if matches > 0:
                normalized = min(type_score / min(matches, 3), 1.0)
                scores[doc_type] = (normalized, type_score, high_confidence_count)

        if not scores:
            return DocumentType.UNKNOWN, 0.5

        # Sort by: 1) normalized score, 2) high confidence count, 3) raw sum
        sorted_types = sorted(
            scores.items(),
            key=lambda x: (x[1][0], x[1][2], x[1][1]),
            reverse=True,
        )

        best_type = sorted_types[0][0]
        best_score = sorted_types[0][1][0]  # normalized score

        if best_score < threshold:
            logger.debug(
                "document_classification_low_confidence",
                best_type=best_type.value,
                score=best_score,
                threshold=threshold,
            )
            return DocumentType.UNKNOWN, best_score

        logger.info(
            "document_classified",
            document_type=best_type.value,
            confidence=round(best_score, 3),
            runner_up=sorted_types[1][0].value if len(sorted_types) > 1 else None,
        )

        return best_type, best_score


# Rest of a heading number that a pattern stopped inside, for example the "101." of "Section 12-101."
_HEADING_NUMBER_TAIL = re.compile(r"[.\-]?[0-9A-Za-z(][0-9A-Za-z()]*(?:[.\-][0-9A-Za-z()]+)*\.?")


def _heading_title(text: str, match: re.Match[str]) -> str:
    """
    Return the full heading token for a matched heading.

    Patterns end on the first character that settles the heading, so a match can stop inside
    a compound number ("Section 12-" of "Section 12-101.", "Sec. 3." of "Sec. 3.4(b)").
    When the match ends on a number separator and more of the number follows, extend the
    title to the end of the number.
    """
    title = match.group()
    if title and title[-1] in "0123456789-.":
        tail = _HEADING_NUMBER_TAIL.match(text, match.end())
        if tail:
            title += tail.group()
    return title.strip()


class UniversalSectionDetector:
    """
    Detects section boundaries across all legal document types.

    Routes to type-specific patterns based on DocumentTypeClassifier result.
    Falls back to generic section detection for UNKNOWN types.

    Cost: $0.00 per document (pattern-based, no API calls)
    """

    # Section patterns organized by document type
    # Each pattern maps to (LegalSectionType, hierarchy_level)
    SECTION_PATTERNS: ClassVar[
        dict[DocumentType, dict[LegalSectionType, list[tuple[re.Pattern[str], int]]]]
    ] = {
        # ═══════════════════════════════════════════════════════════════════════
        # JUDGMENT patterns (US, UK and other common-law courts)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.JUDGMENT: {
            LegalSectionType.HEADNOTE: [
                (re.compile(r"^HEADNOTE\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^IN THE (?:SUPREME COURT|HIGH COURT)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^(?:CIVIL|CRIMINAL) APPEAL NO\.", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BEFORE\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BETWEEN\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:APPELLANT|PETITIONER|PLAINTIFF)(?:\(S\))?\s*[:\-]",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.JUDGMENT: [
                (
                    re.compile(r"^\s*JUDGMENT\s*[:\-]?$", re.MULTILINE | re.IGNORECASE),
                    0,
                ),  # Standalone "judgment" with optional leading whitespace
                (re.compile(r"^JUDGMENT\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^J\s*U\s*D\s*G\s*(?:M\s*)?E\s*N\s*T", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:delivered by|per)\s+(?:Justice|J\.)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^OPINION (?:AND ORDER|OF THE COURT)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^\d+\.\s+(?:This|The present|The)\s+(?:appeal|petition|case|matter)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.ORDER: [
                (
                    re.compile(r"^\s*ORDER\s*[:\-]?$", re.MULTILINE | re.IGNORECASE),
                    0,
                ),  # Standalone "order" with optional leading whitespace
                (re.compile(r"^ORDER\s*[:\-]", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^O\s*R\s*D\s*E\s*R", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^For the foregoing reasons", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^In view of the above", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^In the result", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.OPERATIVE_PART: [
                (
                    re.compile(
                        r"^(?:The appeal is|Appeals are)\s+(?:allowed|dismissed|partly allowed)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:The petition is|Petitions are)\s+(?:allowed|dismissed|disposed)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:We|The Court)\s+(?:direct|order|hold)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^IT IS SO ORDERED", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^Accordingly,", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.DISSENT: [
                (re.compile(r"^DISSENTING (?:OPINION|JUDGMENT)", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^I respectfully dissent", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:Justice|J\.)\s+\w+,?\s*dissenting", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.CONCURRENCE: [
                (re.compile(r"^CONCURRING (?:OPINION|JUDGMENT)", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^I concur", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:Justice|J\.)\s+\w+,?\s*concurring", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # CONTRACT patterns (agreements, NDAs, SLAs, employment, etc.)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.CONTRACT: {
            LegalSectionType.RECITALS: [
                (re.compile(r"^RECITALS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^WHEREAS[,:]", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BACKGROUND\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^PREAMBLE\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.DEFINITIONS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?DEFINITIONS?\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?INTERPRETATION\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?DEFINED TERMS\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r'"[A-Z][^"]+"\s+(?:means|shall mean|has the meaning)', re.IGNORECASE
                    ),
                    1,
                ),
            ],
            LegalSectionType.TERMS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?TERM\s+(?:AND|OF)\s+(?:AGREEMENT|CONTRACT)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?AGREEMENT PERIOD", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^NOW,?\s*THEREFORE", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?SCOPE OF (?:WORK|SERVICES|AGREEMENT)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.REPRESENTATIONS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?REPRESENTATIONS?\s*(?:AND WARRANTIES)?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?(?:MUTUAL )?REPRESENTATIONS", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.WARRANTIES: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?WARRANTIES?\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?(?:EXPRESS|IMPLIED) WARRANTIES?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?WARRANTY DISCLAIMER", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.COVENANTS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?COVENANTS?\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?(?:AFFIRMATIVE|NEGATIVE) COVENANTS?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?CONFIDENTIALITY", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?NON-?(?:COMPETE|SOLICITATION|DISCLOSURE)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.INDEMNIFICATION: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?INDEMNIFICATION\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(r"^(?:\d+[\.\s]+)?INDEMNITY\s*[:\.]?", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?HOLD HARMLESS", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?LIMITATION OF LIABILITY", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.TERMINATION: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?TERM(?:INATION)?\s+(?:AND|OF)?\s*(?:TERMINATION)?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?TERMINATION\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?EFFECT(?:S)? OF TERMINATION", re.MULTILINE | re.IGNORECASE
                    ),
                    1,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?SURVIVAL", re.MULTILINE | re.IGNORECASE), 1),
            ],
            LegalSectionType.DISPUTE_RESOLUTION: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?DISPUTE RESOLUTION\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?ARBITRATION\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?GOVERNING LAW\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?JURISDICTION\s*[:\.]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?CHOICE OF (?:LAW|FORUM)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.MISCELLANEOUS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?(?:MISCELLANEOUS|GENERAL PROVISIONS?)\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?ENTIRE AGREEMENT", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?AMENDMENT", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?WAIVER", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?SEVERABILITY", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?NOTICES?\s*[:\.]?", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?ASSIGNMENT", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?FORCE MAJEURE", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^IN WITNESS WHEREOF", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SCHEDULES: [
                (re.compile(r"^SCHEDULE\s+[A-Z\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(r"^ANNEX(?:URE)?\s+[A-Z\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (re.compile(r"^APPENDIX\s+[A-Z\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.EXHIBITS: [
                (re.compile(r"^EXHIBIT\s+[A-Z\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ATTACHMENT\s+[A-Z\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # M&A patterns (SPAs, APAs, mergers, term sheets)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.MNA: {
            LegalSectionType.RECITALS: [
                (re.compile(r"^RECITALS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^WHEREAS[,:]", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BACKGROUND\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.DEFINITIONS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?DEFINITIONS?\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.TERMS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?(?:THE )?(?:SALE|PURCHASE|MERGER|ACQUISITION)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?PURCHASE PRICE", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:\d+[\.\s]+)?CONSIDERATION", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:\d+[\.\s]+)?CLOSING\s*[:\.]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?CONDITIONS (?:TO )?(?:PRECEDENT|CLOSING)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?ESCROW", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:\d+[\.\s]+)?EARNOUT", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?PURCHASE PRICE ADJUSTMENT", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.REPRESENTATIONS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?REPRESENTATIONS?\s+(?:AND WARRANTIES )?OF\s+(?:THE )?SELLER",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?REPRESENTATIONS?\s+(?:AND WARRANTIES )?OF\s+(?:THE )?(?:BUYER|PURCHASER)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?REPRESENTATIONS?\s+(?:AND WARRANTIES )?OF\s+(?:THE )?(?:TARGET|COMPANY)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.COVENANTS: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?(?:PRE-?CLOSING )?COVENANTS?\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?CONDUCT OF BUSINESS", re.MULTILINE | re.IGNORECASE
                    ),
                    1,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?EXCLUSIVITY", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^(?:\d+[\.\s]+)?NON-?SHOP", re.MULTILINE | re.IGNORECASE), 1),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?POST-?CLOSING COVENANTS?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.INDEMNIFICATION: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?INDEMNIFICATION\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?SURVIVAL OF REPRESENTATIONS", re.MULTILINE | re.IGNORECASE
                    ),
                    1,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?(?:INDEMNIFICATION )?(?:CAPS?|BASKETS?|LIMITATIONS?)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    1,
                ),
            ],
            LegalSectionType.TERMINATION: [
                (
                    re.compile(
                        r"^(?:ARTICLE\s+[IVXLC\d]+[\.\s:\-]*)?TERMINATION\s*[:\.]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?BREAK(?:-UP)? FEE", re.MULTILINE | re.IGNORECASE), 1),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?REVERSE BREAK(?:-UP)? FEE", re.MULTILINE | re.IGNORECASE
                    ),
                    1,
                ),
            ],
            LegalSectionType.SCHEDULES: [
                (re.compile(r"^DISCLOSURE SCHEDULE", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^SCHEDULE\s+[A-Z\d]+", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # STATUTE patterns (Acts, rules, regulations)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.STATUTE: {
            LegalSectionType.PREAMBLE: [
                (re.compile(r"^PREAMBLE\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:AN )?ACT (?:TO|FOR|OF)", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BE IT ENACTED BY", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^ENACTED BY (?:PARLIAMENT|LEGISLATURE)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.CHAPTER: [
                (re.compile(r"^CHAPTER\s+[IVXLC\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.PART: [
                (re.compile(r"^PART\s+[IVXLC\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SECTION: [
                (
                    re.compile(
                        r"^(?:Section|Sec\.?)\s+\d+[A-Z]?\s*[\.\s:\-]", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^\d+\.\s+(?:Short title|Definitions|Commencement|Extent|Application)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^§\s*\d+", re.MULTILINE), 0),
            ],
            LegalSectionType.SUBSECTION: [
                (re.compile(r"^\(\d+\)\s+", re.MULTILINE), 1),
                (re.compile(r"^\([a-z]\)\s+", re.MULTILINE), 2),
                (re.compile(r"^\([ivxlc]+\)\s+", re.MULTILINE | re.IGNORECASE), 2),
            ],
            LegalSectionType.PROVISO: [
                (re.compile(r"(?:^|\n)\s*Provided (?:that|,)", re.IGNORECASE), 1),
                (re.compile(r"(?:^|\n)\s*Provided (?:further|always|however)", re.IGNORECASE), 1),
            ],
            LegalSectionType.EXPLANATION: [
                (re.compile(r"(?:^|\n)\s*Explanation[\.\s:\-]", re.IGNORECASE), 1),
                (re.compile(r"(?:^|\n)\s*Illustrations?[\.\s:\-]", re.IGNORECASE), 1),
            ],
            LegalSectionType.SCHEDULE: [
                (
                    re.compile(
                        r"^(?:THE )?(?:FIRST|SECOND|THIRD|\d+(?:ST|ND|RD|TH))?\s*SCHEDULE",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^\[?Schedule\s+[IVXLC\d]+\]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # ARBITRATION patterns (awards, claims, procedural documents)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.ARBITRATION: {
            LegalSectionType.TRIBUNAL: [
                (
                    re.compile(
                        r"^(?:THE )?(?:ARBITRAL )?TRIBUNAL\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^ARBITRATOR(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^COMPOSITION OF (?:THE )?TRIBUNAL", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.PROCEDURAL_HISTORY: [
                (
                    re.compile(r"^PROCEDURAL (?:HISTORY|BACKGROUND)", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:BRIEF )?HISTORY OF (?:THE )?PROCEEDINGS", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^PROCEDURAL CHRONOLOGY", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.PARTIES: [
                (re.compile(r"^(?:THE )?PARTIES\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^CLAIMANT(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^RESPONDENT(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.FACTS: [
                (re.compile(r"^(?:STATEMENT OF )?FACTS\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^FACTUAL BACKGROUND", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BACKGROUND TO (?:THE )?DISPUTE", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.CLAIMS: [
                (
                    re.compile(
                        r"^(?:STATEMENT OF )?CLAIM(?:S)?\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^RELIEF (?:SOUGHT|REQUESTED)", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^CLAIMANT'?S? (?:POSITION|SUBMISSIONS)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^RESPONDENT'?S? (?:POSITION|SUBMISSIONS|DEFENSE)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:COUNTER)?CLAIM(?:S)?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.ANALYSIS: [
                (re.compile(r"^(?:THE TRIBUNAL'?S? )?ANALYSIS", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^DISCUSSION\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^REASONS FOR (?:THE )?(?:AWARD|DECISION)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(r"^(?:MERITS|JURISDICTION|LIABILITY)", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (re.compile(r"^LEGAL ANALYSIS", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.AWARD: [
                (
                    re.compile(
                        r"^(?:THE )?(?:FINAL )?AWARD\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^DISPOSITIF\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^DECISION\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^FOR (?:THESE|THE (?:ABOVE|FOREGOING)) REASONS",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:THE )?TRIBUNAL (?:HEREBY )?(?:AWARDS?|ORDERS?|DECIDES?)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.COSTS: [
                (re.compile(r"^COSTS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:ALLOCATION|APPORTIONMENT) OF COSTS?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^COSTS OF (?:THE )?ARBITRATION", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # CORPORATE patterns (MOA, AOA, resolutions, minutes)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.CORPORATE: {
            LegalSectionType.ARTICLES: [
                (re.compile(r"^ARTICLE\s+[IVXLC\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?NAME\s+(?:AND\s+)?(?:REGISTERED\s+)?OFFICE",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^(?:\d+[\.\s]+)?OBJECTS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?(?:SHARE )?CAPITAL\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?MEMBERSHIP\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:\d+[\.\s]+)?DIRECTORS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.RESOLUTIONS: [
                (re.compile(r"^RESOLVED\s*(?:THAT|,)", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:ORDINARY|SPECIAL|BOARD) RESOLUTION", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^IT IS HEREBY RESOLVED", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^FURTHER RESOLVED", re.MULTILINE | re.IGNORECASE), 1),
            ],
            LegalSectionType.ATTENDANCE: [
                (
                    re.compile(
                        r"^(?:MEMBERS? )?(?:PRESENT|IN ATTENDANCE)\s*[:\-]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^QUORUM\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ATTENDANCE\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.AGENDA: [
                (re.compile(r"^AGENDA\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ORDER OF BUSINESS", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ITEM\s+(?:NO\.?)?\s*\d+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 1),
            ],
            LegalSectionType.MINUTES: [
                (
                    re.compile(
                        r"^MINUTES OF (?:THE )?(?:BOARD )?MEETING", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:ANNUAL|EXTRAORDINARY) GENERAL MEETING", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^MEETING (?:HELD|CONVENED) (?:ON|AT)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # REGULATORY patterns (agency filings and notices)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.REGULATORY: {
            LegalSectionType.PREAMBLE: [
                (re.compile(r"^(?:CIRCULAR|NOTIFICATION) NO\.?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:TO|ADDRESSED TO)\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SECTION: [
                (re.compile(r"^PART\s+[IVXLC\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ITEM\s+\d+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE), 0),
            ],
            LegalSectionType.SCHEDULES: [
                (re.compile(r"^ANNEXURE\s+[A-Z\d]+", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^SCHEDULE\s+[A-Z\d]+", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # IP patterns (patents, trademarks)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.IP: {
            LegalSectionType.ABSTRACT: [
                (re.compile(r"^ABSTRACT\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^ABSTRACT OF (?:THE )?DISCLOSURE", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SPECIFICATION: [
                (
                    re.compile(
                        r"^(?:DETAILED )?DESCRIPTION\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^SPECIFICATION\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^FIELD OF (?:THE )?INVENTION", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^BACKGROUND OF (?:THE )?INVENTION", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^SUMMARY OF (?:THE )?INVENTION", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^BRIEF DESCRIPTION OF (?:THE )?(?:DRAWINGS|FIGURES)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.CLAIMS: [
                (re.compile(r"^CLAIMS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:WE|I) CLAIM\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^\d+\.\s+(?:A method|An apparatus|A system|A device|A composition)",
                        re.MULTILINE,
                    ),
                    1,
                ),
            ],
            LegalSectionType.DRAWINGS: [
                (
                    re.compile(
                        r"^(?:BRIEF )?DESCRIPTION OF (?:THE )?(?:DRAWINGS|FIGURES)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^FIG(?:URE)?\.?\s*\d+", re.MULTILINE | re.IGNORECASE), 1),
            ],
            LegalSectionType.PRIOR_ART: [
                (
                    re.compile(
                        r"^(?:DISCUSSION OF )?(?:THE )?(?:RELATED|PRIOR) ART",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^REFERENCES CITED", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # REAL_ESTATE patterns (deeds, leases, mortgages)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.REAL_ESTATE: {
            LegalSectionType.PARTIES: [
                (re.compile(r"^(?:THE )?PARTIES\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:VENDOR|SELLER|TRANSFEROR|LESSOR|LICENSOR)\s*[:\-]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:PURCHASER|BUYER|TRANSFEREE|LESSEE|LICENSEE)\s*[:\-]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
            LegalSectionType.RECITALS: [
                (re.compile(r"^WHEREAS[,:]", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^RECITALS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.PROPERTY: [
                (
                    re.compile(
                        r"^(?:SCHEDULE|DESCRIPTION) OF (?:THE )?PROPERTY",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:THE )?(?:DEMISED|LEASED|SCHEDULED) PREMISES",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(r"^ALL THAT (?:PIECE|PARCEL) OF LAND", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (re.compile(r"^PROPERTY DETAILS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.CONSIDERATION: [
                (re.compile(r"^CONSIDERATION\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:SALE|PURCHASE) (?:PRICE|CONSIDERATION)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^(?:RENT|LICENSE FEE)\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.COVENANTS: [
                (re.compile(r"^COVENANTS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:LESSOR'?S?|LESSEE'?S?) (?:COVENANTS?|OBLIGATIONS?)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:TERMS AND )?CONDITIONS\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
            ],
            LegalSectionType.ENCUMBRANCES: [
                (re.compile(r"^ENCUMBRANCES?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(r"^(?:TITLE|CHAIN OF TITLE)\s*[:\-]?", re.MULTILINE | re.IGNORECASE),
                    0,
                ),
                (re.compile(r"^LIENS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # ESTATE patterns (wills, trusts, POA)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.ESTATE: {
            LegalSectionType.DECLARATION: [
                (
                    re.compile(
                        r"^(?:THIS IS (?:MY|THE) )?(?:LAST )?WILL\s*(?:AND TESTAMENT)?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^DECLARATION OF TRUST", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^I,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+(?:HEREBY|DO HEREBY|BEING)",
                        re.MULTILINE,
                    ),
                    0,
                ),
                (re.compile(r"^REVOCATION\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.BEQUESTS: [
                (
                    re.compile(
                        r"^(?:SPECIFIC )?(?:BEQUESTS?|DEVISES?|GIFTS?)\s*[:\-]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (re.compile(r"^I\s+(?:GIVE|BEQUEATH|DEVISE)", re.MULTILINE | re.IGNORECASE), 1),
                (re.compile(r"^DISTRIBUTION(?:S)?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.EXECUTOR: [
                (
                    re.compile(
                        r"^(?:APPOINTMENT OF )?EXECUTOR(?:S)?\s*[:\-]?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:I )?(?:APPOINT|NOMINATE).{0,50}EXECUTOR", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^TRUSTEE(?:S)?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.RESIDUARY: [
                (
                    re.compile(
                        r"^RESIDUARY (?:ESTATE|CLAUSE)\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^RESIDUE\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.POWERS: [
                (re.compile(r"^POWERS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:EXECUTOR'?S?|TRUSTEE'?S?) POWERS?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^POWER OF ATTORNEY\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:GENERAL|SPECIFIC|LIMITED|DURABLE) POWERS?",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # PLEADING patterns (complaints, petitions, motions)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.PLEADING: {
            LegalSectionType.PARTIES: [
                (re.compile(r"^(?:THE )?PARTIES\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^PLAINTIFF(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^DEFENDANT(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^PETITIONER(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^RESPONDENT(?:\(S\))?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.FACTS: [
                (re.compile(r"^(?:STATEMENT OF )?FACTS\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^FACTUAL (?:BACKGROUND|ALLEGATIONS?)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^GENERAL ALLEGATIONS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.CLAIMS: [
                (
                    re.compile(
                        r"^(?:FIRST|SECOND|THIRD|\d+(?:ST|ND|RD|TH))?\s*(?:CAUSE|COUNT|CLAIM)",
                        re.MULTILINE | re.IGNORECASE,
                    ),
                    0,
                ),
                (
                    re.compile(
                        r"^(?:CLAIMS? FOR RELIEF|CAUSES? OF ACTION)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^GROUNDS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.ANALYSIS: [
                (
                    re.compile(
                        r"^(?:LEGAL )?ARGUMENT(?:S)?\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^MEMORANDUM (?:OF LAW|IN SUPPORT)", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^POINTS AND AUTHORITIES", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.AWARD: [  # Prayer for relief
                (re.compile(r"^(?:PRAYER|RELIEF|WHEREFORE)", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^RELIEF (?:SOUGHT|REQUESTED)", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
        # ═══════════════════════════════════════════════════════════════════════
        # CIRCULAR patterns (agency circulars, bulletins, notices)
        # ═══════════════════════════════════════════════════════════════════════
        DocumentType.CIRCULAR: {
            LegalSectionType.PREAMBLE: [
                (re.compile(r"^CIRCULAR NO\.?", re.MULTILINE | re.IGNORECASE), 0),
                (
                    re.compile(
                        r"^(?:MASTER )?(?:CIRCULAR|DIRECTION)", re.MULTILINE | re.IGNORECASE
                    ),
                    0,
                ),
                (re.compile(r"^(?:TO|ADDRESSED TO)\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^(?:SUBJECT|RE)\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SECTION: [
                (re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE), 0),
                (re.compile(r"^PART\s+[IVXLC\d]+\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^PARA(?:GRAPH)?\s+\d+", re.MULTILINE | re.IGNORECASE), 0),
            ],
            LegalSectionType.SCHEDULES: [
                (re.compile(r"^ANNEXURE\s+[A-Z\d]+", re.MULTILINE | re.IGNORECASE), 0),
                (re.compile(r"^APPENDIX\s+[A-Z\d]+", re.MULTILINE | re.IGNORECASE), 0),
            ],
        },
    }

    # Generic fallback patterns for UNKNOWN document type
    GENERIC_PATTERNS: ClassVar[dict[LegalSectionType, list[tuple[re.Pattern[str], int]]]] = {
        LegalSectionType.PREAMBLE: [
            (
                re.compile(
                    r"^(?:PREAMBLE|INTRODUCTION|BACKGROUND)\s*[:\-]?", re.MULTILINE | re.IGNORECASE
                ),
                0,
            ),
        ],
        LegalSectionType.DEFINITIONS: [
            (re.compile(r"^DEFINITIONS?\s*[:\-]?", re.MULTILINE | re.IGNORECASE), 0),
        ],
        LegalSectionType.SECTION: [
            (
                re.compile(r"^(?:ARTICLE|SECTION|PART)\s+[IVXLC\d]+", re.MULTILINE | re.IGNORECASE),
                0,
            ),
            (re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE), 0),
        ],
        LegalSectionType.SCHEDULES: [
            (
                re.compile(
                    r"^(?:SCHEDULE|ANNEX(?:URE)?|APPENDIX|EXHIBIT)\s+[A-Z\d]+",
                    re.MULTILINE | re.IGNORECASE,
                ),
                0,
            ),
        ],
    }

    def __init__(self, jurisdictions: Sequence[str] | None = None):
        """Initialize detector with a classifier (optionally with extra pattern packs)."""
        self._classifier = get_document_type_classifier(jurisdictions)

    def detect(
        self,
        text: str,
        document_type: DocumentType | None = None,
    ) -> list[UniversalSectionBoundary]:
        """
        Detect section boundaries in document.

        Args:
            text: Full document text
            document_type: If known, skip classification. Otherwise auto-detect.

        Returns:
            List of detected section boundaries with confidence scores
        """
        if not text or not text.strip():
            return []

        # Auto-classify if not provided
        if document_type is None:
            document_type, classification_confidence = self._classifier.classify(text)
        else:
            classification_confidence = 1.0

        # Get appropriate patterns
        if document_type == DocumentType.UNKNOWN or document_type not in self.SECTION_PATTERNS:
            patterns = self.GENERIC_PATTERNS
        else:
            patterns = self.SECTION_PATTERNS[document_type]

        # Find all pattern matches
        matches: list[dict] = []

        for section_type, pattern_list in patterns.items():
            for pattern, hierarchy_level in pattern_list:
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "section_type": section_type,
                            "start": match.start(),
                            "end": match.end(),
                            "matched_text": match.group(),
                            "title": _heading_title(text, match),
                            "hierarchy_level": hierarchy_level,
                        }
                    )

        if not matches:
            # No sections detected - treat entire document as UNKNOWN section
            logger.debug(
                "universal_section_detection_no_matches",
                text_length=len(text),
                document_type=document_type.value,
            )
            return [
                UniversalSectionBoundary(
                    section_type=LegalSectionType.UNKNOWN,
                    document_type=document_type,
                    start_char=0,
                    end_char=len(text),
                    content=text,
                    confidence=0.5 * classification_confidence,
                )
            ]

        # Sort by position
        matches.sort(key=lambda x: x["start"])

        # Build section boundaries
        boundaries = self._build_boundaries(matches, text, document_type, classification_confidence)

        logger.info(
            "universal_section_detection_completed",
            document_type=document_type.value,
            section_count=len(boundaries),
            section_types=[b.section_type.value for b in boundaries],
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
        document_type: DocumentType,
        classification_confidence: float,
    ) -> list[UniversalSectionBoundary]:
        """Build section boundaries from pattern matches."""
        boundaries: list[UniversalSectionBoundary] = []
        current_section_type: LegalSectionType | None = None
        section_start = 0
        section_matches: list[dict] = []
        current_hierarchy = 0

        for i, match in enumerate(matches):
            # Start new section if:
            # 1. First match
            # 2. Different section type at same or higher hierarchy level
            # 3. Significant gap from previous match (>300 chars for non-subsection)
            is_new_section = (
                current_section_type is None
                or (
                    match["section_type"] != current_section_type
                    and match["hierarchy_level"] <= current_hierarchy
                )
                or (
                    i > 0
                    and match["start"] - matches[i - 1]["end"] > 300
                    and match["hierarchy_level"] == 0
                )
            )

            if is_new_section:
                # Save previous section if exists
                if current_section_type is not None and section_matches:
                    confidence = self._calculate_confidence(
                        section_matches,
                        text[section_start : match["start"]],
                        classification_confidence,
                    )

                    boundaries.append(
                        UniversalSectionBoundary(
                            section_type=current_section_type,
                            document_type=document_type,
                            start_char=section_start,
                            end_char=match["start"],
                            content=text[section_start : match["start"]],
                            confidence=confidence,
                            title=section_matches[0]["title"],
                            hierarchy_level=current_hierarchy,
                        )
                    )

                # Start new section
                current_section_type = match["section_type"]
                section_start = match["start"]
                section_matches = [match]
                current_hierarchy = match["hierarchy_level"]
            else:
                section_matches.append(match)

        # Don't forget the last section
        if current_section_type is not None and section_matches:
            confidence = self._calculate_confidence(
                section_matches,
                text[section_start:],
                classification_confidence,
            )

            boundaries.append(
                UniversalSectionBoundary(
                    section_type=current_section_type,
                    document_type=document_type,
                    start_char=section_start,
                    end_char=len(text),
                    content=text[section_start:],
                    confidence=confidence,
                    title=section_matches[0]["title"],
                    hierarchy_level=current_hierarchy,
                )
            )

        return boundaries

    def _calculate_confidence(
        self,
        section_matches: list[dict],
        content: str,
        classification_confidence: float,
    ) -> float:
        """Calculate confidence score for section detection."""
        if not section_matches:
            return 0.5 * classification_confidence

        # Base confidence from number of matches (max 0.5)
        match_confidence = min(len(section_matches) * 0.2, 0.5)

        # Bonus for header position (first 500 chars of section)
        first_match_pos = section_matches[0]["start"]
        position_confidence = 0.2 if first_match_pos < 500 else 0.1

        # Bonus for reasonable content length
        content_len = len(content.strip())
        length_confidence = 0.2 if 100 < content_len < 50000 else 0.1

        # Factor in document classification confidence
        total_confidence = (
            match_confidence + position_confidence + length_confidence
        ) * classification_confidence
        return min(total_confidence, 1.0)


# Singletons
_classifiers: dict[tuple[str, ...] | None, DocumentTypeClassifier] = {}
_detectors: dict[tuple[str, ...] | None, UniversalSectionDetector] = {}


def _key(jurisdictions: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if jurisdictions is None else tuple(jurisdictions)


def get_document_type_classifier(
    jurisdictions: Sequence[str] | None = None,
) -> DocumentTypeClassifier:
    """Get a cached document type classifier for a set of jurisdiction pattern packs."""
    key = _key(jurisdictions)
    if key not in _classifiers:
        _classifiers[key] = DocumentTypeClassifier(jurisdictions)
    return _classifiers[key]


def get_universal_section_detector(
    jurisdictions: Sequence[str] | None = None,
) -> UniversalSectionDetector:
    """Get a cached universal section detector for a set of jurisdiction pattern packs."""
    key = _key(jurisdictions)
    if key not in _detectors:
        _detectors[key] = UniversalSectionDetector(jurisdictions)
    return _detectors[key]
