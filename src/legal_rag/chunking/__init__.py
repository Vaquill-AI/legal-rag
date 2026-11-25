"""
Chunkers package - text chunking strategies with universal document type support.

Available chunkers:
- RecursiveLegalChunker: Hybrid recursive + universal legal section awareness with TRUE overlap (recommended)
- SemchunkLegalChunker: Fast semantic chunking with legal section awareness (legacy)
- SemanticChunker: Alias for backward compatibility

Document type classification:
- DocumentType: Enum for document types (JUDGMENT, CONTRACT, STATUTE, ARBITRATION, etc.)
- DocumentTypeClassifier: Pattern-based document type classification

Universal section detection (NEW - supports 12 document types, ~50 section types):
- LegalSectionType: Enum for universal section types across all document categories
- UniversalSectionBoundary: Detected section with document type context
- UniversalSectionDetector: Pattern-based section detection for all document types

Legacy section detection (judgments only):
- LegalSection: Enum for judgment section types (HEADNOTE, JUDGMENT, ORDER, OPERATIVE_PART)
- SectionBoundary: Detected section boundary with confidence score
- LegalSectionDetector: Pattern-based section detection for court judgments

Supported document types:
- JUDGMENT: Court judgments (US, UK, other common-law courts)
- CONTRACT: Commercial contracts, agreements, NDAs
- STATUTE: Acts, rules, regulations, ordinances
- ARBITRATION: Awards, claims, procedural documents
- CORPORATE: Articles, bylaws, resolutions, minutes
- MNA: M&A documents, SPAs, APAs
- REGULATORY: Agency filings, notices and compliance documents
- IP: Patents, trademarks
- REAL_ESTATE: Deeds, leases
- ESTATE: Wills, trusts, POA
- PLEADING: Complaints, petitions
- CIRCULAR: Agency circulars, bulletins and notices

Chunker comparison:
- RecursiveLegalChunker: TRUE overlap, universal doc types, guaranteed clean splits, $0 cost (recommended)
- SemchunkLegalChunker: Semantic boundaries, NO overlap (simulated via neighbor accumulation)
"""

from legal_rag.chunking.base import BaseChunker, Chunk, ChunkerResult
from legal_rag.chunking.document_types import (
    DocumentType,
    DocumentTypeClassifier,
    LegalSectionType,
    UniversalSectionBoundary,
    UniversalSectionDetector,
    get_document_type_classifier,
    get_section_priority,
    get_universal_section_detector,
)
from legal_rag.chunking.email_chunker import EmailChunker, get_email_chunker
from legal_rag.chunking.legal_sections import (
    LegalSection,
    LegalSectionDetector,
    SectionBoundary,
    get_legal_section_detector,
)
from legal_rag.chunking.recursive_legal import (
    LegalChunk,
    RecursiveLegalChunker,
    get_recursive_legal_chunker,
)
from legal_rag.chunking.semchunk_legal import (
    SemchunkLegalChunker,
    get_semchunk_legal_chunker,
)

# Backward compatibility aliases
SemanticChunker = SemchunkLegalChunker
SectionAwareChunker = SemchunkLegalChunker

__all__ = [
    # Base classes
    "BaseChunker",
    "Chunk",
    "ChunkerResult",
    # Document type classification (NEW)
    "DocumentType",
    "DocumentTypeClassifier",
    # Email chunker (conversation-aware)
    "EmailChunker",
    # Recommended chunker (recursive with true overlap + universal doc types)
    "LegalChunk",
    # Legacy section detection (judgments only)
    "LegalSection",
    "LegalSectionDetector",
    # Universal section detection (NEW - supports all document types)
    "LegalSectionType",
    "RecursiveLegalChunker",
    "SectionAwareChunker",
    "SectionBoundary",
    # Backward compatibility aliases
    "SemanticChunker",
    # Legacy chunker (semchunk-based)
    "SemchunkLegalChunker",
    "UniversalSectionBoundary",
    "UniversalSectionDetector",
    "get_document_type_classifier",
    "get_email_chunker",
    "get_legal_section_detector",
    "get_recursive_legal_chunker",
    "get_section_priority",
    "get_semchunk_legal_chunker",
    "get_universal_section_detector",
]
