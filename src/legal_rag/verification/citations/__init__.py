"""US citation extraction, normalization, validation and provenance.

``extract(text)`` is the main entry point.
"""

from legal_rag.verification.citations.code_refs import extract_code_refs, normalise_for_lookup
from legal_rag.verification.citations.extract import extract, extract_unique
from legal_rag.verification.citations.models import Citation, CitationKind
from legal_rag.verification.citations.provenance import unbacked_case_citations
from legal_rag.verification.citations.us_patterns import looks_like_us_citation

__all__ = [
    "Citation",
    "CitationKind",
    "extract",
    "extract_code_refs",
    "extract_unique",
    "looks_like_us_citation",
    "normalise_for_lookup",
    "unbacked_case_citations",
]
