"""
Temporal Precedent Tracking Module.

Tracks the validity of legal precedents over time, including:
- Overruled cases
- Distinguished cases
- Superseded statutes
- Amendment history
- Temporal validation (prevents impossible relationships)

Components:
- PrecedentTracker: Main tracking class
- PrecedentStatus: Status of a precedent
- PrecedentHistory: Change history for a precedent
- TemporalValidationResult: Result of temporal relationship validation
- extract_year_from_citation: Utility to extract year from citations
"""

from legal_rag.chunking.temporal.precedent import (
    PrecedentHistory,
    PrecedentStatus,
    PrecedentTracker,
    PrecedentValidity,
    TemporalValidationResult,
    extract_year_from_citation,
    get_precedent_tracker,
)

__all__ = [
    "PrecedentHistory",
    "PrecedentStatus",
    "PrecedentTracker",
    "PrecedentValidity",
    "TemporalValidationResult",
    "extract_year_from_citation",
    "get_precedent_tracker",
]
