"""Verification: citation extraction and validation, grounding checks and offline evaluation.

* ``legal_rag.verification.citations``: ``extract(text)`` and friends.
* ``legal_rag.verification.grounding``: hallucination, fabricated case and source checks.
* ``legal_rag.verification.eval``: RAGAS-style and retrieval metrics.
* ``legal_rag.verification.resolver``: the ``CaseResolver`` protocol you implement.
"""

from legal_rag.verification.citations import Citation, CitationKind, extract
from legal_rag.verification.resolver import CaseResolver, ResolvedCase

__all__ = ["CaseResolver", "Citation", "CitationKind", "ResolvedCase", "extract"]
