"""Query enrichment: expand legal acronyms and short forms into their full names.

Short legal queries ("fcra violation", "ucc 2-207") retrieve poorly because the index holds the
full names. This module adds the expansion to the query text without removing the original term,
so lexical and dense retrieval both get the better signal.

Example:
    "ccpa penalties" -> "ccpa penalties California Consumer Privacy Act"

The acronym table lives in ``data/enrichment_rules.json`` so it can be extended without code
changes.
"""

import json
import re
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("rag.conversation.query_enricher")


def _load_data() -> dict[str, Any]:
    path = resources.files(__package__).joinpath("data/enrichment_rules.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_DATA = _load_data()


@dataclass
class QueryEnrichmentResult:
    """Result of rule-based query enrichment."""

    original_query: str
    enriched_query: str
    enrichment_applied: bool
    matched_terms: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ============================================
# Legal Enrichment Rules
# ============================================
# Maps acronyms/abbreviations (lowercase) to full forms.
# Organized by jurisdiction for maintainability.

LEGAL_ENRICHMENT_RULES: dict[str, str] = dict(_DATA["LEGAL_ENRICHMENT_RULES"])

# Pre-compile regex patterns for each rule (case-insensitive word boundary)
_COMPILED_PATTERNS: dict[str, re.Pattern] = {
    acronym: re.compile(r"\b" + re.escape(acronym) + r"\b", re.IGNORECASE)
    for acronym in LEGAL_ENRICHMENT_RULES
}


class QueryEnricher:
    """
    Rule-based query enrichment for legal acronym expansion.

    Appends full forms of detected acronyms to improve BM25 retrieval.
    No LLM cost, <1ms latency.
    """

    def enrich(self, query: str, is_in_domain: bool = True) -> QueryEnrichmentResult:
        """
        Enrich a query by expanding legal acronyms to full forms.

        Args:
            query: Original query string
            is_in_domain: Whether the query is classified as legal domain

        Returns:
            QueryEnrichmentResult with enriched query
        """
        start_time = time.time()

        if not is_in_domain:
            return QueryEnrichmentResult(
                original_query=query,
                enriched_query=query,
                enrichment_applied=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        matched_expansions: list[str] = []
        matched_terms: list[str] = []

        for acronym, pattern in _COMPILED_PATTERNS.items():
            if pattern.search(query):
                full_form = LEGAL_ENRICHMENT_RULES[acronym]
                matched_expansions.append(full_form)
                matched_terms.append(acronym)

        if not matched_expansions:
            return QueryEnrichmentResult(
                original_query=query,
                enriched_query=query,
                enrichment_applied=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        enriched = f"{query} {' '.join(matched_expansions)}"
        elapsed = (time.time() - start_time) * 1000

        logger.debug(
            "query_enriched",
            original=query[:80],
            matched_count=len(matched_terms),
            matched_terms=matched_terms[:5],
            elapsed_ms=round(elapsed, 2),
        )

        return QueryEnrichmentResult(
            original_query=query,
            enriched_query=enriched,
            enrichment_applied=True,
            matched_terms=matched_terms,
            processing_time_ms=elapsed,
        )


# Singleton
_enricher_instance: QueryEnricher | None = None


def get_query_enricher() -> QueryEnricher:
    """Get or create QueryEnricher singleton."""
    global _enricher_instance
    if _enricher_instance is None:
        _enricher_instance = QueryEnricher()
    return _enricher_instance
