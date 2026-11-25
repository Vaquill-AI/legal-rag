"""Query type classifier for legal search optimization.

Classifies incoming legal queries into three categories:

- CITATION: exact citation lookups ("347 U.S. 483", "42 U.S.C. 1983")
- HIERARCHICAL: court, date, judge or statute based filters
- CONCEPTUAL: semantic, concept based search

The pattern sets are data (``data/query_patterns.json``) and can be replaced by passing
``patterns`` to ``QueryClassifier``, so another jurisdiction needs a new file, not new code.

Usage:
    result = classify_query("347 U.S. 483")
    # result.query_type == QueryType.CITATION
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib import resources
from typing import Any

logger = logging.getLogger("legal_rag.retrieval.routing.query_classifier")


def _load_data() -> dict[str, Any]:
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/query_patterns.json")
        .open("r", encoding="utf-8") as fh
    ):
        return json.load(fh)


_DATA = _load_data()


class QueryType(Enum):
    """Legal query classification types based on search patterns."""

    CITATION = "citation"
    HIERARCHICAL = "hierarchical"
    CONCEPTUAL = "conceptual"


@dataclass
class ClassificationResult:
    """Result of query classification with confidence and metadata."""

    query_type: QueryType
    confidence: float
    matched_patterns: list[str]
    extracted_entities: dict
    raw_query: str
    processing_time_ms: float

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(type={self.query_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"patterns={self.matched_patterns})"
        )


class QueryClassifier:
    """
    Intelligent query classifier for legal search optimization.

    Classifies queries into CITATION, HIERARCHICAL, or CONCEPTUAL based on
    pattern matching and confidence scoring.
    """

    def __init__(self, patterns: dict[str, dict[str, dict[str, Any]]] | None = None):
        """Compile the injected pattern sets, or the bundled US defaults."""
        patterns = patterns or _DATA
        self.citation_config = patterns["citation"]
        self.hierarchical_config = patterns["hierarchical"]
        self.conceptual_config = patterns["conceptual"]
        # Compile citation patterns
        self.citation_patterns = {
            name: re.compile(data["pattern"], re.IGNORECASE)
            for name, data in self.citation_config.items()
        }

        # Compile hierarchical patterns
        self.hierarchical_patterns = {
            name: re.compile(data["pattern"], re.IGNORECASE)
            for name, data in self.hierarchical_config.items()
        }

        # Compile conceptual patterns
        self.conceptual_patterns = {
            name: re.compile(data["pattern"], re.IGNORECASE)
            for name, data in self.conceptual_config.items()
        }

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a legal query into CITATION, HIERARCHICAL, or CONCEPTUAL.

        Args:
            query: Raw user query string

        Returns:
            ClassificationResult with query type, confidence, and metadata

        Algorithm:
            1. Check for citation patterns (highest priority)
            2. Check for hierarchical filter patterns (medium priority)
            3. Default to conceptual if no strong patterns (lowest priority)
            4. Calculate confidence based on pattern matches and weights
        """
        start_time = datetime.now(UTC)

        # Normalize query
        query_normalized = query.strip()

        # Track all pattern matches
        citation_matches = []
        hierarchical_matches = []
        conceptual_matches = []
        extracted_entities = {}

        # Step 1: Check citation patterns
        citation_score = 0.0
        for pattern_name, pattern in self.citation_patterns.items():
            matches = pattern.findall(query_normalized)
            if matches:
                weight = self.citation_config[pattern_name]["weight"]
                citation_score += weight
                citation_matches.append(pattern_name)

                # Extract citation entities
                if pattern_name not in extracted_entities:
                    extracted_entities[pattern_name] = []
                extracted_entities[pattern_name].extend(matches)

        # Step 2: Check hierarchical patterns
        hierarchical_score = 0.0
        for pattern_name, pattern in self.hierarchical_patterns.items():
            matches = pattern.findall(query_normalized)
            if matches:
                weight = self.hierarchical_config[pattern_name]["weight"]
                hierarchical_score += weight
                hierarchical_matches.append(pattern_name)

                if pattern_name not in extracted_entities:
                    extracted_entities[pattern_name] = []
                extracted_entities[pattern_name].extend(matches)

        # Step 3: Check conceptual patterns
        conceptual_score = 0.0
        for pattern_name, pattern in self.conceptual_patterns.items():
            matches = pattern.findall(query_normalized)
            if matches:
                weight = self.conceptual_config[pattern_name]["weight"]
                conceptual_score += weight
                conceptual_matches.append(pattern_name)

        # Step 4: Determine query type based on scores
        # Citation patterns have highest priority (threshold 0.7 to include case numbers)
        if citation_score >= 0.7:
            query_type = QueryType.CITATION
            confidence = min(0.95, 0.75 + (citation_score * 0.1))  # 0.75-0.95 for citations
            matched_patterns = citation_matches
        elif hierarchical_score >= 0.8:
            query_type = QueryType.HIERARCHICAL
            confidence = min(0.90, 0.65 + (hierarchical_score * 0.1))  # 0.73-0.90 for hierarchical
            matched_patterns = hierarchical_matches
        elif conceptual_score >= 0.5:
            query_type = QueryType.CONCEPTUAL
            confidence = min(0.85, 0.55 + (conceptual_score * 0.1))  # 0.65-0.85 for conceptual
            matched_patterns = conceptual_matches
        elif hierarchical_score > 0:
            # A LONE filter signal still means a filter query. JUDGE_FILTER (0.5)
            # and YEAR_MENTION (0.6) cannot reach the 0.8 bar on their own, so
            # "Cases decided by Justice Roberts" and "Family Court divorce
            # judgments 2021" matched a hierarchical pattern, failed the bar,
            # and fell through to the default branch below -- which reported
            # them CONCEPTUAL with matched_patterns=[], throwing away the very
            # evidence that would have explained the call. The pattern had
            # matched; only the score was short.
            #
            # Placed AFTER the conceptual branch on purpose: a query carrying a
            # real conceptual signal still reads as conceptual, so this can only
            # reclaim queries that would otherwise have matched nothing at all.
            # Confidence is capped below the 0.73 floor of the >= 0.8 branch so
            # a single-signal call stays distinguishable from a confident one.
            query_type = QueryType.HIERARCHICAL
            confidence = min(0.72, 0.55 + (hierarchical_score * 0.1))
            matched_patterns = hierarchical_matches
        else:
            # Default to conceptual with lower confidence
            query_type = QueryType.CONCEPTUAL
            confidence = 0.50  # Lower confidence when no patterns match
            matched_patterns = []

        # Calculate processing time
        end_time = datetime.now(UTC)
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        result = ClassificationResult(
            query_type=query_type,
            confidence=confidence,
            matched_patterns=matched_patterns,
            extracted_entities=extracted_entities,
            raw_query=query,
            processing_time_ms=processing_time_ms,
        )

        logger.debug(
            "query_classified type=%s confidence=%.2f patterns=%s",
            query_type.value,
            confidence,
            matched_patterns,
        )

        return result


_classifier: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """Return a shared classifier built from the bundled US defaults."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier


def classify_query(query: str, classifier: QueryClassifier | None = None) -> ClassificationResult:
    """Classify a query with the given classifier, or the shared default."""
    return (classifier or get_query_classifier()).classify(query)
