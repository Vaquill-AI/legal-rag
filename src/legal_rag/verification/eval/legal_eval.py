"""
Legal-Domain RAG Evaluation.

Specialized evaluation metrics for legal RAG systems:
- Citation Accuracy: Are citations correctly extracted and formatted?
- Jurisdiction Consistency: Are jurisdiction-specific nuances preserved?
- Temporal Validity: Are cited precedents still valid law?
- Legal Term Precision: Are legal terms used correctly?

Complements RAGAS metrics with legal-specific quality measures.
"""

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol, runtime_checkable

from legal_rag.verification._log import get_logger
from legal_rag.verification.citations import Citation, extract

logger = get_logger("legal_rag.verification.legal_eval")

CitationExtractorFn = Callable[[str], Sequence[Citation]]
"""Any callable that returns the citations found in a text. Defaults to ``citations.extract``."""


class PrecedentStatus(str, Enum):
    """Whether a cited authority is still good law."""

    VALID = "valid"
    OVERRULED = "overruled"
    OUTDATED = "outdated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PrecedentValidity:
    """Result of checking one citation for continued validity."""

    status: PrecedentStatus = PrecedentStatus.UNKNOWN
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.status in (PrecedentStatus.VALID, PrecedentStatus.UNKNOWN)


@runtime_checkable
class PrecedentChecker(Protocol):
    """Checks whether a citation is still good law. Implement with a citator of your choice."""

    async def check_validity(self, citation: str, *, context: str = "") -> PrecedentValidity: ...


class LegalMetricType(str, Enum):
    """Types of legal-specific evaluation metrics."""

    CITATION_ACCURACY = "citation_accuracy"
    JURISDICTION_CONSISTENCY = "jurisdiction_consistency"
    TEMPORAL_VALIDITY = "temporal_validity"
    LEGAL_TERM_PRECISION = "legal_term_precision"
    STATUTE_ACCURACY = "statute_accuracy"


@dataclass
class CitationAccuracy:
    """
    Citation accuracy assessment.

    Attributes:
        total_citations: Number of citations found in response
        valid_citations: Citations with correct format
        grounded_citations: Citations found in source context
        invalid_citations: Malformed or fabricated citations
        accuracy_score: Overall citation accuracy (0-1)
    """

    total_citations: int = 0
    valid_citations: int = 0
    grounded_citations: int = 0
    invalid_citations: int = 0
    fabricated_citations: list[str] = field(default_factory=list)
    accuracy_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "grounded_citations": self.grounded_citations,
            "invalid_citations": self.invalid_citations,
            "fabricated_citations": self.fabricated_citations,
            "accuracy_score": round(self.accuracy_score, 3),
        }


@dataclass
class JurisdictionConsistency:
    """
    Jurisdiction consistency assessment.

    Checks if response maintains appropriate jurisdiction context.
    """

    primary_jurisdiction: str | None = None
    jurisdictions_mentioned: list[str] = field(default_factory=list)
    consistency_score: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_jurisdiction": self.primary_jurisdiction,
            "jurisdictions_mentioned": self.jurisdictions_mentioned,
            "consistency_score": round(self.consistency_score, 3),
            "issues": self.issues,
        }


@dataclass
class TemporalValidity:
    """
    Temporal validity assessment for cited precedents.
    """

    citations_checked: int = 0
    valid_citations: int = 0
    overruled_citations: list[str] = field(default_factory=list)
    outdated_citations: list[str] = field(default_factory=list)
    validity_score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations_checked": self.citations_checked,
            "valid_citations": self.valid_citations,
            "overruled_citations": self.overruled_citations,
            "outdated_citations": self.outdated_citations,
            "validity_score": round(self.validity_score, 3),
            "warnings": self.warnings,
        }


@dataclass
class LegalTermPrecision:
    """
    Legal terminology precision assessment.
    """

    terms_found: int = 0
    correctly_used: int = 0
    incorrectly_used: list[str] = field(default_factory=list)
    precision_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms_found": self.terms_found,
            "correctly_used": self.correctly_used,
            "incorrectly_used": self.incorrectly_used,
            "precision_score": round(self.precision_score, 3),
        }


@dataclass
class LegalEvaluationResult:
    """
    Complete legal-domain evaluation result.

    Attributes:
        query: Original query
        response: Generated response
        citation_accuracy: Citation quality metrics
        jurisdiction_consistency: Jurisdiction handling metrics
        temporal_validity: Precedent validity metrics
        legal_term_precision: Terminology precision metrics
        overall_legal_score: Weighted overall score
        passed: Whether evaluation passed threshold
        recommendations: Improvement suggestions
    """

    query: str
    response: str
    citation_accuracy: CitationAccuracy
    jurisdiction_consistency: JurisdictionConsistency
    temporal_validity: TemporalValidity
    legal_term_precision: LegalTermPrecision
    overall_legal_score: float = 0.0
    passed: bool = False
    threshold: float = 0.7
    recommendations: list[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response_preview": (
                self.response[:200] + "..." if len(self.response) > 200 else self.response
            ),
            "citation_accuracy": self.citation_accuracy.to_dict(),
            "jurisdiction_consistency": self.jurisdiction_consistency.to_dict(),
            "temporal_validity": self.temporal_validity.to_dict(),
            "legal_term_precision": self.legal_term_precision.to_dict(),
            "overall_legal_score": round(self.overall_legal_score, 3),
            "passed": self.passed,
            "threshold": self.threshold,
            "recommendations": self.recommendations,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }


class LegalEvaluator:
    """
    Legal-domain specific evaluator for RAG systems.

    Evaluates legal response quality beyond general RAGAS metrics:
    1. Citation Accuracy - Are citations valid and grounded?
    2. Jurisdiction Consistency - Is jurisdiction context maintained?
    3. Temporal Validity - Are precedents still good law?
    4. Legal Term Precision - Are legal terms used correctly?

    Example:
        evaluator = LegalEvaluator()
        result = await evaluator.evaluate(
            query="What does 42 U.S.C. 1983 provide?",
            response="Under 42 U.S.C. 1983, a person acting under color of state law is liable...",
            context="42 U.S.C. 1983. Civil action for deprivation of rights..."
        )

        print(f"Citation Accuracy: {result.citation_accuracy.accuracy_score}")
        print(f"Legal Score: {result.overall_legal_score}")
    """

    # Common legal terms for precision checking
    LEGAL_TERMS: ClassVar[dict[str, str]] = {
        # Procedural terms
        "bail": "temporary release from custody pending trial",
        "summary judgment": "judgment without trial when no material fact is disputed",
        "motion to dismiss": "request to end a case for legal insufficiency",
        "discovery": "pretrial exchange of information between parties",
        "arraignment": "hearing where the accused hears the charges and enters a plea",
        # Court terminology
        "mandamus": "writ commanding public duty",
        "certiorari": "writ to review a lower court decision",
        "habeas corpus": "writ for unlawful detention",
        "injunction": "court order restraining an action",
        "stay": "temporary halt of proceedings",
        "decree": "formal court decision",
        "judgment": "statement of reasons for decision",
        # Parties
        "appellant": "party filing appeal",
        "appellee": "party responding to appeal",
        "respondent": "party responding to appeal or petition",
        "petitioner": "party filing petition",
        "plaintiff": "party initiating civil suit",
        "defendant": "party defending civil suit",
        # Legal principles
        "res judicata": "matter already decided",
        "stare decisis": "follow precedent",
        "ratio decidendi": "reason for decision",
        "obiter dicta": "remarks in passing",
        "ultra vires": "beyond legal power",
        "mala fide": "bad faith",
        "bona fide": "good faith",
        "prima facie": "on first appearance",
        "mens rea": "guilty mind",
        "actus reus": "guilty act",
    }

    # Court hierarchy for jurisdiction checking: lower number is a higher court.
    COURT_HIERARCHY: ClassVar[dict[str, int]] = {
        "supreme court": 1,
        "court of appeals": 2,
        "circuit court": 2,
        "district court": 3,
        "trial court": 3,
        "magistrate": 4,
    }

    def __init__(
        self,
        quality_threshold: float = 0.7,
        citation_weight: float = 0.35,
        jurisdiction_weight: float = 0.20,
        temporal_weight: float = 0.25,
        term_weight: float = 0.20,
        *,
        citation_extractor: CitationExtractorFn = extract,
        precedent_checker: PrecedentChecker | None = None,
        jurisdiction_patterns: Mapping[str, Sequence[str]] | None = None,
    ):
        """
        Initialize legal evaluator.

        Args:
            quality_threshold: Minimum score to pass
            citation_weight: Weight for citation accuracy
            jurisdiction_weight: Weight for jurisdiction consistency
            temporal_weight: Weight for temporal validity
            term_weight: Weight for legal term precision
            citation_extractor: Callable returning the citations in a text
            precedent_checker: Optional good-law checker. Without one, every citation is
                treated as valid and the temporal score is 1.0.
            jurisdiction_patterns: Maps a jurisdiction label to lowercase phrases that
                indicate it. Without patterns the jurisdiction check finds nothing to flag.
        """
        self.quality_threshold = quality_threshold
        self.weights = {
            "citation": citation_weight,
            "jurisdiction": jurisdiction_weight,
            "temporal": temporal_weight,
            "term": term_weight,
        }
        self.citation_extractor = citation_extractor
        self.precedent_checker = precedent_checker
        self.jurisdiction_patterns: dict[str, list[str]] = {
            k: [p.lower() for p in v] for k, v in (jurisdiction_patterns or {}).items()
        }

    async def evaluate(
        self,
        query: str,
        response: str,
        context: str,
        expected_jurisdiction: str | None = None,
    ) -> LegalEvaluationResult:
        """
        Evaluate legal response quality.

        Args:
            query: Original query
            response: Generated response
            context: Retrieved context
            expected_jurisdiction: Expected jurisdiction for consistency check

        Returns:
            LegalEvaluationResult with all metrics
        """

        start_time = time.time()

        if not response or not response.strip():
            return LegalEvaluationResult(
                query=query,
                response=response,
                citation_accuracy=CitationAccuracy(),
                jurisdiction_consistency=JurisdictionConsistency(),
                temporal_validity=TemporalValidity(),
                legal_term_precision=LegalTermPrecision(),
                passed=False,
                threshold=self.quality_threshold,
            )

        # Evaluate each dimension
        citation_accuracy = self._evaluate_citations(response, context)
        jurisdiction_consistency = self._evaluate_jurisdiction(
            query, response, context, expected_jurisdiction
        )
        temporal_validity = await self._evaluate_temporal_validity(response, context)
        legal_term_precision = self._evaluate_legal_terms(response, context)

        # Calculate overall score
        overall_score = (
            self.weights["citation"] * citation_accuracy.accuracy_score
            + self.weights["jurisdiction"] * jurisdiction_consistency.consistency_score
            + self.weights["temporal"] * temporal_validity.validity_score
            + self.weights["term"] * legal_term_precision.precision_score
        )

        passed = overall_score >= self.quality_threshold

        # Generate recommendations
        recommendations = self._generate_recommendations(
            citation_accuracy,
            jurisdiction_consistency,
            temporal_validity,
            legal_term_precision,
        )

        evaluation_time = (time.time() - start_time) * 1000

        logger.info(
            "legal_evaluation_complete",
            overall_score=round(overall_score, 3),
            citation_accuracy=citation_accuracy.accuracy_score,
            jurisdiction_consistency=jurisdiction_consistency.consistency_score,
            temporal_validity=temporal_validity.validity_score,
            passed=passed,
        )

        return LegalEvaluationResult(
            query=query,
            response=response,
            citation_accuracy=citation_accuracy,
            jurisdiction_consistency=jurisdiction_consistency,
            temporal_validity=temporal_validity,
            legal_term_precision=legal_term_precision,
            overall_legal_score=overall_score,
            passed=passed,
            threshold=self.quality_threshold,
            recommendations=recommendations,
            evaluation_time_ms=evaluation_time,
        )

    def _evaluate_citations(
        self,
        response: str,
        context: str,
    ) -> CitationAccuracy:
        """Evaluate citation accuracy in response."""
        # Extract citations from response
        response_citations = self.citation_extractor(response)
        context_citations = self.citation_extractor(context)

        if not response_citations:
            return CitationAccuracy(accuracy_score=1.0)  # No citations to evaluate

        # Build set of context citation strings for matching
        context_citation_strs = set()
        for cit in context_citations:
            context_citation_strs.add(cit.raw.lower())
            context_citation_strs.add(cit.normalized.lower())

        valid_count = 0
        grounded_count = 0
        fabricated = []

        for citation in response_citations:
            # All extracted citations are considered valid format
            # (extraction already validates format through regex patterns)
            valid_count += 1

            # Check if citation is grounded in context
            citation_lower = citation.raw.lower()

            if (
                citation_lower in context_citation_strs
                or citation.normalized.lower() in context_citation_strs
                or citation_lower in context.lower()
            ):
                grounded_count += 1
            else:
                fabricated.append(citation.raw)

        total = len(response_citations)
        invalid_count = total - valid_count

        # Calculate accuracy score
        if total > 0:
            # Weight: 40% valid format, 60% grounded in context
            format_score = valid_count / total
            grounding_score = grounded_count / total if valid_count > 0 else 0
            accuracy_score = 0.4 * format_score + 0.6 * grounding_score
        else:
            accuracy_score = 1.0

        return CitationAccuracy(
            total_citations=total,
            valid_citations=valid_count,
            grounded_citations=grounded_count,
            invalid_citations=invalid_count,
            fabricated_citations=fabricated,
            accuracy_score=accuracy_score,
        )

    def _evaluate_jurisdiction(
        self,
        query: str,
        response: str,
        context: str,
        expected_jurisdiction: str | None,
    ) -> JurisdictionConsistency:
        """Evaluate jurisdiction consistency."""
        response_lower = response.lower()
        query_lower = query.lower()
        context_lower = context.lower()

        # Detect jurisdictions mentioned
        jurisdictions_found = []
        for jurisdiction, patterns in self.jurisdiction_patterns.items():
            for pattern in patterns:
                if pattern in response_lower:
                    jurisdictions_found.append(jurisdiction)
                    break

        # Determine primary jurisdiction from query/context
        primary_jurisdiction = expected_jurisdiction
        if not primary_jurisdiction:
            for jurisdiction, patterns in self.jurisdiction_patterns.items():
                for pattern in patterns:
                    if pattern in query_lower or pattern in context_lower:
                        primary_jurisdiction = jurisdiction
                        break
                if primary_jurisdiction:
                    break

        # Check consistency
        issues = []
        consistency_score = 1.0

        if primary_jurisdiction and jurisdictions_found:
            # Check if response mentions different jurisdictions without context
            for jurisdiction in jurisdictions_found:
                if jurisdiction != primary_jurisdiction:
                    # Check if the mixing is contextual (e.g., comparing jurisdictions)
                    comparison_indicators = [
                        "however",
                        "unlike",
                        "compared to",
                        "in contrast",
                    ]
                    if not any(ind in response_lower for ind in comparison_indicators):
                        issues.append(
                            f"Mixed jurisdiction: query is about {primary_jurisdiction}, "
                            f"but response mentions {jurisdiction}"
                        )
                        consistency_score -= 0.2

        # Check for court hierarchy consistency
        courts_mentioned = []
        for court, level in self.COURT_HIERARCHY.items():
            if court in response_lower:
                courts_mentioned.append((court, level))

        if len(courts_mentioned) > 1:
            # Check if lower court decisions are cited as binding on higher courts
            # This would be an error
            for i, (court1, level1) in enumerate(courts_mentioned):
                for court2, level2 in courts_mentioned[i + 1 :]:
                    if level1 > level2:  # Lower court cited
                        binding_indicators = ["binding", "must follow", "mandatory"]
                        if any(ind in response_lower for ind in binding_indicators):
                            issues.append(
                                f"Hierarchy issue: {court1} decisions are not binding on {court2}"
                            )
                            consistency_score -= 0.3

        consistency_score = max(0.0, consistency_score)

        return JurisdictionConsistency(
            primary_jurisdiction=primary_jurisdiction,
            jurisdictions_mentioned=list(set(jurisdictions_found)),
            consistency_score=consistency_score,
            issues=issues,
        )

    async def _evaluate_temporal_validity(
        self,
        response: str,
        context: str,
    ) -> TemporalValidity:
        """Evaluate temporal validity of cited precedents."""
        # Extract citations from response
        citations = self.citation_extractor(response)

        if not citations:
            return TemporalValidity(validity_score=1.0)

        valid_count = 0
        overruled = []
        outdated = []
        warnings = []

        for citation in citations:
            if self.precedent_checker is None:
                valid_count += 1
                continue
            validity = await self.precedent_checker.check_validity(
                citation.raw,
                context=context,
            )

            if validity.is_valid:
                valid_count += 1
            elif validity.status == PrecedentStatus.OVERRULED:
                overruled.append(citation.raw)
                warnings.append(f"Citation {citation.raw} has been overruled")
            elif validity.status == PrecedentStatus.OUTDATED:
                outdated.append(citation.raw)
                warnings.append(f"Citation {citation.raw} is outdated")

            # Add warnings from validity check
            warnings.extend(validity.warnings)

        total = len(citations)
        validity_score = valid_count / total if total > 0 else 1.0

        return TemporalValidity(
            citations_checked=total,
            valid_citations=valid_count,
            overruled_citations=overruled,
            outdated_citations=outdated,
            validity_score=validity_score,
            warnings=warnings[:10],  # Limit warnings
        )

    def _evaluate_legal_terms(
        self,
        response: str,
        context: str,
    ) -> LegalTermPrecision:
        """Evaluate legal terminology precision."""
        response_lower = response.lower()
        context_lower = context.lower()

        terms_found = []
        correctly_used = []
        incorrectly_used = []

        for term, definition in self.LEGAL_TERMS.items():
            if term in response_lower:
                terms_found.append(term)

                # Check if term is used in correct context
                # Simple heuristic: term should appear near relevant context words
                definition_words = set(definition.lower().split())

                # Find term position and check surrounding context
                term_pos = response_lower.find(term)
                surrounding = response_lower[max(0, term_pos - 100) : term_pos + len(term) + 100]
                surrounding_words = set(surrounding.split())

                # Check if context words overlap with definition
                overlap = definition_words & surrounding_words
                if len(overlap) >= 2 or term in context_lower:
                    correctly_used.append(term)
                else:
                    # Check for obvious misuse patterns
                    misuse_detected = self._check_term_misuse(term, response_lower)
                    if misuse_detected:
                        incorrectly_used.append(f"{term}: {misuse_detected}")
                    else:
                        correctly_used.append(term)  # Assume correct if no misuse detected

        total = len(terms_found)
        precision_score = len(correctly_used) / total if total > 0 else 1.0

        return LegalTermPrecision(
            terms_found=total,
            correctly_used=len(correctly_used),
            incorrectly_used=incorrectly_used,
            precision_score=precision_score,
        )

    def _check_term_misuse(self, term: str, text: str) -> str | None:
        """Check for common misuse patterns of legal terms."""
        misuse_patterns = {
            "mandamus": [
                (r"mandamus.*private", "Mandamus is only for public duties"),
            ],
            "ratio decidendi": [
                (r"ratio decidendi.*not binding", "Ratio decidendi is binding"),
            ],
            "obiter dicta": [
                (r"obiter dicta.*binding", "Obiter dicta is not binding"),
            ],
        }

        if term in misuse_patterns:
            for pattern, explanation in misuse_patterns[term]:
                if re.search(pattern, text, re.IGNORECASE):
                    return explanation

        return None

    def _generate_recommendations(
        self,
        citation_accuracy: CitationAccuracy,
        jurisdiction_consistency: JurisdictionConsistency,
        temporal_validity: TemporalValidity,
        legal_term_precision: LegalTermPrecision,
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if citation_accuracy.accuracy_score < 0.7:
            if citation_accuracy.fabricated_citations:
                recommendations.append(
                    f"Verify citations: {', '.join(citation_accuracy.fabricated_citations[:3])} "
                    "could not be found in source documents"
                )
            if citation_accuracy.invalid_citations > 0:
                recommendations.append("Check citation formats - some citations appear malformed")

        if jurisdiction_consistency.consistency_score < 0.8:
            recommendations.extend(jurisdiction_consistency.issues[:2])

        if temporal_validity.validity_score < 0.8 and temporal_validity.overruled_citations:
            recommendations.append(
                f"Update citations: {', '.join(temporal_validity.overruled_citations[:2])} "
                "may no longer be good law"
            )

        if legal_term_precision.precision_score < 0.8 and legal_term_precision.incorrectly_used:
            recommendations.append(
                f"Review terminology usage: {', '.join(legal_term_precision.incorrectly_used[:2])}"
            )

        return recommendations


async def evaluate_legal_response(
    query: str,
    response: str,
    context: str,
    expected_jurisdiction: str | None = None,
) -> LegalEvaluationResult:
    """
    Convenience function to evaluate a legal response.

    Args:
        query: Original query
        response: Generated response
        context: Retrieved context
        expected_jurisdiction: Expected jurisdiction

    Returns:
        LegalEvaluationResult
    """
    evaluator = get_legal_evaluator()
    return await evaluator.evaluate(query, response, context, expected_jurisdiction)


# Singleton instance
_default_evaluator: LegalEvaluator | None = None


def get_legal_evaluator() -> LegalEvaluator:
    """Get or create default legal evaluator instance."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = LegalEvaluator()
    return _default_evaluator
