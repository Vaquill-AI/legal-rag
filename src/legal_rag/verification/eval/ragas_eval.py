"""
RAGAS-Style RAG Evaluation.

Implements core RAGAS metrics for evaluating RAG pipeline quality:
- Faithfulness: Are claims supported by context?
- Answer Relevancy: Does answer address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Was all needed info retrieved?

References:
- https://docs.ragas.io/en/stable/
- https://arxiv.org/abs/2309.15217
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from legal_rag.verification._log import get_logger

logger = get_logger("legal_rag.verification.ragas_eval")


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    HALLUCINATION_RATE = "hallucination_rate"


@dataclass
class EvaluationMetrics:
    """
    Individual metric scores.

    All scores are normalized to 0-1 range where 1 is best.
    """

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    hallucination_rate: float = 0.0  # Lower is better

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        # Weights tuned for legal domain (faithfulness most important)
        weights = {
            "faithfulness": 0.35,
            "answer_relevancy": 0.25,
            "context_precision": 0.20,
            "context_recall": 0.15,
            "hallucination_penalty": 0.05,
        }

        score = (
            weights["faithfulness"] * self.faithfulness
            + weights["answer_relevancy"] * self.answer_relevancy
            + weights["context_precision"] * self.context_precision
            + weights["context_recall"] * self.context_recall
            + weights["hallucination_penalty"] * (1 - self.hallucination_rate)
        )

        return round(score, 3)

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "hallucination_rate": round(self.hallucination_rate, 3),
            "overall_score": self.overall_score,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a RAG query.

    Attributes:
        query: Original query
        answer: Generated answer
        contexts: Retrieved context chunks
        metrics: Calculated metrics
        claims: Extracted claims from answer
        grounded_claims: Claims supported by context
        ungrounded_claims: Claims not found in context
        passed: Whether evaluation passed quality threshold
        threshold: Quality threshold used
    """

    query: str
    answer: str
    contexts: list[str]
    metrics: EvaluationMetrics
    claims: list[str] = field(default_factory=list)
    grounded_claims: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    passed: bool = False
    threshold: float = 0.7
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer_preview": (
                self.answer[:200] + "..." if len(self.answer) > 200 else self.answer
            ),
            "contexts_count": len(self.contexts),
            "metrics": self.metrics.to_dict(),
            "claims_count": len(self.claims),
            "grounded_claims_count": len(self.grounded_claims),
            "ungrounded_claims_count": len(self.ungrounded_claims),
            "passed": self.passed,
            "threshold": self.threshold,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG systems.

    Evaluates RAG output quality using four core metrics:
    1. Faithfulness - Is the answer grounded in context?
    2. Answer Relevancy - Does the answer address the query?
    3. Context Precision - Are retrieved docs relevant?
    4. Context Recall - Is all needed info retrieved?

    Example:
        evaluator = RAGASEvaluator()
        result = await evaluator.evaluate(
            query="What does 42 U.S.C. 1983 provide?",
            answer="Section 1983 creates a civil action for deprivation of rights...",
            contexts=["42 U.S.C. 1983. Civil action for deprivation of rights..."]
        )

        print(f"Faithfulness: {result.metrics.faithfulness}")
        print(f"Passed: {result.passed}")
    """

    # Claim extraction patterns
    CLAIM_PATTERNS: ClassVar[list[str]] = [
        r"([A-Z][^.!?]*(?:is|are|was|were|has|have|provides|states|holds|requires|mandates|prohibits)[^.!?]*[.!?])",
        r"([A-Z][^.!?]*(?:shall|must|may|can|cannot|should)[^.!?]*[.!?])",
        r"([A-Z][^.!?]*(?:Section|Article|Rule|Regulation)\s+\d+[^.!?]*[.!?])",
    ]

    def __init__(
        self,
        quality_threshold: float = 0.7,
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            quality_threshold: Minimum overall score to pass
        """
        self.quality_threshold = quality_threshold

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG output quality.

        Args:
            query: Original query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer for recall

        Returns:
            EvaluationResult with all metrics
        """

        start_time = time.time()

        if not answer or not answer.strip():
            return EvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                metrics=EvaluationMetrics(),
                passed=False,
                threshold=self.quality_threshold,
            )

        # Extract claims from answer
        claims = self._extract_claims(answer)

        # Calculate metrics
        faithfulness, grounded, ungrounded = self._calculate_faithfulness(answer, contexts, claims)
        answer_relevancy = self._calculate_answer_relevancy(query, answer)
        context_precision = self._calculate_context_precision(query, contexts)
        context_recall = self._calculate_context_recall(answer, contexts, ground_truth)
        hallucination_rate = len(ungrounded) / max(len(claims), 1)

        metrics = EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            hallucination_rate=hallucination_rate,
        )

        passed = metrics.overall_score >= self.quality_threshold

        evaluation_time = (time.time() - start_time) * 1000

        logger.info(
            "ragas_evaluation_complete",
            overall_score=metrics.overall_score,
            faithfulness=faithfulness,
            passed=passed,
            claims_count=len(claims),
        )

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            metrics=metrics,
            claims=claims,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            passed=passed,
            threshold=self.quality_threshold,
            evaluation_time_ms=evaluation_time,
        )

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate multiple RAG samples.

        Args:
            samples: List of {"query", "answer", "contexts", "ground_truth"} dicts

        Returns:
            Aggregated evaluation results. With no samples, ``samples_evaluated``
            is 0, ``individual_results`` is empty, and ``pass_rate`` and
            ``average_metrics`` are ``None`` (nothing was scored), never a perfect score.
        """
        if not samples:
            return {
                "samples_evaluated": 0,
                "pass_rate": None,
                "average_metrics": None,
                "individual_results": [],
            }

        results = []

        for sample in samples:
            result = await self.evaluate(
                query=sample["query"],
                answer=sample["answer"],
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)

        # Aggregate metrics
        avg_metrics = EvaluationMetrics(
            faithfulness=sum(r.metrics.faithfulness for r in results) / len(results),
            answer_relevancy=sum(r.metrics.answer_relevancy for r in results) / len(results),
            context_precision=sum(r.metrics.context_precision for r in results) / len(results),
            context_recall=sum(r.metrics.context_recall for r in results) / len(results),
            hallucination_rate=sum(r.metrics.hallucination_rate for r in results) / len(results),
        )

        pass_rate = sum(1 for r in results if r.passed) / len(results)

        return {
            "samples_evaluated": len(results),
            "pass_rate": round(pass_rate, 3),
            "average_metrics": avg_metrics.to_dict(),
            "individual_results": [r.to_dict() for r in results],
        }

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text."""
        claims = []

        for pattern in self.CLAIM_PATTERNS:
            matches = re.findall(pattern, text)
            claims.extend(matches)

        # Fallback: split by sentences if no patterns match
        if not claims:
            sentences = re.split(r"[.!?]\s+", text)
            claims = [s.strip() + "." for s in sentences if len(s.strip()) > 20]

        # Deduplicate
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_normalized = claim.lower().strip()
            if claim_normalized not in seen:
                seen.add(claim_normalized)
                unique_claims.append(claim)

        return unique_claims[:20]  # Limit to 20 claims

    def _calculate_faithfulness(
        self,
        answer: str,
        contexts: list[str],
        claims: list[str],
    ) -> tuple:
        """
        Calculate faithfulness score.

        Faithfulness = # claims supported by context / total claims
        """
        if not claims:
            return 1.0, [], []

        combined_context = " ".join(contexts).lower()
        grounded = []
        ungrounded = []

        for claim in claims:
            # Check if key terms from claim appear in context
            claim_terms = self._extract_key_terms(claim)

            if not claim_terms:
                grounded.append(claim)  # No key terms, assume grounded
                continue

            # Calculate overlap
            found_terms = sum(1 for term in claim_terms if term.lower() in combined_context)
            overlap_ratio = found_terms / len(claim_terms)

            if overlap_ratio >= 0.5:  # At least 50% of key terms found
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        faithfulness = len(grounded) / len(claims)
        return faithfulness, grounded, ungrounded

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for matching."""
        # Remove common words and extract meaningful terms
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "and",
            "or",
            "but",
            "if",
            "then",
            "than",
            "so",
            "as",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "to",
            "from",
            "up",
            "down",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Filter stopwords and keep meaningful terms
        key_terms = [w for w in words if w not in stopwords]

        # Also extract numbers and legal references
        numbers = re.findall(r"\b\d+\b", text)
        key_terms.extend(numbers)

        return key_terms[:10]  # Limit to 10 key terms

    def _calculate_answer_relevancy(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Calculate answer relevancy score.

        Measures semantic similarity between query and answer.
        """
        query_terms = set(self._extract_key_terms(query))
        answer_terms = set(self._extract_key_terms(answer))

        if not query_terms:
            return 0.8  # Default if no query terms

        # Calculate Jaccard-like overlap
        overlap = len(query_terms & answer_terms)
        relevancy = overlap / len(query_terms)

        # Boost if answer is substantial
        if len(answer) > 100:
            relevancy = min(relevancy + 0.2, 1.0)

        return relevancy

    def _calculate_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> float:
        """
        Calculate context precision score.

        Precision = # relevant contexts / total contexts
        """
        if not contexts:
            return 0.0

        query_terms = set(self._extract_key_terms(query))

        if not query_terms:
            return 0.7  # Default

        relevant_count = 0
        for context in contexts:
            context_terms = set(self._extract_key_terms(context))
            overlap = len(query_terms & context_terms)

            if overlap >= len(query_terms) * 0.3:  # 30% overlap threshold
                relevant_count += 1

        return relevant_count / len(contexts)

    def _calculate_context_recall(
        self,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> float:
        """
        Calculate context recall score.

        If ground_truth provided: How much of ground truth is covered?
        Otherwise: How much answer content comes from context?
        """
        if ground_truth:
            # Check coverage of ground truth
            gt_terms = set(self._extract_key_terms(ground_truth))
            combined_context = " ".join(contexts).lower()

            found = sum(1 for term in gt_terms if term in combined_context)
            return found / max(len(gt_terms), 1)

        # Without ground truth, check answer-context overlap
        answer_terms = set(self._extract_key_terms(answer))
        combined_context = " ".join(contexts).lower()

        found = sum(1 for term in answer_terms if term in combined_context)
        return found / max(len(answer_terms), 1)


# Singleton instance
_default_evaluator: RAGASEvaluator | None = None


def get_ragas_evaluator() -> RAGASEvaluator:
    """Get or create default RAGAS evaluator instance."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = RAGASEvaluator()
    return _default_evaluator
