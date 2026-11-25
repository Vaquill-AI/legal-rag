"""Offline evaluation: RAGAS-style metrics, legal metrics, retrieval metrics and citation support."""

from legal_rag.verification.eval.deterministic_citation import CitationReport, score_citations
from legal_rag.verification.eval.legal_eval import (
    CitationAccuracy,
    LegalEvaluationResult,
    LegalEvaluator,
    PrecedentChecker,
    PrecedentStatus,
    PrecedentValidity,
    get_legal_evaluator,
)
from legal_rag.verification.eval.ragas_eval import (
    EvaluationMetrics,
    EvaluationResult,
    RAGASEvaluator,
    get_ragas_evaluator,
)
from legal_rag.verification.eval.retrieval_bench import (
    BenchReport,
    LabeledQuery,
    StageIds,
    evaluate_retrieval,
    evaluate_retrieval_async,
    load_labeled_set,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    stage_attribution,
    write_bench_results,
)

__all__ = [
    "BenchReport",
    "CitationAccuracy",
    "CitationReport",
    "EvaluationMetrics",
    "EvaluationResult",
    "LabeledQuery",
    "LegalEvaluationResult",
    "LegalEvaluator",
    "PrecedentChecker",
    "PrecedentStatus",
    "PrecedentValidity",
    "RAGASEvaluator",
    "StageIds",
    "evaluate_retrieval",
    "evaluate_retrieval_async",
    "get_legal_evaluator",
    "get_ragas_evaluator",
    "load_labeled_set",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "score_citations",
    "stage_attribution",
    "write_bench_results",
]
