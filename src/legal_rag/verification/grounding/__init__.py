"""Grounding checks: hallucination detection, fabricated case detection, claim and source checks."""

from legal_rag.verification.grounding.case_names import ParsedCaseName, parse_case_name_query
from legal_rag.verification.grounding.citation_grounding import (
    ungrounded_case_citations,
    ungrounded_legal_citations,
)
from legal_rag.verification.grounding.detector import (
    HallucinationCheckResult,
    HallucinationConfig,
    HallucinationDetector,
    HallucinationResult,
    get_hallucination_checker,
    get_hallucination_detector,
)
from legal_rag.verification.grounding.fabricated_case import (
    FlaggedCase,
    StateMismatch,
    detect_fabricated_cases,
    detect_state_mismatches,
)
from legal_rag.verification.grounding.named_case_guard import (
    NamedCaseGuardResult,
    evaluate_named_case_guard,
    evaluate_named_case_guard_with_resolver,
)
from legal_rag.verification.grounding.source_hygiene import (
    SourceHygiene,
    SourceTier,
    classify_source,
    filter_sources,
)

__all__ = [
    "FlaggedCase",
    "HallucinationCheckResult",
    "HallucinationConfig",
    "HallucinationDetector",
    "HallucinationResult",
    "NamedCaseGuardResult",
    "ParsedCaseName",
    "SourceHygiene",
    "SourceTier",
    "StateMismatch",
    "classify_source",
    "detect_fabricated_cases",
    "detect_state_mismatches",
    "evaluate_named_case_guard",
    "evaluate_named_case_guard_with_resolver",
    "filter_sources",
    "get_hallucination_checker",
    "get_hallucination_detector",
    "parse_case_name_query",
    "ungrounded_case_citations",
    "ungrounded_legal_citations",
]
