"""Model-free plan validation, re-exported from the grounding package.

``legal_rag.verification.grounding.deterministic`` already holds the validator.
It checks a ``ResponsePlan`` against retrieved chunks with keyword and source matching, needs no
language model and takes about a millisecond, so it can gate the model-backed validation actor.
"""

from __future__ import annotations

from legal_rag.verification.grounding.deterministic import (
    DeterministicPlanValidator,
    DeterministicValidationResult,
    quick_plan_check,
    validate_plan_deterministically,
)

__all__ = [
    "DeterministicPlanValidator",
    "DeterministicValidationResult",
    "quick_plan_check",
    "validate_plan_deterministically",
]
