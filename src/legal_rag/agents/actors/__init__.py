"""Actors: planning, validation and refinement around an injected language model.

Each actor takes a ``LanguageModel`` and an optional ``ActorConfig``.
The deterministic plan validator needs no model.
"""

from legal_rag.agents.actors.base import (
    ActorConfig,
    ActorError,
    ActorOutputError,
    BaseActor,
    normalize_chunks,
)
from legal_rag.agents.actors.deterministic import (
    DeterministicPlanValidator,
    DeterministicValidationResult,
    quick_plan_check,
    validate_plan_deterministically,
)
from legal_rag.agents.actors.planning import PlanningActor, plan_response, select_response_structure
from legal_rag.agents.actors.refinement import RefinementActor, refine_response
from legal_rag.agents.actors.validation import ValidationActor, validate_response_plan

__all__ = [
    "ActorConfig",
    "ActorError",
    "ActorOutputError",
    "BaseActor",
    "DeterministicPlanValidator",
    "DeterministicValidationResult",
    "PlanningActor",
    "RefinementActor",
    "ValidationActor",
    "normalize_chunks",
    "plan_response",
    "quick_plan_check",
    "refine_response",
    "select_response_structure",
    "validate_plan_deterministically",
    "validate_response_plan",
]
