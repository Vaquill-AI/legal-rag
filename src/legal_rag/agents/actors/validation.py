"""Validation actor: check that a plan can be grounded in the retrieved chunks.

The check runs before any answer is written, so an ungroundable plan is caught at the planning
stage instead of after the answer has hallucinated.
It reports which sub-questions lack supporting chunks and recommends one of ``proceed``,
``retrieve_more``, ``simplify`` or ``fallback_to_general``.
When the reply cannot be validated the result fails closed (``is_valid`` false), while the
recommendation stays ``proceed`` because answer generation has its own guards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from legal_rag.agents.actors.base import (
    ActorConfig,
    ActorOutputError,
    BaseActor,
    normalize_chunks,
    source_name,
)
from legal_rag.providers.protocols import LanguageModel
from legal_rag.types import Message, RetrievedChunk
from legal_rag.verification.grounding.actor_schemas import (
    ActorMetrics,
    ResponsePlan,
    ValidationResult,
)

#: Characters of each chunk shown to the validator. Longer than the planner's preview because
#: validation must check content coverage, not only the topic.
CHUNK_PREVIEW_CHARS = 500

VALIDATION_SYSTEM_PROMPT = """You are a pre-flight validation specialist for legal research systems.

Your role is to decide whether a planned legal research response can be FULLY GROUNDED in the available context chunks.

Produce a validation result with these fields:
- is_valid: true or false.
- missing_support: the sub-questions or claims that lack supporting chunks.
- recommendation: one of proceed, retrieve_more, simplify, fallback_to_general.
- confidence_score: a number from 0.0 to 1.0.
- validation_notes: notes about the result.

Validation criteria:
1. CHUNK COVERAGE: do the chunks contain information for each sub-question?
2. SOURCE AVAILABILITY: are all required sources present in the chunks?
3. DETAIL SUFFICIENCY: is there enough detail to answer comprehensively?
4. AMBIGUITY CHECK: are there conflicting or ambiguous passages?

Decision rules:
- is_valid is true if at least 80% of the priority-1 sub-questions have supporting chunks.
- is_valid is false if coverage is below 80% or a critical source is missing.
- proceed: good chunk support for all sub-questions.
- retrieve_more: the chunks are incomplete and more retrieval is needed.
- simplify: the plan is too ambitious for the available chunks.
- fallback_to_general: no relevant chunks are available."""

VALIDATION_USER_PROMPT = """<plan_to_validate>
<sub_questions>
{sub_questions}
</sub_questions>

<required_sources>
{required_sources}
</required_sources>

<response_structure>{response_structure}</response_structure>

<information_gaps>
{information_gaps}
</information_gaps>
</plan_to_validate>

<available_chunks>
{chunks_context}
</available_chunks>

Validate whether this plan can be fully grounded in the available chunks."""


class ValidationActor(BaseActor):
    """Pre-flight check of a ``ResponsePlan`` against the retrieved chunks."""

    DEFAULT_CONFIG = ActorConfig(max_tokens=2048, temperature=0.2)

    @property
    def actor_name(self) -> str:
        return "validation"

    def build_prompt(
        self, plan: ResponsePlan, chunks: Sequence[Mapping[str, Any]]
    ) -> list[Message]:
        """Build the validation messages. Every chunk is listed, none are dropped."""
        details = []
        for index, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            preview = content[:CHUNK_PREVIEW_CHARS].rstrip()
            if len(content) > CHUNK_PREVIEW_CHARS:
                preview += "..."
            name = source_name(chunk) or "Unknown"
            details.append(
                f"{index}. [{name}] (score: {chunk.get('score', 0.0):.3f})\n   {preview}"
            )

        sub_questions = "\n".join(
            f"{number}. {sq.question} (chunk_indices: {sq.chunk_indices}, "
            f"answerable: {sq.answerable}, priority: {sq.priority})"
            for number, sq in enumerate(plan.sub_questions, 1)
        )
        gaps = (
            "\n".join(f"- {gap}" for gap in plan.information_gaps) or "None identified by planner"
        )
        return [
            Message("system", VALIDATION_SYSTEM_PROMPT),
            Message(
                "user",
                VALIDATION_USER_PROMPT.format(
                    sub_questions=sub_questions,
                    required_sources="\n".join(f"- {s}" for s in plan.required_sources),
                    response_structure=plan.response_structure,
                    information_gaps=gaps,
                    chunks_context="\n\n".join(details),
                ),
            ),
        ]

    async def execute(self, **kwargs: Any) -> ValidationResult:
        """Validate a plan.

        Keyword arguments:
            plan: the ``ResponsePlan`` to check (required)
            available_chunks: the retrieved chunks (required)
        """
        plan = kwargs.get("plan")
        if not plan:
            raise ValueError("plan parameter is required")
        if not kwargs.get("available_chunks"):
            raise ValueError("available_chunks parameter is required")
        chunks = normalize_chunks(kwargs["available_chunks"])

        messages = self.build_prompt(plan, chunks)
        self.logger.info(
            "validation_started",
            sub_questions_count=len(plan.sub_questions),
            chunks_count=len(chunks),
        )

        try:
            validation = await self.call_structured(messages, ValidationResult)
        except ActorOutputError as exc:
            self.logger.error("validation_output_invalid", error=str(exc))
            return ValidationResult(
                is_valid=False,
                missing_support=["Validation failed - unable to parse response"],
                recommendation="proceed",
                confidence_score=0.0,
                validation_notes=None,
            )

        self.logger.info(
            "validation_completed",
            is_valid=validation.is_valid,
            recommendation=validation.recommendation,
            missing_support_count=len(validation.missing_support),
            confidence_score=validation.confidence_score,
        )
        return validation


async def validate_response_plan(
    model: LanguageModel,
    plan: ResponsePlan,
    available_chunks: Sequence[RetrievedChunk | Mapping[str, Any]],
    config: ActorConfig | None = None,
) -> tuple[ValidationResult, ActorMetrics]:
    """Validate a plan and return the result with the actor's metrics."""
    actor = ValidationActor(model, config)
    return await actor.run_with_metrics(plan=plan, available_chunks=available_chunks)
