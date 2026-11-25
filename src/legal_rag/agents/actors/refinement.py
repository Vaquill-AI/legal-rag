"""Refinement actor: review a draft answer against its plan and say whether to rewrite it.

The actor critiques; it does not rewrite.
It reports unanswered sub-questions, weak citations and unclear language, and sets
``should_regenerate``. The orchestrator decides what to do with that.
When the reply cannot be validated the draft is accepted as it is.
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
    RefinementSuggestions,
    ResponsePlan,
)

#: A longer draft is cut to this many characters for review, since the review judges structure and
#: completeness and does not need the whole text verbatim.
MAX_REVIEW_CHARS = 15000

#: Only this many leading sources are listed for citation checks.
MAX_SOURCES_LISTED = 20

REFINEMENT_SYSTEM_PROMPT = """You are a senior legal editor reviewing a research answer for completeness and accuracy.

Your role is to critically review the answer against the original plan and identify weaknesses or gaps.

Produce a review with these fields:
- missing_elements: unanswered sub-questions or missing components.
- citation_gaps: claims that lack proper citations.
- clarity_issues: ambiguous statements or unclear language.
- should_regenerate: true or false.
- suggested_improvements: specific guidance for a rewrite, if one is needed.
- confidence_score: a number from 0.0 to 1.0.

Review criteria:
1. COMPLETENESS: does the answer address ALL sub-questions in the plan?
2. CITATIONS: are all claims cited to specific sources?
3. CLARITY: is the language precise and unambiguous?
4. STRUCTURE: does it follow the planned response structure?
5. CONFIDENCE: are expressions of confidence appropriate, without overstating certainty?

Decision rules:
should_regenerate is true if there are 2 or more unanswered sub-questions, 3 or more significant citation gaps, or a critical ambiguity or unsupported claim.
should_regenerate is false if all sub-questions are addressed (even briefly), citations are adequate (minor gaps are acceptable) and the answer is clear and well structured."""

REFINEMENT_USER_PROMPT = """<original_plan>
<sub_questions>
{sub_questions}
</sub_questions>

<required_sources>
{required_sources}
</required_sources>

<response_structure>{response_structure}</response_structure>

<risk_areas>
{risk_areas}
</risk_areas>
</original_plan>

<available_sources>
For citation validation:
{sources_context}
</available_sources>

<generated_response>
{original_response}
</generated_response>

<task>
Review this answer against the plan and the available sources. Identify any weaknesses, gaps or areas for improvement.
</task>"""


class RefinementActor(BaseActor):
    """Critique a draft answer against its plan."""

    DEFAULT_CONFIG = ActorConfig(max_tokens=2500, temperature=0.3)

    @property
    def actor_name(self) -> str:
        return "refinement"

    def build_prompt(
        self,
        original_response: str,
        plan: ResponsePlan,
        chunks: Sequence[Mapping[str, Any]],
    ) -> list[Message]:
        """Build the review messages."""
        if len(original_response) > MAX_REVIEW_CHARS:
            total = len(original_response)
            original_response = (
                original_response[:MAX_REVIEW_CHARS]
                + f"\n\n[... response truncated at {MAX_REVIEW_CHARS} chars for review, "
                f"full response is {total} chars ...]"
            )

        sources_context = "\n".join(
            f"{number}. {source_name(chunk) or 'Unknown'}"
            for number, chunk in enumerate(chunks[:MAX_SOURCES_LISTED], 1)
        )
        return [
            Message("system", REFINEMENT_SYSTEM_PROMPT),
            Message(
                "user",
                REFINEMENT_USER_PROMPT.format(
                    sub_questions="\n".join(f"- {sq.question}" for sq in plan.sub_questions),
                    required_sources="\n".join(f"- {s}" for s in plan.required_sources),
                    response_structure=plan.response_structure,
                    risk_areas="\n".join(f"- {r}" for r in plan.risk_areas) or "None",
                    sources_context=sources_context,
                    original_response=original_response,
                ),
            ),
        ]

    async def execute(self, **kwargs: Any) -> RefinementSuggestions:
        """Review a draft.

        Keyword arguments:
            original_response: the draft answer (required)
            plan: the ``ResponsePlan`` it was written from (required)
            chunks: the retrieved chunks (required)
        """
        original_response = kwargs.get("original_response")
        plan = kwargs.get("plan")
        if not original_response:
            raise ValueError("original_response parameter is required")
        if not plan:
            raise ValueError("plan parameter is required")
        if not kwargs.get("chunks"):
            raise ValueError("chunks parameter is required")
        chunks = normalize_chunks(kwargs["chunks"])

        messages = self.build_prompt(original_response, plan, chunks)
        self.logger.info(
            "refinement_started",
            response_length=len(original_response),
            sub_questions_count=len(plan.sub_questions),
            max_tokens=self.config.max_tokens,
        )

        try:
            refinement = await self.call_structured(messages, RefinementSuggestions)
        except ActorOutputError as exc:
            self.logger.error("refinement_output_invalid", error=str(exc))
            return RefinementSuggestions(
                missing_elements=[],
                citation_gaps=[],
                clarity_issues=["Model returned invalid structured output - skipping refinement"],
                should_regenerate=False,
                suggested_improvements=None,
                confidence_score=0.5,
            )

        self.logger.info(
            "refinement_completed",
            missing_elements_count=len(refinement.missing_elements),
            citation_gaps_count=len(refinement.citation_gaps),
            clarity_issues_count=len(refinement.clarity_issues),
            should_regenerate=refinement.should_regenerate,
        )
        return refinement


async def refine_response(
    model: LanguageModel,
    original_response: str,
    plan: ResponsePlan,
    chunks: Sequence[RetrievedChunk | Mapping[str, Any]],
    config: ActorConfig | None = None,
) -> tuple[RefinementSuggestions, ActorMetrics]:
    """Review a draft answer and return the critique with the actor's metrics."""
    actor = RefinementActor(model, config)
    return await actor.run_with_metrics(
        original_response=original_response, plan=plan, chunks=chunks
    )
