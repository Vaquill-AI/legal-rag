"""Grounded answer generation: plan, validate, write, refine once, verify.

``generate_answer_buffered`` runs the stages in this order:

1. Plan (deep tier only): the planning actor maps sub-questions to retrieved chunks.
2. Validate: the model-free plan validator, then the validation actor unless the first one found
   the plan critically ungroundable. The recommendation can relax grounding or prune the plan.
3. Write: one buffered call to the answer model.
4. Refine (deep tier only): the refinement actor critiques the draft. At most one rewrite follows.
5. Verify: an injected ``HallucinationDetector`` scores the final answer against the sources.

Every stage except the answer model call fails open: an error is logged and the pipeline carries
on with what it has, so a failing actor never blocks an answer.
The caller owns the model, the detector and the retrieved sources; nothing here reads configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from legal_rag.agents.actors.base import ActorConfig, normalize_chunks, source_name
from legal_rag.agents.actors.deterministic import (
    DeterministicValidationResult,
    validate_plan_deterministically,
)
from legal_rag.agents.actors.planning import PlanningActor
from legal_rag.agents.actors.refinement import RefinementActor
from legal_rag.agents.actors.validation import ValidationActor
from legal_rag.providers.protocols import LanguageModel
from legal_rag.types import Message, RetrievedChunk
from legal_rag.verification._log import get_logger
from legal_rag.verification.grounding.actor_schemas import (
    ActorMetrics,
    RefinementSuggestions,
    ResponsePlan,
    ValidationResult,
)
from legal_rag.verification.grounding.detector import (
    HallucinationCheckResult,
    HallucinationDetector,
    VerificationResult,
)

logger = get_logger(__name__)

#: Upper bound on answer tokens per query complexity, so a simple question gets a short answer.
COMPLEXITY_TOKEN_CAPS: dict[str, int] = {
    "trivial": 500,
    "simple": 4000,
    "moderate": 16000,
    "complex": 30000,
}

#: Cap used when the complexity is unknown or not in ``COMPLEXITY_TOKEN_CAPS``.
DEFAULT_COMPLEXITY_CAP = 30000

#: A rewrite needs the critique to ask for one, to carry guidance, and to report a confidence
#: below this ceiling.
REFINEMENT_CONFIDENCE_CEILING = 0.6

#: At most this many failed claims are copied into the verification summary.
MAX_REPORTED_CLAIMS = 10

#: Added to the plan when validation finds the sources cannot answer the query.
INSUFFICIENT_CONTEXT_GAP = "Retrieved sources are insufficient to answer this query on their own."

STRICT_GROUNDING_RULE = (
    "Use only the sources below. If they do not contain the answer, say so plainly. "
    "Do not rely on outside knowledge and do not invent citations."
)
OPEN_GROUNDING_RULE = (
    "Prefer the sources below and cite them. Where they fall short you may add general legal "
    "background, and you must label that background as general knowledge, not as a citation."
)

ANSWER_SYSTEM_PROMPT = """You are a legal research assistant.

{grounding_rule}
Cite each factual claim to the source it comes from, by source name.
Quote a rule or holding only as the source states it.
{plan_guidance}
<sources>
{context}
</sources>"""

REFINEMENT_GUIDANCE_TEMPLATE = """

<refinement_feedback>
**Quality Review Feedback** (from the refinement pass):

<missing_elements>
{missing_elements}
</missing_elements>

<citation_gaps>
{citation_gaps}
</citation_gaps>

<clarity_issues>
{clarity_issues}
</clarity_issues>

<improvement_guidance>
{improvements}
</improvement_guidance>

**IMPORTANT**: Address the feedback above in your rewritten answer. Ensure all sub-questions are
answered, all claims are cited, and the language is clear and precise.
</refinement_feedback>"""


class SystemPromptBuilder(Protocol):
    """Builds the system message for the answer model. Pass your own to replace the default."""

    def __call__(
        self,
        *,
        context: str,
        plan: ResponsePlan | None,
        strict_grounding: bool,
        legal_task_type: str | None,
    ) -> str: ...


def build_context(sources: Sequence[Mapping[str, Any]]) -> str:
    """Number each source and join them into one context block."""
    blocks = []
    for number, chunk in enumerate(sources, 1):
        name = source_name(chunk) or "Unknown"
        blocks.append(f"[{number}] {name}\n{chunk.get('content', '')}")
    return "\n\n".join(blocks)


def format_plan_guidance(plan: ResponsePlan | None, legal_task_type: str | None = None) -> str:
    """Render a plan as guidance for the answer writer. Empty when there is no plan."""
    if plan is None:
        return ""
    lines = ["<response_plan>", f"Structure: {plan.response_structure}"]
    if legal_task_type:
        lines.append(f"Task type: {legal_task_type}")
    lines.append("Sub-questions to address:")
    for number, sq in enumerate(plan.sub_questions, 1):
        note = "" if sq.answerable else " (the sources do not cover this; say so plainly)"
        lines.append(f"{number}. [priority {sq.priority}] {sq.question}{note}")
    if plan.information_gaps:
        lines.append("Information gaps to acknowledge:")
        lines.extend(f"- {gap}" for gap in plan.information_gaps)
    if plan.risk_areas:
        lines.append("Areas needing care:")
        lines.extend(f"- {risk}" for risk in plan.risk_areas)
    if plan.strategy_notes:
        lines.append(f"Notes: {plan.strategy_notes}")
    lines.append("</response_plan>")
    return "\n".join(lines) + "\n"


def build_system_prompt(
    *,
    context: str,
    plan: ResponsePlan | None,
    strict_grounding: bool,
    legal_task_type: str | None = None,
) -> str:
    """The default answer system prompt: grounding rule, plan guidance and the sources."""
    return ANSWER_SYSTEM_PROMPT.format(
        grounding_rule=STRICT_GROUNDING_RULE if strict_grounding else OPEN_GROUNDING_RULE,
        plan_guidance=format_plan_guidance(plan, legal_task_type),
        context=context,
    )


@dataclass
class GenerationConfig:
    """Inputs for one generation run.

    ``sources`` are chunks as ``RetrievedChunk`` or mappings with ``text`` or ``content``,
    ``metadata`` and ``score``. They are normalized on construction.
    ``tier`` is ``standard`` or ``deep``; planning and refinement run on ``deep`` only.
    ``query_analysis`` may carry ``complexity`` (trivial, simple, moderate or complex) and
    ``legal_task_type``.
    ``confidence`` is not read by any stage and is kept so existing callers still construct.
    The ``enable_*`` flags switch whole stages off.
    """

    query: str
    sources: Sequence[RetrievedChunk | Mapping[str, Any]]
    messages: Sequence[Message] = ()
    context: str = ""
    tier: str = "standard"
    query_analysis: Mapping[str, Any] | None = None
    max_tokens: int = 4096
    temperature: float = 0.3
    strict_grounding: bool = True
    confidence: float = 0.5
    regeneration_max_tokens: int = 3000
    regeneration_temperature: float = 0.7
    enable_planning: bool = True
    enable_validation: bool = True
    enable_refinement: bool = True
    enable_verification: bool = True
    system_prompt_builder: SystemPromptBuilder | None = None

    def __post_init__(self) -> None:
        normalized = normalize_chunks(self.sources)
        self.sources = normalized
        if not self.messages:
            self.messages = [Message("user", self.query)]
        if not self.context:
            self.context = build_context(normalized)

    @property
    def is_deep(self) -> bool:
        return self.tier == "deep"


@dataclass
class PlanningResult:
    """Output of the planning stage. ``response_plan`` is None when planning was skipped or failed."""

    response_plan: ResponsePlan | None = None
    metrics: ActorMetrics | None = None
    query_complexity: str | None = None
    legal_task_type: str | None = None


@dataclass
class ValidationStageResult:
    """Output of the validation stage. Either part is None when it was skipped or failed."""

    deterministic: DeterministicValidationResult | None = None
    llm: ValidationResult | None = None
    metrics: ActorMetrics | None = None

    @property
    def recommendation(self) -> str | None:
        """The model's recommendation when there is one, else the deterministic one."""
        if self.llm is not None:
            return self.llm.recommendation
        if self.deterministic is not None:
            return self.deterministic.recommendation
        return None


@dataclass
class Completion:
    """The messages and limits for one answer model call."""

    messages: list[Message] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.3
    system_message: str = ""


@dataclass
class RefinementResult:
    """Output of the refinement stage. ``regenerated`` means a rewrite is wanted."""

    regenerated: bool = False
    full_response: str = ""
    refinement_guidance: str = ""
    metrics: ActorMetrics | None = None


@dataclass
class GenerationResult:
    """The final answer, the artifacts that produced it and a summary of each stage."""

    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    plan: ResponsePlan | None = None
    actor_metrics: list[ActorMetrics] = field(default_factory=list)
    verification: HallucinationCheckResult | None = None


def _all_web(sources: Sequence[Mapping[str, Any]]) -> bool:
    """True when every source is a web result, which gives a plan nothing corpus-side to check."""
    return all(
        (src.get("metadata") or {}).get("source_type") == "web" or src.get("source_type") == "web"
        for src in sources
    )


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


async def run_planning(config: GenerationConfig, planner: PlanningActor) -> PlanningResult:
    """Run the planning actor for deep-tier, non-trivial queries.

    Returns a result with ``response_plan`` None when planning is skipped or fails.
    """
    analysis = config.query_analysis or {}
    query_complexity = analysis.get("complexity")
    task_type = analysis.get("legal_task_type")
    legal_task_type = _enum_text(task_type) if task_type else None
    result = PlanningResult(query_complexity=query_complexity, legal_task_type=legal_task_type)

    if not (config.enable_planning and config.is_deep) or query_complexity == "trivial":
        return result
    if not config.sources:
        logger.debug("planning_skipped_no_chunks")
        return result

    current_query = config.messages[-1].content if config.messages else config.query
    try:
        plan, metrics = await planner.run_with_metrics(
            query=current_query,
            chunks=config.sources,
            query_metadata={
                "query_type": legal_task_type or "general",
                # The planner reads a 0.0 to 1.0 number here. Analysis gives a label (simple,
                # moderate, complex) and `confidence` is not a complexity, so send none and let
                # the planner use its own default.
                "tier": config.tier,
            },
        )
        result.response_plan = plan
        result.metrics = metrics
    except Exception as exc:  # noqa: BLE001 - planning is optional, so any failure degrades to no plan
        logger.warning("planning_skipped_error", error=str(exc), error_type=type(exc).__name__)
    return result


async def run_validation(
    config: GenerationConfig,
    plan: ResponsePlan | None,
    validator: ValidationActor,
) -> ValidationStageResult:
    """Run the deterministic plan check, then the validation actor.

    The actor is skipped when the deterministic check recommends ``fallback_to_general``.
    A stage that raises is logged and left as None.
    """
    result = ValidationStageResult()
    if not config.enable_validation or plan is None or not config.sources:
        return result
    if _all_web(config.sources):
        logger.debug("validation_skipped_web_only", source_count=len(config.sources))
        return result

    try:
        deterministic = validate_plan_deterministically(plan=plan, chunks=list(config.sources))
        result.deterministic = deterministic
        if not deterministic.is_valid and deterministic.recommendation == "fallback_to_general":
            return result
    except Exception as exc:  # noqa: BLE001 - validation never blocks an answer
        logger.error("deterministic_validation_error", error=str(exc))

    try:
        validation, metrics = await validator.run_with_metrics(
            plan=plan, available_chunks=config.sources
        )
        result.llm = validation
        result.metrics = metrics
    except Exception as exc:  # noqa: BLE001 - validation never blocks an answer
        logger.error("llm_validation_error", error=str(exc))
    return result


def apply_validation_recommendation(
    config: GenerationConfig,
    plan: ResponsePlan | None,
    recommendation: str | None,
) -> tuple[GenerationConfig, ResponsePlan | None]:
    """Adjust the generation inputs to match what validation found.

    ``fallback_to_general`` relaxes strict grounding, marks every sub-question unanswerable and
    records the gap, so the answer says what the sources lack.
    ``retrieve_more`` marks the sub-questions that have no chunk mapping as unanswerable.
    Anything else leaves both inputs unchanged.
    """
    if plan is None:
        return config, plan

    if recommendation == "fallback_to_general":
        adjusted = plan.model_copy(
            update={
                "sub_questions": [
                    sq.model_copy(update={"answerable": False}) for sq in plan.sub_questions
                ],
                "information_gaps": [*plan.information_gaps, INSUFFICIENT_CONTEXT_GAP],
            }
        )
        logger.info("validation_fallback_applied", strict_grounding_overridden=True)
        return replace(config, strict_grounding=False), adjusted

    if recommendation == "retrieve_more":
        pruned = [
            sq.model_copy(update={"answerable": False}) if not sq.chunk_indices else sq
            for sq in plan.sub_questions
        ]
        unsupported = sum(1 for sq in pruned if not sq.answerable)
        if unsupported > 0:
            logger.debug(
                "validation_plan_pruned", total=len(plan.sub_questions), unsupported=unsupported
            )
            return config, plan.model_copy(update={"sub_questions": pruned})

    return config, plan


def build_completion(
    config: GenerationConfig,
    plan: ResponsePlan | None,
    legal_task_type: str | None = None,
    query_complexity: str | None = None,
) -> Completion:
    """Build the system prompt and apply the complexity-aware token cap."""
    builder = config.system_prompt_builder or build_system_prompt
    system_message = builder(
        context=config.context,
        plan=plan,
        strict_grounding=config.strict_grounding,
        legal_task_type=legal_task_type,
    )
    cap = COMPLEXITY_TOKEN_CAPS.get(query_complexity or "complex", DEFAULT_COMPLEXITY_CAP)
    effective = min(config.max_tokens, cap)
    if query_complexity and effective < config.max_tokens:
        logger.debug(
            "complexity_aware_max_tokens",
            query_complexity=query_complexity,
            original=config.max_tokens,
            capped=effective,
        )
    return Completion(
        messages=[Message("system", system_message), *config.messages],
        max_tokens=effective,
        temperature=config.temperature,
        system_message=system_message,
    )


def build_refinement_guidance(refinement: RefinementSuggestions) -> str:
    """Render a critique as feedback to append to the last user message of a rewrite."""
    return REFINEMENT_GUIDANCE_TEMPLATE.format(
        missing_elements="\n".join(f"- {item}" for item in refinement.missing_elements),
        citation_gaps="\n".join(f"- {item}" for item in refinement.citation_gaps),
        clarity_issues="\n".join(f"- {item}" for item in refinement.clarity_issues),
        improvements=refinement.suggested_improvements,
    )


def build_regeneration(
    config: GenerationConfig,
    plan: ResponsePlan | None,
    refinement_guidance: str,
    legal_task_type: str | None = None,
) -> Completion:
    """Build the rewrite call: the system prompt again, with the critique on the last message."""
    builder = config.system_prompt_builder or build_system_prompt
    system_message = builder(
        context=config.context,
        plan=plan,
        strict_grounding=config.strict_grounding,
        legal_task_type=legal_task_type,
    )
    messages = [Message("system", system_message), *config.messages]
    last = messages[-1]
    messages[-1] = Message(last.role, last.content + refinement_guidance)
    return Completion(
        messages=messages,
        max_tokens=config.regeneration_max_tokens,
        temperature=config.regeneration_temperature,
        system_message=system_message,
    )


async def run_refinement(
    config: GenerationConfig,
    full_response: str,
    plan: ResponsePlan | None,
    refiner: RefinementActor,
) -> RefinementResult:
    """Run the refinement actor for deep-tier plans and say whether a rewrite is wanted.

    This function does not rewrite. It returns the guidance and the caller rewrites.
    """
    result = RefinementResult(full_response=full_response)
    if not (config.enable_refinement and config.is_deep and plan and full_response):
        return result
    if config.sources and _all_web(config.sources):
        return result

    try:
        refinement, metrics = await refiner.run_with_metrics(
            original_response=full_response, plan=plan, chunks=config.sources
        )
        result.metrics = metrics
        if (
            refinement.should_regenerate
            and refinement.suggested_improvements
            and refinement.confidence_score < REFINEMENT_CONFIDENCE_CEILING
        ):
            result.regenerated = True
            result.refinement_guidance = build_refinement_guidance(refinement)
    except Exception as exc:  # noqa: BLE001 - refinement is optional, so keep the draft
        logger.error("refinement_error", error=str(exc), error_type=type(exc).__name__)
    return result


async def run_verification(
    detector: HallucinationDetector,
    full_response: str,
    sources: Sequence[Mapping[str, Any]],
) -> HallucinationCheckResult | None:
    """Score an answer against its sources with the injected detector.

    Returns None when there is nothing to check or the detector raises, so verification never
    blocks an answer.
    """
    if not sources or not full_response:
        return None
    try:
        result = await detector.check(full_response, list(sources))
    except Exception as exc:  # noqa: BLE001 - verification is advisory
        logger.warning("verification_failed", error=str(exc), error_type=type(exc).__name__)
        return None
    logger.debug(
        "verification_complete",
        score=result.score,
        verified=result.verified,
        total_claims=result.total_claims,
        supported_claims=result.supported_claims,
    )
    return result


def _verification_summary(result: HallucinationCheckResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "verified": result.verified,
        "total_claims": result.total_claims,
        "supported_claims": result.supported_claims,
        "unsupported_claims": [
            {"claim": detail.claim, "status": detail.result.value}
            for detail in result.claim_details
            if detail.result is not VerificationResult.SUPPORTED
        ][:MAX_REPORTED_CLAIMS],
    }


async def generate_answer_buffered(
    config: GenerationConfig,
    model: LanguageModel,
    *,
    actor_model: LanguageModel | None = None,
    actor_configs: Mapping[str, ActorConfig] | None = None,
    detector: HallucinationDetector | None = None,
) -> GenerationResult:
    """Run the full pipeline without streaming.

    ``model`` writes the answer and any rewrite. ``actor_model`` runs the planning, validation and
    refinement actors and defaults to ``model``. ``actor_configs`` maps ``planning``,
    ``validation`` or ``refinement`` to an ``ActorConfig``. Without a ``detector`` the
    verification stage is skipped.
    Errors from the answer model propagate; a failed rewrite keeps the first draft.
    """
    actor_llm = actor_model or model
    configs = actor_configs or {}
    planner = PlanningActor(actor_llm, configs.get("planning"))
    validator = ValidationActor(actor_llm, configs.get("validation"))
    refiner = RefinementActor(actor_llm, configs.get("refinement"))

    metadata: dict[str, Any] = {}
    actor_metrics: list[ActorMetrics] = []

    planning = await run_planning(config, planner)
    plan = planning.response_plan
    metadata["planning"] = {
        "has_plan": plan is not None,
        "query_complexity": planning.query_complexity,
        "legal_task_type": planning.legal_task_type,
    }
    if planning.metrics:
        actor_metrics.append(planning.metrics)

    validation = await run_validation(config, plan, validator)
    metadata["validation"] = {
        "deterministic": validation.deterministic is not None,
        "llm": validation.llm is not None,
    }
    if validation.metrics:
        actor_metrics.append(validation.metrics)

    effective_config, effective_plan = apply_validation_recommendation(
        config, plan, validation.recommendation
    )
    completion = build_completion(
        effective_config,
        effective_plan,
        legal_task_type=planning.legal_task_type,
        query_complexity=planning.query_complexity,
    )

    full_response = await model.generate(
        completion.messages,
        max_tokens=completion.max_tokens,
        temperature=completion.temperature,
    )

    if plan is not None:
        refinement = await run_refinement(config, full_response, plan, refiner)
        metadata["refinement"] = {"regenerated": refinement.regenerated}
        if refinement.metrics:
            actor_metrics.append(refinement.metrics)

        if refinement.regenerated:
            rewrite = build_regeneration(
                effective_config,
                effective_plan,
                refinement.refinement_guidance,
                planning.legal_task_type,
            )
            try:
                full_response = await model.generate(
                    rewrite.messages,
                    max_tokens=rewrite.max_tokens,
                    temperature=rewrite.temperature,
                )
            except Exception as exc:  # noqa: BLE001 - keep the first draft on a failed rewrite
                logger.error("buffered_regeneration_error", error=str(exc))

    verification: HallucinationCheckResult | None = None
    if config.enable_verification and detector is not None:
        verification = await run_verification(detector, full_response, config.sources)
        if verification is not None:
            metadata["verification"] = _verification_summary(verification)

    metadata["effective_max_tokens"] = completion.max_tokens
    metadata["actor_metrics"] = [m.model_dump() for m in actor_metrics]
    return GenerationResult(
        answer=full_response,
        metadata=metadata,
        plan=plan,
        actor_metrics=actor_metrics,
        verification=verification,
    )
