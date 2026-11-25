"""Generation orchestration: plan, validate, write, refine once, verify, and fail open."""

from __future__ import annotations

import pytest

from legal_rag.agents.actors import ActorConfig
from legal_rag.agents.generation import (
    DEFAULT_COMPLEXITY_CAP,
    INSUFFICIENT_CONTEXT_GAP,
    OPEN_GROUNDING_RULE,
    STRICT_GROUNDING_RULE,
    GenerationConfig,
    apply_validation_recommendation,
    build_completion,
    generate_answer_buffered,
)
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.types import Message
from legal_rag.verification.grounding.actor_schemas import (
    RefinementSuggestions,
    ResponsePlan,
    SubQuestion,
    ValidationResult,
)
from legal_rag.verification.grounding.detector import (
    ClaimVerification,
    HallucinationCheckResult,
    HallucinationDetector,
    VerificationResult,
)

SOURCES = [
    {
        "text": "A tenant may end a lease with thirty days written notice to the landlord.",
        "metadata": {"source_name": "Smith v. Jones"},
        "score": 0.92,
    },
    {
        "text": "The landlord must return the security deposit within twenty one days.",
        "metadata": {"source_name": "Tenancy Act Section 12"},
        "score": 0.85,
    },
]

DRAFT = "A tenant may end a lease with thirty days written notice to the landlord."
REWRITE = "The tenant gives thirty days written notice, and the deposit returns in twenty one days."


def plan_for(sources: list[str] | None = None) -> ResponsePlan:
    return ResponsePlan(
        sub_questions=[
            SubQuestion(question="How does a tenant end a lease?", chunk_indices=[0]),
            SubQuestion(question="When is the deposit returned?", chunk_indices=[1]),
        ],
        required_sources=sources if sources is not None else ["Smith v. Jones"],
        response_structure="Procedural",
        information_gaps=["Notice by email is not addressed."],
        strategy_notes="Lead with the notice period.",
    )


def critique(*, regenerate: bool, confidence: float = 0.4) -> RefinementSuggestions:
    return RefinementSuggestions(
        missing_elements=["deposit timing"],
        citation_gaps=["notice period"],
        should_regenerate=regenerate,
        suggested_improvements="Add the deposit deadline and cite each rule.",
        confidence_score=confidence,
    )


class ScriptedModel(FakeLanguageModel):
    """Answers ``generate`` from a script and records every call, keyed by schema or role."""

    def __init__(self, replies, structured=None, *, fail_generate_after: int | None = None) -> None:
        super().__init__(structured=structured)
        self.replies = list(replies)
        self.fail_generate_after = fail_generate_after
        self.generate_calls: list[dict] = []
        self.structured_calls: list[str] = []
        self.structured_messages: dict[str, list[Message]] = {}

    async def generate(self, messages, *, max_tokens=None, temperature=None):
        index = len(self.generate_calls)
        self.generate_calls.append(
            {"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature}
        )
        if self.fail_generate_after is not None and index >= self.fail_generate_after:
            raise RuntimeError("answer model failed")
        return self.replies[min(index, len(self.replies) - 1)]

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        self.structured_calls.append(schema.__name__)
        self.structured_messages[schema.__name__] = list(messages)
        return await super().generate_structured(messages, schema, max_tokens=max_tokens)


class BrokenActorsModel(ScriptedModel):
    """Every structured call raises, as when the actor model is down."""

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        self.structured_calls.append(schema.__name__)
        raise RuntimeError("actor model down")


class RecordingDetector(HallucinationDetector):
    """The real detector, recording what it was asked to check."""

    def __init__(self) -> None:
        super().__init__()
        self.checked: list[tuple[str, list[dict]]] = []

    async def check(self, response, sources):
        self.checked.append((response, sources))
        return await super().check(response, sources)


class FailingDetector(HallucinationDetector):
    async def check(self, response, sources):
        raise RuntimeError("detector down")


class PartialClaimDetector(HallucinationDetector):
    """Reports what the real detector reports for one grounded and one partly grounded claim."""

    async def check(self, response, sources):
        return HallucinationCheckResult(
            score=1.0,
            verified=True,
            total_claims=2,
            supported_claims=1,
            unsupported_claims=[],
            claim_details=[
                ClaimVerification(
                    claim="Fully grounded claim.",
                    result=VerificationResult.SUPPORTED,
                    confidence=1.0,
                ),
                ClaimVerification(
                    claim="Partly grounded claim.",
                    result=VerificationResult.NEUTRAL,
                    confidence=0.75,
                ),
            ],
        )


def deep_config(**overrides) -> GenerationConfig:
    fields = {"query": "How does a tenant end a lease?", "sources": SOURCES, "tier": "deep"}
    return GenerationConfig(**{**fields, **overrides})


def structured(**extra):
    return {
        "ResponsePlan": plan_for(),
        "ValidationResult": ValidationResult(is_valid=True, recommendation="proceed"),
        "RefinementSuggestions": critique(regenerate=False),
        **extra,
    }


# -- the happy path ------------------------------------------------------------------------------


async def test_the_plan_is_produced_and_passed_to_the_answer_prompt():
    model = ScriptedModel([DRAFT], structured())
    result = await generate_answer_buffered(deep_config(), model)

    assert result.answer == DRAFT
    assert result.plan is not None and result.plan.response_structure == "Procedural"
    assert model.structured_calls == ["ResponsePlan", "ValidationResult", "RefinementSuggestions"]
    assert len(model.generate_calls) == 1

    system = model.generate_calls[0]["messages"][0]
    assert system.role == "system"
    assert "Structure: Procedural" in system.content
    assert "How does a tenant end a lease?" in system.content
    assert "Notice by email is not addressed." in system.content
    assert "[1] Smith v. Jones" in system.content
    assert STRICT_GROUNDING_RULE in system.content
    assert result.metadata["planning"]["has_plan"] is True
    assert result.metadata["refinement"] == {"regenerated": False}


async def test_actor_metrics_are_recorded_for_each_actor_that_ran():
    model = ScriptedModel([DRAFT], structured())
    result = await generate_answer_buffered(deep_config(), model)

    assert [m.actor_name for m in result.actor_metrics] == ["planning", "validation", "refinement"]
    for metrics in result.actor_metrics:
        assert metrics.llm_calls == 1
        assert metrics.tokens_used > 0
        assert metrics.error_occurred is False
    assert [m["actor_name"] for m in result.metadata["actor_metrics"]] == [
        "planning",
        "validation",
        "refinement",
    ]


async def test_the_standard_tier_writes_the_answer_without_any_actor_call():
    model = ScriptedModel([DRAFT], structured())
    result = await generate_answer_buffered(deep_config(tier="standard"), model)

    assert model.structured_calls == []
    assert result.plan is None and result.actor_metrics == []
    assert result.metadata["planning"]["has_plan"] is False
    assert "refinement" not in result.metadata
    assert result.answer == DRAFT


async def test_a_trivial_query_skips_planning_even_on_the_deep_tier():
    model = ScriptedModel([DRAFT], structured())
    config = deep_config(query_analysis={"complexity": "trivial"})
    result = await generate_answer_buffered(config, model)

    assert model.structured_calls == []
    assert result.plan is None
    assert model.generate_calls[0]["max_tokens"] == 500
    assert result.metadata["effective_max_tokens"] == 500


async def test_the_complexity_cap_limits_answer_tokens_and_unknown_complexity_uses_the_default():
    capped = build_completion(deep_config(max_tokens=10000), None, query_complexity="simple")
    assert capped.max_tokens == 4000

    unknown = build_completion(deep_config(max_tokens=99999), None, query_complexity="exotic")
    assert unknown.max_tokens == DEFAULT_COMPLEXITY_CAP

    small = build_completion(deep_config(max_tokens=100), None, query_complexity="complex")
    assert small.max_tokens == 100


async def test_the_planner_gets_no_complexity_taken_from_the_confidence_field():
    model = ScriptedModel([DRAFT], structured())
    config = deep_config(confidence=0.91, query_analysis={"complexity": "complex"})
    await generate_answer_buffered(config, model)

    prompt = model.structured_messages["ResponsePlan"][-1].content
    assert "Estimated Complexity: 0.50" in prompt
    assert "0.91" not in prompt


# -- validation ----------------------------------------------------------------------------------


async def test_a_fallback_recommendation_relaxes_grounding_and_marks_the_gap():
    verdict = ValidationResult(is_valid=False, recommendation="fallback_to_general")
    model = ScriptedModel([DRAFT], structured(ValidationResult=verdict))
    result = await generate_answer_buffered(deep_config(), model)

    system = model.generate_calls[0]["messages"][0].content
    assert OPEN_GROUNDING_RULE in system and STRICT_GROUNDING_RULE not in system
    assert INSUFFICIENT_CONTEXT_GAP in system
    assert "the sources do not cover this" in system
    # The plan returned to the caller is the planner's, not the adjusted copy.
    assert all(sq.answerable for sq in result.plan.sub_questions)


async def test_the_rewrite_after_a_fallback_keeps_the_relaxed_grounding_and_the_gap():
    verdict = ValidationResult(is_valid=False, recommendation="fallback_to_general")
    model = ScriptedModel(
        [DRAFT, REWRITE],
        structured(ValidationResult=verdict, RefinementSuggestions=critique(regenerate=True)),
    )
    result = await generate_answer_buffered(deep_config(), model)

    assert result.answer == REWRITE
    assert len(model.generate_calls) == 2
    for call in model.generate_calls:
        system = call["messages"][0].content
        assert OPEN_GROUNDING_RULE in system and STRICT_GROUNDING_RULE not in system
        assert INSUFFICIENT_CONTEXT_GAP in system


async def test_a_critical_deterministic_failure_skips_the_validation_actor_and_still_applies():
    ungroundable = ResponsePlan(
        sub_questions=[SubQuestion(question="Which sanctions apply to spoliation of evidence?")],
        required_sources=["Doe v. Roe"],
        response_structure="IRAC",
    )
    model = ScriptedModel([DRAFT], structured(ResponsePlan=ungroundable))
    result = await generate_answer_buffered(deep_config(), model)

    assert "ValidationResult" not in model.structured_calls
    assert result.metadata["validation"] == {"deterministic": True, "llm": False}
    assert OPEN_GROUNDING_RULE in model.generate_calls[0]["messages"][0].content


def test_retrieve_more_marks_unmapped_sub_questions_unanswerable():
    plan = ResponsePlan(
        sub_questions=[
            SubQuestion(question="Mapped", chunk_indices=[0]),
            SubQuestion(question="Unmapped", chunk_indices=[]),
        ],
        response_structure="IRAC",
    )
    config = deep_config()
    same_config, adjusted = apply_validation_recommendation(config, plan, "retrieve_more")

    assert same_config is config
    assert [sq.answerable for sq in adjusted.sub_questions] == [True, False]
    assert [sq.answerable for sq in plan.sub_questions] == [True, True]

    untouched_config, untouched = apply_validation_recommendation(config, plan, "proceed")
    assert untouched is plan and untouched_config is config
    assert apply_validation_recommendation(config, None, "fallback_to_general") == (config, None)


async def test_web_only_sources_skip_validation_and_refinement():
    web = [{"text": "A web page about leases.", "metadata": {"source_type": "web"}, "score": 0.7}]
    model = ScriptedModel([DRAFT], structured())
    result = await generate_answer_buffered(deep_config(sources=web), model)

    assert model.structured_calls == ["ResponsePlan"]
    assert result.metadata["validation"] == {"deterministic": False, "llm": False}
    assert result.metadata["refinement"] == {"regenerated": False}


# -- refinement and the single rewrite -----------------------------------------------------------


async def test_a_critique_that_asks_for_a_rewrite_triggers_exactly_one_rewrite():
    model = ScriptedModel(
        [DRAFT, REWRITE], structured(RefinementSuggestions=critique(regenerate=True))
    )
    result = await generate_answer_buffered(deep_config(), model)

    assert model.structured_calls.count("RefinementSuggestions") == 1
    assert len(model.generate_calls) == 2
    assert result.answer == REWRITE
    assert result.metadata["refinement"] == {"regenerated": True}

    rewrite_messages = model.generate_calls[1]["messages"]
    assert rewrite_messages[-1].role == "user"
    assert "<refinement_feedback>" in rewrite_messages[-1].content
    assert "Add the deposit deadline and cite each rule." in rewrite_messages[-1].content
    assert "- deposit timing" in rewrite_messages[-1].content
    assert model.generate_calls[1]["max_tokens"] == 3000
    assert model.generate_calls[1]["temperature"] == 0.7


async def test_no_rewrite_when_the_critique_is_confident_or_does_not_ask_for_one():
    confident = ScriptedModel(
        [DRAFT, REWRITE],
        structured(RefinementSuggestions=critique(regenerate=True, confidence=0.9)),
    )
    result = await generate_answer_buffered(deep_config(), confident)
    assert result.answer == DRAFT and len(confident.generate_calls) == 1

    declined = ScriptedModel([DRAFT, REWRITE], structured())
    result = await generate_answer_buffered(deep_config(), declined)
    assert result.answer == DRAFT and len(declined.generate_calls) == 1


async def test_a_failed_rewrite_keeps_the_first_draft():
    model = ScriptedModel(
        [DRAFT],
        structured(RefinementSuggestions=critique(regenerate=True)),
        fail_generate_after=1,
    )
    result = await generate_answer_buffered(deep_config(), model)

    assert result.answer == DRAFT
    assert len(model.generate_calls) == 2


# -- fail-open -----------------------------------------------------------------------------------


async def test_the_answer_is_written_when_every_actor_call_raises():
    model = BrokenActorsModel([DRAFT])
    result = await generate_answer_buffered(deep_config(), model)

    assert result.answer == DRAFT
    assert result.plan is None and result.actor_metrics == []
    assert result.metadata["planning"]["has_plan"] is False
    assert model.structured_calls == ["ResponsePlan"]
    assert STRICT_GROUNDING_RULE in model.generate_calls[0]["messages"][0].content


async def test_the_answer_is_written_when_only_validation_and_refinement_fail():
    model = ScriptedModel([DRAFT], {"ResponsePlan": plan_for()})
    result = await generate_answer_buffered(deep_config(), model)

    assert result.answer == DRAFT
    assert result.plan is not None
    assert result.metadata["validation"] == {"deterministic": True, "llm": False}
    assert result.metadata["refinement"] == {"regenerated": False}
    assert [m.actor_name for m in result.actor_metrics] == ["planning"]


async def test_an_answer_model_failure_is_not_swallowed():
    model = ScriptedModel([DRAFT], fail_generate_after=0)
    with pytest.raises(RuntimeError, match="answer model failed"):
        await generate_answer_buffered(deep_config(tier="standard"), model)


# -- verification --------------------------------------------------------------------------------


async def test_the_verification_step_calls_the_injected_detector_on_the_final_answer():
    detector = RecordingDetector()
    model = ScriptedModel(
        [DRAFT, REWRITE], structured(RefinementSuggestions=critique(regenerate=True))
    )
    result = await generate_answer_buffered(deep_config(), model, detector=detector)

    assert len(detector.checked) == 1
    checked_answer, checked_sources = detector.checked[0]
    assert checked_answer == REWRITE
    assert [s["text"] for s in checked_sources] == [s["text"] for s in SOURCES]
    assert result.verification is not None
    summary = result.metadata["verification"]
    assert summary["total_claims"] == result.verification.total_claims
    assert summary["verified"] == result.verification.verified
    assert isinstance(summary["unsupported_claims"], list)


async def test_the_summary_passes_the_detector_numbers_through_and_lists_partly_grounded_claims():
    model = ScriptedModel([DRAFT])
    result = await generate_answer_buffered(
        deep_config(tier="standard"), model, detector=PartialClaimDetector()
    )

    assert result.metadata["verification"] == {
        "score": 1.0,
        "verified": True,
        "total_claims": 2,
        "supported_claims": 1,
        "unsupported_claims": [{"claim": "Partly grounded claim.", "status": "neutral"}],
    }


async def test_verification_runs_on_the_standard_tier_and_is_skipped_without_a_detector():
    detector = RecordingDetector()
    model = ScriptedModel([DRAFT])
    result = await generate_answer_buffered(deep_config(tier="standard"), model, detector=detector)
    assert len(detector.checked) == 1 and result.verification is not None

    no_detector = await generate_answer_buffered(
        deep_config(tier="standard"), ScriptedModel([DRAFT])
    )
    assert no_detector.verification is None and "verification" not in no_detector.metadata

    switched_off = RecordingDetector()
    await generate_answer_buffered(
        deep_config(tier="standard", enable_verification=False),
        ScriptedModel([DRAFT]),
        detector=switched_off,
    )
    assert switched_off.checked == []


async def test_a_failing_detector_does_not_block_the_answer():
    model = ScriptedModel([DRAFT])
    result = await generate_answer_buffered(
        deep_config(tier="standard"), model, detector=FailingDetector()
    )
    assert result.answer == DRAFT
    assert result.verification is None and "verification" not in result.metadata


# -- configuration -------------------------------------------------------------------------------


async def test_a_custom_system_prompt_builder_and_actor_configs_are_used():
    seen: dict = {}

    def builder(*, context, plan, strict_grounding, legal_task_type):
        seen.update(plan=plan, strict=strict_grounding)
        return "CUSTOM SYSTEM PROMPT"

    model = ScriptedModel([DRAFT], structured())
    actor_model = ScriptedModel([], structured())
    result = await generate_answer_buffered(
        deep_config(system_prompt_builder=builder),
        model,
        actor_model=actor_model,
        actor_configs={"planning": ActorConfig(max_tokens=77)},
    )

    assert model.generate_calls[0]["messages"][0] == Message("system", "CUSTOM SYSTEM PROMPT")
    assert seen["plan"] is result.plan and seen["strict"] is True
    assert model.structured_calls == []
    assert actor_model.structured_calls == [
        "ResponsePlan",
        "ValidationResult",
        "RefinementSuggestions",
    ]


def test_config_defaults_the_message_and_builds_a_numbered_context():
    config = GenerationConfig(query="Q?", sources=SOURCES)

    assert config.messages == [Message("user", "Q?")]
    assert config.context.startswith("[1] Smith v. Jones\n")
    assert "[2] Tenancy Act Section 12" in config.context
    assert config.is_deep is False
