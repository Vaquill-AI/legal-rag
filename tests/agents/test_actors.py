"""Actors: planning, validation, refinement, the model-free validator and the shared base."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from legal_rag.agents.actors import (
    ActorConfig,
    ActorError,
    PlanningActor,
    RefinementActor,
    ValidationActor,
    normalize_chunks,
    plan_response,
    quick_plan_check,
    refine_response,
    select_response_structure,
    validate_plan_deterministically,
    validate_response_plan,
)
from legal_rag.agents.actors import base as actors_base
from legal_rag.agents.actors.planning import GENERIC_SOURCE
from legal_rag.agents.actors.refinement import MAX_REVIEW_CHARS
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.providers.usage import estimate_tokens
from legal_rag.types import RetrievedChunk
from legal_rag.verification.grounding.actor_schemas import (
    RefinementSuggestions,
    ResponsePlan,
    SubQuestion,
    ValidationResult,
)

CHUNKS = [
    {
        "text": "A lease may be ended by the tenant with thirty days written notice to the landlord.",
        "metadata": {"source_name": "Smith v. Jones", "page_number": 4},
        "score": 0.91,
    },
    {
        "text": "The landlord must return the security deposit within twenty one days.",
        "metadata": {"title": "Residential Tenancy Act, Section 12"},
        "score": 0.84,
    },
]


def make_plan(**overrides) -> ResponsePlan:
    fields = {
        "sub_questions": [
            SubQuestion(question="How does a tenant end a lease?", chunk_indices=[0]),
            SubQuestion(question="When is the deposit returned?", chunk_indices=[1, 9]),
        ],
        "required_sources": ["Smith v. Jones"],
        "response_structure": "IRAC",
    }
    return ResponsePlan(**{**fields, **overrides})


class RaisingModel(FakeLanguageModel):
    """A model whose structured call raises the given exception."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        self.calls.append(messages)
        raise self.error


class RecordingModel(FakeLanguageModel):
    """A model that accepts a temperature argument and records what it was given."""

    def __init__(self, structured) -> None:
        super().__init__(structured=structured)
        self.seen: dict = {}

    async def generate_structured(self, messages, schema, *, max_tokens=None, temperature=None):
        self.seen = {"max_tokens": max_tokens, "temperature": temperature}
        return await super().generate_structured(messages, schema, max_tokens=max_tokens)


def schema_error() -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        ResponsePlan.model_validate({})
    return caught.value


# -- chunk handling ------------------------------------------------------------------------------


def test_normalize_chunks_fills_both_text_keys_and_accepts_retrieved_chunks():
    normalized = normalize_chunks(
        [
            {"text": "alpha", "metadata": {"source_name": "A"}},
            {"content": "beta"},
            RetrievedChunk(id="c3", text="gamma", score=0.5, metadata={"title": "G"}),
        ]
    )
    assert [c["content"] for c in normalized] == ["alpha", "beta", "gamma"]
    assert [c["text"] for c in normalized] == ["alpha", "beta", "gamma"]
    assert normalized[1]["metadata"] == {} and normalized[1]["score"] == 0.0
    assert normalized[2]["score"] == 0.5 and normalized[2]["id"] == "c3"


# -- planning ------------------------------------------------------------------------------------


def test_structure_heuristic_is_confident_only_when_unambiguous():
    assert (
        select_response_structure("Compare the two notice rules, versus the old one")
        == "Comparative"
    )
    assert select_response_structure("What is the meaning of consideration?") == "Direct"
    assert select_response_structure("tenant rights") is None
    # One match each for two structures is ambiguous, so the model decides.
    assert select_response_structure("What is the timeline?") is None


async def test_plan_is_produced_with_a_model_chosen_structure():
    model = FakeLanguageModel(
        structured={"ResponsePlan": make_plan(response_structure="Procedural")}
    )
    plan = await PlanningActor(model).execute(query="Tenant lease termination", chunks=CHUNKS)

    assert isinstance(plan, ResponsePlan)
    assert plan.response_structure == "Procedural"
    assert len(model.calls) == 1
    user_prompt = model.calls[0][1].content
    assert "Tenant lease termination" in user_prompt
    assert "0. [Smith v. Jones] score:0.910 p.4" in user_prompt
    assert "- Residential Tenancy Act, Section 12" in user_prompt
    assert "Choose the best response_structure" in model.calls[0][0].content


async def test_heuristic_structure_overrides_the_model_and_chunk_indices_are_clamped():
    model = FakeLanguageModel(structured={"ResponsePlan": make_plan(response_structure="Direct")})
    plan = await PlanningActor(model).execute(
        query="Compare tenant versus landlord duties", chunks=CHUNKS
    )

    assert plan.response_structure == "Comparative"
    assert 'MUST be "Comparative"' in model.calls[0][0].content
    assert plan.sub_questions[1].chunk_indices == [1]


async def test_planning_does_not_mutate_the_models_reply():
    canned = make_plan(required_sources=[])
    model = FakeLanguageModel(structured={"ResponsePlan": canned})
    plan = await PlanningActor(model).execute(query="Tenant duties", chunks=CHUNKS)

    assert plan.required_sources == ["Residential Tenancy Act, Section 12", "Smith v. Jones"]
    assert canned.required_sources == []
    assert canned.sub_questions[1].chunk_indices == [1, 9]


async def test_required_sources_fall_back_to_a_generic_name_when_chunks_are_unnamed():
    model = FakeLanguageModel(structured={"ResponsePlan": make_plan(required_sources=[])})
    plan = await PlanningActor(model).execute(
        query="Tenant duties", chunks=[{"text": "Some text."}]
    )
    assert plan.required_sources == [GENERIC_SOURCE]


async def test_an_unusable_plan_degrades_to_a_single_question_fallback():
    model = RaisingModel(schema_error())
    plan = await PlanningActor(model).execute(query="Compare notice versus deposit", chunks=CHUNKS)

    assert [sq.question for sq in plan.sub_questions] == ["Compare notice versus deposit"]
    assert plan.sub_questions[0].chunk_indices == [0, 1]
    assert plan.response_structure == "Comparative"
    assert plan.strategy_notes is None
    assert plan.required_sources == ["Residential Tenancy Act, Section 12", "Smith v. Jones"]


async def test_planning_metrics_are_recorded():
    model = FakeLanguageModel(structured={"ResponsePlan": make_plan()})
    plan, metrics = await plan_response(model, "Tenant duties", CHUNKS, {"query_type": "general"})

    assert isinstance(plan, ResponsePlan)
    assert metrics.actor_name == "planning"
    assert metrics.llm_calls == 1
    assert metrics.tokens_used > 0
    assert metrics.execution_time_ms >= 0
    assert metrics.error_occurred is False and metrics.error_message is None


def test_actors_use_the_one_shared_token_estimate():
    assert actors_base.estimate_tokens is estimate_tokens
    assert actors_base.estimate_tokens("abc") == 1


async def test_tokens_used_is_the_shared_estimate_of_the_prompt_plus_the_reply():
    class RecordingModel(FakeLanguageModel):
        async def generate_structured(self, messages, schema, *, max_tokens=None):
            self.sent = list(messages)
            return await super().generate_structured(messages, schema, max_tokens=max_tokens)

    plan = make_plan()
    model = RecordingModel(structured={"ResponsePlan": plan})
    _, metrics = await plan_response(model, "Tenant duties", CHUNKS, {"query_type": "general"})

    expected = sum(estimate_tokens(m.content) for m in model.sent)
    expected += estimate_tokens(plan.model_dump_json())
    assert metrics.tokens_used == expected


async def test_a_model_failure_raises_actor_error_and_missing_input_raises_value_error():
    failing = PlanningActor(RaisingModel(RuntimeError("model down")))
    with pytest.raises(ActorError, match="model down"):
        await failing.run_with_metrics(query="q", chunks=CHUNKS)

    with pytest.raises(ValueError, match="chunks parameter is required"):
        await PlanningActor(FakeLanguageModel()).run_with_metrics(query="q", chunks=[])
    with pytest.raises(ValueError, match="query parameter is required"):
        await PlanningActor(FakeLanguageModel()).run_with_metrics(query="", chunks=CHUNKS)


async def test_actor_config_limits_reach_the_model_and_temperature_only_when_accepted():
    model = RecordingModel({"ResponsePlan": make_plan()})
    config = ActorConfig(max_tokens=123, temperature=0.1)
    await PlanningActor(model, config).execute(query="Tenant duties", chunks=CHUNKS)
    assert model.seen == {"max_tokens": 123, "temperature": 0.1}

    # The fake's signature has no temperature argument, so none is passed and the call succeeds.
    plain = FakeLanguageModel(structured={"ResponsePlan": make_plan()})
    await PlanningActor(plain, config).execute(query="Tenant duties", chunks=CHUNKS)


# -- validation ----------------------------------------------------------------------------------


async def test_validation_returns_the_models_assessment_and_lists_every_chunk():
    verdict = ValidationResult(
        is_valid=False,
        missing_support=["When is the deposit returned?"],
        recommendation="retrieve_more",
    )
    model = FakeLanguageModel(structured={"ValidationResult": verdict})
    result, metrics = await validate_response_plan(model, make_plan(), CHUNKS)

    assert result.recommendation == "retrieve_more" and result.is_valid is False
    assert metrics.actor_name == "validation" and metrics.llm_calls == 1
    prompt = model.calls[0][1].content
    assert "0. [Smith v. Jones]" in prompt and "1. [Residential Tenancy Act, Section 12]" in prompt
    assert "chunk_indices: [1, 9]" in prompt


async def test_validation_fails_closed_but_still_proceeds_on_invalid_output():
    result = await ValidationActor(RaisingModel(schema_error())).execute(
        plan=make_plan(), available_chunks=CHUNKS
    )
    assert result.is_valid is False
    assert result.recommendation == "proceed"
    assert result.confidence_score == 0.0
    assert "unable to parse" in result.missing_support[0]


async def test_validation_requires_a_plan_and_chunks():
    actor = ValidationActor(FakeLanguageModel())
    with pytest.raises(ValueError, match="plan parameter is required"):
        await actor.execute(plan=None, available_chunks=CHUNKS)
    with pytest.raises(ValueError, match="available_chunks parameter is required"):
        await actor.execute(plan=make_plan(), available_chunks=[])


# -- refinement ----------------------------------------------------------------------------------


async def test_refinement_returns_the_critique_and_records_metrics():
    critique = RefinementSuggestions(
        missing_elements=["deposit timing"],
        should_regenerate=True,
        suggested_improvements="Answer the deposit question.",
        confidence_score=0.4,
    )
    model = FakeLanguageModel(structured={"RefinementSuggestions": critique})
    result, metrics = await refine_response(model, "A draft answer.", make_plan(), CHUNKS)

    assert result.should_regenerate is True
    assert metrics.actor_name == "refinement" and metrics.llm_calls == 1
    prompt = model.calls[0][1].content
    assert "A draft answer." in prompt
    assert "1. Smith v. Jones" in prompt


async def test_refinement_accepts_the_draft_on_invalid_output():
    result = await RefinementActor(RaisingModel(schema_error())).execute(
        original_response="A draft answer.", plan=make_plan(), chunks=CHUNKS
    )
    assert result.should_regenerate is False
    assert result.suggested_improvements is None
    assert result.confidence_score == 0.5
    assert "skipping refinement" in result.clarity_issues[0]


async def test_a_long_draft_is_truncated_for_review_with_its_true_length():
    model = FakeLanguageModel(
        structured={"RefinementSuggestions": RefinementSuggestions(should_regenerate=False)}
    )
    draft = "x" * (MAX_REVIEW_CHARS + 500)
    await RefinementActor(model).execute(original_response=draft, plan=make_plan(), chunks=CHUNKS)

    prompt = model.calls[0][1].content
    assert f"full response is {len(draft)} chars" in prompt
    assert draft not in prompt


# -- model-free plan validation ------------------------------------------------------------------


def test_deterministic_validator_flags_a_sub_question_and_source_with_no_support():
    plan = make_plan(
        sub_questions=[SubQuestion(question="Which sanctions apply to spoliation of evidence?")],
        required_sources=["Doe v. Roe"],
    )
    result = validate_plan_deterministically(plan, normalize_chunks(CHUNKS))

    assert result.is_valid is False
    assert result.missing_sub_questions == ["Which sanctions apply to spoliation of evidence?"]
    assert result.missing_sources == ["Doe v. Roe"]
    assert result.recommendation == "fallback_to_general"
    assert quick_plan_check(plan, normalize_chunks(CHUNKS)) is False


def test_deterministic_validator_passes_a_plan_the_chunks_support():
    plan = make_plan(sub_questions=[SubQuestion(question="Ending a lease", chunk_indices=[0])])
    result = validate_plan_deterministically(plan, normalize_chunks(CHUNKS))

    assert result.is_valid is True
    assert result.matched_sources == ["Smith v. Jones"]
    assert result.recommendation == "proceed"
