"""HyDETransformer (the conversation-facing HyDE wrapper) and its old import paths."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from legal_rag.providers.fakes import FakeEmbedder, FakeLanguageModel
from legal_rag.retrieval import conversation, strategies
from legal_rag.retrieval.conversation import hyde as conversation_hyde
from legal_rag.retrieval.strategies import hyde as strategies_hyde
from legal_rag.retrieval.strategies.hyde import (
    HyDEConfig,
    HyDEGenerator,
    HyDETransformer,
    HyDETransformResult,
)
from legal_rag.retrieval.strategies.hyde_prompts import (
    HYDE_GENERAL_SYSTEM_PROMPT,
    HYDE_LEGAL_SYSTEM_PROMPT,
)
from legal_rag.types import Message, UsageEvent

HYPOTHESIS = "The Court held that a covered entity must notify affected individuals promptly."
SPECIFIC_QUERY = "Whether the 2019 amendment to section 12 applies to pending claims in Ohio"


class RecordingLanguageModel(FakeLanguageModel):
    """Remembers the sampling arguments of each call."""

    def __init__(self, reply: str = "") -> None:
        super().__init__(reply=reply)
        self.kwargs: list[dict[str, float | int | None]] = []

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.kwargs.append({"max_tokens": max_tokens, "temperature": temperature})
        return await super().generate(messages, max_tokens=max_tokens, temperature=temperature)


class BrokenLanguageModel(FakeLanguageModel):
    """Fails every call and counts how often it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def generate(self, messages, *, max_tokens=None, temperature=None) -> str:
        self.attempts += 1
        raise RuntimeError("model down")


@pytest.mark.parametrize(
    "query",
    [
        "Explain adverse possession in a residential setting",
        "TELL ME ABOUT easements by necessity in Texas",
        "Overview of the doctrine of laches as applied to trademark claims",
    ],
)
def test_gate_accepts_abstract_pattern_queries(query):
    assert HyDETransformer(FakeLanguageModel()).should_use_hyde(query) is True


@pytest.mark.parametrize("query", ["laches", "ccpa penalties", "fraud statute limitation"])
def test_gate_accepts_queries_shorter_than_four_words(query):
    assert HyDETransformer(FakeLanguageModel()).should_use_hyde(query) is True


def test_gate_rejects_a_specific_multi_word_query():
    assert HyDETransformer(FakeLanguageModel()).should_use_hyde(SPECIFIC_QUERY) is False


async def test_skipped_query_never_calls_the_model_and_is_returned_unchanged():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = await HyDETransformer(lm).transform(SPECIFIC_QUERY)
    assert lm.calls == []
    assert result.used_hyde is False
    assert result.hypothetical_document == ""
    assert result.combined_query == SPECIFIC_QUERY
    assert result.original_query == SPECIFIC_QUERY


async def test_force_calls_the_model_even_when_the_gate_says_no():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = await HyDETransformer(lm).transform(SPECIFIC_QUERY, force=True)
    assert len(lm.calls) == 1
    assert result.used_hyde is True
    assert result.hypothetical_document == HYPOTHESIS


async def test_transform_sends_the_legal_system_prompt_then_the_raw_query():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    await HyDETransformer(lm).transform("explain laches")
    assert lm.calls == [
        [Message("system", HYDE_LEGAL_SYSTEM_PROMPT), Message("user", "explain laches")]
    ]


async def test_combined_query_is_the_query_then_the_stripped_hypothesis():
    lm = FakeLanguageModel(reply=f"  {HYPOTHESIS}\n ")
    result = await HyDETransformer(lm).transform("explain laches")
    assert result.hypothetical_document == HYPOTHESIS
    assert result.combined_query == f"explain laches\n\n{HYPOTHESIS}"
    assert result.used_hyde is True
    assert result.processing_time_ms >= 0


async def test_include_original_false_embeds_the_hypothesis_alone():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = await HyDETransformer(lm, include_original=False).transform("explain laches")
    assert result.combined_query == HYPOTHESIS


async def test_general_prompt_is_used_when_the_legal_prompt_is_off():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    await HyDETransformer(lm, use_legal_prompt=False).transform("explain laches")
    assert lm.calls[0][0] == Message("system", HYDE_GENERAL_SYSTEM_PROMPT)


async def test_prompt_choice_follows_an_attribute_changed_after_construction():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    transformer = HyDETransformer(lm)
    await transformer.transform("explain laches")
    transformer.use_legal_prompt = False
    await transformer.transform("explain laches")
    assert [call[0].content for call in lm.calls] == [
        HYDE_LEGAL_SYSTEM_PROMPT,
        HYDE_GENERAL_SYSTEM_PROMPT,
    ]


async def test_token_budget_and_temperature_reach_the_model():
    lm = RecordingLanguageModel(reply=HYPOTHESIS)
    await HyDETransformer(lm, max_tokens=123, temperature=0.2).transform("explain laches")
    assert lm.kwargs == [{"max_tokens": 123, "temperature": 0.2}]


async def test_a_failing_model_falls_back_to_the_query_after_a_single_attempt():
    lm = BrokenLanguageModel()
    result = await HyDETransformer(lm).transform("explain laches")
    assert lm.attempts == 1
    assert result.used_hyde is False
    assert result.hypothetical_document == ""
    assert result.combined_query == "explain laches"


@pytest.mark.parametrize("reply", ["", "   \n"])
async def test_an_empty_reply_is_not_treated_as_a_hypothesis(reply):
    result = await HyDETransformer(FakeLanguageModel(reply=reply)).transform("explain laches")
    assert result.used_hyde is False
    assert result.combined_query == "explain laches"


async def test_usage_is_metered_once_per_generation():
    events: list[UsageEvent] = []
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    await HyDETransformer(lm, usage_hook=events.append).transform("explain laches")
    assert [e.operation for e in events] == ["hyde_generation"]
    assert events[0].input_tokens and events[0].input_tokens > 0
    assert events[0].output_tokens and events[0].output_tokens > 0


async def test_a_skipped_query_is_not_metered():
    events: list[UsageEvent] = []
    transformer = HyDETransformer(FakeLanguageModel(reply=HYPOTHESIS), usage_hook=events.append)
    await transformer.transform(SPECIFIC_QUERY)
    assert events == []


async def test_a_raising_usage_hook_does_not_lose_the_hypothesis():
    def bad_hook(event: UsageEvent) -> None:
        raise RuntimeError("metering down")

    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = await HyDETransformer(lm, usage_hook=bad_hook).transform("explain laches")
    assert result.used_hyde is True
    assert result.hypothetical_document == HYPOTHESIS


async def test_the_combined_query_embeds_differently_from_the_bare_query():
    embedder = FakeEmbedder(dimension=8)
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = await HyDETransformer(lm).transform("explain laches")
    bare = await embedder.embed_query("explain laches")
    hyde = await embedder.embed_query(result.combined_query)
    assert hyde != bare
    assert hyde == await embedder.embed_query(f"explain laches\n\n{HYPOTHESIS}")


def test_transform_sync_matches_the_async_result():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    result = HyDETransformer(lm).transform_sync("explain laches")
    assert result.used_hyde is True
    assert result.combined_query == f"explain laches\n\n{HYPOTHESIS}"


def test_result_to_dict_carries_every_field():
    result = HyDETransformResult("q", "doc", "q\n\ndoc", 1.5, used_hyde=True)
    assert result.to_dict() == {
        "original_query": "q",
        "hypothetical_document": "doc",
        "combined_query": "q\n\ndoc",
        "processing_time_ms": 1.5,
        "used_hyde": True,
    }


async def test_generator_system_prompt_replaces_the_single_message_template():
    lm = FakeLanguageModel(reply=HYPOTHESIS)
    config = HyDEConfig(system_prompt="Write like a statute.", prompt_template="ignored {query}")
    result = await HyDEGenerator(lm, config).generate("liens")
    assert result.success is True
    assert lm.calls == [[Message("system", "Write like a statute."), Message("user", "liens")]]


def test_old_conversation_import_paths_resolve_to_the_one_implementation():
    assert conversation_hyde.HyDETransformer is strategies_hyde.HyDETransformer
    assert conversation.HyDETransformer is strategies_hyde.HyDETransformer
    assert strategies.HyDETransformer is strategies_hyde.HyDETransformer
    assert conversation_hyde.HYDE_LEGAL_SYSTEM_PROMPT is HYDE_LEGAL_SYSTEM_PROMPT
    assert conversation_hyde.HYDE_GENERAL_SYSTEM_PROMPT is HYDE_GENERAL_SYSTEM_PROMPT


def test_the_two_hyde_result_types_stay_distinct_under_their_historic_names():
    assert conversation.HyDEResult is HyDETransformResult
    assert conversation_hyde.HyDEResult is HyDETransformResult
    assert strategies.HyDEResult is strategies_hyde.HyDEResult
    assert strategies.HyDEResult is not conversation.HyDEResult
    assert [f.name for f in dataclasses.fields(conversation.HyDEResult)] == [
        "original_query",
        "hypothetical_document",
        "combined_query",
        "processing_time_ms",
        "used_hyde",
    ]
