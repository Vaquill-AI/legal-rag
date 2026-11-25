import asyncio

import pytest
from pydantic import BaseModel

from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.retrieval.strategies import (
    HyDEConfig,
    HyDEGenerator,
    MultiQueryConfig,
    MultiQueryGenerator,
    QueryDecomposer,
    QueryVariations,
    generate_rewrites,
)
from legal_rag.retrieval.strategies.query_decomposer import DecompositionAnalysis
from legal_rag.types import Message, UsageEvent

LONG_PASSAGE = (
    "Under the statute, a covered entity shall notify affected individuals promptly. " * 2
)


class BrokenLanguageModel:
    """Fails every call."""

    async def generate(self, messages, *, max_tokens=None, temperature=None) -> str:
        raise RuntimeError("model down")

    async def generate_structured(self, messages, schema: type[BaseModel], *, max_tokens=None):
        raise RuntimeError("model down")


async def test_hyde_generates_and_meters():
    events: list[UsageEvent] = []
    lm = FakeLanguageModel(reply=LONG_PASSAGE)
    gen = HyDEGenerator(lm, usage_hook=events.append)
    result = await gen.generate("breach notification deadline")
    assert result.success
    assert result.primary_hypothesis == LONG_PASSAGE.strip()
    assert events and events[0].operation == "hyde_generation"
    sent = lm.calls[0][0].content
    assert "breach notification deadline" in sent
    assert "Indian" not in sent


async def test_hyde_short_replies_fall_back_to_the_query():
    gen = HyDEGenerator(FakeLanguageModel(reply="too short"), HyDEConfig(max_retries=1))
    result = await gen.generate("what is estoppel")
    assert not result.success
    assert result.primary_hypothesis == "what is estoppel"
    assert gen.stats["failed"] == 1


async def test_hyde_custom_template_and_multi_hypothesis():
    lm = FakeLanguageModel(reply=LONG_PASSAGE)
    cfg = HyDEConfig(
        prompt_template="Write about {query}.", use_multi_hypothesis=True, num_hypotheses=3
    )
    result = await HyDEGenerator(lm, cfg).generate("liens")
    assert len(result.hypotheses) == 3
    assert lm.calls[0][0].content == "Write about liens."


async def test_multi_query_filters_and_dedups():
    variations = QueryVariations(
        queries=[
            "Qualified immunity doctrine for police officers",
            "qualified immunity doctrine for police officers",
            "too short",
            "Section 1983 immunity defense standard of review",
        ]
    )
    lm = FakeLanguageModel(structured={"QueryVariations": variations})
    gen = MultiQueryGenerator(lm, MultiQueryConfig(num_queries=4))
    result = await gen.generate("qualified immunity police")
    assert result.success
    assert result.all_queries[0] == "qualified immunity police"
    assert result.generated_queries == [
        "Qualified immunity doctrine for police officers",
        "Section 1983 immunity defense standard of review",
    ]
    prompt = lm.calls[0][0].content
    assert "3" not in prompt.split("Generate")[1][:3] or "4" in prompt


async def test_multi_query_failure_returns_original_only():
    result = await MultiQueryGenerator(BrokenLanguageModel()).generate("anything at all here")
    assert not result.success
    assert result.all_queries == ["anything at all here"]


async def test_generate_rewrites_caches_and_degrades():
    variations = QueryVariations(
        queries=["rewrite one of the question", "rewrite two of the question"]
    )
    lm = FakeLanguageModel(structured={"QueryVariations": variations})
    cache: dict[str, list[str]] = {}
    first = await generate_rewrites(lm, "  How is venue transferred?  ", n=2, cache=cache)
    second = await generate_rewrites(lm, "how is venue transferred?", n=2, cache=cache)
    assert first == second == ["rewrite one of the question", "rewrite two of the question"]
    assert len(lm.calls) == 1
    assert await generate_rewrites(BrokenLanguageModel(), "a question", cache={}) == []
    assert await generate_rewrites(lm, "   ") == []


async def test_generate_rewrites_times_out_to_empty():
    class Slow(FakeLanguageModel):
        async def generate_structured(self, messages, schema, *, max_tokens=None):
            await asyncio.sleep(1)

    assert await generate_rewrites(Slow(), "a real question", timeout=0.01) == []


async def test_decomposer_splits_multi_track_queries():
    analysis = DecompositionAnalysis(
        requires_decomposition=True,
        tracks=["Title VII", "42 U.S.C. 1983"],
        sub_queries=["Title VII employment discrimination remedies", "Section 1983 claim employer"],
        reasoning="two tracks",
    )
    lm = FakeLanguageModel(structured={"DecompositionAnalysis": analysis})
    result = await QueryDecomposer(lm).decompose(
        "employment discrimination remedies for a public employee"
    )
    assert result.requires_decomposition
    assert result.all_queries[0].startswith("employment discrimination")
    assert len(result.all_queries) == 3


async def test_decomposer_short_query_and_failure_paths():
    lm = FakeLanguageModel()
    short = await QueryDecomposer(lm).decompose("bail")
    assert not short.requires_decomposition and not lm.calls
    failed = await QueryDecomposer(BrokenLanguageModel()).decompose(
        "a long enough research query here"
    )
    assert not failed.success and failed.all_queries == ["a long enough research query here"]


def test_decomposition_analysis_coerces_nulls():
    parsed = DecompositionAnalysis.model_validate(
        {"requires_decomposition": None, "tracks": None, "sub_queries": None, "reasoning": None}
    )
    assert parsed.requires_decomposition is False and parsed.tracks == []


@pytest.mark.parametrize("message", [Message("user", "x")])
def test_message_is_frozen(message):
    with pytest.raises(AttributeError):
        message.content = "y"  # type: ignore[misc]
