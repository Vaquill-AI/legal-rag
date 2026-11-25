from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.retrieval.strategies import (
    CRAGConfig,
    CRAGCorrector,
    GapAnalysisResult,
    LLMGapAnalyzer,
    MixStrategy,
    MultiHopConfig,
    MultiHopRetriever,
    RetrievalQuality,
    SelfRAGConfig,
    SelfRAGController,
    SourceMixer,
    SourceMixerConfig,
)
from legal_rag.retrieval.strategies.crag import RelevanceScore
from legal_rag.retrieval.strategies.crag_prompts import (
    PromptVariant,
    format_mixed_context_for_prompt,
    get_prompt_with_partial_knowledge_note,
    get_system_prompt,
)
from legal_rag.retrieval.strategies.gap_analysis import InformationGapAnalysis
from legal_rag.retrieval.strategies.self_rag import AnswerCritique, RetrievalAssessment
from legal_rag.types import RetrievedChunk


def chunk(cid: str, text: str = "", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text or f"text {cid}", score=score)


class ScriptedLM(FakeLanguageModel):
    """Returns queued structured replies in order."""

    def __init__(self, replies):
        super().__init__()
        self.queue = list(replies)

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        self.calls.append(messages)
        reply = self.queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# ---- multi-hop ----


async def test_multi_hop_stops_when_enough_chunks():
    async def retrieve(query: str, k: int):
        return [chunk(f"{query}-{i}") for i in range(6)]

    result = await MultiHopRetriever(retrieve).retrieve("some question about duty of care")
    assert result.success and result.completed_early
    assert result.total_hops == 1 and len(result.chunks) == 6


async def test_multi_hop_follows_gap_analyzer_then_dedups_chunks():
    calls: list[str] = []

    async def retrieve(query: str, k: int):
        calls.append(query)
        return [chunk("shared"), chunk(f"only-{len(calls)}")]

    async def analyzer(query, chunks):
        return GapAnalysisResult(False, ["missing remedy"], "remedies for breach", 0.2)

    cfg = MultiHopConfig(max_hops=3, sufficient_chunks_threshold=100)
    events = []

    async def on_event(event):
        events.append(event.event_type)

    result = await MultiHopRetriever(retrieve, cfg, gap_analyzer=analyzer).retrieve(
        "duty of care", on_hop_event=on_event
    )
    assert calls[:2] == ["duty of care", "remedies for breach"]
    assert [c.id for c in result.chunks].count("shared") == 1
    assert events[0] == "multi_hop_started" and events[-1] == "multi_hop_complete"
    assert "multi_hop_gap_analysis" in events


async def test_multi_hop_sufficient_verdict_ends_early():
    async def retrieve(query: str, k: int):
        return [chunk("a")]

    async def analyzer(query, chunks):
        return GapAnalysisResult(True, [], "", 0.95)

    cfg = MultiHopConfig(sufficient_chunks_threshold=50)
    result = await MultiHopRetriever(retrieve, cfg, gap_analyzer=analyzer).retrieve(
        "q one two three four"
    )
    assert result.completed_early and result.total_hops == 1


async def test_multi_hop_timeout_and_failure_do_not_raise():
    async def slow(query: str, k: int):
        import asyncio

        await asyncio.sleep(1)
        return []

    cfg = MultiHopConfig(hop_timeout_seconds=0.01)
    timed_out = await MultiHopRetriever(slow, cfg).retrieve("slow query here")
    assert timed_out.success and timed_out.completed_early and timed_out.total_hops == 0

    async def boom(query: str, k: int):
        raise RuntimeError("store down")

    failed = await MultiHopRetriever(boom).retrieve("another query")
    assert not failed.success


async def test_llm_gap_analyzer_falls_back_to_rule_on_error():
    analyzer = LLMGapAnalyzer(ScriptedLM([RuntimeError("down")]))
    verdict = await analyzer("what is the rule", [chunk("a"), chunk("b"), chunk("c")])
    assert verdict.has_sufficient_info and verdict.confidence == 0.7
    empty = await analyzer("what is the rule", [])
    assert not empty.has_sufficient_info
    ok = await LLMGapAnalyzer(
        ScriptedLM(
            [InformationGapAnalysis(is_sufficient=False, suggested_followup="next", confidence=0.4)]
        )
    )("q", [chunk("a")])
    assert ok.suggested_followup == "next"


# ---- self-rag ----


async def test_self_rag_skips_retrieval_for_chitchat_and_extracts_count():
    lm = ScriptedLM(
        [
            RetrievalAssessment(decision="SKIP", requested_count=None),
            AnswerCritique(verdict="GOOD"),
        ]
    )

    async def generate(query, context, previous_answer=None):
        return "hello back"

    result = await SelfRAGController(lm).process("hello there", generate_func=generate)
    assert not result.retrieval_performed and result.answer == "hello back"

    lm2 = ScriptedLM(
        [
            RetrievalAssessment(decision="retrieve", requested_count=5),
            AnswerCritique(verdict="GOOD"),
        ]
    )

    async def retrieve(query):
        return "context text"

    res2 = await SelfRAGController(lm2).process("give me 5 cases", retrieve, generate)
    assert res2.retrieval_performed and res2.requested_count == 5 and res2.context == "context text"


async def test_self_rag_regenerates_until_good():
    lm = ScriptedLM(
        [
            RetrievalAssessment(decision="retrieve"),
            AnswerCritique(verdict="NEEDS_IMPROVEMENT"),
            AnswerCritique(verdict="GOOD"),
        ]
    )
    answers = iter(["draft", "better"])

    async def generate(query, context, previous_answer=None):
        return next(answers)

    async def retrieve(query):
        return "ctx"

    result = await SelfRAGController(lm).process("explain the statute", retrieve, generate)
    assert result.num_regenerations == 1 and result.answer == "better"


async def test_self_rag_keyword_fallback_when_router_fails():
    lm = ScriptedLM([RuntimeError("down")])
    cfg = SelfRAGConfig(enable_critique=False)
    result = await SelfRAGController(lm, cfg).process("what is the definition of a statute")
    assert result.success and result.retrieval_performed


# ---- crag ----


async def test_crag_filters_medium_quality_and_flags_degraded_inputs():
    scores = [0.9, 0.9, 0.1, 0.1, 0.1]
    lm = ScriptedLM([RelevanceScore(score=s) for s in scores])
    chunks = [chunk(str(i)) for i in range(5)]
    result = await CRAGCorrector(lm).evaluate_and_correct("q", chunks)
    assert result.quality is RetrievalQuality.MEDIUM
    assert [c.id for c in result.corrected_chunks] == ["0", "1"]


async def test_crag_high_low_and_error_paths():
    high = await CRAGCorrector(ScriptedLM([RelevanceScore(score=0.9)] * 3)).evaluate_and_correct(
        "q", [chunk("a"), chunk("b"), chunk("c")]
    )
    assert high.quality is RetrievalQuality.HIGH and high.corrections_applied == ["none_needed"]

    cfg = CRAGConfig(preserve_count=1)
    low = await CRAGCorrector(
        ScriptedLM([RelevanceScore(score=s) for s in (0.0, 0.2, 0.1)]), cfg
    ).evaluate_and_correct("q", [chunk("a"), chunk("b"), chunk("c")])
    assert low.quality is RetrievalQuality.LOW and [c.id for c in low.corrected_chunks] == ["b"]

    empty = await CRAGCorrector(ScriptedLM([])).evaluate_and_correct("q", [])
    assert empty.quality is RetrievalQuality.LOW and empty.corrections_applied == ["no_results"]

    # A failing grader keeps the retrieval signal instead of dropping chunks.
    fallback = await CRAGCorrector(
        ScriptedLM([RuntimeError("x"), RuntimeError("x")])
    ).evaluate_and_correct("q", [chunk("a", score=0.9), chunk("b", score=0.8)])
    assert fallback.quality is RetrievalQuality.HIGH


def test_source_mixer_budgets_and_formatting():
    mixer = SourceMixer(
        SourceMixerConfig(max_total_tokens=400, min_primary_tokens=50, min_supplementary_tokens=50),
        token_counter=lambda t: len(t) // 4,
    )
    primary = [chunk("p1", "alpha " * 100), chunk("p2", "beta " * 100)]
    supp = [chunk("s1", "gamma " * 40)]
    mixed = mixer.mix(primary, supp, primary_confidence=0.9)
    assert mixed.mix_strategy is MixStrategy.PRIMARY_LED
    assert mixed.truncation_applied
    assert mixed.source_summary["supplementary"] == 1
    text = mixer.format_mixed_context(mixed)
    assert "PRIMARY SOURCES" in text and "SUPPLEMENTARY SOURCES" in text
    assert mixer.choose_strategy([], supp, 0.5) is MixStrategy.SUPPLEMENTARY_LED
    assert mixer.mix([], []).total_tokens == mixer.config.header_overhead


def test_crag_prompts_are_neutral():
    for variant in PromptVariant:
        prompt = get_system_prompt(variant)
        assert "upload" not in prompt.lower() and "[WEB]" not in prompt
    ctx = format_mixed_context_for_prompt(
        [{"content": "a", "source": "doc"}], [{"content": "b", "title": "T"}]
    )
    assert "[PRIMARY]" in ctx and "[SUPPLEMENTARY]" in ctx
    assert get_prompt_with_partial_knowledge_note("base", True).startswith("base")
    assert get_prompt_with_partial_knowledge_note("base", False) == "base"
