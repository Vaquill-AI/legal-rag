"""The analyze stage: follow-up condensation, tier gates, explicit scope, fail-open and hygiene."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline.protocols import JurisdictionPlugin
from legal_rag.pipeline.stages.analyze import (
    ANALYZE_EXTRA_KEYS,
    AnalyzeConfig,
    analyze_stage,
    detect_recency_gap,
    detect_recency_intent,
    has_explicit_scope,
    llm_legal_domain_check,
    retrieval_wanted,
)
from legal_rag.pipeline.tiers import get_tier
from legal_rag.pipeline.types import QueryContext, UsageLedger
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.retrieval.conversation.condenser import (
    CondensationResult,
    CondensedQueryOutput,
    QueryType,
)
from legal_rag.retrieval.conversation.intent_router import IntentRoutingResult, UserIntent
from legal_rag.retrieval.conversation.query_analyzer import (
    LegalTaskType,
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)
from legal_rag.retrieval.conversation.query_expander import QueryExpansionResult
from legal_rag.types import Message

HISTORY = [
    {
        "role": "user",
        "content": "What is the statute of limitations for breach of contract in Texas?",
    },
    {"role": "assistant", "content": "Four years under the state civil practice code."},
]
FOLLOW_UP = "what about California?"
STANDALONE = "What is the statute of limitations for breach of contract in California?"
LONG_QUERY = " ".join(["please look at this notice and tell me what it means for the hearing"] * 2)


def _ctx(query: str = FOLLOW_UP, *, history: Sequence[Any] = HISTORY, **kw: Any) -> QueryContext:
    return QueryContext(query=query, chat_history=list(history), **kw)


def _condense_llm(reply: str = STANDALONE) -> FakeLanguageModel:
    return FakeLanguageModel(
        reply="yes", structured={"CondensedQueryOutput": CondensedQueryOutput(primary_query=reply)}
    )


def _analysis(**kw: Any) -> QueryAnalysisResult:
    values: dict[str, Any] = {
        "primary_intent": QueryIntent.FACTUAL,
        "complexity": QueryComplexity.TRIVIAL,
        "confidence": 0.9,
        "has_multiple_intents": False,
        "intents_detected": [QueryIntent.FACTUAL],
        "is_in_domain": True,
        "is_ambiguous": False,
        "is_out_of_scope": False,
        "should_retrieve": True,
        "should_expand_query": False,
        "should_decompose": False,
        "legal_entities_found": [],
        "ambiguity_score": 0.0,
        "processing_time_ms": 0.1,
        "legal_task_type": LegalTaskType.GENERAL,
    }
    values.update(kw)
    return QueryAnalysisResult(**values)


class FixedAnalyzer:
    def __init__(self, **kw: Any) -> None:
        self.result = _analysis(**kw)

    def analyze(self, query: str, chat_history: Any = None, intent_routing: Any = None):
        return self.result


class RaisingAnalyzer:
    def analyze(self, *args: Any, **kwargs: Any):
        raise RuntimeError("analyzer exploded")


class StubCondenser:
    def __init__(self, *, requires_retrieval: bool = True) -> None:
        self.requires_retrieval = requires_retrieval
        self.scopes: list[str] = []

    async def condense(self, query: str, chat_history: Any, force: bool = False, scope: str = ""):
        self.scopes.append(scope)
        return CondensationResult(
            condensed_query="standalone " + query,
            original_query=query,
            query_type=QueryType.CONVERSATIONAL,
            confidence=0.9,
            requires_retrieval=self.requires_retrieval,
            entities_resolved=[],
            processing_time_ms=0.0,
        )


class StubExpander:
    def __init__(self) -> None:
        self.calls = 0

    async def expand(self, query: str, ambiguity_score: float = 0.0, is_legal: bool = True):
        self.calls += 1
        return QueryExpansionResult(
            original_query=query,
            expanded_queries=["expanded " + query, "broader " + query],
            expansion_applied=True,
            expansion_type="specificity",
            confidence=0.8,
            processing_time_ms=0.0,
        )


class StubRouter:
    def __init__(self, intent: UserIntent, confidence: float = 0.9) -> None:
        self.intent, self.confidence = intent, confidence

    async def route(self, message: str) -> IntentRoutingResult:
        return IntentRoutingResult(
            user_intent=self.intent,
            instruction_text=message,
            content_text="pasted text",
            should_retrieve=False,
            confidence=self.confidence,
            routing_reason="stub",
            instruction_position="before",
            content_word_count=80,
            processing_time_ms=0.0,
        )


class RaisingModel:
    """A language model whose every call fails."""

    async def generate(self, messages: Sequence[Message], **kwargs: Any) -> str:
        raise ConnectionError("model offline")

    async def generate_structured(self, messages: Sequence[Message], schema: Any, **kwargs: Any):
        raise ConnectionError("model offline")


class StubPlugin:
    condenser_key = "xx"

    def __init__(self, matches: bool = True) -> None:
        self._matches = matches

    def matches(self, filters: Mapping[str, Any]) -> bool:
        return self._matches


def _tier_without(**flags: bool):
    standard = get_tier("standard")
    return dataclasses.replace(standard, features=dataclasses.replace(standard.features, **flags))


# ---------------------------------------------------------------------------
# Condensation and tier gates
# ---------------------------------------------------------------------------


async def test_follow_up_question_is_condensed_against_history():
    ctx = _ctx()
    llm = _condense_llm()
    out = await analyze_stage(ctx, llm=llm)
    assert out.extra["condensed_query"] == STANDALONE
    assert out.extra["original_query"] == FOLLOW_UP
    assert out.extra["condensation_confidence"] == 0.9
    assert retrieval_wanted(out)
    assert len(llm.calls) == 1
    assert out.query == FOLLOW_UP
    assert "condensed_query" not in ctx.extra, "the input context must not be mutated"


async def test_every_documented_key_is_written():
    out = await analyze_stage(_ctx(), llm=_condense_llm())
    assert set(ANALYZE_EXTRA_KEYS) <= set(out.extra)


async def test_tier_gate_off_skips_condensation():
    llm = _condense_llm()
    out = await analyze_stage(_ctx(), llm=llm, tier=_tier_without(use_query_condensation=False))
    assert out.extra["condensed_query"] == FOLLOW_UP
    assert out.extra["query_type"] == QueryType.STANDALONE.value
    assert out.extra["requires_retrieval"] is True
    assert llm.calls == []


async def test_condensation_needs_history_and_a_model():
    llm = _condense_llm()
    no_history = await analyze_stage(_ctx(history=[]), llm=llm)
    assert no_history.extra["condensed_query"] == FOLLOW_UP and llm.calls == []
    rule_based = await analyze_stage(_ctx())
    assert rule_based.extra["condensed_query"] == FOLLOW_UP


async def test_history_may_be_message_objects():
    history = [Message("user", HISTORY[0]["content"]), Message("assistant", "Four years.")]
    out = await analyze_stage(_ctx(history=history), llm=_condense_llm())
    assert out.extra["condensed_query"] == STANDALONE


async def test_condenser_failure_falls_back_to_the_raw_query():
    out = await analyze_stage(_ctx(), llm=RaisingModel())
    assert out.extra["original_query"] == FOLLOW_UP
    assert out.extra["requires_retrieval"] is True
    assert out.extra["condensed_query"]


async def test_expansion_runs_only_when_tier_allows_and_nothing_is_scoped():
    analyzer = FixedAnalyzer(should_expand_query=True, is_ambiguous=True, ambiguity_score=0.8)
    expander = StubExpander()
    out = await analyze_stage(_ctx(history=[]), analyzer=analyzer, expander=expander)
    assert out.extra["expanded_query"] == "expanded " + FOLLOW_UP
    assert out.extra["query_expansion"]["applied"] is True
    assert out.extra["condensed_query"] == "expanded " + FOLLOW_UP

    gated = StubExpander()
    off = _tier_without(use_query_expansion=False)
    out = await analyze_stage(_ctx(history=[]), analyzer=analyzer, expander=gated, tier=off)
    assert out.extra["expanded_query"] is None and gated.calls == 0

    scoped = StubExpander()
    out = await analyze_stage(
        _ctx(history=[], scope_lock_ids=["case-1"]), analyzer=analyzer, expander=scoped
    )
    assert out.extra["expanded_query"] is None and scoped.calls == 0


async def test_enrichment_appends_acronym_full_forms():
    query = "ERISA fiduciary duty claims"
    out = await analyze_stage(_ctx(query, history=[]), analyzer=FixedAnalyzer(is_in_domain=True))
    assert out.extra["enriched_query"] == query + " Employee Retirement Income Security Act"
    assert out.extra["condensed_query"] == out.extra["enriched_query"]

    out_of_domain = FixedAnalyzer(is_in_domain=False)
    out = await analyze_stage(_ctx(query, history=[]), analyzer=out_of_domain)
    assert out.extra["enriched_query"] is None


# ---------------------------------------------------------------------------
# Explicit scope overrides every skip-retrieval heuristic
# ---------------------------------------------------------------------------


async def test_scope_helpers():
    assert not has_explicit_scope(_ctx())
    assert has_explicit_scope(_ctx(pinned_ids=["d1"]))
    assert has_explicit_scope(_ctx(scope_lock_ids=["c1"]))
    assert not has_explicit_scope(_ctx(boost_ids=["b1"])), "boost ids are not a lock"


async def test_analyzer_no_retrieve_verdict_is_overridden_by_scope():
    analyzer = FixedAnalyzer(should_retrieve=False, is_in_domain=False)
    free = await analyze_stage(_ctx(history=[]), analyzer=analyzer)
    assert free.extra["query_analysis"]["should_retrieve"] is False
    assert not retrieval_wanted(free)

    pinned = await analyze_stage(_ctx(history=[], pinned_ids=["doc-1"]), analyzer=analyzer)
    assert pinned.extra["query_analysis"]["should_retrieve"] is True
    assert pinned.extra["query_analysis"]["complexity"] == "moderate"
    assert pinned.extra["query_analysis"]["is_in_domain"] is True
    assert retrieval_wanted(pinned)


async def test_general_knowledge_skips_retrieval_unless_scoped():
    query = "write a python function to sort a list"
    free = await analyze_stage(_ctx(query, history=[]))
    assert free.extra["query_analysis"]["is_general_knowledge"] is True
    assert free.extra["web_search_suggestion"] is True
    assert not retrieval_wanted(free)

    scoped = await analyze_stage(_ctx(query, history=[], pinned_ids=["doc-1"]))
    assert scoped.extra["query_analysis"]["is_general_knowledge"] is False
    assert retrieval_wanted(scoped)


async def test_identity_hook_never_beats_scope():
    def about_the_assistant(query: str) -> bool:
        return "who are you" in query.lower()

    ctx = _ctx("Who are you?", history=[])
    out = await analyze_stage(ctx, identity_check=about_the_assistant)
    assert out.extra["query_analysis"]["intent"] == "identity"
    assert not retrieval_wanted(out)

    pinned = await analyze_stage(
        dataclasses.replace(ctx, pinned_ids=["doc-who-are-you.pdf"]),
        identity_check=about_the_assistant,
    )
    assert pinned.extra["query_analysis"]["intent"] != "identity"
    assert retrieval_wanted(pinned)


async def test_text_processing_intent_skips_retrieval_unless_scoped():
    router = StubRouter(UserIntent.TEXT_PROCESSING)
    analyzer = FixedAnalyzer()
    free = await analyze_stage(
        _ctx(LONG_QUERY, history=[]), intent_router=router, analyzer=analyzer
    )
    assert free.extra["query_analysis"]["intent"] == "text_processing"
    assert free.extra["query_analysis"]["has_pasted_content"] is True
    assert not retrieval_wanted(free)

    scoped = await analyze_stage(
        _ctx(LONG_QUERY, history=[], pinned_ids=["doc-1"]), intent_router=router, analyzer=analyzer
    )
    assert scoped.extra["query_analysis"]["intent"] == "factual"
    assert retrieval_wanted(scoped)


async def test_condenser_no_retrieval_verdict_is_overridden_by_scope():
    analyzer = FixedAnalyzer()
    free = await analyze_stage(
        _ctx(), analyzer=analyzer, condenser=StubCondenser(requires_retrieval=False)
    )
    assert free.extra["requires_retrieval"] is False
    assert not retrieval_wanted(free)

    scoped = await analyze_stage(
        _ctx(scope_lock_ids=["case-1"]),
        analyzer=analyzer,
        condenser=StubCondenser(requires_retrieval=False),
    )
    assert scoped.extra["requires_retrieval"] is True
    assert retrieval_wanted(scoped)


async def test_pure_scope_skips_condensation_but_shared_corpus_keeps_it():
    pure = _ctx(pinned_ids=["doc-1"], enable_shared_corpus=False)
    llm = _condense_llm()
    out = await analyze_stage(pure, llm=llm)
    assert out.extra["condensed_query"] == FOLLOW_UP
    assert out.extra["query_type"] == QueryType.FOLLOW_UP.value
    assert llm.calls == []

    broader = _ctx(pinned_ids=["doc-1"], enable_shared_corpus=True)
    out = await analyze_stage(broader, llm=llm)
    assert out.extra["condensed_query"] == STANDALONE
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


async def test_unexpected_error_returns_the_input_context_unchanged():
    ctx = _ctx(pinned_ids=["d1"], extra={"caller": "kept"})
    before = dataclasses.replace(ctx)
    out = await analyze_stage(ctx, llm=_condense_llm(), analyzer=RaisingAnalyzer())
    assert out is ctx
    assert ctx == before
    assert not set(ANALYZE_EXTRA_KEYS) & set(out.extra)


async def test_unknown_tier_returns_the_input_context():
    ctx = _ctx(tier="not-a-tier")
    assert await analyze_stage(ctx) is ctx


async def test_failing_identity_hook_returns_the_input_context():
    def broken(query: str) -> bool:
        raise ValueError("hook bug")

    ctx = _ctx()
    assert await analyze_stage(ctx, identity_check=broken) is ctx


async def test_llm_that_raises_never_blocks_the_stage():
    ctx = _ctx(LONG_QUERY, history=HISTORY)
    analyzer = FixedAnalyzer(should_retrieve=False, is_in_domain=False)
    out = await analyze_stage(ctx, llm=RaisingModel(), analyzer=analyzer)
    assert out.extra["query_analysis"]["should_retrieve"] is True
    assert out.metrics["analyze"]["domain_check"] == "failed"
    assert retrieval_wanted(out)


# ---------------------------------------------------------------------------
# Identity hook, domain check, recency, plugin
# ---------------------------------------------------------------------------


async def test_identity_check_short_circuit_when_supplied():
    llm = _condense_llm()
    out = await analyze_stage(
        _ctx("Who are you?", history=HISTORY), llm=llm, identity_check=lambda q: True
    )
    assert out.extra["query_analysis"]["intent"] == "identity"
    assert out.extra["query_analysis"]["should_retrieve"] is False
    assert out.extra["condensed_query"] == "Who are you?"
    assert out.metrics["analyze"]["short_circuit"] == "identity"
    assert llm.calls == []


async def test_without_identity_check_nothing_is_treated_as_identity():
    out = await analyze_stage(_ctx("Who are you?", history=[]))
    assert out.extra["query_analysis"]["intent"] != "identity"


async def test_domain_check_overrides_a_no_retrieve_verdict_for_long_queries():
    analyzer = FixedAnalyzer(should_retrieve=False, is_in_domain=False)
    ledger = UsageLedger()
    yes = await analyze_stage(
        _ctx(LONG_QUERY, history=[]),
        llm=FakeLanguageModel(reply="Yes"),
        analyzer=analyzer,
        usage=ledger,
    )
    assert yes.extra["query_analysis"]["should_retrieve"] is True
    assert yes.extra["query_analysis"]["complexity"] == "moderate"
    assert ledger.counts_by_operation().get("legal_domain_check") == 1

    no = await analyze_stage(
        _ctx(LONG_QUERY, history=[]), llm=FakeLanguageModel(reply="no"), analyzer=analyzer
    )
    assert no.extra["query_analysis"]["should_retrieve"] is False

    short = FakeLanguageModel(reply="yes")
    out = await analyze_stage(_ctx("hi there", history=[]), llm=short, analyzer=analyzer)
    assert out.extra["query_analysis"]["should_retrieve"] is False and short.calls == []


async def test_domain_check_without_a_model_retrieves_anyway():
    analyzer = FixedAnalyzer(should_retrieve=False, is_in_domain=False)
    out = await analyze_stage(_ctx(LONG_QUERY, history=[]), analyzer=analyzer)
    assert out.extra["query_analysis"]["should_retrieve"] is True
    assert out.metrics["analyze"]["domain_check"] == "unavailable"


async def test_domain_check_truncates_and_treats_empty_answer_as_yes():
    llm = FakeLanguageModel(reply="")
    assert await llm_legal_domain_check("x" * 5000, llm) is True
    assert len(llm.calls[0][1].content) == AnalyzeConfig().domain_check_max_chars
    assert await llm_legal_domain_check("hello", FakeLanguageModel(reply="no")) is False


async def test_recency_gap_flag_uses_the_configured_cutoff():
    query = "latest case law on adverse possession from 2026"
    on = await analyze_stage(_ctx(query, history=[]), config=AnalyzeConfig(corpus_cutoff_year=2024))
    assert on.extra["temporal_recency_gap"] is True
    assert on.extra["temporal_requested_year"] == 2026
    assert on.extra["web_search_suggestion"] is True

    inside = await analyze_stage(
        _ctx("adverse possession cases from 2022", history=[]),
        config=AnalyzeConfig(corpus_cutoff_year=2024),
    )
    assert inside.extra["temporal_recency_gap"] is False
    assert inside.extra["temporal_requested_year"] == 2022
    assert inside.extra["web_search_suggestion"] is False

    off = await analyze_stage(_ctx(query, history=[]))
    assert off.extra["temporal_recency_gap"] is False, "no cutoff configured means no claim"
    assert off.extra["temporal_requested_year"] == 2026


def test_recency_helpers():
    assert detect_recency_gap("rulings from 2026", 2025) == (True, 2026)
    assert detect_recency_gap("rulings from 2025", 2025) == (False, 2025)
    assert detect_recency_gap("rulings", 2025) == (False, None)
    assert detect_recency_gap("rulings from 2030", None) == (False, 2030)
    assert detect_recency_intent("what are the recent developments in zoning law")
    assert detect_recency_intent("cases from the last 5 years")
    assert not detect_recency_intent("what is adverse possession")


async def test_plugin_condenser_key_scopes_the_condenser():
    condenser = StubCondenser()
    analyzer = FixedAnalyzer()
    plugin: JurisdictionPlugin = StubPlugin()  # type: ignore[assignment]
    out = await analyze_stage(_ctx(), analyzer=analyzer, condenser=condenser, plugin=plugin)
    assert out.extra["condenser_key"] == "xx"
    assert condenser.scopes == ["xx"]

    other = StubCondenser()
    out = await analyze_stage(
        _ctx(),
        analyzer=analyzer,
        condenser=other,
        plugin=StubPlugin(matches=False),  # type: ignore[arg-type]
    )
    assert out.extra["condenser_key"] is None and other.scopes == [""]

    plain = StubCondenser()
    out = await analyze_stage(_ctx(), analyzer=analyzer, condenser=plain)
    assert out.extra["condenser_key"] is None and plain.scopes == [""]


async def test_caller_extra_is_preserved_and_input_is_not_mutated():
    ctx = _ctx(extra={"caller": "kept"})
    out = await analyze_stage(ctx, llm=_condense_llm())
    assert out.extra["caller"] == "kept"
    assert ctx.extra == {"caller": "kept"}
    assert out is not ctx


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

ANALYZE_FILE = (
    Path(__file__).resolve().parents[2] / "src" / "legal_rag" / "pipeline" / "stages" / "analyze.py"
)
FORBIDDEN = {
    "vaquill": re.compile(r"vaquill", re.IGNORECASE),
    "supabase": re.compile(r"supabase", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "organization_id": re.compile(r"organization_id"),
    "user_id": re.compile(r"user_id"),
    "matter_id": re.compile(r"matter_id"),
    "private app import": re.compile(r"(?<![A-Za-z_])app\."),
    "os.environ": re.compile(r"os\.environ|getenv"),
    "settings": re.compile(r"settings", re.IGNORECASE),
    "gpt-": re.compile(r"gpt-", re.IGNORECASE),
    "courtlistener": re.compile(r"courtlistener", re.IGNORECASE),
    "voyage": re.compile(r"voyage", re.IGNORECASE),
    "em dash": re.compile(f"{chr(0x2014)}|{chr(0x2013)}"),
    "ticket tag": re.compile(r"Round \d|bug-?fix #\d|\bF\d{1,2}\b|EDGE \d|GAP \d", re.IGNORECASE),
    "direct chat model": re.compile(r"ChatOpenAI|langchain"),
    "cost tracker": re.compile(r"cost_tracking|track_openai|track_technique"),
}


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_analyze_module_has_no_forbidden_text(label: str):
    text = ANALYZE_FILE.read_text(encoding="utf-8")
    hits = [
        f"line {n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        if FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found in analyze.py: {hits[:5]}"
