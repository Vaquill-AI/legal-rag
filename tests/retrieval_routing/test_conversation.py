"""Conversation handling: message split, intent routing, condensing, memory, expansion."""

from __future__ import annotations

import asyncio

from legal_rag.providers.fakes import FakeCache, FakeLanguageModel
from legal_rag.retrieval.conversation.condenser import CondensedQueryOutput, QueryCondenser
from legal_rag.retrieval.conversation.intent_router import (
    IntentClassification,
    IntentRouter,
    UserIntent,
    extract_instruction_and_content,
)
from legal_rag.retrieval.conversation.memory import ConversationManager
from legal_rag.retrieval.conversation.message_split import split_message
from legal_rag.retrieval.conversation.query_analyzer import (
    QueryAnalyzer,
    QueryComplexity,
    QueryIntent,
)
from legal_rag.retrieval.conversation.query_enricher import QueryEnricher
from legal_rag.retrieval.conversation.query_expander import QueryExpander, QueryExpansions


def run(coro):
    return asyncio.run(coro)


PASTE = " ".join(["The parties agree to the following terms and conditions."] * 15)


def test_split_message_leaves_a_plain_question_alone():
    instruction, body = split_message("What is the statute of limitations for fraud?")
    assert "statute of limitations" in instruction
    assert body is None


def test_instruction_and_pasted_content_are_separated():
    instruction, content, _reason = extract_instruction_and_content(
        f"Proofread this brief:\n\n{PASTE}"
    )
    assert "Proofread" in instruction
    assert content and "parties agree" in content


def test_short_message_is_not_a_paste_and_retrieves():
    result = run(IntentRouter().route("What is the statute of limitations for fraud?"))
    assert result.should_retrieve


def test_pasted_text_with_editing_instruction_skips_retrieval():
    llm = FakeLanguageModel(
        structured={
            "IntentClassification": IntentClassification(intent="text_processing", confidence=0.95)
        }
    )
    result = run(IntentRouter(llm).route(f"Proofread this brief:\n\n{PASTE}"))
    assert result.user_intent == UserIntent.TEXT_PROCESSING
    assert not result.should_retrieve


def test_intent_router_fails_open_to_research():
    result = run(IntentRouter(FakeLanguageModel()).route(f"Proofread this brief:\n\n{PASTE}"))
    assert result.should_retrieve


def test_condenser_rewrites_a_follow_up_with_the_injected_model():
    llm = FakeLanguageModel(
        structured={
            "CondensedQueryOutput": CondensedQueryOutput(
                primary_query="What is the statute of limitations for fraud in Texas?"
            )
        }
    )
    history = [
        {"role": "user", "content": "Tell me about fraud limitation periods."},
        {"role": "assistant", "content": "They vary by state."},
    ]
    result = run(QueryCondenser(llm).condense("what about Texas?", history, force=True))
    assert result.condensed_query == "What is the statute of limitations for fraud in Texas?"
    assert result.original_query == "what about Texas?"


def test_memory_is_isolated_by_scope():
    manager = ConversationManager(cache=FakeCache())
    assert run(manager.add_turn("scope-a", "s1", "hi", "hello"))
    assert run(manager.get_history("scope-a", "s1"))
    assert not run(manager.get_history("scope-b", "s1"))


def test_clearing_a_session_deletes_its_key_and_drops_it_from_the_listing():
    cache = FakeCache()
    manager = ConversationManager(cache=cache)
    assert run(manager.add_turn("scope-a", "s1", "hi", "hello"))
    assert run(manager.add_turn("scope-a", "s2", "again", "hello"))
    session_key = manager._make_key("scope-a", "s1")
    assert session_key in cache.data
    assert run(manager.clear_session("scope-a", "s1"))
    assert session_key not in cache.data
    assert run(manager.get_history("scope-a", "s1")) == []
    assert run(manager.list_sessions("scope-a")) == ["s2"]


def test_query_expander_uses_the_injected_model():
    llm = FakeLanguageModel(
        structured={
            "QueryExpansions": QueryExpansions(expansions=["fraud elements", "fraud damages"])
        }
    )
    result = run(QueryExpander(llm).expand("fraud", ambiguity_score=0.9))
    assert result.original_query == "fraud"
    assert isinstance(result.expanded_queries, list)


def test_enricher_expands_a_known_acronym():
    result = QueryEnricher().enrich("ccpa penalties")
    assert "California Consumer Privacy Act" in result.enriched_query


def test_analyzer_treats_a_statutory_lookup_as_a_simple_retrievable_query():
    result = QueryAnalyzer().analyze("What does 42 U.S.C. 1983 provide?")
    assert result.primary_intent == QueryIntent.FACTUAL
    assert result.complexity == QueryComplexity.SIMPLE
    assert result.should_retrieve is True
    assert result.is_out_of_scope is False
    assert result.should_decompose is False
