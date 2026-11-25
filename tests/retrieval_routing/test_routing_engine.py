"""The taxonomy-driven router: prefilter, decision rules, classification and the eval harness."""

from __future__ import annotations

import asyncio
import json

from legal_rag.providers.fakes import FakeEmbedder, FakeLanguageModel
from legal_rag.retrieval.routing.config import ActionCost, Destination, Taxonomy, example_taxonomy
from legal_rag.retrieval.routing.decision import decide
from legal_rag.retrieval.routing.eval import evaluate, format_report
from legal_rag.retrieval.routing.example_cases import EXAMPLE_CASES
from legal_rag.retrieval.routing.prefilter import prefilter
from legal_rag.retrieval.routing.router import route, shortlist
from legal_rag.retrieval.routing.shadow_report import build_report, load
from legal_rag.retrieval.routing.state import PinnedDoc, RouterState, RoutingInput

TAX = example_taxonomy()


def run(coro):
    return asyncio.run(coro)


def _ri(text: str) -> RoutingInput:
    return RoutingInput(instruction=text, full_message=text)


def test_example_taxonomy_has_default_and_small_talk():
    assert TAX.has(TAX.default_intent)
    assert TAX.small_talk_intent and TAX.has(TAX.small_talk_intent)
    assert TAX.cost("export_report") is ActionCost.EXPENSIVE


def test_taxonomy_is_injected_configuration():
    custom = Taxonomy(
        destinations=(
            Destination(intent="answer", description="Answer the question"),
            Destination(intent="lookup", description="Look up a record by id"),
        ),
        small_talk_intent=None,
    )
    assert custom.intents() == ("answer", "lookup")
    assert not custom.has("compare")


def test_prefilter_routes_greeting_to_small_talk():
    d = prefilter("hi", RouterState(), TAX)
    assert d is not None
    assert d.intent == "small_talk"


def test_route_without_models_falls_back_to_default_intent():
    d = run(route(_ri("what is the limitations period for fraud"), RouterState(), taxonomy=TAX))
    assert d.intent == TAX.default_intent


def test_compare_with_one_document_degrades():
    state = RouterState(pinned_docs=(PinnedDoc("d1"),))
    d = decide(TAX, "compare", 0.95, state)
    assert d.intent != "compare"
    assert d.precondition_note


def test_compare_with_two_documents_is_kept():
    state = RouterState(pinned_docs=(PinnedDoc("d1"), PinnedDoc("d2")))
    d = decide(TAX, "compare", 0.95, state)
    assert d.intent == "compare"


def test_shortlist_returns_known_intents():
    picked = run(shortlist("compare these two contracts", TAX, FakeEmbedder(), k=3))
    assert TAX.default_intent in picked
    assert all(TAX.has(i) for i in picked)


def test_llm_classification_is_used():
    llm = FakeLanguageModel(reply=json.dumps({"intent": "find_authority", "confidence": 0.97}))
    d = run(
        route(
            _ri("find me cases that apply the economic loss rule"),
            RouterState(),
            taxonomy=TAX,
            embedder=FakeEmbedder(),
            llm=llm,
        )
    )
    assert llm.calls
    assert TAX.has(d.intent)


def test_eval_harness_runs_on_example_cases():
    report = run(evaluate(EXAMPLE_CASES, TAX, llm=None))
    assert report.total == len(EXAMPLE_CASES)
    assert 0 <= report.hits <= report.total
    assert isinstance(format_report(report), str)


def test_shadow_report_handles_empty_input():
    assert isinstance(build_report(load([])), str)
