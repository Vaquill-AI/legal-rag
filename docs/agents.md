# Agents

The agents component writes a grounded answer from retrieved sources by running a plan, validate, write, refine and verify loop around a language model you inject.
It also ships the tool schemas, as plain data, that a tool-calling legal agent offers its model, plus the executor contract that runs them.

## Generation order

`generate_answer_buffered` in `legal_rag.agents.generation` runs these stages in this order.

```text
query + retrieved sources
  |
  1. plan       PlanningActor            deep tier only, skipped for a trivial query
  |
  2. validate   deterministic check, then ValidationActor
  |             recommendation: proceed | retrieve_more | simplify | fallback_to_general
  |
  3. write      model.generate           one buffered call, answer tokens capped by complexity
  |
  4. refine     RefinementActor          deep tier only, at most one rewrite
  |
  5. verify     HallucinationDetector    optional, scores the final answer
  |
GenerationResult(answer, metadata, plan, actor_metrics, verification)
```

Every stage except the answer call fails open: an error is logged and the run carries on.
An error from the answer model propagates to you.

## The actors

| Actor | Model needed | What it decides |
| --- | --- | --- |
| `PlanningActor` | yes | Sub-questions mapped to chunk indices, the response structure, information gaps and risk areas, with keyword heuristics picking the structure when confident. |
| `validate_plan_deterministically` | no | Whether each sub-question and required source has keyword or source support in the chunks, and it gates the next actor. |
| `ValidationActor` | yes | `is_valid`, `missing_support` and a recommendation. |
| `RefinementActor` | yes | A critique of the draft: missing elements, citation gaps, clarity issues and `should_regenerate`, and it never rewrites. |

A recommendation changes the write stage.
`fallback_to_general` relaxes strict grounding, marks every sub-question unanswerable and records the gap.
`retrieve_more` marks sub-questions with no chunk mapping as unanswerable.
`proceed` and `simplify` change nothing.
A rewrite happens when the critique sets `should_regenerate`, carries suggested improvements and reports `confidence_score` below 0.6.

## Quick start

A canned model, two sources and the real detector, with no network.
The standard tier skips planning and refinement, so the answer is one model call.

```python
import asyncio

from legal_rag.agents.generation import GenerationConfig, generate_answer_buffered
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.verification.grounding.detector import HallucinationDetector

SOURCES = [
    {"text": "A landlord must give a tenant 14 days written notice before an eviction.",
     "metadata": {"title": "Tenancies Act, section 12"}, "score": 0.82},
    {"text": "The written notice must state the amount owed and the date it is due.",
     "metadata": {"title": "Tenancies Act, section 13"}, "score": 0.74},
]
ANSWER = "A landlord must give 14 days written notice [1]. It must state the amount owed [2]."


async def main() -> None:
    config = GenerationConfig("What notice comes before an eviction?", SOURCES)
    model = FakeLanguageModel(reply=ANSWER)
    result = await generate_answer_buffered(config, model, detector=HallucinationDetector())
    print(result.answer)
    print(result.plan, result.actor_metrics)
    print(result.metadata["planning"], result.metadata["effective_max_tokens"])
    check = result.verification
    print(check.verified, check.score, check.total_claims, check.supported_claims)


asyncio.run(main())
```

It prints the answer, `None []` (no plan, no actor metrics), `has_plan` false with `4096`, then `True 1.0 2 1`.
The run is verified although only one of two claims is fully supported, because the detector counts a partly grounded claim toward `score` but not toward `supported_claims` (see Known limits).
Read the per-claim summary in `metadata["verification"]` and not only `verified`.
`GenerationResult` holds `answer`, `metadata`, `plan`, `actor_metrics` and `verification`.
`metadata` always carries `planning`, `validation`, `effective_max_tokens` and `actor_metrics`, plus `refinement` when a plan exists and `verification` when the detector returned a result.

The deep tier adds the three actors.
A `FakeLanguageModel` answers each actor from `structured`, keyed by schema name.

```python
import asyncio

from legal_rag.agents.generation import GenerationConfig, generate_answer_buffered
from legal_rag.providers.fakes import FakeLanguageModel
from legal_rag.verification.grounding.actor_schemas import (
    RefinementSuggestions,
    ResponsePlan,
    SubQuestion,
    ValidationResult,
)

SOURCES = [
    {"text": "A tenant ends a lease with thirty days written notice.", "metadata": {"title": "Act s1"}},
    {"text": "The deposit is returned within twenty one days.", "metadata": {"title": "Act s2"}},
]
PLAN = ResponsePlan(
    sub_questions=[SubQuestion(question="How does a tenant end a lease?", chunk_indices=[0])],
    required_sources=["Act s1"],
    response_structure="Procedural",
)
STRUCTURED = {
    "ResponsePlan": PLAN,
    "ValidationResult": ValidationResult(is_valid=True, recommendation="proceed"),
    "RefinementSuggestions": RefinementSuggestions(should_regenerate=False, confidence_score=0.9),
}


async def main() -> None:
    config = GenerationConfig(
        "How does a tenant end a lease?", SOURCES, tier="deep", query_analysis={"complexity": "simple"}
    )
    model = FakeLanguageModel(reply="Thirty days notice [1].", structured=STRUCTURED)
    result = await generate_answer_buffered(config, model)
    print(result.metadata["refinement"], result.metadata["effective_max_tokens"])
    for m in result.actor_metrics:
        print(m.actor_name, m.llm_calls, m.tokens_used, m.error_occurred)


asyncio.run(main())
```

It prints `{'regenerated': False} 4000`, then one metrics line each for `planning`, `validation` and `refinement`.
The `simple` complexity caps answer tokens at 4000.
The other caps are `trivial` 500, `moderate` 16000 and `complex` 30000, which is also the cap when the complexity is unknown.
Pass `actor_model` to run the actors on a cheaper model than the one that writes the answer, and `actor_configs` (name to `ActorConfig`) to set their limits.

## Agentic search tools

`legal_rag.agents.tools` holds ten tool schemas in the function-calling shape and the rules for which to offer.
Tools are always listed by tier, because a model tends to pick the first tool that fits: exact lookup, then semantic search, then URL fetch, then web search.

| Tool | Tier | Flag | Limited to | Your executor backs it with |
| --- | --- | --- | --- | --- |
| `lookup_us_statute_exact` | exact | `shared_corpus` | US | A section lookup by corpus (`usc`, `cfr`, `state`), title and section |
| `lookup_us_case_by_citation` | exact | `shared_corpus` | US | Opinion text and an `opinion_id` for a reporter citation |
| `verify_us_citation` | exact | `shared_corpus` | US | A light validity check that normalizes a citation |
| `find_us_citing_cases` | exact | `shared_corpus` | US | Later cases that cite an `opinion_id` or `cluster_id` |
| `resolve_citation` | exact | `shared_corpus` | US | A citation string resolved to its authority and aliases |
| `search_us_statutes` | semantic | `shared_corpus` | US | Semantic search over federal statutes and regulations |
| `search_us_case_law` | semantic | `shared_corpus` | US | Semantic search over opinions, with court and date filters |
| `search_documents` | semantic | `documents` | any | Search over the user's own documents, optionally one document |
| `fetch_url` | fetch | `fetch_url` | any | An HTTP fetch of a public page |
| `web_search` | web | `web_search` | any | A web search |

Nothing here runs a tool.
You implement `ToolExecutor`, one async method `call(name, arguments) -> str`, and own authorization, tenancy filters and rate limits.
Return an error string for an expected failure so the model can recover.

```python
import asyncio
from collections.abc import Mapping
from typing import Any

from legal_rag.agents.tools import DynamicToolSpec, ToolExecutor, tools_for_request


def names(tools: list[dict[str, Any]]) -> list[str]:
    return [t["function"]["name"] for t in tools]


print(names(tools_for_request()))

rule_tool = DynamicToolSpec(
    name="lookup_local_rule",
    description="Look up a local court rule by number.",
    parameters={"rule": {"type": "string", "description": "The rule number."}},
    required=("rule",),
    tier="exact",
)
print(names(tools_for_request(jurisdiction="US", enable_shared_corpus=False, extra_specs=[rule_tool])))


class LocalExecutor:
    """Your code: authorize, apply tenant filters, call your backends, return text."""

    async def call(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name == "lookup_local_rule":
            return f"Rule {arguments['rule']}: (text from your own store)"
        return f"error: tool {name!r} is not wired up"


executor = LocalExecutor()
print(isinstance(executor, ToolExecutor))
print(asyncio.run(executor.call("lookup_local_rule", {"rule": "12(b)"})))
```

The first line prints only `search_documents`, `fetch_url` and `web_search`, because a request with no jurisdiction gets no jurisdiction-limited tools.
The second turns the US corpus tools off with `enable_shared_corpus=False` and puts `lookup_local_rule` first because its tier is `exact`.
Pass `jurisdiction="US"` with every flag on to get all ten, in the table's order.
`ToolRegistry`, `default_registry` and `build_dynamic_tools` let you keep your own registry.

## How tokens_used is estimated

The `LanguageModel` protocol returns text and no usage, so each actor estimates tokens at four characters per token.
`tokens_used` on an actor's `ActorMetrics` is the sum over the messages sent of `estimate_tokens(text)`, plus `estimate_tokens(reply_json)` for the parsed structured reply.
There is one estimator: `legal_rag.providers.usage.estimate_tokens`, which is `max(1, ceil(len(text) / 4))`.

```python
from legal_rag.agents.actors.base import estimate_tokens
from legal_rag.providers.usage import estimate_tokens as metering_estimate

print(estimate_tokens is metering_estimate)
print(estimate_tokens(""), estimate_tokens("abc"), estimate_tokens("a" * 400), estimate_tokens("a" * 401))
```

This prints `True` and then `1 1 100 101`: the actors and the `UsageHook` helper share one function, which rounds up with a minimum of 1.
For billing, treat `tokens_used` as a relative cost signal, not an invoice.

- It covers the three actors only.
  The answer call and any rewrite are not in `actor_metrics`, so meter them from your provider's own usage report.
- It uses text length, not a tokenizer, and measures the parsed reply re-serialized, so reasoning tokens and formatting are missed.
- A reply that fails schema validation still counts its prompt tokens and one call.
  Any other model failure counts nothing and returns no metrics, although your provider may have billed it.
- `cache_hit` is always false.

## Known limits

- Answers are buffered only.
  There is no streaming path.
- No tool-calling loop ships. `LanguageModel.generate` takes no tools argument, so you wire the schemas into your own client and drive the loop with your `ToolExecutor`.
- Planning and refinement run on the deep tier only, and refinement needs a plan.
  A `trivial` complexity or an empty source list skips planning.
  Web-only sources skip validation and refinement.
- At most one rewrite happens.
  It is not refined or validated again, and verification sees only the final text.
- The rewrite is built from the validation-adjusted config and plan, so after `fallback_to_general` it keeps open grounding and the gap note.
  It uses `regeneration_max_tokens` (3000) and `regeneration_temperature` (0.7) instead of the complexity cap.
  The refinement critique still reads the planner's original plan, not the adjusted one.
- The planner prompt's "Estimated Complexity" is always the planner's own default, 0.50.
  `query_analysis["complexity"]` is a label, not the 0.0 to 1.0 number the planner reads, and `GenerationConfig.confidence` is not read by any stage.
- Unusable model output degrades quietly.
  Planning falls back to a one-question plan, validation reports `is_valid` false with the recommendation `proceed`, and refinement accepts the draft.
- Prompts see excerpts: 300 characters per chunk for planning, 500 for validation.
  Refinement cuts a draft at 15000 characters and lists the first 20 sources.
- `FakeLanguageModel.generate_structured` raises `KeyError` for a schema with no canned reply, which an actor reports as `ActorError`.
- `verified` and `score` come from the injected `HallucinationDetector`, and the summary lists at most 10 claims that are not fully supported, including `neutral` (partly grounded) ones.
- The detector's `check` scores `(grounded + partly grounded) / claims analysed` and sets `verified` when that is at least 0.7, but `supported_claims` counts grounded claims only (match confidence 0.8 or more).
  One grounded and one partly grounded claim therefore gives `verified` true, `score` 1.0 and `supported_claims` 1 of 2.
  This is detector behavior in `verification/grounding/detector.py`, not something this component changes, so its owner decides whether partly grounded claims should count.
- The built-in tools describe US law, so other jurisdictions need your own `DynamicToolSpec` entries.

## Tests

The tests live in `tests/agents/`: `test_generation.py` (stage order, fallback, rewrite, fail-open), `test_actors.py` (the actors and the deterministic validator) and `test_tools.py` (schemas, ordering, executor protocol).
`test_agents_hygiene.py` and `test_tools_hygiene.py` scan the source for private names and for imports beyond the standard library, pydantic and this library.
Run them with `.venv/bin/python -m pytest tests/agents`.
