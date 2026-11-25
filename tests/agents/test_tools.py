"""Behavioural tests for the agent tool schemas, registry, selection and dynamic tools."""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from typing import Any

import pytest

from legal_rag.agents.tools import (
    CAP_SHARED_CORPUS,
    DEFAULT_TOOL_NAMES,
    TIER_ORDER,
    DynamicToolSpec,
    ToolExecutor,
    ToolRegistry,
    build_dynamic_tools,
    default_registry,
    spec_from_tool,
    tools_for_request,
    validate_tool_schema,
)

KEPT = {
    "web_search",
    "fetch_url",
    "search_documents",
    "lookup_us_statute_exact",
    "lookup_us_case_by_citation",
    "verify_us_citation",
    "find_us_citing_cases",
    "search_us_statutes",
    "search_us_case_law",
    "resolve_citation",
}
LEFT_OUT = {
    "search_drafts",
    "diff_draft_versions",
    "review_against_playbook",
    "extract_obligations",
    "ask_across_portfolio",
    "read_matter_analysis",
    "screen_counterparty",
    "analyze_vendor_policy",
    "resolve_location",
    "generate_case_brief",
    "search_legal_corpus",
}
CORPUS_TOOLS = {
    "lookup_us_statute_exact",
    "lookup_us_case_by_citation",
    "verify_us_citation",
    "find_us_citing_cases",
    "search_us_statutes",
    "search_us_case_law",
    "resolve_citation",
}
FLAG_NAMES = (
    "enable_web_search",
    "enable_fetch_url",
    "enable_document_search",
    "enable_shared_corpus",
)


def _names(tools: list[dict[str, Any]]) -> list[str]:
    return [t["function"]["name"] for t in tools]


def _tiers(registry: ToolRegistry, tools: list[dict[str, Any]]) -> list[int]:
    return [TIER_ORDER.index(registry.info(n).tier) for n in _names(tools)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_default_registry_holds_exactly_the_kept_tools() -> None:
    registry = default_registry()
    assert set(registry.names()) == KEPT
    assert not (set(registry.names()) & LEFT_OUT)
    assert tuple(registry.names()) == DEFAULT_TOOL_NAMES


@pytest.mark.parametrize("name", sorted(KEPT))
def test_every_schema_is_valid_json_schema_shape(name: str) -> None:
    tool = default_registry().get(name)
    assert validate_tool_schema(tool) == name
    assert tool["type"] == "function"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert isinstance(params["properties"], dict)
    assert set(params["required"]) <= set(params["properties"])
    assert params["additionalProperties"] is False
    assert tool["function"]["description"].strip()
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for prop, schema in params["properties"].items():
        assert schema["description"].strip(), f"{name}.{prop} has no description"
        for value in schema.get("enum", []):
            assert isinstance(value, types[schema["type"]])
        if "default" in schema and "minimum" in schema:
            assert schema["minimum"] <= schema["default"] <= schema["maximum"]


def test_names_are_unique_and_duplicates_are_refused() -> None:
    registry = default_registry()
    names = registry.names()
    assert len(names) == len(set(names))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("fetch_url"))
    registry.register(registry.get("fetch_url"), tier="fetch", replace=True)
    assert registry.names() == names


def test_registry_returns_copies() -> None:
    registry = default_registry()
    registry.get("web_search")["function"]["name"] = "changed"
    registry.filter()[0]["function"]["description"] = "changed"
    assert registry.get("web_search")["function"]["name"] == "web_search"
    assert registry.get("lookup_us_statute_exact")["function"]["description"] != "changed"
    with pytest.raises(KeyError):
        registry.get("search_drafts")


def test_registry_filter_by_name_and_predicate() -> None:
    registry = default_registry()
    assert _names(registry.filter(names=["web_search", "fetch_url"])) == ["fetch_url", "web_search"]
    exact = registry.filter(lambda info: info.tier == "exact")
    assert set(_names(exact)) < CORPUS_TOOLS | {"resolve_citation"}
    with pytest.raises(KeyError):
        registry.filter(names=["nope"])


def test_web_search_jurisdiction_is_a_free_string() -> None:
    prop = default_registry().get("web_search")["function"]["parameters"]["properties"]
    assert prop["jurisdiction"]["type"] == "string"
    assert "enum" not in prop["jurisdiction"]
    assert default_registry().get("web_search")["function"]["parameters"]["required"] == ["query"]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"type": "function"},
        {
            "type": "function",
            "function": {"name": "has space", "description": "d", "parameters": {}},
        },
        {"type": "function", "function": {"name": "x", "description": " ", "parameters": {}}},
        {
            "type": "function",
            "function": {
                "name": "x",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "x",
                "description": "d",
                "parameters": {"type": "object", "properties": {"a": {}}, "required": []},
            },
        },
    ],
)
def test_validate_tool_schema_rejects_malformed_tools(bad: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_tool_schema(bad)


# ---------------------------------------------------------------------------
# Ordering rule
# ---------------------------------------------------------------------------


def test_order_rule_is_exact_then_semantic_then_fetch_then_web() -> None:
    assert TIER_ORDER == ("exact", "semantic", "fetch", "web")


def test_default_listing_follows_the_order_rule() -> None:
    registry = default_registry()
    tiers = [TIER_ORDER.index(registry.info(n).tier) for n in DEFAULT_TOOL_NAMES]
    assert tiers == sorted(tiers)
    position = {n: i for i, n in enumerate(DEFAULT_TOOL_NAMES)}
    exact = [n for n in DEFAULT_TOOL_NAMES if registry.info(n).tier == "exact"]
    semantic = [n for n in DEFAULT_TOOL_NAMES if registry.info(n).tier == "semantic"]
    assert exact and semantic
    assert max(position[n] for n in exact) < min(position[n] for n in semantic)
    assert max(position[n] for n in semantic) < position["fetch_url"] < position["web_search"]
    assert position["lookup_us_statute_exact"] < position["search_us_statutes"]
    assert position["lookup_us_case_by_citation"] < position["search_us_case_law"]


@pytest.mark.parametrize("jurisdiction", [None, "US", "us", "GB"])
def test_every_flag_combination_keeps_the_order_rule(jurisdiction: str | None) -> None:
    registry = default_registry()
    for combo in itertools.product([True, False], repeat=len(FLAG_NAMES)):
        tools = tools_for_request(
            jurisdiction=jurisdiction, **dict(zip(FLAG_NAMES, combo, strict=True))
        )
        tiers = _tiers(registry, tools)
        assert tiers == sorted(tiers), (jurisdiction, combo)


def test_semantic_tools_point_at_the_exact_tool_first() -> None:
    registry = default_registry()
    statutes = registry.get("search_us_statutes")["function"]["description"]
    cases = registry.get("search_us_case_law")["function"]["description"]
    assert "lookup_us_statute_exact" in statutes
    assert "lookup_us_case_by_citation" in cases


def test_descriptions_only_name_registered_tools() -> None:
    registry = default_registry()
    pattern = re.compile(r"\b(?:lookup|search|verify|find|resolve|fetch|web)_[a-z_]+\b")
    for name in registry.names():
        params = registry.get(name)["function"]["parameters"]["properties"]
        text = registry.get(name)["function"]["description"]
        for match in pattern.findall(text):
            assert match in registry or match in params, f"{name} mentions unknown tool {match}"


# ---------------------------------------------------------------------------
# Request selection
# ---------------------------------------------------------------------------


def test_full_request_for_us_returns_every_tool_in_default_order() -> None:
    assert _names(tools_for_request(jurisdiction="US")) == list(DEFAULT_TOOL_NAMES)


def test_web_search_is_dropped_when_not_allowed() -> None:
    names = _names(tools_for_request(jurisdiction="US", enable_web_search=False))
    assert "web_search" not in names
    assert "fetch_url" in names
    assert set(names) == KEPT - {"web_search"}


def test_corpus_tools_are_dropped_when_the_shared_corpus_is_off() -> None:
    names = set(_names(tools_for_request(jurisdiction="US", enable_shared_corpus=False)))
    assert names == {"search_documents", "fetch_url", "web_search"}
    registry = default_registry()
    assert {
        n for n in registry.names() if registry.info(n).capability == CAP_SHARED_CORPUS
    } == CORPUS_TOOLS


def test_jurisdiction_limited_tools_need_a_matching_jurisdiction() -> None:
    assert set(_names(tools_for_request())) == {"search_documents", "fetch_url", "web_search"}
    assert set(_names(tools_for_request(jurisdiction="GB"))) == {
        "search_documents",
        "fetch_url",
        "web_search",
    }
    assert _names(tools_for_request(jurisdiction=" us ")) == list(DEFAULT_TOOL_NAMES)


def test_document_and_fetch_flags() -> None:
    assert "search_documents" not in _names(
        tools_for_request(jurisdiction="US", enable_document_search=False)
    )
    assert "fetch_url" not in _names(tools_for_request(jurisdiction="US", enable_fetch_url=False))


def test_selection_returns_copies() -> None:
    first = tools_for_request(jurisdiction="US")
    first[0]["function"]["name"] = "changed"
    assert _names(tools_for_request(jurisdiction="US"))[0] == "lookup_us_statute_exact"


# ---------------------------------------------------------------------------
# Dynamic tools
# ---------------------------------------------------------------------------

SPECS = [
    DynamicToolSpec(
        name="lookup_client_file",
        description="Look up a document in the caller's own system.",
        parameters={
            "file_number": {"type": "string", "description": "The file number."},
            "limit": {"type": "integer", "description": "Max rows.", "default": 5},
        },
        required=("file_number",),
        tier="exact",
    ),
    DynamicToolSpec(name="ping", description="A tool with no parameters."),
]


def test_build_dynamic_tools_round_trips() -> None:
    tools = build_dynamic_tools(SPECS)
    assert _names(tools) == ["lookup_client_file", "ping"]
    for tool in tools:
        validate_tool_schema(tool)
    first = tools[0]["function"]
    assert first["description"] == SPECS[0].description
    assert first["parameters"]["required"] == ["file_number"]
    assert first["parameters"]["properties"] == SPECS[0].parameters
    assert tools[1]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    rebuilt = [spec_from_tool(t) for t in tools]
    assert [(s.name, s.description, dict(s.parameters), s.required) for s in rebuilt] == [
        (s.name, s.description, dict(s.parameters), s.required) for s in SPECS
    ]
    assert build_dynamic_tools(rebuilt) == tools


def test_build_dynamic_tools_accepts_mappings_and_a_prefix() -> None:
    tools = build_dynamic_tools(
        [
            {
                "name": "a",
                "description": "d",
                "parameters": {"q": {"type": "string"}},
                "required": ["q"],
            }
        ],
        name_prefix="team__",
    )
    assert _names(tools) == ["team__a"]
    assert tools[0]["function"]["parameters"]["required"] == ["q"]


@pytest.mark.parametrize(
    "specs",
    [
        [{"name": "a", "description": "d", "bogus": 1}],
        [{"name": "bad name", "description": "d"}],
        [{"name": "a", "description": ""}],
        [{"name": "a", "description": "d", "required": ["missing"]}],
        [{"name": "a", "description": "d", "parameters": {"q": {"description": "no type"}}}],
        [{"name": "a", "description": "d"}, {"name": "a", "description": "e"}],
        [{"name": "a", "description": "d", "tier": "nonsense"}],
        [{"name": "a", "description": "d", "capability": "nonsense"}],
    ],
)
def test_build_dynamic_tools_rejects_bad_specs(specs: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        build_dynamic_tools(specs)


def test_dynamic_specs_register_with_their_selection_data() -> None:
    registry = default_registry()
    added = registry.register_specs(
        [
            DynamicToolSpec(
                name="lookup_court_rule",
                description="Exact court rule lookup.",
                tier="exact",
                capability=CAP_SHARED_CORPUS,
                jurisdictions=frozenset({"us"}),
            )
        ]
    )
    assert added == ["lookup_court_rule"]
    info = registry.info("lookup_court_rule")
    assert (info.tier, info.capability, info.jurisdictions) == (
        "exact",
        CAP_SHARED_CORPUS,
        frozenset({"US"}),
    )
    assert (
        registry.get("lookup_court_rule")["function"]["description"] == "Exact court rule lookup."
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.register_specs([{"name": "web_search", "description": "clash"}])


def test_extra_specs_are_sorted_into_their_tier_and_do_not_leak() -> None:
    extra = [
        {"name": "late_tool", "description": "A web tier tool.", "tier": "web"},
        {"name": "early_tool", "description": "An exact tier tool.", "tier": "exact"},
    ]
    names = _names(tools_for_request(jurisdiction="US", extra_specs=extra))
    assert names[0] == "lookup_us_statute_exact"
    assert names.index("early_tool") < names.index("search_us_statutes")
    assert names.index("late_tool") > names.index("fetch_url")
    assert "early_tool" not in _names(tools_for_request(jurisdiction="US"))
    registry = default_registry()
    tools = tools_for_request(registry=registry, jurisdiction="US", extra_specs=extra)
    assert "early_tool" in _names(tools) and "early_tool" not in registry


def test_a_custom_registry_is_honoured() -> None:
    registry = ToolRegistry()
    registry.register(build_dynamic_tools([{"name": "only", "description": "d"}])[0])
    assert _names(tools_for_request(registry=registry, jurisdiction="US")) == ["only"]


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, name: str, arguments: Mapping[str, Any]) -> str:
        self.calls.append((name, dict(arguments)))
        return f"ran {name}"


class NotAnExecutor:
    def run(self, name: str) -> str:
        return name


async def test_a_fake_executor_satisfies_the_protocol() -> None:
    executor = FakeExecutor()
    assert isinstance(executor, ToolExecutor)
    assert not isinstance(NotAnExecutor(), ToolExecutor)
    result = await executor.call("web_search", {"query": "qualified immunity"})
    assert result == "ran web_search"
    assert executor.calls == [("web_search", {"query": "qualified immunity"})]
