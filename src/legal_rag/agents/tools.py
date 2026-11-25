"""Tool schemas for tool-calling legal agents, kept as data.

Each tool is a plain dict in the function-calling shape
``{"type": "function", "function": {"name", "description", "parameters"}}``,
where ``parameters`` is a JSON Schema object.
Nothing here runs a tool. Execution belongs to a :class:`ToolExecutor` that the caller supplies.

Ordering rule
-------------
When two tools could answer the same question, a model tends to pick the one listed first.
The tools are therefore always listed by tier, in the order of :data:`TIER_ORDER`:
exact lookup, then semantic search, then URL fetch, then web search.
An explicit citation such as ``42 U.S.C. § 1983`` should reach a deterministic lookup
before it can reach a similarity search that returns adjacent sections.
Every tool is registered with a tier, and :func:`tools_for_request` sorts by it.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Ordering rule and capabilities, as data
# ---------------------------------------------------------------------------

TIER_ORDER: tuple[str, ...] = ("exact", "semantic", "fetch", "web")
"""Tool tiers from most specific to least specific. Earlier tiers are listed first."""

DEFAULT_TIER = "semantic"

CAP_WEB_SEARCH = "web_search"
CAP_FETCH_URL = "fetch_url"
CAP_DOCUMENTS = "documents"
CAP_SHARED_CORPUS = "shared_corpus"
KNOWN_CAPABILITIES: frozenset[str] = frozenset(
    {CAP_WEB_SEARCH, CAP_FETCH_URL, CAP_DOCUMENTS, CAP_SHARED_CORPUS}
)
"""Request-level switches a tool can depend on. A tool with no capability is always offered."""

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_TYPES = frozenset({"string", "integer", "number", "boolean", "array", "object", "null"})


def _tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL: dict[str, Any] = _tool(
    "web_search",
    "Search the web for current legal information: recent decisions, statutory or regulatory "
    "updates, agency notices, or news. Use when the user asks about recent events, or when the "
    "available documents and corpus are not enough. Prefer the exact lookup and search tools "
    "first whenever they apply.",
    {
        "query": {
            "type": "string",
            "description": (
                "The search query. Be specific: include case names, statute numbers, "
                "or relevant legal terms."
            ),
        },
        "jurisdiction": {
            "type": "string",
            "description": (
                "Optional. Jurisdiction to focus the search on, as free text "
                "(for example 'US', 'federal', 'California'). "
                "Match the jurisdiction the user is asking about."
            ),
        },
    },
    ["query"],
)

FETCH_URL_TOOL: dict[str, Any] = _tool(
    "fetch_url",
    "Fetch and read the content of a specific URL, such as a court order PDF, an agency notice, "
    "a regulatory filing, or any public legal document. Use when the user provides a specific "
    "URL or when you need the exact contents of a page.",
    {
        "url": {
            "type": "string",
            "description": "The full URL to fetch (must be http:// or https://).",
        },
    },
    ["url"],
)

SEARCH_DOCUMENTS_TOOL: dict[str, Any] = _tool(
    "search_documents",
    "Search the documents the user has provided for this conversation. Use proactively to find "
    "clauses, provisions, or facts before answering. To confine the search to ONE document, pass "
    "its `document_id`. To read a whole document exhaustively (for example to summarize it or "
    "answer what the entire document says), set `whole_document: true` WITH a `document_id`; "
    "this reads every section rather than a few top matches, so it will not miss parts.",
    {
        "query": {
            "type": "string",
            "description": "What to search for. Be specific. Ignored when whole_document is true.",
        },
        "document_id": {
            "type": "string",
            "description": (
                "Optional. Restrict the search to this single document (an identifier from an "
                "earlier search result or the user's reference). Omit to search all documents."
            ),
        },
        "whole_document": {
            "type": "boolean",
            "description": (
                "Optional. When true (requires document_id), read the ENTIRE document section by "
                "section instead of returning top matches. Use for summaries or "
                "whole-document questions."
            ),
        },
    },
    ["query"],
)

LOOKUP_US_STATUTE_EXACT_TOOL: dict[str, Any] = _tool(
    "lookup_us_statute_exact",
    "Retrieve the EXACT text of a specific USC, CFR, or state-code section by title and section "
    "number. ALWAYS use this tool BEFORE `search_us_statutes` whenever the user has cited a "
    "section explicitly, for example '7 U.S.C. § 8307', '42 USC 1983', '29 C.F.R. § 825.110', "
    "'8 Del. C. § 102', or 'Cal. Civ. Code § 1559'. This is a deterministic lookup, not a "
    "similarity search, so it returns the cited section if it exists.\n\n"
    "Semantic search (`search_us_statutes`) can return adjacent or related sections instead of "
    "the one cited. For an explicit citation, always prefer this exact lookup.\n\n"
    "Returns the section text, citation, title, source URL, and related metadata. If the section "
    "is not found, fall back to `search_us_statutes` or `web_search`.\n\n"
    "Examples:\n"
    "- user: 'give me the full text of 7 U.S.C. § 8307' "
    "-> lookup_us_statute_exact(corpus='usc', title=7, section='8307')\n"
    "- user: 'what does 42 USC 1983 say' "
    "-> lookup_us_statute_exact(corpus='usc', title=42, section='1983')\n"
    "- user: 'show me 29 CFR 825.110' "
    "-> lookup_us_statute_exact(corpus='cfr', title=29, section='825.110')\n"
    "- user: 'what does 8 Del. C. § 102 say' "
    "-> lookup_us_statute_exact(corpus='state', title=8, section='102', state='DE')",
    {
        "corpus": {
            "type": "string",
            "enum": ["usc", "cfr", "state"],
            "description": (
                "'usc' for the United States Code (federal statutes), as in '7 U.S.C. § 8307'. "
                "'cfr' for the Code of Federal Regulations, as in '29 C.F.R. § 825.110'. "
                "'state' for any state code, as in '8 Del. C. § 102' or 'Tex. Bus. & Com. Code "
                "§ 503.001'."
            ),
        },
        "title": {
            "type": "integer",
            "description": (
                "The title number. For USC, 1 to 54 (for example 18 Crimes and Criminal "
                "Procedure, 26 Internal Revenue Code, 42 Public Health and Welfare). For CFR, "
                "1 to 50 (for example 17 Securities, 29 Labor, 40 Environment). For state codes "
                "that use title numbering (Delaware, Texas), the title number. For state codes "
                "organized by name rather than title, use 1."
            ),
        },
        "section": {
            "type": "string",
            "description": (
                "The section identifier as a string. Keep subsection notation if the user gave "
                "it: '8307', '1983', '1962(c)', '825.110', '240.10b-5'. The lookup normalizes "
                "subsection brackets, so both '1962(c)' and '1962' find section 1962."
            ),
        },
        "state": {
            "type": "string",
            "description": (
                "REQUIRED when corpus='state'. The two-letter state code (for example 'DE', "
                "'CA', 'NY', 'TX'). Case-insensitive. Leave OUT entirely for corpus='usc' or "
                "'cfr'."
            ),
        },
    },
    ["corpus", "title", "section"],
)

LOOKUP_US_CASE_BY_CITATION_TOOL: dict[str, Any] = _tool(
    "lookup_us_case_by_citation",
    "Look up a US case by its reporter citation. Use this BEFORE `search_us_case_law` whenever "
    "the user gives a specific reporter cite, for example '373 U.S. 83', '550 U.S. 544', or "
    "'123 F.3d 456'. This is a deterministic exact match on the citation, not a similarity "
    "search.\n\n"
    "`search_us_case_law` ranks by query relevance and may surface cases that cite the requested "
    "case rather than the case itself. For an explicit citation, this exact lookup is the right "
    "tool.\n\n"
    "Returns the case name, court, year, opinion text, judges, the full citation list, and an "
    "`opinion_id`. If the citation is not found, returns an empty result.\n\n"
    "Examples:\n"
    "- user: 'what did Brady v. Maryland, 373 U.S. 83 hold' "
    "-> lookup_us_case_by_citation(citation='373 U.S. 83')\n"
    "- user: 'give me the holding from 550 U.S. 544' "
    "-> lookup_us_case_by_citation(citation='550 U.S. 544')",
    {
        "citation": {
            "type": "string",
            "description": (
                "The reporter citation to look up. Common formats: '373 U.S. 83', '373 US 83', "
                "'550 U.S. 544 (2007)', '123 F.3d 456', '129 N.E.3d 1197'. Spacing, "
                "punctuation, and a parenthetical year are normalized away."
            ),
        },
    },
    ["citation"],
)

VERIFY_US_CITATION_TOOL: dict[str, Any] = _tool(
    "verify_us_citation",
    "DETERMINISTIC citation validity check. Use this BEFORE quoting a US case in your final "
    "answer when you are not certain the cite is real (for example it came from the user's "
    "earlier text or from your own recall). Returns whether the citation is valid, the "
    "normalized citation, the case name, court, year, and an `opinion_id`.\n\n"
    "Key uses:\n"
    "- HALLUCINATION GUARD: confirm a cite exists before you put it in your answer. "
    "If the citation is not valid, do NOT cite it.\n"
    "- NORMALIZATION: convert '550 US 544' to '550 U.S. 544' for the response.\n"
    "- GET opinion_id: needed as input to `find_us_citing_cases` to discover later cases that "
    "interpret this one.\n\n"
    "Distinct from `lookup_us_case_by_citation`, which returns the full opinion text and is "
    "heavy. This tool is a light validity probe with a small response.\n\n"
    "Examples:\n"
    "- 'I am about to cite 550 U.S. 544' -> verify_us_citation(citation='550 U.S. 544')\n"
    "- 'is 999 F.4th 100 real?' -> verify_us_citation(citation='999 F.4th 100')",
    {
        "citation": {
            "type": "string",
            "description": (
                "The reporter citation to verify. Common formats: '550 U.S. 544', '373 US 83'. "
                "Spacing and punctuation are normalized."
            ),
        },
    },
    ["citation"],
)

FIND_US_CITING_CASES_TOOL: dict[str, Any] = _tool(
    "find_us_citing_cases",
    "Forward-citation lookup: given a target opinion's id, return up to 10 LATER cases that cite "
    "it. Each result carries the case name, court, year, citation, a snippet, and a citation "
    "count, ordered by citation count (most cited first). Provide `opinion_id` or "
    "`cluster_id`.\n\n"
    "WHEN TO USE: the user asks what cases interpret X, how courts have applied Y, circuit "
    "treatment of Z, or which cases cite a named decision. This is purpose-built for that "
    "question and far more precise than `search_us_case_law`, which only matches on topic.\n\n"
    "PIPELINE: call `lookup_us_case_by_citation` or `verify_us_citation` first, take the "
    "`opinion_id` from the result, then call this tool with it.\n\n"
    "Example: 'what cases cite Brady v. Maryland' -> "
    "lookup_us_case_by_citation(citation='373 U.S. 83'), then "
    "find_us_citing_cases(opinion_id=<that id>).",
    {
        "opinion_id": {
            "type": "integer",
            "description": (
                "Numeric opinion id (NOT a citation string). Get it from "
                "`lookup_us_case_by_citation` or `verify_us_citation` first."
            ),
        },
        "cluster_id": {
            "type": "integer",
            "description": (
                "Numeric cluster id. Use as an alternative to opinion_id when only the "
                "cluster id is known."
            ),
        },
        "limit": {
            "type": "integer",
            "description": (
                "Max citing cases to return (1 to 20, default 10). Use a higher limit for all "
                "cases interpreting X questions and a lower one for a few examples."
            ),
            "default": 10,
            "minimum": 1,
            "maximum": 20,
        },
    },
    [],
)

SEARCH_US_STATUTES_TOOL: dict[str, Any] = _tool(
    "search_us_statutes",
    "PROBABILISTIC semantic search over US federal statutes (USC) and regulations (CFR). Use ONLY "
    "when the user did NOT cite a specific section. If they cited one ('7 USC § 8307', "
    "'29 CFR 825.110'), call `lookup_us_statute_exact` first, because semantic similarity "
    "routinely returns adjacent sections instead of the literal target.\n\n"
    "Returns the section title, text, citation (for example 42 U.S.C. 1983), a link to the "
    "official source, and a relevance `score`. Low scores signal a weak match; consider "
    "refining the query.\n\n"
    "Examples (open-ended queries, no explicit cite):\n"
    "- 'Section 1983 civil rights' (finds 42 U.S.C. 1983)\n"
    "- 'FCRA adverse action notice requirements' (finds 15 U.S.C. 1681m)\n"
    "- 'SEC Rule 10b-5' (finds 17 C.F.R. 240.10b-5)\n"
    "- 'HIPAA privacy rule' (finds 45 C.F.R. Part 164)",
    {
        "query": {
            "type": "string",
            "description": (
                "Search query. Use legal terms, statute names, or regulatory topics. "
                "Examples: 'Title VII employment discrimination', 'mail fraud', "
                "'Clean Air Act emission standards'."
            ),
        },
    },
    ["query"],
)

SEARCH_US_CASE_LAW_TOOL: dict[str, Any] = _tool(
    "search_us_case_law",
    "PROBABILISTIC semantic search over US federal and state court opinions. Use when the "
    "question involves US case law, constitutional law, circuit decisions, or Supreme Court "
    "rulings and the user did NOT give an explicit reporter citation. If they did, call "
    "`lookup_us_case_by_citation` first (deterministic).\n\n"
    "Results typically include the case name, citation, court, date, judge, an opinion text "
    "excerpt, a link to the full opinion, and a relevance `score`. Low scores signal a weak "
    "match: re-query with different keywords, or if the user cited a case, fall back to "
    "`lookup_us_case_by_citation`.",
    {
        "query": {
            "type": "string",
            "description": (
                "Legal search query. Be specific: include case names, legal doctrines, statute "
                "references, or constitutional amendments. Examples: 'Fourth Amendment "
                "reasonable expectation of privacy', 'Section 1983 qualified immunity police', "
                "'patent claim construction Markman'."
            ),
        },
        "court": {
            "type": "string",
            "description": (
                "Filter by court identifier, comma-separated for several: 'scotus', 'ca9', "
                "'ca2,ca3', 'nysd'. Omit to search across all courts."
            ),
        },
        "filed_after": {
            "type": "string",
            "description": "Cases filed after this date (YYYY-MM-DD), for example '2020-01-01'.",
        },
        "filed_before": {
            "type": "string",
            "description": "Cases filed before this date (YYYY-MM-DD), for example '2025-12-31'.",
        },
    },
    ["query"],
)

RESOLVE_CITATION_TOOL: dict[str, Any] = _tool(
    "resolve_citation",
    "Resolve a legal citation string to the authority it identifies and list its alternative "
    "citation forms. Use when the user mentions a citation such as '347 U.S. 483', "
    "'74 S. Ct. 686', or '42 U.S.C. § 1983' and you need to confirm what it refers to, or find "
    "its parallel citations.\n\n"
    "Returns the title, court, year, the primary citation, all known aliases (for example "
    "'347 U.S. 483' is also '74 S. Ct. 686' and '98 L. Ed. 873'), and how the citation was "
    "resolved. This is a cheap lookup on the citation string only. It does NOT search by topic: "
    "use `search_us_case_law` or `search_us_statutes` for that, and "
    "`lookup_us_case_by_citation` when you need the opinion text.",
    {
        "citation": {
            "type": "string",
            "description": (
                "The citation to resolve. Common formats: '347 U.S. 483', '74 S. Ct. 686', "
                "'98 L. Ed. 873', '42 U.S.C. § 1983'."
            ),
        },
    },
    ["citation"],
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ToolSchemaError(ValueError):
    """A tool schema or spec is malformed."""


def validate_tool_schema(tool: Mapping[str, Any]) -> str:
    """Raise ``ValueError`` unless ``tool`` is a well-formed function tool; return its name."""
    if tool.get("type") != "function":
        raise ValueError("tool 'type' must be 'function'")
    fn = tool.get("function")
    if not isinstance(fn, Mapping):
        raise ToolSchemaError("tool needs a 'function' object")
    name = fn.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid tool name: {name!r}")
    description = fn.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"tool {name!r} needs a non-empty description")
    params = fn.get("parameters")
    if not isinstance(params, Mapping) or params.get("type") != "object":
        raise ValueError(f"tool {name!r} parameters must be a JSON Schema object")
    props = params.get("properties")
    if not isinstance(props, Mapping):
        raise ToolSchemaError(f"tool {name!r} parameters need a 'properties' object")
    for prop_name, schema in props.items():
        if not isinstance(schema, Mapping) or schema.get("type") not in _JSON_TYPES:
            raise ValueError(f"tool {name!r} property {prop_name!r} needs a JSON Schema 'type'")
    required = params.get("required", [])
    if not isinstance(required, list) or not set(required) <= set(props):
        raise ValueError(f"tool {name!r} 'required' must be a subset of its properties")
    return name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolInfo:
    """Selection data kept beside a tool schema."""

    name: str
    tier: str = DEFAULT_TIER
    capability: str | None = None
    jurisdictions: frozenset[str] | None = None


def _norm_jurisdictions(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(v.strip().upper() for v in values)


class ToolRegistry:
    """Named tool schemas with their tier, capability and jurisdiction data.

    Registration order is the order within a tier.
    Schemas are copied on the way in and on the way out, so callers cannot mutate the registry.
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._info: dict[str, ToolInfo] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def register(
        self,
        tool: Mapping[str, Any],
        *,
        tier: str = DEFAULT_TIER,
        capability: str | None = None,
        jurisdictions: Iterable[str] | None = None,
        replace: bool = False,
    ) -> ToolInfo:
        """Validate and add a tool. A duplicate name raises unless ``replace`` is true."""
        name = validate_tool_schema(tool)
        if tier not in TIER_ORDER:
            raise ValueError(f"unknown tier {tier!r}; expected one of {TIER_ORDER}")
        if capability is not None and capability not in KNOWN_CAPABILITIES:
            raise ValueError(f"unknown capability {capability!r}")
        if name in self._tools and not replace:
            raise ValueError(f"tool {name!r} is already registered")
        info = ToolInfo(name, tier, capability, _norm_jurisdictions(jurisdictions))
        self._tools[name] = copy.deepcopy(dict(tool))
        self._info[name] = info
        return info

    def get(self, name: str) -> dict[str, Any]:
        """Return a copy of the schema for ``name``. Raises ``KeyError`` if unknown."""
        return copy.deepcopy(self._tools[name])

    def info(self, name: str) -> ToolInfo:
        return self._info[name]

    def names(self) -> list[str]:
        """Tool names in registration order."""
        return list(self._tools)

    def filter(
        self,
        predicate: Callable[[ToolInfo], bool] | None = None,
        *,
        names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return schema copies in registration order, optionally narrowed by name or predicate."""
        wanted = None if names is None else set(names)
        if wanted is not None and not wanted <= set(self._tools):
            raise KeyError(f"unknown tools: {sorted(wanted - set(self._tools))}")
        return [
            self.get(n)
            for n, info in self._info.items()
            if (wanted is None or n in wanted) and (predicate is None or predicate(info))
        ]

    def copy(self) -> ToolRegistry:
        other = ToolRegistry()
        other._tools = copy.deepcopy(self._tools)
        other._info = dict(self._info)
        return other

    def register_specs(
        self, specs: Iterable[DynamicToolSpec | Mapping[str, Any]], *, name_prefix: str = ""
    ) -> list[str]:
        """Build tools from caller-supplied specs and register each with its own selection data."""
        normalized = [_coerce_spec(s) for s in specs]
        tools = [tool_from_spec(s, name_prefix=name_prefix) for s in normalized]
        _reject_duplicate_names(tools)
        for spec, tool in zip(normalized, tools, strict=True):
            self.register(
                tool,
                tier=spec.tier,
                capability=spec.capability,
                jurisdictions=spec.jurisdictions,
            )
        return [t["function"]["name"] for t in tools]


# ---------------------------------------------------------------------------
# Dynamic tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicToolSpec:
    """A caller-supplied tool: a name, a description and a set of parameters.

    ``parameters`` maps a parameter name to its JSON Schema (each needs a ``type``).
    ``tier``, ``capability`` and ``jurisdictions`` decide where and when the tool is offered.
    """

    name: str
    description: str
    parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    tier: str = DEFAULT_TIER
    capability: str | None = None
    jurisdictions: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "jurisdictions", _norm_jurisdictions(self.jurisdictions))
        if self.tier not in TIER_ORDER:
            raise ValueError(f"unknown tier {self.tier!r}; expected one of {TIER_ORDER}")
        if self.capability is not None and self.capability not in KNOWN_CAPABILITIES:
            raise ValueError(f"unknown capability {self.capability!r}")


_SPEC_KEYS = frozenset(DynamicToolSpec.__dataclass_fields__)


def _coerce_spec(spec: DynamicToolSpec | Mapping[str, Any]) -> DynamicToolSpec:
    if isinstance(spec, DynamicToolSpec):
        return spec
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown tool spec keys: {sorted(unknown)}")
    return DynamicToolSpec(**spec)


def tool_from_spec(spec: DynamicToolSpec, *, name_prefix: str = "") -> dict[str, Any]:
    """Build one validated tool schema from a spec."""
    tool = _tool(
        name_prefix + spec.name,
        spec.description,
        {k: copy.deepcopy(dict(v)) for k, v in spec.parameters.items()},
        spec.required,
    )
    validate_tool_schema(tool)
    return tool


def spec_from_tool(tool: Mapping[str, Any]) -> DynamicToolSpec:
    """Inverse of :func:`tool_from_spec` for the schema fields (selection data is not stored)."""
    validate_tool_schema(tool)
    fn = tool["function"]
    params = fn["parameters"]
    return DynamicToolSpec(
        name=fn["name"],
        description=fn["description"],
        parameters=copy.deepcopy(dict(params["properties"])),
        required=tuple(params.get("required", [])),
    )


def _reject_duplicate_names(tools: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for tool in tools:
        name = tool["function"]["name"]
        if name in seen:
            raise ValueError(f"duplicate tool name {name!r}")
        seen.add(name)


def build_dynamic_tools(
    specs: Iterable[DynamicToolSpec | Mapping[str, Any]], *, name_prefix: str = ""
) -> list[dict[str, Any]]:
    """Turn caller-supplied specs into tool schemas.

    A spec is a :class:`DynamicToolSpec` or a mapping with the same keys.
    Unknown keys, bad names, duplicate names and a ``required`` entry that is not a
    parameter all raise ``ValueError``, because a silently dropped field is a wrong tool.
    """
    tools = [tool_from_spec(_coerce_spec(s), name_prefix=name_prefix) for s in specs]
    _reject_duplicate_names(tools)
    return tools


# ---------------------------------------------------------------------------
# Default registry and request-time selection
# ---------------------------------------------------------------------------

_US = ("US",)
_DEFAULTS: tuple[tuple[dict[str, Any], str, str | None, tuple[str, ...] | None], ...] = (
    # (schema, tier, capability, jurisdictions); this order is the order within each tier.
    (LOOKUP_US_STATUTE_EXACT_TOOL, "exact", CAP_SHARED_CORPUS, _US),
    (LOOKUP_US_CASE_BY_CITATION_TOOL, "exact", CAP_SHARED_CORPUS, _US),
    (VERIFY_US_CITATION_TOOL, "exact", CAP_SHARED_CORPUS, _US),
    (FIND_US_CITING_CASES_TOOL, "exact", CAP_SHARED_CORPUS, _US),
    (RESOLVE_CITATION_TOOL, "exact", CAP_SHARED_CORPUS, _US),
    (SEARCH_US_STATUTES_TOOL, "semantic", CAP_SHARED_CORPUS, _US),
    (SEARCH_US_CASE_LAW_TOOL, "semantic", CAP_SHARED_CORPUS, _US),
    (SEARCH_DOCUMENTS_TOOL, "semantic", CAP_DOCUMENTS, None),
    (FETCH_URL_TOOL, "fetch", CAP_FETCH_URL, None),
    (WEB_SEARCH_TOOL, "web", CAP_WEB_SEARCH, None),
)

DEFAULT_TOOL_NAMES: tuple[str, ...] = tuple(s["function"]["name"] for s, *_ in _DEFAULTS)
"""The built-in tools in listing order. It satisfies :data:`TIER_ORDER` by construction."""


def default_registry() -> ToolRegistry:
    """A fresh registry holding the built-in tools."""
    registry = ToolRegistry()
    for schema, tier, capability, jurisdictions in _DEFAULTS:
        registry.register(schema, tier=tier, capability=capability, jurisdictions=jurisdictions)
    return registry


def tools_for_request(
    *,
    registry: ToolRegistry | None = None,
    enable_web_search: bool = True,
    enable_fetch_url: bool = True,
    enable_document_search: bool = True,
    enable_shared_corpus: bool = True,
    jurisdiction: str | None = None,
    extra_specs: Sequence[DynamicToolSpec | Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Select the tools to offer the model for one request.

    A tool is dropped when its capability flag is off, or when it is limited to certain
    jurisdictions and the request's ``jurisdiction`` (case-insensitive) is not one of them.
    A request with no jurisdiction therefore gets no jurisdiction-limited tools.
    ``extra_specs`` adds caller-supplied tools for this request only.
    The result is sorted by :data:`TIER_ORDER`, and by registration order within a tier.
    """
    reg = registry if registry is not None else default_registry()
    if extra_specs:
        reg = reg.copy()
        reg.register_specs(extra_specs)
    flags = {
        CAP_WEB_SEARCH: enable_web_search,
        CAP_FETCH_URL: enable_fetch_url,
        CAP_DOCUMENTS: enable_document_search,
        CAP_SHARED_CORPUS: enable_shared_corpus,
    }
    requested = jurisdiction.strip().upper() if jurisdiction else None

    def allowed(info: ToolInfo) -> bool:
        if info.capability is not None and not flags[info.capability]:
            return False
        return info.jurisdictions is None or requested in info.jurisdictions

    rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    selected = reg.filter(allowed)
    selected.sort(key=lambda t: rank[reg.info(t["function"]["name"]).tier])
    return selected


# ---------------------------------------------------------------------------
# Executor contract
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolExecutor(Protocol):
    """Runs a tool the model asked for and returns its result as text for the model.

    The caller owns execution: authorization, tenancy filters, rate limits and error text.
    Return a readable error string rather than raising for expected failures, so the model
    can recover on its next turn.
    """

    async def call(self, name: str, arguments: Mapping[str, Any]) -> str: ...
