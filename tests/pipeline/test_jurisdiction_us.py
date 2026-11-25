"""US jurisdiction plug-in: matching, adjusters, exact lookups, labels, registry, hygiene."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from legal_rag.pipeline import jurisdictions
from legal_rag.pipeline.jurisdictions import us as us_module
from legal_rag.pipeline.jurisdictions.us import (
    CURRENCY_FIELD,
    JURISDICTION_TAG_KEY,
    NAMED_KEY,
    NAMED_PIN_SCORE,
    PIN_KEY,
    PRE_ADJUST_SCORE_KEY,
    NamedCaseLookup,
    NamedStatuteLookup,
    USJurisdiction,
    apply_court_prestige,
    collapse_alias_duplicates,
    downweight_unknown_currency,
    extract_case_candidates,
    extract_statute_citations,
    merge_pinned_chunks,
)
from legal_rag.pipeline.protocols import ExactLookup, JurisdictionPlugin
from legal_rag.retrieval.routing.jurisdiction_classifier import Classification
from legal_rag.types import RetrievedChunk
from legal_rag.verification.resolver import ResolvedCase

# ---------------------------------------------------------------------------
# Fakes and synthetic data
# ---------------------------------------------------------------------------

MARBURY = ResolvedCase(
    case_name="Marbury v. Madison",
    citation="5 U.S. 137",
    court="Supreme Court of the United States",
    year=1803,
)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None, error: Exception | None = None):
        self.chunks = list(chunks or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Any = None,
        sub_queries: Any = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "sub_queries": sub_queries}
        )
        if self.error is not None:
            raise self.error
        return [RetrievedChunk(c.id, c.text, c.score, dict(c.metadata)) for c in self.chunks]


class FakeResolver:
    def __init__(self, table: dict[str, ResolvedCase], error: Exception | None = None):
        self.table = table
        self.error = error
        self.queries: list[str] = []

    async def resolve(self, query: str) -> ResolvedCase | None:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        for needle, case in self.table.items():
            if needle in query:
                return case
        return None


def chunk(chunk_id: str, text: str = "text", score: float = 0.5, **metadata: Any) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, text=text, score=score, metadata=dict(metadata))


def by_id(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {c.id: c for c in chunks}


# ---------------------------------------------------------------------------
# Protocol conformance and matching
# ---------------------------------------------------------------------------


def test_satisfies_the_jurisdiction_plugin_protocol():
    plugin = USJurisdiction()
    assert isinstance(plugin, JurisdictionPlugin)
    assert plugin.condenser_key == "US"
    assert all(callable(a) for a in plugin.adjusters())


@pytest.mark.parametrize(
    "filters",
    [
        {"country_code": "US"},
        {"country_code": "us"},
        {"jurisdiction": "United States"},
        {"jurisdiction": " USA "},
        {"country_code": "GB", "jurisdiction": "US"},
    ],
)
def test_matches_us_filters(filters: dict[str, Any]):
    assert USJurisdiction().matches(filters) is True


@pytest.mark.parametrize(
    "filters",
    [{}, {"country_code": "GB"}, {"jurisdiction": "Canada"}, {"jurisdiction_scope": ["CA"]}],
)
def test_does_not_match_other_or_missing_jurisdictions(filters: dict[str, Any]):
    assert USJurisdiction().matches(filters) is False


def test_recency_gap_uses_the_corpus_cutoff():
    assert USJurisdiction().recency_gap("give me 2026 judgments") == (True, 2026)
    assert USJurisdiction(corpus_cutoff_year=2030).recency_gap("give me 2026 judgments")[0] is False


# ---------------------------------------------------------------------------
# Adjusters
# ---------------------------------------------------------------------------


def test_default_adjusters_are_currency_then_alias_collapse_and_prestige_is_opt_in():
    assert len(USJurisdiction().adjusters()) == 2
    assert len(USJurisdiction(unknown_currency_penalty=None).adjusters()) == 1
    assert len(USJurisdiction(court_prestige=True).adjusters()) == 3
    with pytest.raises(ValueError):
        USJurisdiction(unknown_currency_penalty=0.0)


def test_currency_downweight_demotes_unknown_below_vouched_law_of_another_family():
    pool = [
        chunk("good", score=0.80, corpus_family="usc", **{CURRENCY_FIELD: "current"}),
        chunk("manual", score=0.90, corpus_family="agency_manual", **{CURRENCY_FIELD: "unknown"}),
        chunk("plain", score=0.70),
    ]
    out = downweight_unknown_currency(pool, 0.8)
    assert [c.id for c in out] == ["good", "manual", "plain"]
    manual = by_id(out)["manual"]
    assert manual.score == pytest.approx(0.72)
    assert manual.metadata[PRE_ADJUST_SCORE_KEY] == pytest.approx(0.90)
    assert by_id(out)["good"].score == pytest.approx(0.80)
    assert by_id(out)["plain"].score == pytest.approx(0.70)
    # The input chunk is not mutated.
    assert pool[1].score == pytest.approx(0.90)
    assert PRE_ADJUST_SCORE_KEY not in pool[1].metadata


def test_currency_downweight_never_fires_without_a_vouched_chunk_of_another_family():
    only_unknown = [chunk("a", score=0.9, **{CURRENCY_FIELD: "unknown"})]
    same_family = [
        chunk("a", score=0.9, corpus_family="usc", **{CURRENCY_FIELD: "unknown"}),
        chunk("b", score=0.5, corpus_family="usc", **{CURRENCY_FIELD: "current"}),
    ]
    dead = [
        chunk("a", score=0.9, corpus_family="x", **{CURRENCY_FIELD: "repealed"}),
        chunk("b", score=0.5, corpus_family="y", **{CURRENCY_FIELD: "current"}),
    ]
    for pool in (only_unknown, same_family, dead):
        out = downweight_unknown_currency(pool, 0.5)
        assert [(c.id, c.score) for c in out] == [(c.id, c.score) for c in pool]


def test_unmeasured_chunk_is_not_penalized_and_counts_as_vouched():
    pool = [
        chunk("state", score=0.7, corpus_family="state_code", **{CURRENCY_FIELD: "unmeasured"}),
        chunk("session", score=0.9, corpus_family="session_law", **{CURRENCY_FIELD: "unknown"}),
    ]
    out = by_id(downweight_unknown_currency(pool, 0.5))
    assert out["state"].score == pytest.approx(0.7)
    assert out["session"].score == pytest.approx(0.45)


def test_currency_adjuster_from_the_plugin_reorders_the_pool():
    pool = [
        chunk("manual", score=0.9, corpus_family="agency_manual", **{CURRENCY_FIELD: "unknown"}),
        chunk("good", score=0.8, corpus_family="usc", **{CURRENCY_FIELD: "current"}),
    ]
    currency = USJurisdiction(unknown_currency_penalty=0.5).adjusters()[0]
    out = currency("q", pool, {})
    assert [c.id for c in out] == ["good", "manual"]


def test_alias_collapse_keeps_one_of_each_alias_group_and_merges_pin_flags():
    text_a = "No person shall  possess the substance.\n"
    pool = [
        chunk("a1", text_a, 0.50, citation="Miss. Code Ann. 41-29-168", **{PIN_KEY: True}),
        chunk("x1", "Different section text.", 0.60, citation="Miss. Code Ann. 41-29-170"),
        chunk(
            "a2",
            "No person shall possess the substance.",
            0.90,
            citation="miss. code ann. 41-29-168",
        ),
        chunk("a3", text_a, 0.70, citation="Miss. Code Ann. 41-29-168"),
        chunk("b1", "Second group body.", 0.40, citation="5 U.S.C. 552"),
        chunk("b2", "Second group body.", 0.30, citation="5 U.S.C. 552"),
    ]
    out, dropped = collapse_alias_duplicates(pool)
    assert dropped == 3
    assert [c.id for c in out] == ["a2", "x1", "b1"]
    survivor = out[0]
    assert survivor.score == pytest.approx(0.90)
    assert survivor.metadata[PIN_KEY] is True
    assert pool[2].metadata.get(PIN_KEY) is None


def test_alias_collapse_never_merges_uncited_or_different_text():
    pool = [
        chunk("d1", "Boilerplate clause.", 0.5),
        chunk("d2", "Boilerplate clause.", 0.4),
        chunk("s1", "Part one of the section.", 0.6, citation="42 U.S.C. 1983"),
        chunk("s2", "Part two of the section.", 0.5, citation="42 U.S.C. 1983"),
    ]
    out, dropped = collapse_alias_duplicates(pool)
    assert dropped == 0
    assert [c.id for c in out] == ["d1", "d2", "s1", "s2"]


def test_alias_collapse_adjuster_from_the_plugin():
    plugin = USJurisdiction(unknown_currency_penalty=None)
    (collapse,) = plugin.adjusters()
    pool = [
        chunk("a", "Same body.", 0.5, citation="N.Y. Penal Law 10.00"),
        chunk("b", "Same body.", 0.6, citation="N.Y. Penal Law 10.00"),
    ]
    assert [c.id for c in collapse("q", pool, {})] == ["b"]


def test_court_prestige_boosts_by_question_jurisdiction_and_spares_pins_and_untagged():
    federal = {JURISDICTION_TAG_KEY: "federal"}
    pool = [
        chunk("lower", score=0.90, court_id="not-a-court", **federal),
        chunk("scotus", score=0.40, court_id="scotus", **federal),
        chunk("pinned", score=0.20, court_id="scotus", **{PIN_KEY: True}, **federal),
        chunk("doc", score=0.10),
    ]
    out = by_id(apply_court_prestige(pool))
    assert out["scotus"].score == pytest.approx(1.2)
    assert out["lower"].score == pytest.approx(0.72)
    assert out["pinned"].score == pytest.approx(0.20)
    assert out["doc"].score == pytest.approx(0.10)
    assert next(c.id for c in apply_court_prestige(pool)) == "scotus"

    state_pool = [chunk("scotus", score=0.40, court_id="scotus", **{JURISDICTION_TAG_KEY: "state"})]
    assert apply_court_prestige(state_pool)[0].score == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Exact lookups
# ---------------------------------------------------------------------------


def test_extraction_finds_citations_captions_and_statutes():
    query = "Does Marbury v. Madison, 5 U.S. 137 (1803) matter under 42 U.S.C. 1983?"
    assert extract_case_candidates(query) == ["5 U.S. 137", "Marbury v. Madison"]
    assert extract_statute_citations(query) == ["42 U.S.C. 1983"]
    assert extract_statute_citations("See 42 U.S.C. § 1983.") == ["42 U.S.C. § 1983"]
    both = extract_statute_citations("Compare 7 C.F.R. 205.1 with Nev. Rev. Stat. § 587.877.")
    assert both == ["7 C.F.R. 205.1", "Nev. Rev. Stat. § 587.877"]
    assert extract_case_candidates("What is the standard for summary judgment?") == []
    assert extract_statute_citations("What is the standard for summary judgment?") == []


async def test_named_case_lookup_pins_the_named_case():
    resolver = FakeResolver({"5 U.S. 137": MARBURY, "Marbury v. Madison": MARBURY})
    lookup = NamedCaseLookup(resolver)
    assert isinstance(lookup, ExactLookup)
    chunks = await lookup.lookup("Does Marbury v. Madison, 5 U.S. 137 (1803) establish review?")
    assert resolver.queries == ["5 U.S. 137", "Marbury v. Madison"]
    assert len(chunks) == 1
    pinned = chunks[0]
    assert pinned.score == NAMED_PIN_SCORE
    assert pinned.metadata[PIN_KEY] is True
    assert pinned.metadata[NAMED_KEY] is True
    assert pinned.metadata["citation"] == "5 U.S. 137"
    assert pinned.metadata["court"] == "Supreme Court of the United States"
    assert "Marbury v. Madison" in pinned.text
    assert "verify the holding" in pinned.text


async def test_named_case_lookup_pins_only_the_first_loaded_chunk():
    async def loader(case: ResolvedCase) -> list[RetrievedChunk]:
        return [chunk(f"op:{i}", f"Opinion part {i}", 0.3, document_id="op-1") for i in range(3)]

    lookup = NamedCaseLookup(FakeResolver({"5 U.S. 137": MARBURY}), text_loader=loader)
    chunks = await lookup.lookup("What did Marbury v. Madison, 5 U.S. 137 (1803) hold?")
    assert [c.id for c in chunks] == ["op:0", "op:1", "op:2"]
    assert [c.metadata.get(PIN_KEY, False) for c in chunks] == [True, False, False]
    assert all(c.metadata[NAMED_KEY] and c.score == NAMED_PIN_SCORE for c in chunks)
    assert chunks[0].metadata["title"] == "Marbury v. Madison"


async def test_named_case_lookup_is_fail_soft_and_silent_without_citations():
    unresolved = FakeResolver({})
    assert await NamedCaseLookup(unresolved).lookup("Explain 5 U.S. 137") == []
    untouched = FakeResolver({"x": MARBURY})
    assert await NamedCaseLookup(untouched).lookup("What is promissory estoppel?") == []
    assert untouched.queries == []
    broken = FakeResolver({}, error=RuntimeError("upstream down"))
    assert await NamedCaseLookup(broken).lookup("Marbury v. Madison, 5 U.S. 137") == []


async def test_named_statute_lookup_pins_the_named_section():
    seen: list[str] = []

    async def section_lookup(citation: str) -> list[RetrievedChunk]:
        seen.append(citation)
        if citation == "42 U.S.C. 1983":
            return [
                chunk("usc-42-1983-0", "Every person who, under color of law...", 0.2),
                chunk("usc-42-1983-1", "Second chunk.", 0.2),
            ]
        return []

    lookup = NamedStatuteLookup(section_lookup)
    assert isinstance(lookup, ExactLookup)
    out = await lookup.lookup("Can I sue under 42 U.S.C. 1983 for a First Amendment claim?")
    assert seen == ["42 U.S.C. 1983"]
    assert [c.id for c in out] == ["usc-42-1983-0", "usc-42-1983-1"]
    assert out[0].metadata[PIN_KEY] is True
    assert PIN_KEY not in out[1].metadata
    assert all(c.metadata[NAMED_KEY] and c.score == NAMED_PIN_SCORE for c in out)
    assert out[0].metadata["__named_citation_raw"] == "42 U.S.C. 1983"
    assert await lookup.lookup("How do I draft a lease?") == []


async def test_named_statute_lookup_skips_a_failing_citation():
    async def section_lookup(citation: str) -> list[RetrievedChunk]:
        if "C.F.R." in citation:
            raise RuntimeError("store unavailable")
        return [chunk("sec", "Body.", 0.1)]

    out = await NamedStatuteLookup(section_lookup).lookup(
        "Compare 7 C.F.R. 205.1 and 15 U.S.C. 78j"
    )
    assert [c.id for c in out] == ["sec"]


def test_exact_lookups_follow_the_injected_backends():
    async def section_lookup(citation: str) -> list[RetrievedChunk]:
        return []

    assert USJurisdiction().exact_lookups() == []
    full = USJurisdiction(
        case_resolver=FakeResolver({}), statute_section_lookup=section_lookup
    ).exact_lookups()
    assert [type(x) for x in full] == [NamedCaseLookup, NamedStatuteLookup]


def test_merge_pinned_chunks_upgrades_existing_copies_and_prepends_new_ones():
    found = chunk("hit-1", "Section text.", 0.4, document_id="act-1")
    found_b = chunk("hit-2", "More text.", 0.3, document_id="act-1")
    other = chunk("hit-3", "Other.", 0.2, document_id="act-2")
    pinned = [
        chunk("pin-1", "Section text.", NAMED_PIN_SCORE, document_id="act-1", **{PIN_KEY: True}),
        chunk("pin-2", "New section.", NAMED_PIN_SCORE, document_id="act-9", **{PIN_KEY: True}),
    ]
    merged = merge_pinned_chunks([found, found_b, other], pinned)
    assert [c.id for c in merged] == ["pin-2", "hit-1", "hit-2", "hit-3"]
    by = by_id(merged)
    assert by["hit-1"].metadata[PIN_KEY] is True and by["hit-1"].metadata[NAMED_KEY] is True
    assert PIN_KEY not in by["hit-2"].metadata and by["hit-2"].metadata[NAMED_KEY] is True
    assert PIN_KEY not in by["hit-3"].metadata
    assert PIN_KEY not in found.metadata


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def test_retrieve_runs_both_sources_tags_and_dedupes():
    cases = FakeRetriever([chunk("c1", score=0.9, source_type="case_law"), chunk("dup", score=0.5)])
    statutes = FakeRetriever([chunk("s1", score=0.8), chunk("dup", score=0.4)])
    plugin = USJurisdiction(case_retriever=cases, statute_retriever=statutes)
    filters = {"country_code": "US", "corpus_types": ["state_statute"], "sources_filter": ""}
    out = await plugin.retrieve(
        "how does adverse possession work",
        top_k=5,
        filters=filters,
        sub_queries=["elements of reliance", " ", "damages remedy"],
    )
    assert [c.id for c in out] == ["c1", "dup", "s1"]
    assert all(JURISDICTION_TAG_KEY in c.metadata for c in out)
    assert cases.calls[0]["sub_queries"] is None
    assert statutes.calls[0]["sub_queries"] == ["elements of reliance", "damages remedy"]
    assert cases.calls[0]["filters"]["corpus_types"] == ["state_statute"]
    assert cases.calls[0]["top_k"] == 5


async def test_retrieve_honors_sources_filter_and_needs_two_sub_queries():
    cases = FakeRetriever([chunk("c1")])
    statutes = FakeRetriever([chunk("s1")])
    plugin = USJurisdiction(case_retriever=cases, statute_retriever=statutes)
    await plugin.retrieve("negligence", top_k=3, filters={"sources_filter": "cases_only"})
    assert (len(cases.calls), len(statutes.calls)) == (1, 0)
    await plugin.retrieve(
        "negligence", top_k=3, filters={"sources_filter": "statutes_only"}, sub_queries=["one"]
    )
    assert (len(cases.calls), len(statutes.calls)) == (1, 1)
    assert statutes.calls[0]["sub_queries"] is None
    assert await plugin.retrieve("negligence", top_k=0) == []
    assert await plugin.retrieve("   ", top_k=3) == []


async def test_retrieve_survives_one_failing_source_but_raises_when_all_fail():
    good = FakeRetriever([chunk("s1")])
    bad = FakeRetriever(error=RuntimeError("case source down"))
    out = await USJurisdiction(case_retriever=bad, statute_retriever=good).retrieve(
        "negligence", top_k=3
    )
    assert [c.id for c in out] == ["s1"]
    with pytest.raises(RuntimeError, match="case source down"):
        await USJurisdiction(case_retriever=bad).retrieve("negligence", top_k=3)


async def test_retrieve_pins_the_canonical_authority_for_a_named_doctrine():
    plugin = USJurisdiction(case_retriever=FakeRetriever([chunk("c1", score=0.4)]))
    out = await plugin.retrieve("What is the Brady rule?", top_k=5, filters={"country_code": "US"})
    pins = [c for c in out if c.metadata.get(PIN_KEY)]
    assert pins and out[0].metadata.get(PIN_KEY)
    assert any("Brady v. Maryland" in c.text for c in pins)
    assert all(c.metadata[JURISDICTION_TAG_KEY] == "federal" for c in out)
    assert out[-1].id == "c1"


async def test_state_scope_keeps_only_in_state_canonicals():
    query = "Explain the business judgment rule in Delaware"
    plugin = USJurisdiction(case_retriever=FakeRetriever())
    unscoped = await plugin.retrieve(query, top_k=5)
    assert unscoped and all(c.metadata[JURISDICTION_TAG_KEY] == "state" for c in unscoped)
    in_state = await plugin.retrieve(query, top_k=5, filters={"jurisdiction_scope": ["de"]})
    assert len(in_state) == len(unscoped)
    other_state = await plugin.retrieve(query, top_k=5, filters={"jurisdiction_scope": "TX"})
    assert all("delaware" not in str(c.metadata.get("court", "")).lower() for c in other_state)
    assert len(other_state) < len(unscoped)


async def test_state_scope_swaps_federal_canonicals_for_the_state_equivalent(monkeypatch):
    async def state_verdict(query: str, **_: Any) -> Classification:
        return Classification("state", 0.9, "test")

    monkeypatch.setattr(us_module, "classify_jurisdiction", state_verdict)
    plugin = USJurisdiction(case_retriever=FakeRetriever([chunk("c1")]))
    out = await plugin.retrieve(
        "What is the summary judgment standard?",
        top_k=5,
        filters={"jurisdiction_scope": ["LA"]},
    )
    assert out[0].metadata["is_state_equivalent"] is True
    assert out[0].metadata["citation"] == "La. C.C.P. art. 966"
    assert not any("Celotex" in c.text for c in out)
    assert out[-1].id == "c1"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "source_type": "case_law",
                "title": "Marbury v. Madison",
                "citation": "5 U.S. 137",
                "year": 1803,
            },
            "Marbury v. Madison, 5 U.S. 137 (1803)",
        ),
        (
            {
                "court": "Supreme Court",
                "title": "Marbury v. Madison, 5 U.S. 137 (1803)",
                "citation": "5 U.S. 137",
                "year": 1803,
            },
            "Marbury v. Madison, 5 U.S. 137 (1803)",
        ),
        (
            {"source_type": "case_law", "case_name": "Roe v. Wade", "citations": ["410 U.S. 113"]},
            "Roe v. Wade, 410 U.S. 113",
        ),
        (
            {
                "source_type": "statute",
                "citation": "42 U.S.C. 1983",
                "title": "Civil action for deprivation of rights",
            },
            "42 U.S.C. 1983: Civil action for deprivation of rights",
        ),
        ({"source_type": "statute", "citation": "42 U.S.C. 1983"}, "42 U.S.C. 1983"),
        (
            {
                "source_type": "statute",
                "title": "42 U.S.C. 1983 Civil action",
                "citation": "42 U.S.C. 1983",
            },
            "42 U.S.C. 1983 Civil action",
        ),
        ({"filename": "lease.pdf"}, "lease.pdf"),
        ({}, "chunk-id"),
    ],
)
def test_label_for(metadata: dict[str, Any], expected: str):
    assert USJurisdiction().label_for(chunk("chunk-id", **metadata)) == expected


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_returns_the_us_plugin():
    assert "us" in jurisdictions.available()
    plugin = jurisdictions.get("us")
    assert isinstance(plugin, USJurisdiction)
    assert isinstance(plugin, JurisdictionPlugin)
    assert isinstance(jurisdictions.get(" US "), USJurisdiction)
    cases = FakeRetriever()
    assert jurisdictions.get("us", case_retriever=cases)._case_retriever is cases


def test_registry_rejects_an_unknown_name():
    with pytest.raises(jurisdictions.UnknownJurisdictionError, match="unknown jurisdiction 'xx'"):
        jurisdictions.get("xx")
    with pytest.raises(KeyError):
        jurisdictions.get("xx")
    with pytest.raises(ValueError):
        jurisdictions.get("  ")


def test_registry_register_replace_and_unregister():
    assert jurisdictions.available() == tuple(sorted(jurisdictions.available()))
    jurisdictions.register("Demo", lambda **kw: USJurisdiction(**kw))
    try:
        assert "demo" in jurisdictions.available()
        assert isinstance(jurisdictions.get("demo"), USJurisdiction)
        with pytest.raises(ValueError, match="already registered"):
            jurisdictions.register("demo", lambda **kw: USJurisdiction(**kw))
        jurisdictions.register("demo", lambda **kw: USJurisdiction(**kw), replace=True)
        with pytest.raises(TypeError):
            jurisdictions.register("bad", "not callable")  # type: ignore[arg-type]
    finally:
        jurisdictions.unregister("demo")
    assert "demo" not in jurisdictions.available()
    with pytest.raises(jurisdictions.UnknownJurisdictionError):
        jurisdictions.unregister("demo")


def test_registry_rejects_a_factory_that_returns_a_non_plugin():
    jurisdictions.register("broken", lambda **kw: object())  # type: ignore[arg-type,return-value]
    try:
        with pytest.raises(TypeError, match="did not return a JurisdictionPlugin"):
            jurisdictions.get("broken")
    finally:
        jurisdictions.unregister("broken")


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------

JURISDICTIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "legal_rag" / "pipeline" / "jurisdictions"
)

FORBIDDEN = {
    "vaquill": re.compile(r"vaquill", re.IGNORECASE),
    "supabase": re.compile(r"supabase", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "courtlistener": re.compile(r"courtlistener", re.IGNORECASE),
    "organization_id": re.compile(r"organization_id"),
    "user_id": re.compile(r"user_id"),
    "matter_id": re.compile(r"matter_id"),
    "private app import": re.compile(r"(?<![A-Za-z_])app\."),
    "os.environ": re.compile(r"os\.environ|getenv"),
    "settings": re.compile(r"settings", re.IGNORECASE),
    "gpt-": re.compile(r"gpt-", re.IGNORECASE),
    "india": re.compile(r"india", re.IGNORECASE),
    "voyage": re.compile(r"voyage", re.IGNORECASE),
    "em or en dash": re.compile("—|–"),
}


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
@pytest.mark.parametrize("name", ["us.py", "__init__.py"])
def test_plugin_files_have_no_forbidden_text(name: str, label: str):
    path = JURISDICTIONS_DIR / name
    hits = [
        f"{name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN[label].search(line)
    ]
    assert not hits, f"{label} found:\n" + "\n".join(hits)
