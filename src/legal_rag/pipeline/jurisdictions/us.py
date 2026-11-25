"""US jurisdiction plug-in: opt-in retrieval behavior for United States primary law.

The plug-in is wiring around the routing library in ``legal_rag.retrieval.routing``.
It owns no client code: the case-law source, the statute source, the exact lookups and the
opinion text loader are all injected, so any backend that satisfies the protocols works.

Filter keys it reads (all optional, all opaque to the rest of the pipeline):

- ``country_code`` or ``jurisdiction``: ``matches`` is true when either names the US.
- ``jurisdiction_scope``: a state code or a list of state codes the matter is scoped to.
- ``corpus_types``: passed through to the injected retrievers untouched.
- ``sources_filter``: ``"cases_only"`` skips the statute source, ``"statutes_only"`` skips the
  case-law source.

Metadata keys it reads or writes are module constants so a rerank stage can share them.
Tenant isolation is the caller's job: bake it into the injected retrievers and lookups.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from importlib import resources
from typing import Any

from legal_rag.pipeline._log import get_logger
from legal_rag.pipeline.protocols import Cache, ChunkAdjuster, ExactLookup, LanguageModel, Retriever
from legal_rag.pipeline.types import PINNED_KEY
from legal_rag.retrieval.routing.court_prestige import boost_factor, classify_court
from legal_rag.retrieval.routing.doctrine_detector import (
    CaseResolver as OpinionResolver,
)
from legal_rag.retrieval.routing.doctrine_detector import (
    DoctrineMatch,
    build_canonical_chunks,
    detect_doctrine,
    detect_doctrines_multi,
    enrich_canonical_chunks_with_opinions,
)
from legal_rag.retrieval.routing.jurisdiction_classifier import classify_jurisdiction
from legal_rag.retrieval.routing.query_classification import detect_temporal_recency_gap
from legal_rag.retrieval.routing.state_doctrine_equivalents import build_state_equivalent_chunks
from legal_rag.retrieval.routing.us_citation_patterns import (
    US_CASE_PATTERN,
    US_REPORTER_CITATION_PATTERN,
    US_STATE_CITATION_PATTERN,
)
from legal_rag.types import RetrievedChunk
from legal_rag.verification.resolver import CaseResolver, ResolvedCase

logger = get_logger("pipeline.jurisdictions.us")

PIN_KEY = "is_canonical_authority"
"""Metadata flag: the chunk is a canonical authority (adjusters leave its score alone).

``merge_pinned_chunks`` also sets ``legal_rag.pipeline.types.PINNED_KEY``, the one flag the
rerank and context stages read to skip the reranker and survive the relevance gate."""

NAMED_KEY = "__named_citation"
"""Metadata flag: the user named this authority outright in the query."""

JURISDICTION_TAG_KEY = "__jurisdiction_question"
"""Metadata tag written by ``retrieve``: ``federal``, ``state`` or ``ambiguous``."""

CURRENCY_FIELD = "currency_status"
"""Metadata field the currency adjuster reads."""

PRE_ADJUST_SCORE_KEY = "pre_adjust_score"
"""Metadata key holding a chunk's score before the first adjuster changed it."""

NAMED_PIN_SCORE = 0.97
"""Score given to a chunk pinned by an exact lookup, so it survives alongside canonical pins."""

CURRENT_STATUSES = frozenset({"current", "in_force"})
UNKNOWN_STATUSES = frozenset({"unknown"})
UNMEASURED_STATUSES = frozenset({"unmeasured"})

_STICKY_FLAGS = (PIN_KEY, NAMED_KEY)
_US_VALUES = frozenset({"us", "usa", "u.s.", "united states"})
_CASE_SOURCE_TYPES = frozenset({"case_law", "case", "opinion", "court_opinion"})
_MAX_SUB_QUERIES = 4

_FEDERAL_STATUTE_CITE = re.compile(
    r"\b\d{1,3}\s+U\.?\s?S\.?\s?C\.?(?:\s?A\.?)?\s*(?:§§?\s*|secs?\.\s*|sections?\s+)?"
    r"\d+[\w.\-]*(?:\([A-Za-z0-9]+\))*"
    r"|\b\d{1,3}\s+C\.?\s?F\.?\s?R\.?\s*(?:(?:parts?|pts?\.)\s*|§§?\s*)?"
    r"\d+(?:\.\d+[\w\-]*)?(?:\([A-Za-z0-9]+\))*",
    re.IGNORECASE,
)
_CODE_CITE = re.compile(r"\bU\.?\s?S\.?\s?C\.?|\bC\.?\s?F\.?\s?R\.?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _state_names() -> dict[str, str]:
    """Two-letter code to full name, from the routing library's state table."""
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/us_states.json")
        .open("r", encoding="utf-8") as fh
    ):
        return {code.upper(): name for code, name in json.load(fh).items()}


def _meta(chunk: RetrievedChunk) -> dict[str, Any]:
    return chunk.metadata or {}


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _with_meta(chunk: RetrievedChunk, **extra: Any) -> RetrievedChunk:
    return replace(chunk, metadata={**_meta(chunk), **extra})


def _rescored(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    meta = _meta(chunk)
    return replace(
        chunk,
        score=score,
        metadata={**meta, PRE_ADJUST_SCORE_KEY: meta.get(PRE_ADJUST_SCORE_KEY, chunk.score)},
    )


def _by_score(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _citation_of(meta: Mapping[str, Any]) -> str:
    for key in ("primary_citation", "citation"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if value:
            return str(value).strip()
    citations = meta.get("citations")
    if isinstance(citations, (list, tuple)) and citations:
        return str(citations[0]).strip()
    return ""


def _scope_codes(filters: Mapping[str, Any]) -> tuple[str, ...]:
    """State codes named by the ``jurisdiction_scope`` filter, in order, without duplicates."""
    raw = filters.get("jurisdiction_scope")
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, str) else list(raw)
    names = _state_names()
    codes: list[str] = []
    for item in items:
        code = str(item).strip().upper()
        if code in names and code not in codes:
            codes.append(code)
    return tuple(codes)


# ---------------------------------------------------------------------------
# Adjusters (pure functions over chunks)
# ---------------------------------------------------------------------------


def collapse_alias_duplicates(chunks: Sequence[RetrievedChunk]) -> tuple[list[RetrievedChunk], int]:
    """Collapse chunks that are the same content stored under alias ids.

    A statute section can be indexed under several document ids with byte-identical text.
    Each alias would take a top-k slot and a slice of the context budget for nothing.
    The key is (citation, normalized text hash) and both must be present, so different sections,
    different chunks of one section, and uncited user documents are never merged.
    The highest scoring alias survives in the first alias's position.
    Pin flags merge onto the survivor, so a pinned alias is never lost to a higher scoring copy.
    Returns ``(collapsed, number_dropped)``.
    """
    seen: dict[tuple[str, str], int] = {}
    out: list[RetrievedChunk] = []
    dropped = 0
    for chunk in chunks:
        citation = _norm(_citation_of(_meta(chunk)))
        text = _norm(chunk.text)
        if not citation or not text:
            out.append(chunk)
            continue
        key = (citation, hashlib.sha256(text.encode("utf-8")).hexdigest())
        index = seen.get(key)
        if index is None:
            seen[key] = len(out)
            out.append(chunk)
            continue
        dropped += 1
        keeper = out[index]
        sticky = {f for f in _STICKY_FLAGS if _meta(chunk).get(f) or _meta(keeper).get(f)}
        winner = chunk if chunk.score > keeper.score else keeper
        out[index] = _with_meta(winner, **dict.fromkeys(sticky, True)) if sticky else winner
    return out, dropped


def _family(meta: Mapping[str, Any]) -> str:
    return str(meta.get("corpus_family") or meta.get("corpus_type") or "").strip().lower()


def downweight_unknown_currency(
    chunks: Sequence[RetrievedChunk], penalty: float, *, field: str = CURRENCY_FIELD
) -> list[RetrievedChunk]:
    """Rank law we cannot vouch for below law we can, without ever emptying an answer.

    Reads ``metadata[field]``: ``current`` (vouched for), ``unknown`` (penalized), and
    ``unmeasured`` (not penalized, and counts as vouched for).
    ``unmeasured`` is for a document whose publisher marks it in force and where only our own
    measurement is missing.
    Any other value, and a missing field, leaves the chunk alone.

    The load-bearing guard: an ``unknown`` chunk is penalized only when a vouched chunk from a
    DIFFERENT corpus family (``corpus_family``, else ``corpus_type``) is in the pool.
    Within one family the status says which publisher posts withdrawal notices, not which
    document is good, so it is not a ranking signal.
    A pool whose only good answers are unknown is therefore reordered never, and hollowed out never.
    Scores are multiplied by ``penalty`` for ordering only, and the pre-adjustment score is kept
    in ``metadata["pre_adjust_score"]`` so a relevance gate can still use the true score.
    """
    unknown: list[int] = []
    vouched: set[str] = set()
    for index, chunk in enumerate(chunks):
        meta = _meta(chunk)
        status = str(meta.get(field) or "").strip().lower()
        if status in CURRENT_STATUSES or status in UNMEASURED_STATUSES:
            vouched.add(_family(meta))
        elif status in UNKNOWN_STATUSES:
            unknown.append(index)
    penalized = {i for i in unknown if vouched - {_family(_meta(chunks[i]))}}
    if not penalized:
        return list(chunks)
    out = [_rescored(c, c.score * penalty) if i in penalized else c for i, c in enumerate(chunks)]
    return _by_score(out)


def apply_court_prestige(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Multiply the score of tagged chunks by the court-prestige factor, then re-sort.

    A text-similarity ranker can put a lower court quoting a rule at length above the rule's own
    source. Federal questions favor federal appellate courts and the Supreme Court, state
    questions favor the state's highest court.
    Only chunks carrying the ``__jurisdiction_question`` tag (written by ``retrieve``) are
    touched, and pinned chunks are exempt, so user documents and pins keep their scores.
    """
    out: list[RetrievedChunk] = []
    changed = False
    for chunk in chunks:
        meta = _meta(chunk)
        if JURISDICTION_TAG_KEY not in meta or meta.get(PIN_KEY):
            out.append(chunk)
            continue
        tier = classify_court(court_id=meta.get("court_id"), court_name=meta.get("court"))
        label = "state" if meta.get(JURISDICTION_TAG_KEY) == "state" else "federal"
        out.append(_rescored(chunk, chunk.score * boost_factor(tier, jurisdiction=label)))
        changed = True
    return _by_score(out) if changed else list(chunks)


def merge_pinned_chunks(
    chunks: Sequence[RetrievedChunk], pinned: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Merge exact-lookup chunks into a retrieved pool without duplicating a section.

    A chunk is the same authority when its ``document_id`` (else ``cluster_id``, else id)
    matches. If search already surfaced the named authority, that copy is upgraded in place
    (first copy pinned, all copies flagged named) and the lookup copy is dropped, so the section
    does not compete with itself for top-k. Authorities search did not find are put first.
    """

    def key(chunk: RetrievedChunk) -> str:
        meta = _meta(chunk)
        return str(meta.get("document_id") or meta.get("cluster_id") or chunk.id)

    pinned_keys = {key(c) for c in pinned}
    present: set[str] = set()
    out: list[RetrievedChunk] = []
    for chunk in chunks:
        k = key(chunk)
        if k in pinned_keys:
            flags: dict[str, Any] = {NAMED_KEY: True}
            if k not in present:
                flags[PIN_KEY] = True
                flags[PINNED_KEY] = True
            present.add(k)
            chunk = _with_meta(chunk, **flags)
        out.append(chunk)
    return [c for c in pinned if key(c) not in present] + out


# ---------------------------------------------------------------------------
# Exact lookups
# ---------------------------------------------------------------------------

CaseTextLoader = Callable[[ResolvedCase], Awaitable[Sequence[RetrievedChunk]]]
"""Loads opinion text chunks for a resolved case. Optional: without it a reference chunk is used."""

StatuteSectionLookup = Callable[[str], Awaitable[Sequence[RetrievedChunk]]]
"""Maps one statute or regulation citation string to the chunks of that section."""


def extract_case_candidates(query: str, limit: int = 4) -> list[str]:
    """Reporter citations first, then case captions, as strings to hand to a resolver."""
    citations = [
        m.group(0).strip()
        for m in US_REPORTER_CITATION_PATTERN.finditer(query or "")
        if not _CODE_CITE.search(m.group(0))
    ]
    captions = [m.group(0).strip() for m in US_CASE_PATTERN.finditer(query or "")]
    seen: set[str] = set()
    out: list[str] = []
    for candidate in citations + captions:
        key = _norm(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out[:limit]


def extract_statute_citations(query: str) -> list[str]:
    """Federal code and state code citations named in the query, without overlaps."""
    text = query or ""
    found: list[tuple[int, int, str]] = []
    for m in _FEDERAL_STATUTE_CITE.finditer(text):
        found.append((m.start(), m.end(), m.group(0)))
    for m in US_STATE_CITATION_PATTERN.finditer(text):
        if not any(m.start() < end and start < m.end() for start, end, _ in found):
            found.append((m.start(), m.end(), m.group(0)))
    seen: set[str] = set()
    out: list[str] = []
    for _, _, raw in sorted(found):
        cite = raw.strip().rstrip(".,;:")
        if _norm(cite) not in seen:
            seen.add(_norm(cite))
            out.append(cite)
    return out


def _case_slug(case: ResolvedCase) -> str:
    digest = hashlib.sha256(f"{_norm(case.case_name)}|{_norm(case.citation)}".encode()).hexdigest()
    return digest[:12]


class NamedCaseLookup:
    """Pins the case a query names by caption or reporter citation.

    Dense and keyword retrieval are worst at exact identifiers, so a case the user names outright
    can be missing from the pool. This resolves the named case through an injected
    ``CaseResolver`` and force-includes it. The first chunk per case is pinned, the rest ride
    normal reranking. Resolution failures are logged and skipped, never raised.
    """

    def __init__(
        self,
        resolver: CaseResolver,
        *,
        text_loader: CaseTextLoader | None = None,
        max_cases: int = 2,
        max_chunks_per_case: int = 6,
    ) -> None:
        self._resolver = resolver
        self._text_loader = text_loader
        self._max_cases = max_cases
        self._max_chunks = max_chunks_per_case

    async def lookup(self, query: str) -> list[RetrievedChunk]:
        candidates = extract_case_candidates(query, limit=self._max_cases * 2)
        if not candidates:
            return []
        resolved = await asyncio.gather(
            *(self._resolver.resolve(c) for c in candidates), return_exceptions=True
        )
        cases: list[ResolvedCase] = []
        seen: set[tuple[str, str]] = set()
        for item in resolved:
            if isinstance(item, BaseException):
                logger.warning("us_case_resolve_failed", error=repr(item))
                continue
            if item is None:
                continue
            key = (_norm(item.case_name), _norm(item.citation))
            if key not in seen:
                seen.add(key)
                cases.append(item)
        out: list[RetrievedChunk] = []
        for case in cases[: self._max_cases]:
            out.extend(await self._chunks_for(case))
        return out

    async def _chunks_for(self, case: ResolvedCase) -> list[RetrievedChunk]:
        slug = _case_slug(case)
        defaults = {
            "source_type": "case_law",
            "document_id": f"cited:{slug}",
            "title": case.case_name,
            "case_name": case.case_name,
            "citation": case.citation,
            "court": case.court,
            "year": case.year,
        }
        meta = {**{k: v for k, v in defaults.items() if v is not None}, **dict(case.metadata)}
        loaded: list[RetrievedChunk] = []
        if self._text_loader is not None:
            try:
                loaded = list(await self._text_loader(case))[: self._max_chunks]
            except Exception:
                logger.warning("us_case_text_load_failed", case=case.case_name, exc_info=True)
        if not loaded:
            cite = f", {case.citation}" if case.citation else ""
            year = f" ({case.year})" if case.year else ""
            note = (
                f"{case.case_name}{cite}{year}. Full opinion text was not available from the "
                "source; cite cautiously and verify the holding."
            )
            loaded = [RetrievedChunk(id=f"cited:{slug}:0", text=note, score=NAMED_PIN_SCORE)]
        out: list[RetrievedChunk] = []
        for index, chunk in enumerate(loaded):
            flags: dict[str, Any] = {NAMED_KEY: True}
            if index == 0:
                flags[PIN_KEY] = True
            out.append(
                replace(
                    chunk,
                    score=NAMED_PIN_SCORE,
                    metadata={**meta, **_meta(chunk), **flags},
                )
            )
        return out


class NamedStatuteLookup:
    """Pins the statute or regulation sections a query names, such as ``42 U.S.C. 1983``.

    Citations are found with the routing library's patterns and each one is handed to the
    injected ``section_lookup``, which returns the chunks of that section or an empty list when
    it cannot resolve it unambiguously. The first chunk per citation is pinned.
    Lookup failures are logged and skipped, never raised.
    """

    def __init__(
        self,
        section_lookup: StatuteSectionLookup,
        *,
        max_cites: int = 3,
        max_chunks_per_section: int = 4,
    ) -> None:
        self._section_lookup = section_lookup
        self._max_cites = max_cites
        self._max_chunks = max_chunks_per_section

    async def lookup(self, query: str) -> list[RetrievedChunk]:
        cites = extract_statute_citations(query)[: self._max_cites]
        if not cites:
            return []
        results = await asyncio.gather(
            *(self._section_lookup(c) for c in cites), return_exceptions=True
        )
        out: list[RetrievedChunk] = []
        for cite, result in zip(cites, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("us_statute_lookup_failed", citation=cite, error=repr(result))
                continue
            for index, chunk in enumerate(list(result)[: self._max_chunks]):
                flags: dict[str, Any] = {NAMED_KEY: True, "__named_citation_raw": cite}
                if index == 0:
                    flags[PIN_KEY] = True
                out.append(
                    replace(chunk, score=NAMED_PIN_SCORE, metadata={**_meta(chunk), **flags})
                )
        return out


# ---------------------------------------------------------------------------
# The plug-in
# ---------------------------------------------------------------------------


class USJurisdiction:
    """``JurisdictionPlugin`` for US primary law.

    Inject whichever pieces you have. A missing retriever or lookup simply disables that part.
    ``language_model`` and ``cache`` are optional and only feed the routing library's model
    fallbacks for jurisdiction and doctrine detection, which run keyword-first.
    ``opinion_resolver`` lets canonical case cards carry real opinion excerpts.
    """

    condenser_key = "US"

    def __init__(
        self,
        *,
        case_retriever: Retriever | None = None,
        statute_retriever: Retriever | None = None,
        case_resolver: CaseResolver | None = None,
        case_text_loader: CaseTextLoader | None = None,
        statute_section_lookup: StatuteSectionLookup | None = None,
        opinion_resolver: OpinionResolver | None = None,
        language_model: LanguageModel | None = None,
        cache: Cache | None = None,
        unknown_currency_penalty: float | None = 0.8,
        court_prestige: bool = False,
        corpus_cutoff_year: int | None = None,
    ) -> None:
        if unknown_currency_penalty is not None and not 0.0 < unknown_currency_penalty <= 1.0:
            raise ValueError("unknown_currency_penalty must be in (0, 1] or None")
        self._case_retriever = case_retriever
        self._statute_retriever = statute_retriever
        self._case_resolver = case_resolver
        self._case_text_loader = case_text_loader
        self._statute_section_lookup = statute_section_lookup
        self._opinion_resolver = opinion_resolver
        self._language_model = language_model
        self._cache = cache
        self._penalty = unknown_currency_penalty
        self._court_prestige = court_prestige
        self._corpus_cutoff_year = corpus_cutoff_year

    def matches(self, filters: Mapping[str, Any]) -> bool:
        """True when ``country_code`` or ``jurisdiction`` names the US."""
        for key in ("country_code", "jurisdiction"):
            if str(filters.get(key) or "").strip().lower() in _US_VALUES:
                return True
        return False

    def recency_gap(self, query: str) -> tuple[bool, int | None]:
        """Whether the query asks for a year past the corpus cutoff, and the year asked for."""
        if self._corpus_cutoff_year is None:
            return detect_temporal_recency_gap(query)
        return detect_temporal_recency_gap(query, self._corpus_cutoff_year)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        sub_queries: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search the case-law and statute sources, tag the pool, and pin doctrine authorities.

        Sub-queries broaden statute recall only: the primary query carries the case-law search.
        If one source fails the other's results are still returned, and the error is raised only
        when every configured source failed. Named-citation pins come from ``exact_lookups``
        and are merged by the caller with ``merge_pinned_chunks``.
        """
        if top_k <= 0 or not (query or "").strip():
            return []
        flt = dict(filters or {})
        sources = str(flt.get("sources_filter") or "")
        subs = [s.strip() for s in (sub_queries or ()) if s and s.strip()]
        legs: list[Awaitable[list[RetrievedChunk]]] = []
        if self._case_retriever is not None and sources != "statutes_only":
            legs.append(self._case_retriever.retrieve(query, top_k=top_k, filters=flt))
        if self._statute_retriever is not None and sources != "cases_only":
            legs.append(
                self._statute_retriever.retrieve(
                    query,
                    top_k=top_k,
                    filters=flt,
                    sub_queries=subs[:_MAX_SUB_QUERIES] if len(subs) >= 2 else None,
                )
            )
        gathered = await asyncio.gather(
            *legs, self._classify(query), self._detect(query), return_exceptions=True
        )
        results = gathered[: len(legs)]
        label = str(gathered[len(legs)])
        doctrine = gathered[len(legs) + 1]
        failures = [r for r in results if isinstance(r, BaseException)]
        if legs and len(failures) == len(legs):
            raise failures[0]
        for failure in failures:
            logger.warning("us_source_failed", error=repr(failure))

        pool: list[RetrievedChunk] = []
        seen: set[str] = set()
        for result in results:
            if isinstance(result, BaseException):
                continue
            for chunk in result:
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    pool.append(_with_meta(chunk, **{JURISDICTION_TAG_KEY: label}))
        pins = await self._doctrine_pins(
            query, doctrine if isinstance(doctrine, DoctrineMatch) else None, label, flt
        )
        return pins + pool

    async def _classify(self, query: str) -> str:
        try:
            verdict = await classify_jurisdiction(
                query, llm=self._language_model, cache=self._cache
            )
            return str(verdict.label)
        except Exception:
            logger.warning("us_jurisdiction_classify_failed", exc_info=True)
            return "ambiguous"

    async def _detect(self, query: str) -> DoctrineMatch | None:
        try:
            return await detect_doctrine(query, llm=self._language_model, cache=self._cache)
        except Exception:
            logger.warning("us_doctrine_detect_failed", exc_info=True)
            return None

    async def _doctrine_pins(
        self,
        query: str,
        primary: DoctrineMatch | None,
        label: str,
        filters: Mapping[str, Any],
    ) -> list[RetrievedChunk]:
        """Canonical authorities for the doctrines the query names, with a state-scope gate.

        Doctrine canonicals are mostly Supreme Court and federal statutes. For a matter scoped to
        a state and read as a state-law question, only canonicals from a court of that state are
        kept. If none remain, the registered state equivalent of the doctrine is used, and if
        there is none the pins are suppressed so state cases can surface through retrieval.
        """
        if primary is None:
            return []
        matches = detect_doctrines_multi(query, max_matches=3)
        if primary.doctrine.id not in {m.doctrine.id for m in matches}:
            matches.insert(0, primary)
        canonical = [c for m in matches for c in build_canonical_chunks(m)]
        scope = _scope_codes(filters)
        state_scoped = bool(scope) and label == "state"
        if state_scoped:
            names = [_state_names()[code].lower() for code in scope]
            canonical = [
                c
                for c in canonical
                if any(n in str(_meta(c).get("court") or "").lower() for n in names)
            ]
        if canonical:
            if self._opinion_resolver is not None:
                try:
                    await enrich_canonical_chunks_with_opinions(
                        canonical, self._opinion_resolver, cache=self._cache
                    )
                except Exception:
                    logger.warning("us_canonical_enrich_failed", exc_info=True)
            pins = canonical
        elif state_scoped:
            pins = [
                c
                for code in scope
                for c in build_state_equivalent_chunks(
                    doctrine_id=primary.doctrine.id,
                    doctrine_name=primary.doctrine.doctrine_name,
                    us_state=code,
                    query=query,
                )
            ]
        else:
            pins = []
        return [_with_meta(c, **{JURISDICTION_TAG_KEY: label}) for c in pins]

    def adjusters(self) -> Sequence[ChunkAdjuster]:
        """Court prestige (opt-in), then the unknown-currency downweight, then the alias collapse.

        The order matters: scores are adjusted before alias duplicates are collapsed, so the best
        scoring alias survives.
        """
        out: list[ChunkAdjuster] = []
        if self._court_prestige:

            def prestige(
                query: str, chunks: Sequence[RetrievedChunk], filters: Mapping[str, Any]
            ) -> list[RetrievedChunk]:
                return apply_court_prestige(chunks)

            out.append(prestige)
        penalty = self._penalty
        if penalty is not None:

            def currency(
                query: str, chunks: Sequence[RetrievedChunk], filters: Mapping[str, Any]
            ) -> list[RetrievedChunk]:
                return downweight_unknown_currency(chunks, penalty)

            out.append(currency)

        def collapse(
            query: str, chunks: Sequence[RetrievedChunk], filters: Mapping[str, Any]
        ) -> list[RetrievedChunk]:
            return collapse_alias_duplicates(chunks)[0]

        out.append(collapse)
        return out

    def exact_lookups(self) -> Sequence[ExactLookup]:
        """Named case and named statute section pinning, for whichever backends were injected."""
        out: list[ExactLookup] = []
        if self._case_resolver is not None:
            out.append(NamedCaseLookup(self._case_resolver, text_loader=self._case_text_loader))
        if self._statute_section_lookup is not None:
            out.append(NamedStatuteLookup(self._statute_section_lookup))
        return out

    def label_for(self, chunk: RetrievedChunk) -> str:
        """A citation-style display label.

        Cases read ``Name, 5 U.S. 137 (1803)``, statutes and rules read ``42 U.S.C. 1983: Title``.
        A chunk with no title, citation or filename falls back to its id.
        """
        meta = _meta(chunk)
        name = str(meta.get("case_name") or meta.get("title") or meta.get("filename") or "").strip()
        cite = _citation_of(meta)
        year = meta.get("year")
        is_case = str(meta.get("source_type") or "").lower() in _CASE_SOURCE_TYPES or bool(
            meta.get("court")
        )
        if is_case:
            label = name
            if cite and cite.lower() not in name.lower():
                label = f"{name}, {cite}" if name else cite
            if label and year and f"({year})" not in label:
                label = f"{label} ({year})"
            return label or chunk.id
        if cite and name and cite.lower() not in name.lower():
            return f"{cite}: {name}"
        return name or cite or chunk.id


__all__ = [
    "CURRENCY_FIELD",
    "JURISDICTION_TAG_KEY",
    "NAMED_KEY",
    "NAMED_PIN_SCORE",
    "PIN_KEY",
    "PRE_ADJUST_SCORE_KEY",
    "CaseTextLoader",
    "NamedCaseLookup",
    "NamedStatuteLookup",
    "StatuteSectionLookup",
    "USJurisdiction",
    "apply_court_prestige",
    "collapse_alias_duplicates",
    "downweight_unknown_currency",
    "extract_case_candidates",
    "extract_statute_citations",
    "merge_pinned_chunks",
]
