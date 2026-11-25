"""
Doctrine detector and canonical-case force-include.

Maps a US legal question to a curated canonical-doctrine entry (loaded from
``app/rag/data/canonical_doctrines.json``) and produces synthetic
``RetrievedChunk`` objects representing the foundational SCOTUS / circuit
cases for that doctrine. Those chunks are merged into the retrieval result
ahead of the similarity-ranked list so that the LLM and the source panel
always surface the case-of-the-doctrine - Brady for Brady, Strickland for
Strickland, Erie for Erie - instead of trial-court applications.

Detection is two-stage, mirroring :mod:`legal_rag.retrieval.routing.jurisdiction_classifier`:

1. **Fast alias path** matches the question against each doctrine's
   ``aliases`` list (substring, case-insensitive, word-boundary aware).
   This covers the explicit cases - "What is the Brady rule?", "Strickland
   standard", "FRCP 56 summary judgment".

2. **LLM path** (injected language model) handles paraphrased intent - "When can my
   client claim her lawyer was constitutionally ineffective?" -> Strickland.
   The LLM is given the full doctrine list (id + name + description) and
   asked to pick at most one. Results are LRU-cached.

Detection returns at most one doctrine. If neither path finds a hit, the
question is treated as not-canonical-bound and the existing similarity-only
retrieval is left unchanged.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Final, Protocol, runtime_checkable

from legal_rag.providers.protocols import Cache, LanguageModel
from legal_rag.types import Message, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (Python-side mirror of canonical_doctrines.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalCase:
    case_name: str
    citation: str
    year: int
    court: str
    summary: str
    source_url: str | None
    # Mirror opinion-cluster id (resolved + backfilled by
    # scripts/backfill_canonical_cluster_ids.py). When present, the source panel
    # deep-fetches the full opinion/PDF deterministically -- no runtime lookup.
    # Distinct from the id in source_url, which is the case source's public
    # id and does NOT match the mirror.


@dataclass(frozen=True)
class CanonicalStatute:
    """A statute / regulation / Restatement reference that should be pinned
    alongside canonical cases for a doctrine.

    Round 3 addition: some doctrines (FMLA, parol-evidence, civil-rights
    employment) have foundational statutory text the LLM must cite -
    cases alone aren't enough. This is a separate force-include channel
    so the answer can quote the actual code section.

    Fields mirror CanonicalCase shape so the chunk builder can treat them
    uniformly. ``citation`` is the canonical short cite (e.g. "29 U.S.C.
    § 2611" or "Restatement (Second) of Contracts § 213"); ``text``
    carries the operative clause (kept short - under ~600 chars); ``url``
    deep-links to the OFFICIAL publisher (uscode.house.gov, ecfr.gov,
    uscourts.gov, constitution.congress.gov) and is None where no official
    free publisher exists, as for the Restatements and the UCC. Never an
    aggregator.
    """

    citation: str
    title: str  # short label, e.g. "FMLA eligibility - definitions"
    text: str  # operative clause / paraphrase
    source_type: str  # "usc" | "cfr" | "restatement" | "ucc" | "other"
    title_number: int | None  # for USC/CFR title-based filtering
    url: str | None


@dataclass(frozen=True)
class CanonicalDoctrine:
    id: str
    doctrine_name: str
    description: str
    aliases: tuple[str, ...]
    canonical_cases: tuple[CanonicalCase, ...]
    canonical_statutes: tuple[CanonicalStatute, ...] = ()
    # Enrichment (Part 2 canon-unification): the single source of truth for a
    # doctrine's jurisdiction and substantive area. jurisdiction is "federal" or
    # "state"; substantive_area is a SubstantiveArea value. Both drive the
    # derived jurisdiction / topic keyword triggers so the three US classifiers
    # agree on any query that names this doctrine.
    jurisdiction: str = "federal"
    substantive_area: str = "general"


@dataclass(frozen=True)
class DoctrineMatch:
    doctrine: CanonicalDoctrine
    confidence: float  # 0.0-1.0
    source: str  # "alias" | "llm"
    matched_alias: str | None  # for alias hits, the alias text


# ---------------------------------------------------------------------------
# Loader (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_doctrines() -> tuple[CanonicalDoctrine, ...]:
    """Read canonical_doctrines.json once and cache for the process lifetime."""
    raw = json.loads(
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/canonical_doctrines.json")
        .read_text(encoding="utf-8")
    )
    out: list[CanonicalDoctrine] = []
    for d in raw["doctrines"]:
        cases = tuple(
            CanonicalCase(
                case_name=c["case_name"],
                citation=c["citation"],
                year=int(c["year"]),
                court=c.get("court", ""),
                summary=c.get("summary", ""),
                source_url=c.get("url"),
            )
            for c in d["canonical_cases"]
        )
        statutes = tuple(
            CanonicalStatute(
                citation=s["citation"],
                title=s.get("title", s["citation"]),
                text=s.get("text", ""),
                source_type=s.get("source_type", "other"),
                title_number=s.get("title_number"),
                url=s.get("url"),
            )
            for s in d.get("canonical_statutes", [])
        )
        out.append(
            CanonicalDoctrine(
                id=d["id"],
                doctrine_name=d["doctrine_name"],
                description=d["description"],
                aliases=tuple(a.lower() for a in d["aliases"]),
                canonical_cases=cases,
                canonical_statutes=statutes,
                jurisdiction=d.get("jurisdiction", "federal"),
                substantive_area=d.get("substantive_area", "general"),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Stage 1 - alias / keyword detection (fast, deterministic)
# ---------------------------------------------------------------------------


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Build a word-boundary regex for an alias string.

    We need word boundaries so that ``"erie"`` doesn't match ``"experience"``
    and ``"rico"`` doesn't match ``"Puerto Rico"``. Punctuation inside the
    alias (like ``"sherwood v. walker"``) is escaped.

    Round 11 fix: hyphen-tolerance. The alias ``"bristol myers squibb"``
    must match against the natural-language phrasing ``"Bristol-Myers
    Squibb"``. We treat any run of whitespace OR hyphens in the alias as
    matching the same combination in the input.
    """
    escaped = re.escape(alias.lower())
    # Allow flexible whitespace and hyphens inside the alias. The pattern
    # collapses '\ ' and '\-' runs into '[-\s]+', so "bristol myers"
    # matches "bristol-myers", "Bristol  Myers", "BRISTOL_MYERS" (with
    # underscore? - no, only hyphen + whitespace).
    flexible = re.sub(r"(?:\\\s+|\\-)+", r"[-\\s]+", escaped)
    return re.compile(rf"\b{flexible}\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _alias_index() -> tuple[tuple[re.Pattern[str], CanonicalDoctrine, str], ...]:
    """Pre-compile every alias regex paired with its owning doctrine."""
    pairs: list[tuple[re.Pattern[str], CanonicalDoctrine, str]] = []
    for d in load_doctrines():
        for alias in d.aliases:
            pairs.append((_alias_pattern(alias), d, alias))
    # Longest aliases first - "rule 56 summary judgment" should beat "rule 56".
    pairs.sort(key=lambda p: len(p[2]), reverse=True)
    return tuple(pairs)


def detect_doctrine_fast(question: str) -> DoctrineMatch | None:
    """Return the best alias-match doctrine, or ``None`` if no alias fires."""
    q = (question or "").strip()
    if not q:
        return None
    for pattern, doctrine, alias in _alias_index():
        if pattern.search(q):
            return DoctrineMatch(
                doctrine=doctrine,
                confidence=0.95,
                source="alias",
                matched_alias=alias,
            )
    return None


def detect_doctrines_multi(question: str, *, max_matches: int = 3) -> list[DoctrineMatch]:
    """Round 9: return UP TO ``max_matches`` distinct doctrines that match the
    question by alias. Multi-issue practice questions ("Delaware LLC operating
    agreement waiving fiduciary duties and the implied covenant") often touch
    several doctrines simultaneously; returning just the first match force-
    includes the wrong canonical set.

    Doctrines are ordered by longest matched alias (specificity proxy). Each
    doctrine appears at most once even if multiple of its aliases hit.
    """
    q = (question or "").strip()
    if not q or max_matches <= 0:
        return []
    seen: set[str] = set()
    out: list[DoctrineMatch] = []
    for pattern, doctrine, alias in _alias_index():
        if doctrine.id in seen:
            continue
        if pattern.search(q):
            seen.add(doctrine.id)
            out.append(
                DoctrineMatch(
                    doctrine=doctrine,
                    confidence=0.95,
                    source="alias",
                    matched_alias=alias,
                )
            )
            if len(out) >= max_matches:
                break
    return out


# ---------------------------------------------------------------------------
# Stage 2 - LLM-backed intent matching
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT: Final[str] = (
    "You match US legal questions to a curated list of canonical doctrines.\n"
    "\n"
    "Each doctrine has an `id`, `doctrine_name`, and short `description`.\n"
    "Read the user's question and pick the SINGLE doctrine that best "
    "captures the legal rule the user is asking about. If no doctrine "
    "matches, output `null`.\n"
    "\n"
    'Output strict JSON: {"doctrine_id": "<id>" | null, '
    '"confidence": 0.0-1.0, "reason": "≤25 words"}.\n'
    "Only emit a doctrine_id that appears in the provided list. "
    "Be conservative: when in doubt, output null with low confidence."
)


def _doctrine_catalog_for_prompt() -> str:
    """Compact catalog the LLM sees as user-context: ``id | name | description``."""
    lines = ["# Canonical doctrine catalog\n"]
    for d in load_doctrines():
        lines.append(f"- {d.id} | {d.doctrine_name} | {d.description}")
    return "\n".join(lines)


def _cache_key(q: str) -> str:
    norm = re.sub(r"\s+", " ", (q or "").strip().lower())[:512]
    return hashlib.sha256(norm.encode()).hexdigest()[:32]


# Minimum confidence for an LLM-sourced doctrine to be trusted enough to
# force-include canonical authority into retrieval. Alias/keyword hits (0.95)
# bypass this; only the model fallback is gated. Matches the 0.6 LLM floor
# the sibling jurisdiction classifier uses.
_LLM_DOCTRINE_MIN_CONFIDENCE = 0.6


async def _llm_detect(question: str, llm: LanguageModel) -> DoctrineMatch | None:
    """Ask the injected model to pick a doctrine from the catalog."""
    try:
        catalog = _doctrine_catalog_for_prompt()
        raw = await asyncio.wait_for(
            llm.generate(
                [
                    Message("system", _LLM_SYSTEM_PROMPT),
                    Message("user", f"{catalog}\n\n## Question\n{question[:1000]}"),
                ],
                max_tokens=1024,
                temperature=0.0,
            ),
            timeout=10.0,
        )
        data = json.loads(raw or "{}")
        doctrine_id = data.get("doctrine_id")
        if not doctrine_id:
            return None
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        if conf < _LLM_DOCTRINE_MIN_CONFIDENCE:
            # A low-confidence guess must not force-include canonical authority into retrieval.
            logger.info(
                "doctrine_detector.llm_below_floor",
                extra={"id": str(doctrine_id)[:60], "confidence": round(conf, 2)},
            )
            return None
        for doc in load_doctrines():
            if doc.id == doctrine_id:
                return DoctrineMatch(
                    doctrine=doc, confidence=conf, source="llm", matched_alias=None
                )
        logger.warning("doctrine_detector.llm_unknown_id", extra={"id": str(doctrine_id)[:60]})
        return None
    except Exception as exc:
        logger.warning("doctrine_detector.llm_failed", extra={"err": str(exc)[:120]})
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_doctrine(
    question: str,
    *,
    llm: LanguageModel | None = None,
    cache: Cache | None = None,
    cache_ttl_seconds: int = 3600,
) -> DoctrineMatch | None:
    """Detect the canonical doctrine that best matches the user question.

    The alias path runs first. When no alias fires and a language model is supplied, the
    model picks from the doctrine catalog. Model verdicts (including misses) are stored in
    the injected cache when one is given.
    """
    fast = detect_doctrine_fast(question)
    if fast is not None or llm is None:
        return fast

    key = "doctrine:" + _cache_key(question)
    if cache is not None:
        try:
            hit = await cache.get(key)
            if hit is not None:
                cached = json.loads(hit)
                if cached.get("id") is None:
                    return None
                for doc in load_doctrines():
                    if doc.id == cached["id"]:
                        return DoctrineMatch(doc, float(cached["confidence"]), "llm", None)
        except Exception:
            logger.warning("doctrine_detector.cache_read_failed")
    match = await _llm_detect(question, llm)
    if cache is not None:
        try:
            payload = (
                {"id": None}
                if match is None
                else {"id": match.doctrine.id, "confidence": match.confidence}
            )
            await cache.set(key, json.dumps(payload).encode(), ttl_seconds=cache_ttl_seconds)
        except Exception:
            logger.warning("doctrine_detector.cache_write_failed")
    return match


# ---------------------------------------------------------------------------
# Synthetic chunk builder (force-include into retrieval result)
# ---------------------------------------------------------------------------


def build_canonical_chunks(match: DoctrineMatch) -> list[RetrievedChunk]:
    """Build :class:`RetrievedChunk` objects for the canonical authorities of a doctrine.

    The chunks carry a high score so a reranker or fusion stage respects them, and metadata
    shaped like ordinary case-law and statute results. The chunk text is the curated holding
    distillation; ``enrich_canonical_chunks_with_opinions`` can splice opinion text on top.
    """
    chunks: list[RetrievedChunk] = []
    base_score = 0.99  # canonical chunks rank above similarity hits
    for i, c in enumerate(match.doctrine.canonical_cases):
        # Plain text only: no markdown symbols, so a UI that renders the body verbatim is clean.
        body = (
            f"{c.case_name}, {c.citation} ({c.year})\n"
            f"{c.court}\n\n"
            f"Holding (canonical statement of the {match.doctrine.doctrine_name}): "
            f"{c.summary}\n\n"
            "This is the foundational US authority for this doctrine."
        )
        meta: dict[str, Any] = {
            "source_type": "case_law",
            "is_canonical_authority": True,
            "doctrine_id": match.doctrine.id,
            "title": c.case_name,
            "citation": c.citation,
            "citations": [c.citation],
            "court": c.court,
            "year": c.year,
            "decision_date": f"{c.year}-01-01",
            "url": c.source_url,
            "document_type": "case_law",
            "document_id": f"canonical:{match.doctrine.id}:{i}",
        }
        chunks.append(
            RetrievedChunk(
                id=f"canon_{match.doctrine.id}_{i}",
                text=body,
                score=base_score - 0.001 * i,
                metadata=meta,
            )
        )

    # Canonical statutes (USC, CFR, Restatement, UCC) for doctrines whose answers need
    # statutory text.
    statute_base = base_score - 0.001 * max(0, len(match.doctrine.canonical_cases))
    for j, s in enumerate(match.doctrine.canonical_statutes):
        statute_body = (
            f"{s.citation}\n"
            f"{s.title}\n\n"
            f"Operative text (canonical statute or regulation cited under "
            f"the {match.doctrine.doctrine_name} doctrine): {s.text}\n\n"
            "This is the foundational statutory authority for this doctrine."
        )
        statute_meta: dict[str, Any] = {
            "source_type": "statute",
            "is_canonical_authority": True,
            "doctrine_id": match.doctrine.id,
            "title": s.citation,
            "citation": s.citation,
            "citations": [s.citation],
            "url": s.url,
            "document_type": s.source_type,
            "statute_type": s.source_type,
            "title_number": s.title_number,
            "document_id": f"canonical:{match.doctrine.id}:stat:{j}",
        }
        chunks.append(
            RetrievedChunk(
                id=f"canon_{match.doctrine.id}_stat_{j}",
                text=statute_body,
                score=statute_base - 0.001 * j,
                metadata=statute_meta,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Opinion-text enrichment (best-effort, cached, time-bounded)
# ---------------------------------------------------------------------------
#
# build_canonical_chunks emits a curated holding distillation. That guarantees
# the landmark case surfaces, but the source panel then shows only the summary,
# not the opinion's own language. This step splices a live opinion excerpt onto
# each canonical CASE card so the panel shows real text. It is deliberately
# bounded (a slow upstream must never stall retrieval) and aggressively cached
# (opinions are immutable). Any failure leaves the curated summary intact.


@runtime_checkable
class CaseResolver(Protocol):
    """Resolves a canonical case to an opinion in some case-law store.

    The detector ships no case-law backend. A deployment supplies one through this protocol.
    ``resolve_case`` must verify that the case it finds is the one asked for, and return
    ``None`` on any ambiguity, so a wrong opinion is never attached.
    """

    async def resolve_case(
        self, case_name: str, citation: str | None, year: int | None
    ) -> str | None:
        """Return an opaque case id, or ``None`` when the case cannot be resolved."""
        ...

    async def fetch_excerpt(self, case_id: str) -> str:
        """Return opinion text for a case id, or ``""`` when unavailable."""
        ...


_OPINION_EXCERPT_CHARS = 4000
_ENRICH_MAX = 3
_ENRICH_BUDGET_S = 6.0
_OPINION_CACHE_TTL_S = 60 * 60 * 24 * 30


async def enrich_canonical_chunks_with_opinions(
    chunks: list[RetrievedChunk], resolver: CaseResolver, *, cache: Cache | None = None
) -> int:
    """Enrich canonical case chunks in place with opinion text from an injected resolver.

    For up to ``_ENRICH_MAX`` canonical case chunks, resolve the case id, stash it on the
    chunk metadata, then splice an excerpt into the chunk text. Time-bounded, and cached when a
    cache is given. Statutes and unresolved cards are untouched. Returns the count of cards
    that got a case id.
    """
    targets = [
        c
        for c in chunks
        if (c.metadata or {}).get("is_canonical_authority")
        and (c.metadata or {}).get("source_type") == "case_law"
        and (c.metadata or {}).get("citation")
    ][:_ENRICH_MAX]
    if not targets:
        return 0

    async def _cached(key: str) -> dict[str, Any] | None:
        if cache is None:
            return None
        try:
            hit = await cache.get(key)
            return json.loads(hit) if hit else None
        except Exception:
            return None

    async def _store(key: str, value: dict[str, Any]) -> None:
        if cache is None:
            return
        try:
            await cache.set(key, json.dumps(value).encode(), ttl_seconds=_OPINION_CACHE_TTL_S)
        except Exception:
            logger.warning("doctrine_detector.cache_write_failed")

    async def _one(chunk: RetrievedChunk) -> bool:
        meta = chunk.metadata
        citation = str(meta.get("citation") or "")
        expected = str(meta.get("title") or "")
        year_raw = meta.get("year")
        year = (
            int(year_raw) if isinstance(year_raw, int | str) and str(year_raw).isdigit() else None
        )
        id_key = "canonical:case_id:" + re.sub(r"\s+", " ", citation.strip().lower())
        hit = await _cached(id_key)
        case_id = hit.get("case_id") if hit else None
        if case_id is None:
            case_id = await resolver.resolve_case(expected, citation, year)
            if case_id is None:
                return False
            await _store(id_key, {"case_id": case_id})
        chunk.metadata["resolved_case_id"] = case_id
        ex_key = f"canonical:excerpt:{case_id}"
        hit = await _cached(ex_key)
        excerpt = hit.get("text") if hit else None
        if excerpt is None:
            excerpt = (await resolver.fetch_excerpt(case_id) or "")[:_OPINION_EXCERPT_CHARS].strip()
            if excerpt:
                await _store(ex_key, {"text": excerpt})
        if excerpt:
            chunk.text = f"{chunk.text}\n\nOpinion excerpt (full text):\n\n{excerpt}"
            chunk.metadata["opinion_enriched"] = True
        return True

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_one(c) for c in targets], return_exceptions=True),
            timeout=_ENRICH_BUDGET_S,
        )
    except TimeoutError:
        logger.warning("doctrine_detector.enrich_timeout", extra={"count": len(targets)})
        return 0
    return sum(1 for r in results if r is True)


__all__ = [
    "CanonicalCase",
    "CanonicalDoctrine",
    "CanonicalStatute",
    "DoctrineMatch",
    "build_canonical_chunks",
    "detect_doctrine",
    "detect_doctrine_fast",
    "enrich_canonical_chunks_with_opinions",
    "load_doctrines",
]
