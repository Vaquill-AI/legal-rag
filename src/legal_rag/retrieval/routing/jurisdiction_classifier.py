"""
Federal-vs-state classifier for US legal questions.

Two-stage:

1. **Fast keyword/regex path** catches the obvious cases - explicit citations
   to USC / CFR / FRCP, named federal doctrines, named SCOTUS cases, and
   state-law topics like "duty of loyalty" or "adverse possession". Hits
   here return immediately with high confidence and no API cost.

2. **LLM fallback** (an injected ``LanguageModel``) handles the ambiguous middle - questions that don't trigger any keyword
   but might still be clearly federal or clearly state once a model reads
   them ("Can a sheriff search my car at a checkpoint?" -> federal Fourth
   Amendment; "How do I challenge my speeding ticket?" -> state).

Results can be cached through an injected ``Cache`` keyed on the normalised question text
so follow-up turns and re-asks do not repay the model cost.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any, Final, Literal

from legal_rag.providers.protocols import Cache, LanguageModel
from legal_rag.types import Message

logger = logging.getLogger(__name__)


def _load_data() -> dict[str, Any]:
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/jurisdiction_rules.json")
        .open("r", encoding="utf-8") as fh
    ):
        return json.load(fh)


_DATA = _load_data()

JurisdictionLabel = Literal["federal", "state", "ambiguous"]

# Confidence below this threshold triggers the LLM fallback. The fast path
# always reaches at least 0.7 when it's confident, so 0.6 is a safe gate.
_LLM_FALLBACK_THRESHOLD: Final[float] = 0.6


@dataclass(frozen=True)
class Classification:
    label: JurisdictionLabel
    confidence: float  # 0.0 - 1.0
    reason: str  # human-readable signal that drove the decision
    source: Literal["keyword", "llm", "default"] = "keyword"


# ---------------------------------------------------------------------------
# Federal cues
# ---------------------------------------------------------------------------

_FED_CITATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d+\s*U\.?\s*S\.?\s*C\.?\s*§?\s*\d", re.IGNORECASE),
    re.compile(r"\b\d+\s*C\.?\s*F\.?\s*R\.?\s*§?", re.IGNORECASE),
    re.compile(r"\bF\.?\s*R\.?\s*(?:Civ|Crim|App|Bankr|E?vid)\.?\s*P?\.?\s*\d", re.IGNORECASE),
    re.compile(r"\bFRCP\s*\d|\bFRE\s*\d|\bFRAP\s*\d", re.IGNORECASE),
    re.compile(
        r"\b(?:section|§)\s*\d+\s*(?:of\s+)?(?:the\s+)?(?:U\.?S\.?C\.?|USC)\b",
        re.IGNORECASE,
    ),
)

_FED_DOCTRINE_TERMS: Final[frozenset[str]] = frozenset(_DATA["_FED_DOCTRINE_TERMS"])

_FED_CASE_NAMES: Final[frozenset[str]] = frozenset(_DATA["_FED_CASE_NAMES"])

# ---------------------------------------------------------------------------
# State cues
# ---------------------------------------------------------------------------

_STATE_NAMES: Final[frozenset[str]] = frozenset(_DATA["_STATE_NAMES"])

_STATE_DOCTRINE_TERMS: Final[frozenset[str]] = frozenset(_DATA["_STATE_DOCTRINE_TERMS"])


def _word_boundary_re(terms: frozenset[str]) -> re.Pattern[str]:
    """Compile a word-boundary alternation over ``terms``.

    Word boundaries so short acronyms (``sec``, ``ada``, ``rico``, ``fre``) and
    short case names (``erie``) match only as whole words, never as substrings
    inside common words: ``sec`` no longer fires on "security"/"section", ``ada``
    not on "nevada"/"canada", ``fre`` not on "freedom", ``erie`` not on
    "experience". Longest-first so an overlapping term matches the fuller phrase.
    """
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


_FED_DOCTRINE_RE: Final = _word_boundary_re(_FED_DOCTRINE_TERMS)
_FED_CASE_RE: Final = _word_boundary_re(_FED_CASE_NAMES)
_STATE_DOCTRINE_RE: Final = _word_boundary_re(_STATE_DOCTRINE_TERMS)


# ---------------------------------------------------------------------------
# Stage 1: fast keyword path (deterministic, dependency-free).
# ---------------------------------------------------------------------------


def _doctrine_jurisdiction(question: str) -> tuple[str, str] | None:
    """(jurisdiction, doctrine_id) if the query names a canonical doctrine.

    Reads the curated jurisdiction off the doctrine canon
    (`canonical_doctrines.json`), the single source of truth, via the shared
    fast alias matcher. This keeps jurisdiction, doctrine detection, and
    substantive area agreeing on any query that names a doctrine. Pure: the
    only I/O is the one-time cached canon load. Returns None when no doctrine
    matches or its jurisdiction tag is not one of the two known labels.
    """
    from legal_rag.retrieval.routing.doctrine_detector import detect_doctrine_fast

    match = detect_doctrine_fast(question)
    if match is None:
        return None
    jur = match.doctrine.jurisdiction
    if jur not in ("federal", "state"):
        return None
    return jur, match.doctrine.id


def _classify_keyword(question: str) -> Classification:
    """Return a fast keyword-based classification.

    May return ``ambiguous`` with low confidence when no strong cue fires.
    Synchronous / no network I/O (only the cached doctrine-canon load) - safe
    to call from anywhere including LLM-prohibited contexts.
    """
    q = (question or "").strip().lower()
    if not q:
        return Classification("ambiguous", 0.0, "empty question", "default")

    for pat in _FED_CITATION_PATTERNS:
        if pat.search(q):
            return Classification(
                "federal",
                0.95,
                f"federal-statute citation matched: {pat.pattern[:40]}…",
                "keyword",
            )

    # Canonical-doctrine consultation (single source of truth). When the query
    # names a doctrine in the canon, use that doctrine's curated jurisdiction so
    # the three US signals agree. Runs before the legacy hand-listed term sets,
    # which remain as a fallback for federal signals that are not doctrines.
    canon = _doctrine_jurisdiction(question)
    if canon is not None:
        label, doctrine_id = canon
        confidence = 0.9 if label == "federal" else 0.85
        return Classification(label, confidence, f"canonical doctrine: {doctrine_id}", "canon")

    fed_term = _FED_DOCTRINE_RE.search(q)
    if fed_term:
        return Classification(
            "federal", 0.92, f"federal doctrine term: {fed_term.group()!r}", "keyword"
        )

    fed_case = _FED_CASE_RE.search(q)
    if fed_case:
        return Classification(
            "federal", 0.95, f"federal case name: {fed_case.group()!r}", "keyword"
        )

    state_topic = _STATE_DOCTRINE_RE.search(q)
    if state_topic:
        return Classification("state", 0.85, f"state-law topic: {state_topic.group()!r}", "keyword")

    state_hits = [s for s in _STATE_NAMES if s in q]
    if state_hits and any(
        kw in q
        for kw in (
            "law",
            "statute",
            "code",
            "supreme court of",
            "court of appeals of",
            "appellate court of",
        )
    ):
        return Classification(
            "state",
            0.7,
            f"state-name + jurisdictional cue: {state_hits[0]!r}",
            "keyword",
        )

    return Classification("ambiguous", 0.4, "no strong cue; needs LLM", "default")


# ---------------------------------------------------------------------------
# Stage 2: LLM fallback (async, with cache).
# ---------------------------------------------------------------------------

_LLM_PROMPT: Final[str] = (
    "You classify US legal questions as 'federal', 'state', or 'ambiguous'.\n"
    "\n"
    "Most everyday legal questions are governed by STATE law. Choose 'federal' "
    "only when the question genuinely turns on federal law.\n"
    "\n"
    "- 'state' (the common case) = state statutes/codes and state common law: "
    "landlord-tenant and leases, security deposits, evictions, most contracts, "
    "torts and personal injury, real property and HOAs, family law and divorce, "
    "wills/trusts/probate, business formation (LLCs, corporations, director "
    "duties), non-competes, at-will employment, and most crimes.\n"
    "- 'federal' = the question turns on the US Constitution (a 1st/2nd/4th/5th/"
    "14th-Amendment, equal-protection, commerce-clause, or due-process "
    "question), a federal statute or agency (patent, copyright, federal "
    "trademark, securities, bankruptcy, immigration, Title VII, ERISA, "
    "antitrust, RICO), federal court rules or procedure (FRCP/FRE), or a named "
    "federal doctrine (qualified immunity, Brady, Twombly, Erie, Chevron/Loper "
    "Bright).\n"
    "- 'ambiguous' = genuinely unclear with no clear federal hook (e.g., a "
    'general question about "the law on AI", or a matter that varies and could '
    "be either).\n"
    "\n"
    'Output JSON: {"label": "federal"|"state"|"ambiguous", '
    '"confidence": 0.0-1.0, "reason": "≤25 words"}.\n'
    "When there is no clear federal hook, prefer 'state' for a private-law "
    "matter and 'ambiguous' otherwise; do NOT default to 'federal'."
)


def _normalize_for_cache(q: str) -> str:
    """Stable cache key: lowercase + collapse whitespace + truncate."""
    return re.sub(r"\s+", " ", (q or "").strip().lower())[:512]


async def _llm_classify(question: str, llm: LanguageModel) -> Classification | None:
    """Ask the injected model for a verdict and parse the structured response.

    Returns ``None`` on any failure (timeout, parse). The caller falls back to the
    conservative keyword default.
    """
    try:
        raw = await asyncio.wait_for(
            llm.generate(
                [Message("system", _LLM_PROMPT), Message("user", question[:1000])],
                max_tokens=1024,
                temperature=0.0,
            ),
            timeout=8.0,
        )
        data = json.loads(raw or "{}")
        label = data.get("label", "ambiguous")
        if label not in ("federal", "state", "ambiguous"):
            label = "ambiguous"
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        reason = str(data.get("reason", ""))[:200]
        return Classification(label, conf, reason or "llm verdict", "llm")
    except Exception as exc:
        logger.warning("jurisdiction_classifier.llm_failed", extra={"err": str(exc)[:120]})
        return None


def _cache_key(q: str) -> str:
    return "jurisdiction:" + hashlib.sha256(_normalize_for_cache(q).encode()).hexdigest()[:32]


async def classify_jurisdiction(
    question: str,
    *,
    llm: LanguageModel | None = None,
    cache: Cache | None = None,
    cache_ttl_seconds: int = 3600,
) -> Classification:
    """Classify a US legal question as federal-law, state-law or ambiguous.

    The keyword path runs first. When its confidence is below the fallback threshold and
    a language model is supplied, the model decides. Results from the model are stored in
    the injected cache when one is given.
    """
    fast = _classify_keyword(question)
    if fast.confidence >= _LLM_FALLBACK_THRESHOLD or llm is None:
        return fast

    key = _cache_key(question)
    if cache is not None:
        try:
            hit = await cache.get(key)
            if hit:
                d = json.loads(hit)
                return Classification(d["label"], d["confidence"], d["reason"], "llm")
        except Exception:
            logger.warning("jurisdiction_classifier.cache_read_failed")

    verdict = await _llm_classify(question, llm)
    if verdict is None:
        return fast
    if cache is not None:
        try:
            await cache.set(
                key, json.dumps(asdict(verdict)).encode(), ttl_seconds=cache_ttl_seconds
            )
        except Exception:
            logger.warning("jurisdiction_classifier.cache_write_failed")
    return verdict


def classify_jurisdiction_sync(question: str) -> Classification:
    """Keyword-only entry point.

    Use from sync contexts (e.g., tests, simple offline tooling). Will
    never reach the LLM and will never block.
    """
    return _classify_keyword(question)


def _classification_to_dict(c: Classification) -> dict[str, object]:
    """Serialise to a dict for logging / structured tracing."""
    return asdict(c)


__all__ = [
    "Classification",
    "JurisdictionLabel",
    "classify_jurisdiction",
    "classify_jurisdiction_sync",
]
