"""
General-knowledge and legal-domain query classifier.

Single source of truth for query classification:
- is_general_knowledge_query(): Detects non-domain queries that should bypass RAG
- is_legal_domain_query(): Detects legal domain queries (from pipeline.py)
- is_non_legal_query(): Orchestrator combining heuristic + LLM tiebreaker for API callers
- classify_query_domain_llm(): language-model tiebreaker for ambiguous queries
- detect_temporal_recency_gap(): Detects when query asks for years beyond the corpus cutoff

Design principles:
1. Legal guard runs FIRST - prevents false positives on legal queries
2. Conservative default (False) - false negatives (extra compute) are better than false positives
3. All heuristic, no LLM calls - sub-1ms execution target for primary classifier
4. frozenset/tuple constants - immutable, O(1) lookups, computed once at import
5. LLM tiebreaker only fires for "gray zone" queries (~10-20% of traffic)
"""
# ruff: noqa: SIM103

from __future__ import annotations

import asyncio
import json
import logging
import re
import string
from importlib import resources
from typing import Any

from legal_rag.providers.protocols import LanguageModel
from legal_rag.retrieval.routing.us_citation_patterns import US_SIGNAL_PATTERNS
from legal_rag.types import Message

logger = logging.getLogger(__name__)


def _load_data() -> dict[str, Any]:
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/query_classification.json")
        .open("r", encoding="utf-8") as fh
    ):
        return json.load(fh)


_DATA = _load_data()

# Punctuation translation table for stripping from words (computed once)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# ============================================================================
# Legal Guard (checked FIRST - overrides any GK signal)
# ============================================================================

_LEGAL_GUARD_KEYWORDS: frozenset[str] = frozenset(_DATA["_LEGAL_GUARD_KEYWORDS"])

_LEGAL_GUARD_PHRASES: tuple[str, ...] = tuple(_DATA["_LEGAL_GUARD_PHRASES"])

# Pre-compiled legal citation regex patterns
_LEGAL_CITATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsection\s+\d+", re.IGNORECASE),
    re.compile(r"\barticle\s+\d+", re.IGNORECASE),
    re.compile(r"\bclause\s+\d+", re.IGNORECASE),
    re.compile(r"\brule\s+\d+", re.IGNORECASE),
    re.compile(r"\border\s+\d+", re.IGNORECASE),
    re.compile(r"\bschedule\s+\w+", re.IGNORECASE),
    # US formats (federal, state, reporters and statute acronyms).
    # The loose set is deliberate: a false positive costs one unnecessary retrieval, a false
    # negative costs an answer with no sources.
    # The LOOSE set is deliberate here. A false positive costs one unnecessary
    # retrieval ("who is Ada Lovelace" trips the ADA acronym); a false negative
    # costs a lawyer a legal answer with no sources. Those are not comparable.
    *US_SIGNAL_PATTERNS,
)


# ============================================================================
# General Knowledge Categories (13 categories, ~250+ patterns)
# ============================================================================

_GK_BIOGRAPHICAL: frozenset[str] = frozenset(_DATA["_GK_BIOGRAPHICAL"])

_GK_TEMPORAL: frozenset[str] = frozenset(_DATA["_GK_TEMPORAL"])

_GK_GEOGRAPHIC: frozenset[str] = frozenset(_DATA["_GK_GEOGRAPHIC"])

_GK_MATH_CONVERSION: frozenset[str] = frozenset(_DATA["_GK_MATH_CONVERSION"])

_GK_TECH: frozenset[str] = frozenset(_DATA["_GK_TECH"])

_GK_ENTERTAINMENT: frozenset[str] = frozenset(_DATA["_GK_ENTERTAINMENT"])

_GK_HEALTH_SCIENCE: frozenset[str] = frozenset(_DATA["_GK_HEALTH_SCIENCE"])

_GK_SHOPPING: frozenset[str] = frozenset(_DATA["_GK_SHOPPING"])

_GK_GREETINGS: frozenset[str] = frozenset(_DATA["_GK_GREETINGS"])

_GK_META: frozenset[str] = frozenset(_DATA["_GK_META"])

_GK_POLITICAL: frozenset[str] = frozenset(_DATA["_GK_POLITICAL"])

_GK_EDUCATION: frozenset[str] = frozenset(_DATA["_GK_EDUCATION"])

_GK_TRAVEL: frozenset[str] = frozenset(_DATA["_GK_TRAVEL"])

# Combined: union of all 13 categories (computed once at import)
_ALL_GK_PHRASES: frozenset[str] = (
    _GK_BIOGRAPHICAL
    | _GK_TEMPORAL
    | _GK_GEOGRAPHIC
    | _GK_MATH_CONVERSION
    | _GK_TECH
    | _GK_ENTERTAINMENT
    | _GK_HEALTH_SCIENCE
    | _GK_SHOPPING
    | _GK_GREETINGS
    | _GK_META
    | _GK_POLITICAL
    | _GK_EDUCATION
    | _GK_TRAVEL
)

_GK_PHRASE_COLLISIONS: frozenset[str] = frozenset(_DATA["_GK_PHRASE_COLLISIONS"])

# Narrower replacements for a collision whose trivia form is still worth keeping:
# "what time is it in New York?" is unmistakably trivia, while a bare "what time"
# is ordinary legal English ("within what time must the board meet").
_GK_PHRASE_REFINEMENTS: frozenset[str] = frozenset({"what time is it"})

_ALL_GK_PHRASES = (_ALL_GK_PHRASES - _GK_PHRASE_COLLISIONS) | _GK_PHRASE_REFINEMENTS

# The GK phrases are matched at WORD BOUNDARIES, not as raw substrings. As bare
# substrings "age of" fires inside "coverage of" and "percentage of", "convert"
# inside "convertible note", and "causes of" inside "causes of action" - that is,
# ordinary legal English was read as trivia and answered with no retrieval.
_GK_PHRASE_RE: re.Pattern[str] = re.compile(
    r"(?<!\w)(?:"
    + "|".join(sorted((re.escape(p) for p in _ALL_GK_PHRASES), key=len, reverse=True))
    + r")(?!\w)"
)

# Trivia is SHORT and self-contained ("calories in a banana", "currency of
# Japan?", "capital of France"). A real legal question is long and specific
# ("What is the coverage of the policy for flood damage?"). That difference is
# STRUCTURAL rather than lexical, so it holds for vocabulary nobody enumerated -
# which is the only way this gate can be safe. No keyword list can contain all of
# legal English, and every phrase added to catch more trivia is another collision
# with a real question.
#
# This is defence in depth, not the primary fix. With the measured collisions
# removed above, 0 of 1,367 legal prompts are deflected at ANY cap - but that
# corpus is finite, and the failure it guards against is unbounded: some phrase
# nobody has measured yet will collide with some legal question nobody has asked
# yet. A long, specific question is never trivia, whatever words it contains.
#
# 8 is the measured free choice: it deflects exactly as much trivia as no cap at
# all (2 of 19 either way), so it costs nothing today while bounding tomorrow.
_GK_MAX_WORDS = 8

_GK_WORD_TRIGGERS: frozenset[str] = frozenset(_DATA["_GK_WORD_TRIGGERS"])


def _stem_words(words: set[str]) -> set[str]:
    """Naive plural stemming: add de-pluralized forms for better keyword matching.

    Handles common English plural suffixes relevant to legal terms:
    - "statutes" -> "statute", "cases" -> "case", "laws" -> "law"
    - "parties" -> "parti" (won't match, but "parties" itself may be added)

    Only strips trailing 's' (and 'es'/'ies' selectively) - intentionally
    conservative to avoid false matches.
    """
    stemmed = set(words)
    for w in words:
        if len(w) > 3 and w.endswith("s"):
            stemmed.add(w[:-1])  # "statutes" -> "statute", "laws" -> "law"
            if w.endswith("es") and len(w) > 4:
                stemmed.add(w[:-2])  # "offences" -> "offenc" (+ "offence" from -1)
    return stemmed


_FOLLOWUP_PHRASES: frozenset[str] = frozenset(_DATA["_FOLLOWUP_PHRASES"])


def is_general_knowledge_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> bool:
    """
    Detect if a query is general knowledge that should bypass RAG.

    Uses a two-layer approach:
    1. Legal guard (keywords, phrases, citations) - if any match, return False
    2. GK patterns (13 categories, word triggers) - if any match, return True
    3. Default: False (conservative - better to run RAG unnecessarily)

    Follow-up handling: when chat_history is non-empty, this is a
    second-or-later turn in a conversation. Short queries that look
    like follow-ups ("tell me more", "explain that", "why?") are
    treated as in-domain regardless of GK-trigger matches because
    they ride on the prior turn's legal context. Longer queries still
    run the full classifier so a genuine "by the way, what's the
    weather" mid-thread is still caught.

    Performance: Sub-1ms for all queries (all heuristic, no LLM calls).
    """
    query_lower = query.lower().strip()
    has_history = bool(chat_history)

    # -- Follow-up short-circuit (history-aware) --------------------
    # A bare "tell me more" or "why?" mid-conversation is a legal
    # follow-up. Without this branch, it would trip the word-trigger
    # layer below ("explain", "why" are GK-leaning words) and skip
    # retrieval, breaking conversational follow-ups.
    if has_history:
        if not query_lower:
            return False
        if query_lower in _FOLLOWUP_PHRASES:
            return False
        # Short queries (≤ 4 words) inside an ongoing conversation
        # that contain a follow-up phrase as a substring also pass.
        word_count = len(query_lower.split())
        if word_count <= 4 and any(phrase in query_lower for phrase in _FOLLOWUP_PHRASES):
            return False

    # Quick-fail: very short queries are ambiguous
    if len(query_lower.split()) < 2:
        # Single words that are clearly greetings/casual.
        # When in conversation, a single-word greeting/ack is a follow-up
        # ("ok", "thanks") not a fresh GK query, so don't classify as GK.
        if query_lower in _GK_WORD_TRIGGERS and not has_history:
            return True
        return False

    # -- Layer 1: Legal guard (runs FIRST) --

    # 1a. Legal keyword check (set intersection, O(1)) with naive plural stemming
    query_words = {w.translate(_PUNCT_TABLE) for w in query_lower.split()} - {""}
    query_words_stemmed = _stem_words(query_words)
    if query_words_stemmed & _LEGAL_GUARD_KEYWORDS:
        return False

    # 1b. Legal phrase check (substring)
    if any(phrase in query_lower for phrase in _LEGAL_GUARD_PHRASES):
        return False

    # 1c. Legal citation regex
    if any(pattern.search(query) for pattern in _LEGAL_CITATION_PATTERNS):
        return False

    # -- Layer 2: GK pattern matching --

    # 2a. Length gate (runs BEFORE any GK match). Only a short, self-contained
    # question can be trivia. This is what stops a legal question from being
    # deflected by a phrase that merely appears inside it, and it is the
    # property that makes the layer below safe to keep at all.
    if len(query_lower.split()) > _GK_MAX_WORDS:
        return False

    # 2b. GK phrase patterns (word-boundary check across all 13 categories).
    # These are unambiguous ("what's the weather like in delhi"), so we
    # fire them even mid-conversation - a user genuinely asking about
    # weather mid-thread still gets short-circuited.
    if _GK_PHRASE_RE.search(query_lower):
        return True

    # 2c. GK word triggers (set intersection). These are weaker signals
    # ("explain", "describe", "tell me about") that often co-occur with
    # legal follow-ups. Suppress this layer mid-conversation so a
    # follow-up like "explain that finding" doesn't get short-circuited.
    if query_words & _GK_WORD_TRIGGERS and not has_history:
        return True

    # Default: not general knowledge (conservative)
    return False


# ---------------------------------------------------------------------------
# Document-scoped query detection
# ---------------------------------------------------------------------------
# Queries that are commands directed at the user's uploaded documents
# (e.g. "summarize the attached documents", "review my contract",
# "compare these two files"). These should NEVER trigger web search
# or corpus gap fallback - the user wants analysis of their own docs,
# not external case law or web results.

_DOC_SCOPED_VERBS: frozenset[str] = frozenset(_DATA["_DOC_SCOPED_VERBS"])

_DOC_SCOPED_OBJECT_PHRASES: frozenset[str] = frozenset(_DATA["_DOC_SCOPED_OBJECT_PHRASES"])


def is_document_scoped_query(query: str) -> bool:
    """Detect if a query is a command directed at the user's uploaded documents.

    These queries should skip web search and corpus gap fallback because
    the user wants analysis of their own docs, not external results.

    Performance: Sub-1ms (all heuristic, no LLM calls).
    """
    query_lower = query.lower().strip()
    words = {w.translate(_PUNCT_TABLE) for w in query_lower.split()} - {""}

    # Must contain a document-action verb
    if not (words & _DOC_SCOPED_VERBS):
        return False

    # Must reference documents/files as the object
    if any(phrase in query_lower for phrase in _DOC_SCOPED_OBJECT_PHRASES):
        return True

    return False


def is_legal_domain_query(query: str) -> bool:
    """
    Detect if a query is likely related to the legal domain.

    Uses keyword matching, citation pattern detection, and legal concept
    recognition to classify queries. Designed to be conservative -- when
    in doubt, returns True (better to retrieve unnecessarily than miss
    a legal query).

    Returns:
        True if the query appears to be legal in nature, False otherwise.
    """
    query_lower = query.lower().strip()

    # Short queries (<=3 words) are ambiguous -- treat as potentially legal
    if len(query_lower.split()) <= 3:
        return True

    # Legal keyword check (set intersection, O(1)) with naive plural stemming
    query_words = {w.translate(_PUNCT_TABLE) for w in query_lower.split()} - {""}
    query_words_stemmed = _stem_words(query_words)
    if query_words_stemmed & _LEGAL_GUARD_KEYWORDS:
        return True

    # Multi-word legal phrases
    if any(phrase in query_lower for phrase in _LEGAL_GUARD_PHRASES):
        return True

    # Legal citation patterns
    if any(pattern.search(query) for pattern in _LEGAL_CITATION_PATTERNS):
        return True

    # No legal indicators found
    return False


# ============================================================================
# Language-model tiebreaker (for ambiguous queries)
# ============================================================================

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a query classifier for a legal research platform. "
    "Determine if the user's query is about law, legal topics, statutes, case law, "
    "legal procedures, rights, regulations, or legal concepts - in any jurisdiction. "
    'Respond with ONLY a JSON object: {"is_legal": true} or {"is_legal": false}. '
    "Be inclusive: queries about rights, compliance, regulations, contracts, disputes, "
    "government policies, and legal-adjacent topics are legal. "
    "Only reject clearly non-legal topics like "
    "recipes, sports scores, entertainment, science trivia, weather, etc."
)

# Timeout for LLM classification call (seconds)
_LLM_CLASSIFY_TIMEOUT = 2.0


async def classify_query_domain_llm(query: str, llm: LanguageModel) -> bool:
    """Use an injected language model to classify whether a query is legal domain.

    Called only when the heuristic classifier is uncertain (gray zone).
    On timeout or failure it returns True (conservative: assume legal).
    """
    try:
        raw = await asyncio.wait_for(
            llm.generate(
                [Message("system", _CLASSIFY_SYSTEM_PROMPT), Message("user", query)],
                max_tokens=1024,
                temperature=0.0,
            ),
            timeout=_LLM_CLASSIFY_TIMEOUT,
        )
        result = json.loads(raw or "{}")
        return bool(result.get("is_legal", True))
    except TimeoutError:
        logger.warning("classify_query_domain_llm_timeout", extra={"query": query[:50]})
        return True
    except Exception:
        logger.warning(
            "classify_query_domain_llm_error", extra={"query": query[:50]}, exc_info=True
        )
        return True


def _heuristic_legal_guard_fires(query: str) -> bool:
    """Check if the heuristic legal guard matches (any legal keyword/phrase/citation).

    Returns True if query has legal indicators, False otherwise.
    """
    query_lower = query.lower().strip()
    if len(query_lower.split()) < 2:
        return False

    query_words = {w.translate(_PUNCT_TABLE) for w in query_lower.split()} - {""}
    query_words_stemmed = _stem_words(query_words)

    if query_words_stemmed & _LEGAL_GUARD_KEYWORDS:
        return True
    if any(phrase in query_lower for phrase in _LEGAL_GUARD_PHRASES):
        return True
    if any(pattern.search(query) for pattern in _LEGAL_CITATION_PATTERNS):
        return True
    return False


# ============================================================================
# Temporal Recency Gap Detection
# ============================================================================

# Corpus data cutoff: cases in the indexed corpus are available up to this year.
# Queries asking for cases from years AFTER this cutoff cannot be answered from
# the corpus alone and need web search supplementation.
_CORPUS_CUTOFF_YEAR: int = 2025

# Compiled once at import - matches 4-digit years in range 2020-2099
_YEAR_RE: re.Pattern[str] = re.compile(r"\b(20[2-9]\d)\b")

_RECENCY_PHRASES: frozenset[str] = frozenset(_DATA["_RECENCY_PHRASES"])


def detect_temporal_recency_gap(
    query: str, cutoff_year: int = _CORPUS_CUTOFF_YEAR
) -> tuple[bool, int | None]:
    """
    Detect if the query asks for case law from years beyond the corpus cutoff.

    Returns (gap_detected, requested_year):
    - gap_detected=True when the query asks for a year > cutoff_year
    - requested_year is the highest year found in the query (or None)

    Performance: Sub-1ms (regex + set lookup, no LLM calls).

    Examples:
        "latest case law of 2027"  -> (True, 2027)   (cutoff_year=2025)
        "give me 2025 opinions"    -> (False, 2025)  (cutoff_year=2025)
        "give me 2025 opinions"    -> (True, 2025)   (cutoff_year=2024)
        "case law from 2020"       -> (False, 2020)
        "landmark cases on bail"   -> (False, None)
    """
    query_lower = query.lower().strip()

    # Find all explicit years in the query
    year_matches = _YEAR_RE.findall(query)
    explicit_years = [int(y) for y in year_matches]

    # Check for recency phrase triggers (e.g., "latest", "recent", "2026")
    query_words = set(query_lower.split())
    has_recency_phrase = bool(query_words & _RECENCY_PHRASES)

    if explicit_years:
        max_year = max(explicit_years)
        if max_year > cutoff_year:
            return True, max_year
        # Year in query but within corpus range - no gap
        return False, max_year

    # No explicit year but strong recency signal (e.g., "latest 2026 cases")
    # Check if any 2025/2026/2027 strings appear (already captured by regex above,
    # but explicit year strings in _RECENCY_PHRASES handle the case where the year
    # is a phrase member, e.g., "2025" was matched above)
    if has_recency_phrase:
        # Recency phrases without an explicit future year - not a corpus gap per se,
        # but signals user wants the most recent results. Return False (no definitive
        # gap - the corpus may still have the latest available cases).
        return False, None

    return False, None


async def is_non_legal_query(query: str, llm: LanguageModel | None = None) -> bool:
    """Orchestrator: determine whether a query is non-legal.

    Three tiers:
    1. Heuristic general-knowledge detection, which returns True immediately.
    2. Heuristic legal guard, which returns False immediately.
    3. Gray zone: the injected language model decides. Without a model the query is
       treated as legal (conservative).
    """
    if is_general_knowledge_query(query):
        logger.info("non_legal_heuristic_gk", extra={"query": query[:80]})
        return True
    if _heuristic_legal_guard_fires(query):
        return False
    if llm is None:
        return False
    is_legal = await classify_query_domain_llm(query, llm)
    if not is_legal:
        logger.info("non_legal_llm_classifier", extra={"query": query[:80]})
    return not is_legal


# ============================================================================
# Under-specified Query Detection (clarification gate)
# ============================================================================
# When a user types a clearly broken template ("What defenses to a claim of"),
# a fill-in-the-blank with the blank left empty, or a trailing preposition
# with no object, we don't want to bulldoze ahead and produce a 30K-character
# generic answer - that wastes tokens, hides the typo, and sometimes ends up
# treating retrieved user-doc bullets as legal authority. Instead we return
# a single targeted clarifying question. Bias is conservative: false-
# negatives (let the query through to RAG) are preferred to false-positives
# (asking the user back when not needed).

_DANGLING_PREPS: frozenset[str] = frozenset(_DATA["_DANGLING_PREPS"])

_FRAGMENT_TAIL_PHRASES: tuple[str, ...] = tuple(_DATA["_FRAGMENT_TAIL_PHRASES"])

# Bracket placeholders the user forgot to fill in. Matches a `[...]` whose
# contents are alphabetic / whitespace / underscores AND at least two characters
# (e.g. `[BLANK]`, `[CASE NAME]`, `[topic]`). Numeric `[1]` citations are NOT
# matched (digit-only), and single-letter brackets (`[X]`, `[N]`) are NOT matched
# either - those are variables / examples in real questions ("the fee is [X]% of
# net sales", "if a value is a placeholder like [X]"), not forgotten descriptive
# slots, which are always multi-character.
_PLACEHOLDER_RE = re.compile(r"\[\s*[A-Za-z_][A-Za-z_\s]+\s*\]")

# The user is asking ABOUT placeholders / template fields (very common for the
# Drafting workspace, where generated drafts are full of [X] / [BLANK] slots),
# not forgetting to fill one in. When the query descriptively references
# placeholders, don't gate it - answer it.
_REFERS_TO_PLACEHOLDER_RE = re.compile(
    r"\b(?:placeholders?|fill(?:ed|s|ing)?(?:\s+in)?|unfilled|not\s+filled|"
    r"left\s+blank|blank\s+fields?|template\s+fields?|square\s+brackets?|bracketed)\b",
    re.IGNORECASE,
)

# Mid-string template-restart patterns. The user typed half a template, then
# started a fresh clause without filling in the slot. Both audit examples
# look like "FRAGMENT + capital-word verb starting a new instruction".
_BROKEN_TEMPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "...a claim of For each defense, provide..."  ->  fragment + restart
    re.compile(
        r"\b(?:a\s+claim\s+of|a\s+cause\s+of|a\s+basis\s+under|a\s+defense\s+to|a\s+theory\s+of)"
        r"\s+(?:for\s+each|include|provide|give|list|analyze|describe|explain|outline|summarize)\b",
        re.IGNORECASE,
    ),
    # "Find leading court decisions on include the key holdings..."
    re.compile(
        r"\b(?:decisions|cases|holdings|judgments)\s+on"
        r"\s+(?:include|provide|give|list|analyze|describe|explain|outline|summarize)\b",
        re.IGNORECASE,
    ),
)

# Generic clarifier wording. Short, lawyer-tone, no scolding.
_CLARIFIER_DEFAULT = (
    "Your message looks cut off - could you complete the thought? "
    "What claim, topic, or jurisdiction did you mean to specify?"
)
_CLARIFIER_PLACEHOLDER = (
    "Your message contains a placeholder that wasn't filled in. "
    "What topic or claim did you mean to put there?"
)


def is_underspecified_query(query: str) -> tuple[bool, str | None]:
    """Detect prompts that are clearly broken templates or cut off mid-thought.

    Returns ``(is_underspecified, suggested_clarifier)``.

    Bias is to false-negatives: when in doubt, let the query through to RAG.
    A false clarification interrupts a real question; a false negative just
    produces a slightly more generic answer than we'd like.

    Triggers on:
      1. Bracket placeholders the user forgot to fill in (``[BLANK]``,
         ``[topic]``, ``[claim type]``, etc.).
      2. Trailing dangling preposition (``of``, ``on``, ``for`` …) on
         queries with at least 6 tokens - long enough that the user clearly
         intended to keep going.
      3. Trailing fragment phrases like "a claim of", "leading cases on",
         "for each" - template skeletons left unfilled.

    Does NOT trigger on:
      - Short legitimate queries ("What is Miranda?", "Section 1983?")
      - Long detailed queries even if the tail is awkward (> 500 chars)
      - Queries ending in plain content words, named entities, or numbers
    """
    if not query:
        return False, None
    q = query.strip()
    if not q or len(q) > 500:
        return False, None

    # Placeholder check uses the raw (case-preserving) string so "[BLANK]"
    # and "[Blank]" both match. Skip when the user is describing placeholders
    # rather than leaving one unfilled (e.g. asking about a template draft).
    if _PLACEHOLDER_RE.search(q) and not _REFERS_TO_PLACEHOLDER_RE.search(q):
        return True, _CLARIFIER_PLACEHOLDER

    # Mid-string broken-template restart (fragment + verb-restart). Catches
    # the audit example "a claim of For each defense, provide..." where the
    # user typed a partial template, didn't fill the slot, and then began
    # the next instruction. Trailing-tail checks below miss this because
    # the trailing tail is "the legal basis" (legitimate-looking).
    for pat in _BROKEN_TEMPLATE_PATTERNS:
        if pat.search(q):
            return True, _CLARIFIER_DEFAULT

    # A message ending in sentence-terminal punctuation (. ! ?) is a finished
    # thought, not a cut-off fragment. The tail heuristics below (fragment
    # phrases, dangling prepositions) only make sense on unterminated input, so
    # a complete sentence like "...is what I'm looking for." must not be gated.
    # Placeholder + broken-template restarts were already checked above and are
    # unaffected.
    if q[-1] in ".!?":
        return False, None

    # Strip a single trailing punctuation cluster for tail-token analysis.
    q_tail = q.rstrip("?.!,;:").strip()
    if not q_tail:
        return False, None

    q_lower = q_tail.lower()

    # Fragment tail phrases: "...a claim of" -> trigger
    for frag in _FRAGMENT_TAIL_PHRASES:
        if q_lower.endswith(frag):
            return True, _CLARIFIER_DEFAULT

    # Trailing dangling preposition.
    tokens = q_lower.split()
    if len(tokens) >= 6:
        last_token = tokens[-1].strip(".,;:()")
        if last_token in _DANGLING_PREPS:
            return True, _CLARIFIER_DEFAULT

    return False, None
