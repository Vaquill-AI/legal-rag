"""Conversation-aware query condensation.

Rewrites a follow-up question into a standalone retrieval query using the chat history.
The language model, the cache and the metering hook are injected:

- ``llm``: a ``LanguageModel`` used with structured output.
- ``cache``: an optional ``Cache``. The ``scope`` string passed to ``condense`` is part of the
  cache key, so a caller can isolate entries per user, tenant or session without this module
  knowing what those are.
- ``usage_hook``: an optional callback that receives a ``UsageEvent``.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import Cache, LanguageModel, UsageHook
from legal_rag.retrieval.conversation.text_tokens import count_words
from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import Message, UsageEvent

logger = get_logger("conversation.condenser")


class CondensedQueryOutput(BaseModel):
    """Structured output of the condensation call."""

    primary_query: str
    sub_queries: list[str] = Field(default_factory=list)
    resolved_references: list[str] = Field(default_factory=list)
    response_language: str | None = None


class QueryType(Enum):
    """Classification of query types for handling strategy."""

    STANDALONE = "standalone"  # Query is already self-contained
    FOLLOW_UP = "follow_up"  # Requires context from history
    CLARIFICATION = "clarification"  # Asking for simpler explanation
    ELABORATION = "elaboration"  # Asking for more details
    TOPIC_SHIFT = "topic_shift"  # Changing to new topic
    REFERENCE = "reference"  # Referring to something earlier
    CONVERSATIONAL = "conversational"  # Greetings, small talk, no retrieval needed


@dataclass
class CondensationResult:
    """Result of query condensation process."""

    condensed_query: str
    original_query: str
    query_type: QueryType
    confidence: float
    requires_retrieval: bool
    entities_resolved: list[str]
    processing_time_ms: float
    cache_hit: bool = False
    sub_queries: list[str] = field(default_factory=list)
    # Output language the user explicitly asked for, rescued before
    # meta-instruction stripping discards it. None when not requested.
    response_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condensed_query": self.condensed_query,
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "requires_retrieval": self.requires_retrieval,
            "entities_resolved": self.entities_resolved,
            "processing_time_ms": self.processing_time_ms,
            "cache_hit": self.cache_hit,
            "sub_queries": self.sub_queries,
        }


# Prompts for query contextualization.
# Each prompt handles: anaphora resolution, structural references,
# meta-instruction stripping, and multi-topic decomposition in a single LLM
# call with structured output.
#
# The invariant scaffold (role, objective, rules, decomposition contract) is
# identical across jurisdictions. Only the meta_instruction_stripping court-
# abbreviation map and the few-shot <examples> block change. Keeping the two
# prompts as fully-formed strings (rather than templated fragments) keeps the
# system-prompt cache key stable and makes diffs reviewable.


CONDENSER_SYSTEM_PROMPT = """<system>
<role>Legal query transformation specialist for conversational RAG retrieval systems serving United States jurisdiction.</role>

<objective>
Transform the user's latest message into one or more standalone, retrieval-optimized search queries by:
1. Resolving ALL references (pronouns, demonstratives, structural references like "Point 2", "Part A")
2. Stripping meta-instructions (instructional wrappers irrelevant for document search)
3. Decomposing multi-topic queries into focused sub-queries for optimal vector search
</objective>

<reference_resolution_rules>
<rule priority="critical">PRESERVE VERBATIM: Case numbers (e.g., 1:23-cv-04567, 2024 WL 123456, 600 U.S. 412), section citations (42 U.S.C. 1983, 18 U.S.C. 1341, FRCP Rule 12(b)(6)), article references (U.S. Const. amend. XIV), act names (FRCP, FRE, FOIA, ADA, FLSA), dates, and legal terminology</rule>
<rule priority="critical">PRESERVE CORPORATE / TRANSACTIONAL TERMS: Keep contract and commercial terminology intact when present, including indemnification, limitation of liability, MSA (Master Services Agreement), DPA (Data Processing Addendum), SLA (Service Level Agreement), NDA, SOW, change of control, assignment, termination for convenience, auto-renewal, governing law, and named clause references (e.g., "Section 9.2", "Clause 14"). Do not drop or paraphrase these into generic case-law phrasing.</rule>
<rule priority="critical">Replace all pronouns (it, this, that, they, he, she, them) with explicit entity references from chat history</rule>
<rule priority="critical">Replace demonstratives (the case, the section, the judgment, the clause, the agreement) with specific identifiers</rule>
<rule priority="critical">STRUCTURAL REFERENCES: When user refers to "Point 2", "Part A", "the third issue", "heading 3", "para 4" etc. from a previous assistant response, find that numbered/labeled content in chat history and replace the reference with the actual legal topic(s) described there</rule>
<rule priority="medium">If query is already standalone with no context-dependent references, return AS-IS without modification</rule>
</reference_resolution_rules>

<meta_instruction_stripping>
Strip instructional language that is NOT relevant for document retrieval. Extract only WHAT to search for:
- "prepare a brief memo" -> remove (instruction to LLM, not search term)
- "list top 20 cases" -> remove the number/format instruction, keep the topic
- "find authorities on" -> keep the topic, remove "find authorities on"
- "draft a research note on" -> remove instructional wrapper
- "8-12 SCOTUS and 2nd Circuit authorities" -> transform to "Supreme Court Second Circuit case law"
- Court name abbreviations: "SCOTUS" -> "Supreme Court", "SDNY" -> "Southern District of New York", "EDNY" -> "Eastern District of New York", "2d Cir." -> "Second Circuit", "9th Cir." -> "Ninth Circuit", "DC Cir." -> "DC Circuit", "Fed. Cir." -> "Federal Circuit"
</meta_instruction_stripping>

<output_language_rescue priority="critical">
An explicit request for the ANSWER's language is a meta-instruction, so the rules above correctly
strip it from the retrieval query. But it is NOT disposable: it is what the user asked for.
RESCUE it into `response_language` BEFORE stripping, then remove it from `primary_query`.
- "...and translate it into Chinese" -> primary_query without it; response_language: "Chinese"
- "explain in Spanish" -> primary_query without it; response_language: "Spanish"
- "reply in French" -> response_language: "French"
- no output language named -> response_language: null
Writing the query in a different language than the user is NOT a language request. Only set
`response_language` when the user explicitly asked for the answer in a named language.
</output_language_rescue>

<multi_topic_decomposition>
When the resolved query covers MULTIPLE DISTINCT legal topics, produce focused sub_queries:
- Each sub_query should target ONE legal concept for optimal vector similarity
- Maximum 4 sub_queries
- If query focuses on a SINGLE topic, leave sub_queries EMPTY
- Each sub_query should be a complete, standalone search query (not a fragment)
- Include jurisdiction/court context in each sub_query when specified by the user
</multi_topic_decomposition>

<examples>
<example type="pronoun_resolution">
<history>User: "Explain 18 U.S.C. Section 1341 mail fraud"</history>
<follow_up>What's the punishment?</follow_up>
<primary_query>What is the punishment prescribed under 18 U.S.C. Section 1341 mail fraud?</primary_query>
<sub_queries>[]</sub_queries>
</example>

<example type="structural_reference">
<history>Assistant: "...Point 2: Cyber-harassment and online defamation under Section 230 CDA..."</history>
<follow_up>find cases on Point 2</follow_up>
<primary_query>cyber-harassment online defamation case law Section 230 Communications Decency Act</primary_query>
<sub_queries>[]</sub_queries>
</example>

<example type="meta_instruction_plus_structural_plus_decomposition">
<history>Assistant: "...Point 2: The legal issues include (a) tortious interference with contract, (b) breach of fiduciary duty, (c) admissibility of electronically stored information under FRCP Rule 34..."</history>
<follow_up>prepare brief memo of 8-12 federal circuit and SDNY authorities on all legal issues in Point 2</follow_up>
<primary_query>Federal Circuit SDNY case law tortious interference fiduciary duty electronic discovery</primary_query>
<sub_queries>["tortious interference with contract case law Federal Circuit SDNY", "breach of fiduciary duty case law federal courts", "electronically stored information admissibility FRCP Rule 34 federal courts"]</sub_queries>
</example>

<example type="standalone">
<history>Any or none</history>
<follow_up>What is the statute of limitations for breach of contract in New York?</follow_up>
<primary_query>What is the statute of limitations for breach of contract in New York?</primary_query>
<sub_queries>[]</sub_queries>
</example>

<example type="case_reference">
<history>User: "Case No. 1:23-cv-04567 was filed in the Southern District of New York"</history>
<follow_up>Who won?</follow_up>
<primary_query>Who was the successful party in Case No. 1:23-cv-04567 filed in the Southern District of New York?</primary_query>
<sub_queries>[]</sub_queries>
</example>

<example type="multi_topic_no_structural_ref">
<history>User asked about a contract dispute</history>
<follow_up>find cases on both force majeure and frustration of purpose</follow_up>
<primary_query>case law force majeure frustration of purpose contract</primary_query>
<sub_queries>["force majeure clause contract law case law", "frustration of purpose doctrine contract law case law"]</sub_queries>
</example>

<example type="contract_clause_interpretation">
<history>User: "Reviewing a Master Services Agreement with our vendor"</history>
<follow_up>does the limitation of liability cap apply to the indemnification obligations?</follow_up>
<primary_query>whether limitation of liability cap applies to indemnification obligations in a Master Services Agreement</primary_query>
<sub_queries>["limitation of liability cap carve-out for indemnification obligations contract", "indemnification obligations excluded from liability cap MSA"]</sub_queries>
</example>

<example type="compliance_regulatory_lookup">
<history>User: "We process personal data of California residents"</history>
<follow_up>what are our breach notification obligations?</follow_up>
<primary_query>data breach notification obligations for processing California residents personal data under CCPA CPRA</primary_query>
<sub_queries>[]</sub_queries>
</example>

<example type="vendor_sla_dpa">
<history>User: "Negotiating a SaaS vendor agreement"</history>
<follow_up>what uptime commitment and service credits should the SLA include, and does the DPA cover sub-processor notice?</follow_up>
<primary_query>SaaS vendor agreement service level agreement uptime commitment service credits and data processing addendum sub-processor notice requirements</primary_query>
<sub_queries>["service level agreement uptime commitment service credit remedies SaaS vendor", "data processing addendum sub-processor notification and approval requirements"]</sub_queries>
</example>
</examples>
</system>"""


FAST_SYSTEM_PROMPT = """<task>Rewrite as standalone legal query using chat context.</task>
<rules>Preserve: case numbers, section citations, dates, legal terms verbatim. Replace pronouns with explicit references.</rules>
<output>Query only, no explanation.</output>"""


class QueryCondenser:
    """
    Condenses context-dependent queries into standalone queries.

    Condenses a follow-up into a standalone query using the conversation history.
    to preserve BM25 keyword matching for:
    - Case numbers (1239/2023)
    - Section citations (42 U.S.C. 1983)
    - Dates and amounts
    - Legal terminology
    """

    # Pronouns, demonstratives, and structural reference terms
    CONTEXT_INDICATORS: ClassVar[set[str]] = {
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "theirs",
        "he",
        "she",
        "his",
        "her",
        "hers",
        "him",
        "above",
        "previous",
        "earlier",
        "mentioned",
        "the same",
        "such",
        "former",
        "latter",
    }

    # Regex pattern for structural references like "Point 2", "Part A", "Para 3"
    STRUCTURAL_REF_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:point|part|para|paragraph|issue|item|heading)\s+(?:\d+|[a-z])\b",
        re.IGNORECASE,
    )

    # Short queries that likely need context
    SHORT_QUERY_THRESHOLD = 5  # words

    # Phrases indicating elaboration/follow-up
    ELABORATION_PHRASES: ClassVar[set[str]] = {
        "tell me more",
        "more about",
        "elaborate",
        "expand on",
        "what else",
        "anything else",
        "more details",
        "go on",
        "continue",
        "keep going",
        "and?",
        "more?",
        "what about",
    }

    # Phrases indicating clarification
    CLARIFICATION_PHRASES: ClassVar[set[str]] = {
        "explain that",
        "what do you mean",
        "can you clarify",
        "simpler",
        "easier terms",
        "in other words",
        "rephrase",
        "don't understand",
        "confused",
        "not clear",
    }

    # Exact-match whitelist for skipping retrieval.
    # The ENTIRE normalized query must match one of these - no substring matching,
    # no partial matching, no complex interaction with legal marker counter-checks.
    # This makes false positives (skipping retrieval for a real query) impossible.
    SKIP_RETRIEVAL_EXACT: ClassVar[set[str]] = {
        # Greetings
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        # Small talk
        "how are you",
        "how are you doing",
        "whats up",
        "hows it going",
        "sup",
        # Acknowledgments
        "thanks",
        "thank you",
        "thank u",
        "thx",
        "ok",
        "okay",
        "got it",
        "understood",
        "i see",
        "alright",
        # Farewells
        "bye",
        "goodbye",
        "see you",
        # Simple yes/no
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "cool",
        "great",
        # Compound confirmations (production: users confirm LLM suggestions)
        "yes please",
        "yes confirmed",
        "confirmed",
        "go ahead",
        "proceed",
        "do it",
        "please do",
        "will do",
        "sounds good",
        "thats fine",
        "fine",
        "absolutely",
        "of course",
        "right",
        # UI option selections
        "option 1",
        "option 2",
        "option 3",
        "option 4",
        # Search/enable commands (not legal queries)
        "search now",
        "enable web search",
        "enable web access",
        "turn on web search",
        # Pleasantries
        "nice to meet you",
        "pleasure to meet you",
    }

    def __init__(
        self,
        llm: LanguageModel,
        cache: Cache | None = None,
        *,
        use_fast_prompt: bool = False,
        max_history_turns: int = 5,
        system_prompt: str | None = None,
        usage_hook: UsageHook | None = None,
        max_tokens: int = 1024,
        cache_ttl_seconds: int = 3600,
    ):
        """Initialize the condenser with injected providers.

        Args:
            llm: Language model used for the rewrite (structured output).
            cache: Optional cache for condensed queries.
            use_fast_prompt: Use a shorter prompt for speed (less accuracy).
            max_history_turns: Max conversation turns to include.
            system_prompt: Replace the default US system prompt.
            usage_hook: Optional metering callback.
            max_tokens: Output token budget for the rewrite call.
            cache_ttl_seconds: Lifetime of cached rewrites.
        """
        self.llm = llm
        self.cache = cache
        self.use_fast_prompt = use_fast_prompt
        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt or CONDENSER_SYSTEM_PROMPT
        self.usage_hook = usage_hook
        self.max_tokens = max_tokens
        self.cache_ttl_seconds = cache_ttl_seconds

    def _is_pure_conversational(self, query: str) -> bool:
        """
        Detect pure conversational queries that don't need document retrieval.

        Uses exact-match against a whitelist: the ENTIRE query (after stripping
        punctuation and normalizing whitespace) must match a known phrase.
        No substring matching, no partial matching, no counter-check logic.

        This makes false positives impossible - a query can only skip retrieval
        if it is literally just "hello", "thanks", "how are you", etc.

        Examples:
        - "hello" -> True (exact match)
        - "thank you!" -> True (normalized to "thank you")
        - "hi, tell me about Section 420" -> False (not an exact match)
        - "Neethu v Trijo Joseph in OP(Crl.) No. 226 of 2022" -> False
        - "no" -> True (exact match - but only when the entire query is just "no")
        """
        # Strip punctuation, collapse whitespace, lowercase
        normalized = re.sub(r"[^\w\s]", "", query.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized in self.SKIP_RETRIEVAL_EXACT

    def needs_condensation(self, query: str, chat_history: list[dict]) -> bool:
        """
        Quick check if query needs condensation.

        Returns True if:
        - Query contains pronouns/demonstratives
        - Query contains structural references (Point 2, Part A, Para 3)
        - Query is very short (likely follow-up)
        - Query contains elaboration/clarification phrases
        - Chat history exists and query seems dependent
        """
        if not chat_history:
            return False

        query_lower = query.lower().strip()

        # Check for context indicators
        if any(indicator in query_lower for indicator in self.CONTEXT_INDICATORS):
            return True

        # Check for structural references (Point 2, Part A, Para 3, etc.)
        if self.STRUCTURAL_REF_PATTERN.search(query):
            return True

        # Check for short queries. Script-aware: `.split()` counts every CJK
        # sentence as 1 word, so EVERY Chinese query used to land here.
        if count_words(query_lower) < self.SHORT_QUERY_THRESHOLD:
            return True

        # Check for elaboration phrases
        if any(phrase in query_lower for phrase in self.ELABORATION_PHRASES):
            return True

        # Check for clarification phrases
        return bool(any(phrase in query_lower for phrase in self.CLARIFICATION_PHRASES))

    def _classify_query_type(self, query: str, chat_history: list[dict]) -> QueryType:
        """Rule-based query type classification (fast, no LLM call)."""
        # CRITICAL: Check conversational FIRST before anything else
        # This works with or without chat history
        if self._is_pure_conversational(query):
            return QueryType.CONVERSATIONAL

        if not chat_history:
            return QueryType.STANDALONE

        query_lower = query.lower().strip()

        # Check clarification
        if any(phrase in query_lower for phrase in self.CLARIFICATION_PHRASES):
            return QueryType.CLARIFICATION

        # Check elaboration
        if any(phrase in query_lower for phrase in self.ELABORATION_PHRASES):
            return QueryType.ELABORATION

        # Check for explicit references (including structural: Point 2, Part A)
        if any(word in query_lower for word in ["earlier", "previous", "mentioned", "above"]):
            return QueryType.REFERENCE
        if self.STRUCTURAL_REF_PATTERN.search(query):
            return QueryType.REFERENCE

        # Check for pronouns
        if any(indicator in query_lower for indicator in self.CONTEXT_INDICATORS):
            return QueryType.FOLLOW_UP

        # Short queries are likely follow-ups (script-aware; see above)
        if count_words(query_lower) < self.SHORT_QUERY_THRESHOLD:
            return QueryType.FOLLOW_UP

        return QueryType.STANDALONE

    def _to_messages(self, chat_history: list[dict]) -> list[Message]:
        """Convert chat history to messages.

        Supports ``{"role": "user"|"assistant", "content": "..."}`` and
        ``{"user": "...", "assistant": "..."}`` turns.
        """
        messages: list[Message] = []
        for turn in chat_history[-self.max_history_turns :]:
            if "role" in turn:
                if turn["role"] in ("user", "assistant"):
                    messages.append(Message(turn["role"], turn.get("content", "")))
            elif "user" in turn:
                messages.append(Message("user", turn["user"]))
                if turn.get("assistant"):
                    messages.append(Message("assistant", turn["assistant"]))
        return messages

    def _make_cache_key(self, query: str, chat_history: list[dict], scope: str = "") -> str:
        """Generate a scope-keyed cache key.

        ``scope`` is an opaque string the caller chooses (a user, tenant or session id, or
        empty). It is part of the key so entries from different scopes never collide.
        """
        history_parts = []
        for turn in chat_history[-self.max_history_turns :]:
            if "role" in turn:
                history_parts.append(f"{turn['role']}:{turn.get('content', '')}")
            elif "user" in turn:
                history_parts.append(f"user:{turn['user']}")
                if turn.get("assistant"):
                    history_parts.append(f"assistant:{turn['assistant']}")
        combined = f"{scope}:{self.use_fast_prompt}:{query}:{'|'.join(history_parts)}"
        return f"condense:{hashlib.sha256(combined.encode()).hexdigest()[:32]}"

    def _extract_entities_from_history(self, chat_history: list[dict]) -> list[str]:
        """
        Extract key entities (case numbers, sections, names) from history.
        These are important for BM25 keyword matching.
        """
        entities = []

        for turn in chat_history[-self.max_history_turns :]:
            content = ""
            if "content" in turn:
                content = turn["content"]
            elif "user" in turn:
                content = turn["user"] + " " + turn.get("assistant", "")

            # Extract case numbers (e.g., 1239/2023, 2024 WL 123)
            case_patterns = [
                r"\d+/\d{4}",  # 1239/2023 (year-numbered docket)
                r"\d{4}\s+\w+\s+\d+",  # 2024 WL 123
                r"Case\s+No\.?\s*[\w\d/-]+",  # Case No. XYZ
            ]
            for pattern in case_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                entities.extend(matches)

            # Extract section references (Section 420, Article 21)
            section_patterns = [
                r"Section\s+\d+[\w]*",  # Section 420, Section 302A
                r"Article\s+\d+[\w]*",  # Article 21
                r"Rule\s+\d+[\w]*",  # Rule 37
            ]
            for pattern in section_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                entities.extend(matches)

            # Extract capitalized proper nouns (likely names/entities)
            words = content.split()
            for word in words:
                clean_word = word.strip(".,!?:;\"'()[]{}")
                if (
                    clean_word
                    and len(clean_word) > 2
                    and clean_word[0].isupper()
                    and clean_word
                    not in {
                        "The",
                        "This",
                        "That",
                        "What",
                        "How",
                        "Why",
                        "When",
                        "Where",
                        "User",
                        "Assistant",
                    }
                ):
                    entities.append(clean_word)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities[:15]  # Top 15 entities

    async def condense(
        self,
        query: str,
        chat_history: list[dict],
        force: bool = False,
        scope: str = "",
    ) -> CondensationResult:
        """Condense a query using conversation history.

        Args:
            query: The user's current query.
            chat_history: Previous conversation turns.
            force: Force condensation even if not needed.
            scope: Opaque cache scope (see ``_make_cache_key``).
        """
        start_time = time.time()

        def _result(**kw: Any) -> CondensationResult:
            kw.setdefault("original_query", query)
            kw.setdefault("entities_resolved", [])
            kw.setdefault("cache_hit", False)
            kw["processing_time_ms"] = (time.time() - start_time) * 1000
            return CondensationResult(**kw)

        # Conversational queries skip retrieval even without chat history.
        if self._is_pure_conversational(query):
            logger.debug("condense_skipped", reason="conversational_query")
            return _result(
                condensed_query=query,
                query_type=QueryType.CONVERSATIONAL,
                confidence=0.95,
                requires_retrieval=False,
            )

        if not force and not self.needs_condensation(query, chat_history):
            return _result(
                condensed_query=query,
                query_type=QueryType.STANDALONE,
                confidence=1.0,
                requires_retrieval=True,
            )

        cache_key = self._make_cache_key(query, chat_history, scope)
        if self.cache is not None:
            try:
                cached = await self.cache.get(cache_key)
                if cached:
                    return _result(
                        condensed_query=cached.decode("utf-8"),
                        query_type=self._classify_query_type(query, chat_history),
                        confidence=0.95,
                        requires_retrieval=True,
                        entities_resolved=self._extract_entities_from_history(chat_history),
                        cache_hit=True,
                    )
            except Exception as e:
                logger.warning("condense_cache_error", error=str(e))

        query_type = self._classify_query_type(query, chat_history)

        # Clarification requests need no new retrieval.
        if query_type == QueryType.CLARIFICATION:
            return _result(
                condensed_query=query,
                query_type=query_type,
                confidence=0.9,
                requires_retrieval=False,
            )

        history = self._to_messages(chat_history)
        system = FAST_SYSTEM_PROMPT if self.use_fast_prompt else self.system_prompt
        try:
            output = await self.llm.generate_structured(
                [Message("system", system), *history, Message("user", query)],
                CondensedQueryOutput,
                max_tokens=self.max_tokens,
            )
            condensed_query = output.primary_query.strip()
            sub_queries = [sq.strip() for sq in output.sub_queries if sq.strip()]

            if self.usage_hook is not None:
                history_text = " ".join(m.content for m in history)
                self.usage_hook(
                    UsageEvent(
                        operation="query_condensation",
                        input_tokens=(len(history_text) + len(query)) // 4 + 200,
                        output_tokens=(len(condensed_query) + sum(map(len, sub_queries))) // 4,
                    )
                )

            if not condensed_query or len(condensed_query) < 3:
                condensed_query = query

            if self.cache is not None:
                try:
                    await self.cache.set(
                        cache_key,
                        condensed_query.encode("utf-8"),
                        ttl_seconds=self.cache_ttl_seconds,
                    )
                except Exception as e:
                    logger.warning("condense_cache_set_error", error=str(e))

            entities = self._extract_entities_from_history(chat_history)
            logger.debug(
                "condense_completed",
                query_type=query_type.value,
                entities_count=len(entities),
                sub_queries_count=len(sub_queries),
                history_turns=len(chat_history),
            )
            return _result(
                condensed_query=condensed_query,
                query_type=query_type,
                confidence=0.9,
                requires_retrieval=True,
                entities_resolved=entities,
                sub_queries=sub_queries,
                response_language=(output.response_language or None),
            )
        except Exception as e:
            logger.error("condense_failed", error=str(e))
            return _result(
                condensed_query=self._simple_fallback(query, chat_history),
                query_type=query_type,
                confidence=0.5,
                requires_retrieval=True,
            )

    def _simple_fallback(self, query: str, chat_history: list[dict]) -> str:
        """
        Simple rule-based fallback when LLM fails.
        Concatenates recent context with current query.
        """
        if not chat_history:
            return query

        # Get last user message
        last_user_msg = ""
        for turn in reversed(chat_history):
            if turn.get("role") == "user":
                last_user_msg = turn.get("content", "")
                break
            elif turn.get("user"):
                last_user_msg = turn["user"]
                break

        if last_user_msg and len(query.split()) < 5:
            # For very short queries, combine with last message
            return f"{last_user_msg} {query}"

        return query

    def condense_sync(
        self, query: str, chat_history: list[dict], force: bool = False, scope: str = ""
    ) -> CondensationResult:
        """Synchronous version of condense for contexts without an event loop."""
        import asyncio

        return asyncio.run(self.condense(query, chat_history, force=force, scope=scope))
