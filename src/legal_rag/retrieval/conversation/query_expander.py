"""Query expansion for ambiguous or incomplete queries.

Uses an injected language model to add missing context and specificity to vague queries, with a
rule-based synonym fallback when the model fails. The cache and metering hook are injected.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import Cache, LanguageModel, UsageHook
from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import Message, UsageEvent

logger = get_logger("conversation.query_expander")


class QueryExpansions(BaseModel):
    """Structured output of the expansion call."""

    expansions: list[str] = Field(default_factory=list)


@dataclass
class QueryExpansionResult:
    """Result of query expansion."""

    original_query: str
    expanded_queries: list[str]
    expansion_applied: bool
    expansion_type: str  # "synonym", "specificity", "decomposition", "none"
    confidence: float
    processing_time_ms: float


EXPANSION_SYSTEM_PROMPT = """<system>
<role>Legal query expansion specialist for legal research RAG retrieval optimization.</role>

<expansion_protocol>
<objective>Transform ambiguous or incomplete queries into precise, retrieval-optimized variations for legal document search.</objective>

<rules>
<rule name="jurisdiction_specificity">Add jurisdiction or specific statute context to generic legal terms when identifiable from the query</rule>
<rule name="domain_enrichment">Map vague terms to legal domain terminology (e.g., "slip and fall" -> "premises liability under state tort law")</rule>
<rule name="citation_preservation">NEVER modify case numbers, section references, or citation formats</rule>
<rule name="variation_ordering">Most specific variation first, progressively broader</rule>
<rule name="context_dependency">If query requires chat history (e.g., "that section", "what about it"), return unchanged</rule>
</rules>

<output_format>
- Generate 2-3 numbered variations
- Format: 1. [most specific] 2. [alternative framing] 3. [broader context]
- Use standard legal terminology appropriate to the jurisdiction
</output_format>
</expansion_protocol>

<examples>
<example input="damages">
1. damages for breach of contract in civil law
2. calculation and assessment of damages in civil suits
3. types of damages - compensatory, punitive, liquidated, nominal
</example>

<example input="punishment for murder">
1. punishment for murder under criminal law statutes
2. sentencing guidelines for murder convictions
3. death penalty vs life imprisonment for murder
</example>

<example input="company registration">
1. incorporation and registration of company under companies legislation
2. procedure for company registration
3. documents required for private limited company registration
</example>

<example input="that section" context_required="true">
[return unchanged - requires conversation context]
</example>
</examples>
</system>"""


class QueryExpander:
    """
    Expands ambiguous queries for better retrieval.

    Uses LLM to add missing context and specificity to vague queries.
    """

    def __init__(
        self,
        llm: LanguageModel,
        cache: Cache | None = None,
        usage_hook: UsageHook | None = None,
        max_tokens: int = 512,
        cache_ttl_seconds: int = 7200,
    ):
        """
        Args:
            llm: Language model used for expansion (structured output).
            cache: Optional cache for expanded queries.
            usage_hook: Optional metering callback.
            max_tokens: Output token budget.
            cache_ttl_seconds: Lifetime of cached expansions.
        """
        self.llm = llm
        self.cache = cache
        self.usage_hook = usage_hook
        self.max_tokens = max_tokens
        self.cache_ttl_seconds = cache_ttl_seconds

    # NOTE: _parse_expansion_result() removed - structured output handles parsing

    async def expand(
        self,
        query: str,
        ambiguity_score: float = 0.0,
        is_legal: bool = True,
    ) -> QueryExpansionResult:
        """
        Expand an ambiguous query.

        Args:
            query: Original query
            ambiguity_score: Ambiguity score (0.0-1.0)
            is_legal: Whether query is in legal domain

        Returns:
            QueryExpansionResult with expanded variations
        """
        start_time = time.time()

        # Only expand if ambiguous and in-domain
        if ambiguity_score < 0.4 or not is_legal:
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=[query],
                expansion_applied=False,
                expansion_type="none",
                confidence=1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Check cache
        if self.cache:
            cache_key = f"expand:{query}"
            try:
                raw = await self.cache.get(cache_key)
                cached = json.loads(raw) if raw else None
                if cached:
                    logger.debug("expand_cache_hit", query_preview=query[:50])
                    return QueryExpansionResult(
                        original_query=query,
                        expanded_queries=cached,
                        expansion_applied=True,
                        expansion_type="cached",
                        confidence=0.9,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
            except Exception as e:
                logger.warning("expand_cache_error", error=str(e))

        # Generate expansions using LLM with structured output
        try:
            result = await self.llm.generate_structured(
                [Message("system", EXPANSION_SYSTEM_PROMPT), Message("user", query)],
                QueryExpansions,
                max_tokens=self.max_tokens,
            )
            if self.usage_hook is not None:
                self.usage_hook(
                    UsageEvent(
                        operation="query_expansion",
                        input_tokens=len(query) // 4 + 200,
                        output_tokens=80,
                    )
                )

            if result is None or not result.expansions:
                raise ValueError("LLM returned empty structured output")
            expanded_queries = result.expansions[:3]

            # Add original as fallback
            if query not in expanded_queries:
                expanded_queries.append(query)

            # Cache result
            if self.cache and expanded_queries:
                try:
                    await self.cache.set(
                        cache_key,
                        json.dumps(expanded_queries).encode("utf-8"),
                        ttl_seconds=self.cache_ttl_seconds,
                    )
                except Exception as e:
                    logger.warning("expand_cache_set_error", error=str(e))

            logger.debug(
                "expand_completed",
                original_preview=query[:50],
                num_variations=len(expanded_queries),
                ambiguity_score=round(ambiguity_score, 2),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

            return QueryExpansionResult(
                original_query=query,
                expanded_queries=expanded_queries,
                expansion_applied=True,
                expansion_type="specificity",
                confidence=0.8,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("expand_failed", error=str(e), query_preview=query[:50])

            # Fallback: simple synonym expansion
            fallback_queries = self._simple_synonym_expansion(query)

            return QueryExpansionResult(
                original_query=query,
                expanded_queries=fallback_queries,
                expansion_applied=len(fallback_queries) > 1,
                expansion_type="synonym_fallback",
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _simple_synonym_expansion(self, query: str) -> list[str]:
        """
        Simple rule-based synonym expansion (fallback when LLM fails).

        Adds legal domain context to common terms.
        """
        query_lower = query.lower()
        variations = [query]  # Always include original

        # Legal domain expansions
        expansions = {
            "damages": [
                f"{query} in contract law",
                f"{query} under tort law",
                f"calculation of {query}",
            ],
            "punishment": [
                f"{query} under criminal law statutes",
                f"{query} provisions in criminal law",
            ],
            "liability": [
                f"{query} in civil law",
                f"{query} under tort law",
            ],
            "jurisdiction": [
                f"{query} of courts",
                f"{query} under civil procedure",
            ],
        }

        for term, expansions_list in expansions.items():
            if term in query_lower:
                variations.extend(expansions_list[:2])
                break

        return variations[:3]
