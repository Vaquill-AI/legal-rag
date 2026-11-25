"""Multi-query expansion.

A model rewrites one query into several variations so retrieval can fan out and the
results can be merged (see ``legal_rag.retrieval.fusion``).
Two entry points share one implementation:

* ``MultiQueryGenerator`` for faceted expansion with a prompt template.
* ``generate_rewrites`` for plain paraphrase rewrites with an optional cache and timeout.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, fingerprint, safe_error_fields
from legal_rag.retrieval.strategies.multi_query_prompts import (
    GENERAL_MULTI_QUERY_PROMPT,
    LEGAL_FACET_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)
from legal_rag.types import Message

logger = logging.getLogger("legal_rag.retrieval.multi_query")


class QueryVariations(BaseModel):
    """Structured output: the generated query variations."""

    queries: list[str] = Field(default_factory=list)


@dataclass
class MultiQueryConfig:
    """Configuration for multi-query generation."""

    max_tokens: int = 512
    num_queries: int = 3
    include_original: bool = True
    domain: str = "legal"
    prompt_template: str | None = None
    """Overrides the built-in template for ``domain``. Uses ``{query}`` and ``{num_queries}``."""
    min_query_words: int = 5
    deduplicate: bool = True


@dataclass
class MultiQueryResult:
    """Outcome of multi-query generation. On failure ``all_queries`` is the original alone."""

    original_query: str
    query_id: str
    generated_queries: list[str]
    all_queries: list[str]
    generation_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiQueryGenerator:
    """Generate query variations with an injected language model."""

    def __init__(
        self,
        llm: LanguageModel,
        config: MultiQueryConfig | None = None,
        *,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or MultiQueryConfig()
        self.usage_hook = usage_hook
        self.stats = {"generated": 0, "failed": 0}

    def _template(self) -> str:
        if self.config.prompt_template:
            return self.config.prompt_template
        return LEGAL_FACET_PROMPT if self.config.domain == "legal" else GENERAL_MULTI_QUERY_PROMPT

    async def generate(self, query: str, context: dict[str, Any] | None = None) -> MultiQueryResult:
        """Generate variations. Never raises: failure returns the original query alone."""
        start = time.monotonic()
        query_id = fingerprint(query)
        try:
            prompt = (
                self._template()
                .replace("{query}", query)
                .replace("{num_queries}", str(self.config.num_queries))
            )
            result = await self.llm.generate_structured(
                [Message("user", prompt)], QueryVariations, max_tokens=self.config.max_tokens
            )
            emit_usage(
                self.usage_hook,
                "multi_query_generation",
                input_text=prompt,
                output_text="\n".join(result.queries),
            )
            generated = result.queries[: self.config.num_queries]
            generated = [q for q in generated if len(q.split()) >= self.config.min_query_words]
            if self.config.deduplicate:
                generated = _deduplicate(generated, seed=query)
            all_queries = [query, *generated] if self.config.include_original else generated
            self.stats["generated"] += 1
            return MultiQueryResult(
                original_query=query,
                query_id=query_id,
                generated_queries=generated,
                all_queries=all_queries,
                generation_time_ms=(time.monotonic() - start) * 1000,
                success=True,
                metadata={"domain": self.config.domain, "context_provided": context is not None},
            )
        except Exception as error:  # noqa: BLE001 - fall back to the original query
            self.stats["failed"] += 1
            logger.warning(
                "multi-query generation failed query_id=%s %s", query_id, safe_error_fields(error)
            )
            return MultiQueryResult(
                original_query=query,
                query_id=query_id,
                generated_queries=[],
                all_queries=[query],
                generation_time_ms=(time.monotonic() - start) * 1000,
                success=False,
                metadata={"error": str(error)[:160]},
            )


def _deduplicate(queries: list[str], *, seed: str | None = None) -> list[str]:
    """Drop case-insensitive duplicates, and anything equal to ``seed`` when it is given."""
    seen = {seed.lower().strip()} if seed else set()
    unique: list[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _rewrite_cache_key(question: str) -> str:
    normalised = re.sub(r"\s+", " ", question.strip().lower())[:512]
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


async def generate_rewrites(
    llm: LanguageModel,
    question: str,
    *,
    n: int = 3,
    timeout: float = 18.0,
    cache: MutableMapping[str, list[str]] | None = None,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
    usage_hook: UsageHook | None = None,
) -> list[str]:
    """Return up to ``n`` paraphrased rewrites of ``question``.

    Any failure (timeout, model error, empty output) returns ``[]`` so the caller proceeds
    with the original query alone. Pass a mapping as ``cache`` to avoid paying twice for the same text.
    """
    text = (question or "").strip()
    if not text or n <= 0:
        return []
    key = _rewrite_cache_key(text)
    if cache is not None and key in cache:
        return cache[key]
    try:
        messages = [
            Message("system", system_prompt.replace("{num_queries}", str(n))),
            Message("user", f"Question:\n{text[:1000]}"),
        ]
        result = await asyncio.wait_for(
            llm.generate_structured(messages, QueryVariations, max_tokens=1600), timeout=timeout
        )
        emit_usage(
            usage_hook,
            "multi_query_rewrite",
            input_text=text,
            output_text="\n".join(result.queries),
        )
        rewrites = [r[:500] for r in _deduplicate([q.strip() for q in result.queries], seed=text)]
        rewrites = rewrites[:n]
    except Exception as error:  # noqa: BLE001 - degrade to the original query
        logger.warning("rewrite generation failed %s", safe_error_fields(error))
        rewrites = []
    if cache is not None:
        cache[key] = rewrites
    return rewrites
