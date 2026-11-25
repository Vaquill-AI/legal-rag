"""HyDE: Hypothetical Document Embeddings.

A model writes a passage that would answer the query, and that passage (not the question)
is embedded for retrieval. Answer-shaped text sits closer to answer-bearing documents than a
question does, which reduces vocabulary mismatch.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022).

This module is the single HyDE implementation. It has two layers:

* ``HyDEGenerator`` writes the hypotheses: prompt, model call, retries, length check, metering,
  and a fall back to the raw query. It returns a ``HyDEResult``.
* ``HyDETransformer`` is the conversation-facing wrapper. It decides whether a query is worth
  a model call (``should_use_hyde``), delegates the generation to ``HyDEGenerator`` and returns
  the text to embed as a ``HyDETransformResult``.

``legal_rag.retrieval.conversation.hyde`` re-exports the transformer under its old names. Its
``HyDEResult`` is ``HyDETransformResult``, a different shape from this module's ``HyDEResult``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from legal_rag.providers.protocols import LanguageModel, UsageHook
from legal_rag.retrieval.strategies._common import emit_usage, fingerprint, safe_error_fields
from legal_rag.retrieval.strategies.hyde_prompts import (
    GENERAL_HYDE_PROMPT,
    HYDE_GENERAL_SYSTEM_PROMPT,
    HYDE_LEGAL_SYSTEM_PROMPT,
    LEGAL_HYDE_PROMPT,
)
from legal_rag.types import Message

logger = logging.getLogger("legal_rag.retrieval.hyde")


@dataclass
class HyDEConfig:
    """Configuration for hypothesis generation."""

    temperature: float = 0.7
    max_tokens: int = 300
    num_hypotheses: int = 1
    use_multi_hypothesis: bool = False
    domain: str = "legal"
    prompt_template: str | None = None
    """Overrides the built-in template for ``domain``. Must contain ``{query}``."""
    system_prompt: str | None = None
    """When set, sent as a system message with the raw query as the user message.
    Takes precedence over ``prompt_template`` and ``domain``."""
    min_length: int = 50
    max_retries: int = 2


@dataclass
class HyDEResult:
    """Outcome of hypothesis generation. On failure the original query stands in."""

    original_query: str
    hypotheses: list[str]
    primary_hypothesis: str
    generation_time_ms: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class HyDEGenerator:
    """Generate hypothetical documents for a query with an injected language model."""

    def __init__(
        self,
        llm: LanguageModel,
        config: HyDEConfig | None = None,
        *,
        usage_hook: UsageHook | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or HyDEConfig()
        self.usage_hook = usage_hook
        self.stats = {"generated": 0, "failed": 0, "retries": 0}

    def _template(self) -> str:
        if self.config.prompt_template:
            return self.config.prompt_template
        return LEGAL_HYDE_PROMPT if self.config.domain == "legal" else GENERAL_HYDE_PROMPT

    def _messages(self, query: str) -> list[Message]:
        if self.config.system_prompt:
            return [Message("system", self.config.system_prompt), Message("user", query)]
        return [Message("user", self._template().replace("{query}", query))]

    async def generate(self, query: str, context: dict[str, Any] | None = None) -> HyDEResult:
        """Generate hypotheses for ``query``. Never raises: failure returns the query itself."""
        start = time.monotonic()
        try:
            if self.config.use_multi_hypothesis and self.config.num_hypotheses > 1:
                hypotheses = await self._generate_multiple(query)
            else:
                hypotheses = [await self._generate_single(query)]
            hypotheses = [h for h in hypotheses if len(h.strip()) >= self.config.min_length]
            if not hypotheses:
                raise ValueError("No valid hypotheses generated (all below min_length)")
            elapsed = (time.monotonic() - start) * 1000
            self.stats["generated"] += 1
            return HyDEResult(
                original_query=query,
                hypotheses=hypotheses,
                primary_hypothesis=hypotheses[0],
                generation_time_ms=elapsed,
                success=True,
                metadata={
                    "num_hypotheses": len(hypotheses),
                    "domain": self.config.domain,
                    "context_provided": context is not None,
                },
            )
        except Exception as error:  # noqa: BLE001 - fall back to the raw query
            self.stats["failed"] += 1
            logger.warning(
                "hyde generation failed query_id=%s %s",
                fingerprint(query),
                safe_error_fields(error),
            )
            return HyDEResult(
                original_query=query,
                hypotheses=[query],
                primary_hypothesis=query,
                generation_time_ms=(time.monotonic() - start) * 1000,
                success=False,
                metadata={"error": str(error)[:160]},
            )

    async def _generate_single(self, query: str) -> str:
        messages = self._messages(query)
        prompt = "\n".join(m.content for m in messages)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                text = (
                    await self.llm.generate(
                        messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                ).strip()
                emit_usage(self.usage_hook, "hyde_generation", input_text=prompt, output_text=text)
                if len(text) >= self.config.min_length:
                    return text
                self.stats["retries"] += 1
            except Exception as error:  # noqa: BLE001 - retry, then surface the last error
                last_error = error
                self.stats["retries"] += 1
        raise ValueError(
            f"No valid hypothesis after {self.config.max_retries + 1} attempts"
        ) from last_error

    async def _generate_multiple(self, query: str) -> list[str]:
        results = await asyncio.gather(
            *(self._generate_single(query) for _ in range(self.config.num_hypotheses)),
            return_exceptions=True,
        )
        good = [r for r in results if isinstance(r, str)]
        return good or [query]


@dataclass
class HyDETransformResult:
    """Outcome of ``HyDETransformer.transform``: the text to embed in place of the query.

    ``used_hyde`` is False when the gate skipped the model call or generation failed. In both
    cases ``combined_query`` is the original query, so callers can embed it unconditionally.
    """

    original_query: str
    hypothetical_document: str
    combined_query: str  # For retrieval
    processing_time_ms: float
    used_hyde: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "hypothetical_document": self.hypothetical_document,
            "combined_query": self.combined_query,
            "processing_time_ms": self.processing_time_ms,
            "used_hyde": self.used_hyde,
        }


class HyDETransformer:
    """Query transform for conversations: gate, generate through ``HyDEGenerator``, combine.

    Process:
    1. Decide whether the query benefits from HyDE (``should_use_hyde``), unless forced.
    2. Generate a hypothetical answer with ``HyDEGenerator`` (one attempt, system prompt).
    3. Combine it with the original query, or return it alone, for embedding.

    This helps when:
    - Query is vague or abstract
    - Semantic gap between query and documents
    - "Tell me about X" type queries

    Any generation failure, including an empty reply, returns the original query unchanged.
    """

    # Query patterns that benefit from HyDE
    HYDE_BENEFICIAL_PATTERNS: ClassVar[set[str]] = {
        "tell me about",
        "explain",
        "what is",
        "describe",
        "overview of",
        "summary of",
        "how does",
        "why is",
    }

    def __init__(
        self,
        llm: LanguageModel,
        use_legal_prompt: bool = True,
        include_original: bool = True,
        usage_hook: UsageHook | None = None,
        max_tokens: int = 400,
        temperature: float = 0.7,
    ):
        """
        Args:
            llm: Language model that writes the hypothetical document.
            use_legal_prompt: Use the legal-domain prompt.
            include_original: Include the original query in the combined output.
            usage_hook: Optional metering callback.
            max_tokens: Output token budget.
            temperature: Sampling temperature for the hypothesis.
        """
        self.llm = llm
        self.use_legal_prompt = use_legal_prompt
        self.include_original = include_original
        self.usage_hook = usage_hook
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _generator(self) -> HyDEGenerator:
        # Built per call so that changing an attribute after construction still takes effect.
        system = HYDE_LEGAL_SYSTEM_PROMPT if self.use_legal_prompt else HYDE_GENERAL_SYSTEM_PROMPT
        config = HyDEConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            domain="legal" if self.use_legal_prompt else "general",
            system_prompt=system,
            min_length=1,
            max_retries=0,
        )
        return HyDEGenerator(self.llm, config, usage_hook=self.usage_hook)

    def should_use_hyde(self, query: str) -> bool:
        """
        Determine if query would benefit from HyDE.

        HyDE is most useful for:
        - Abstract/vague queries
        - "Tell me about" type queries
        - When initial retrieval quality is low
        """
        query_lower = query.lower()

        # Check for beneficial patterns
        for pattern in self.HYDE_BENEFICIAL_PATTERNS:
            if pattern in query_lower:
                return True

        # Very short queries might benefit
        return len(query.split()) < 4

    async def transform(self, query: str, force: bool = False) -> HyDETransformResult:
        """
        Transform query using HyDE.

        Args:
            query: Original query
            force: Force HyDE even if not recommended

        Returns:
            HyDETransformResult with the hypothetical document
        """
        start_time = time.monotonic()

        if not force and not self.should_use_hyde(query):
            return self._unchanged(query, start_time)

        generated = await self._generator().generate(query)
        if not generated.success:
            return self._unchanged(query, start_time)

        hypothetical = generated.primary_hypothesis
        combined = f"{query}\n\n{hypothetical}" if self.include_original else hypothetical
        return HyDETransformResult(
            original_query=query,
            hypothetical_document=hypothetical,
            combined_query=combined,
            processing_time_ms=(time.monotonic() - start_time) * 1000,
            used_hyde=True,
        )

    @staticmethod
    def _unchanged(query: str, start_time: float) -> HyDETransformResult:
        return HyDETransformResult(
            original_query=query,
            hypothetical_document="",
            combined_query=query,
            processing_time_ms=(time.monotonic() - start_time) * 1000,
            used_hyde=False,
        )

    def transform_sync(self, query: str, force: bool = False) -> HyDETransformResult:
        """Synchronous version."""
        return asyncio.run(self.transform(query, force))
