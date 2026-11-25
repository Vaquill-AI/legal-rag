"""Source mixer: combine primary retrieval with supplementary results under a token budget.

Used after CRAG grades a retrieval as thin. Primary chunks are never replaced, only joined
by supplementary ones, and both are truncated to fit the budget.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from legal_rag.types import RetrievedChunk

logger = logging.getLogger("legal_rag.retrieval.source_mixer")

TokenCounter = Callable[[str], int]


def default_token_counter() -> TokenCounter:
    """Count tokens with tiktoken when installed, else about four characters per token."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoder.encode(text))
    except Exception:  # noqa: BLE001 - optional dependency or offline encoding data
        return lambda text: max(1, len(text) // 4)


class MixStrategy(StrEnum):
    """How the budget is shared between the two kinds of source."""

    PRIMARY_LED = "primary_led"
    SUPPLEMENTARY_LED = "supplementary_led"
    BALANCED = "balanced"


@dataclass
class MixedContext:
    """Truncated chunks from both kinds of source, with attribution counts."""

    primary_chunks: list[RetrievedChunk]
    supplementary_chunks: list[RetrievedChunk]
    mix_strategy: MixStrategy
    source_summary: dict[str, int]
    total_tokens: int
    primary_token_budget: int
    supplementary_token_budget: int
    truncation_applied: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceMixerConfig:
    """Budget and truncation settings."""

    max_total_tokens: int = 16000
    primary_token_ratio: float = 0.6
    min_primary_tokens: int = 500
    min_supplementary_tokens: int = 300
    header_tokens: int = 20
    header_overhead: int = 50
    smart_truncation: bool = True
    chars_per_token: int = 4
    led_share: float = 0.7
    """Share of the budget given to the leading source for the two ``*_LED`` strategies."""


class SourceMixer:
    """Share a token budget between primary and supplementary chunks."""

    def __init__(
        self,
        config: SourceMixerConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or SourceMixerConfig()
        self._count = token_counter or default_token_counter()

    def _chunk_tokens(self, chunk: RetrievedChunk) -> int:
        return self._count(chunk.text) + self.config.header_tokens

    def _truncate(self, chunks: list[RetrievedChunk], max_tokens: int) -> list[RetrievedChunk]:
        """Keep whole chunks while they fit, then one truncated chunk (start and end kept)."""
        result: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            tokens = self._chunk_tokens(chunk)
            if used + tokens > max_tokens:
                remaining_chars = (max_tokens - used) * self.config.chars_per_token
                text = chunk.text
                if (
                    self.config.smart_truncation
                    and remaining_chars >= 300
                    and len(text) > remaining_chars
                ):
                    first = int(remaining_chars * 0.6)
                    last = int(remaining_chars * 0.4) - 20
                    cut = (
                        text[:first] + "\n...[truncated]...\n" + text[-last:]
                        if last > 0
                        else text[:first] + "...[truncated]"
                    )
                    result.append(
                        replace(
                            chunk,
                            text=cut,
                            metadata={
                                **chunk.metadata,
                                "truncated": True,
                                "original_length": len(text),
                            },
                        )
                    )
                break
            result.append(chunk)
            used += tokens
        return result

    def choose_strategy(
        self,
        primary: list[RetrievedChunk],
        supplementary: list[RetrievedChunk],
        primary_confidence: float,
    ) -> MixStrategy:
        """Pick a strategy from what exists and how much the primary results can be trusted."""
        if not primary:
            return MixStrategy.SUPPLEMENTARY_LED
        if not supplementary or primary_confidence >= 0.7:
            return MixStrategy.PRIMARY_LED
        if primary_confidence <= 0.3:
            return MixStrategy.SUPPLEMENTARY_LED
        return MixStrategy.BALANCED

    def mix(
        self,
        primary_chunks: list[RetrievedChunk],
        supplementary_chunks: list[RetrievedChunk],
        strategy: MixStrategy | None = None,
        primary_confidence: float = 0.5,
    ) -> MixedContext:
        """Truncate both lists to their budgets and return them with attribution counts."""
        strategy = strategy or self.choose_strategy(
            primary_chunks, supplementary_chunks, primary_confidence
        )
        cfg = self.config
        available = cfg.max_total_tokens - cfg.header_overhead
        if strategy is MixStrategy.PRIMARY_LED:
            p_budget = int(available * cfg.led_share)
        elif strategy is MixStrategy.SUPPLEMENTARY_LED:
            p_budget = available - int(available * cfg.led_share)
        else:
            p_budget = int(available * cfg.primary_token_ratio)
        s_budget = available - p_budget
        p_budget = max(p_budget, cfg.min_primary_tokens) if primary_chunks else 0
        s_budget = max(s_budget, cfg.min_supplementary_tokens) if supplementary_chunks else 0

        primary = self._truncate(primary_chunks, p_budget)
        supplementary = self._truncate(supplementary_chunks, s_budget)
        truncated = (
            len(primary) < len(primary_chunks)
            or len(supplementary) < len(supplementary_chunks)
            or any(c.metadata.get("truncated") for c in primary + supplementary)
        )
        total = sum(self._chunk_tokens(c) for c in primary + supplementary) + cfg.header_overhead
        return MixedContext(
            primary_chunks=primary,
            supplementary_chunks=supplementary,
            mix_strategy=strategy,
            source_summary={"primary": len(primary), "supplementary": len(supplementary)},
            total_tokens=total,
            primary_token_budget=p_budget,
            supplementary_token_budget=s_budget,
            truncation_applied=truncated,
            metadata={
                "original_primary_count": len(primary_chunks),
                "original_supplementary_count": len(supplementary_chunks),
                "primary_confidence": primary_confidence,
            },
        )

    def format_mixed_context(self, mixed: MixedContext) -> str:
        """Render the mix for a model, with labelled sections and a short note on the balance."""
        sections: list[str] = []
        if mixed.primary_chunks:
            parts = []
            for i, chunk in enumerate(mixed.primary_chunks, 1):
                source = chunk.metadata.get("filename", chunk.metadata.get("document_id", chunk.id))
                note = " [truncated]" if chunk.metadata.get("truncated") else ""
                parts.append(f"[Source {i}: {source}{note}]\n{chunk.text}")
            sections.append("## PRIMARY SOURCES:\n\n" + "\n\n---\n\n".join(parts))
        if mixed.supplementary_chunks:
            parts = []
            for i, chunk in enumerate(mixed.supplementary_chunks, 1):
                url = chunk.metadata.get("url", "Unknown source")
                title = chunk.metadata.get("title", "Supplementary result")
                note = " [truncated]" if chunk.metadata.get("truncated") else ""
                parts.append(f"[Supplementary {i}: {title}{note}]\nURL: {url}\n{chunk.text}")
            sections.append("## SUPPLEMENTARY SOURCES:\n\n" + "\n\n---\n\n".join(parts))
        if mixed.primary_chunks and mixed.supplementary_chunks:
            notes = {
                MixStrategy.PRIMARY_LED: "The primary sources hold the main information. "
                "Supplementary results add context.",
                MixStrategy.SUPPLEMENTARY_LED: "The primary sources are limited on this topic. "
                "Supplementary results add context.",
                MixStrategy.BALANCED: "Information is combined from primary and supplementary sources.",
            }
            sections.append("> Note: " + notes[mixed.mix_strategy])
        return "\n\n".join(sections)
