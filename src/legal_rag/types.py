"""Shared data types used across legal-rag components.

These types carry no tenancy keys and no product fields.
Anything specific to a deployment goes in the free-form ``metadata`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """One chat message passed to a language model."""

    role: str
    content: str


@dataclass
class RetrievedChunk:
    """A unit of text returned by retrieval, with a score and free-form metadata."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking one document, by its index in the input list."""

    index: int
    score: float


@dataclass(frozen=True)
class UsageEvent:
    """Optional metering event. Providers and components may emit these."""

    operation: str
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
