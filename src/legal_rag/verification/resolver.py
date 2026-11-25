"""Case resolution protocol used by the verification components.

``legal_rag.providers`` holds the generic provider protocols.
Looking up a court case by name or citation is specific to verification, so its protocol lives here.
Implement it with your own case-law index, API client or database.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ResolvedCase:
    """A case that a resolver could confirm exists."""

    case_name: str
    citation: str | None = None
    court: str | None = None
    year: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class CaseResolver(Protocol):
    """Resolves a case name or reporter citation to a real case, or ``None``."""

    async def resolve(self, query: str) -> ResolvedCase | None:
        """Return the matching case, or ``None`` when no case matches ``query``."""
        ...
