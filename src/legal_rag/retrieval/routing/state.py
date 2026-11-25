"""The state the router sees, and how the raw request is prepared for routing.

Many intents are only valid in certain states: a comparison needs at least two documents,
an export may need a capability flag.
Conditioning the decision on state removes ambiguity.
``RouterState`` is plain data so the router is a pure function of (query, state).
It carries no tenancy keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PinnedDoc:
    """A document in scope for this turn."""

    doc_id: str
    filename: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class RouterState:
    """Everything non-textual the router should condition on."""

    pinned_docs: tuple[PinnedDoc, ...] = field(default_factory=tuple)
    # Free-form capability flags the caller sets, matched against ``Destination.requires``.
    flags: frozenset[str] = field(default_factory=frozenset)
    country_code: str = "US"
    skill_slug: str | None = None
    is_follow_up: bool = False
    has_prior_answer: bool = False

    @property
    def doc_count(self) -> int:
        return len(self.pinned_docs)

    @property
    def has_documents(self) -> bool:
        return bool(self.pinned_docs)


@dataclass(frozen=True)
class RoutingInput:
    """The prepared text for routing: the instruction, and the full message.

    Routing decides from the instruction slice, never the pasted body, so a transcript that
    happens to contain a tool keyword does not vote.
    """

    instruction: str
    full_message: str
    has_pasted_body: bool = False

    @property
    def routing_text(self) -> str:
        return self.instruction or self.full_message
