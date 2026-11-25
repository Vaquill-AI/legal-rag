"""Routing configuration: the taxonomy of intents and destinations.

The router has no built-in notion of what a deployment can do.
The caller injects a ``Taxonomy``: the set of destinations a turn can be routed to,
how costly a wrong execution is for each, and which state each one needs.
``example_taxonomy`` is a small neutral set that shows the shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum


class ActionCost(IntEnum):
    """How much a wrong execution of a destination hurts. Higher means a higher execute bar."""

    REVERSIBLE = 1
    MODERATE = 2
    EXPENSIVE = 3
    DESTRUCTIVE = 4


@dataclass(frozen=True)
class Destination:
    """One thing a turn can be routed to."""

    intent: str
    description: str
    examples: tuple[str, ...] = ()
    cost: ActionCost = ActionCost.REVERSIBLE
    needs_documents: bool = False
    min_documents: int = 0
    # Names of state flags (see ``RouterState.flags``) that must be present.
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Taxonomy:
    """The injected routing configuration."""

    destinations: tuple[Destination, ...]
    default_intent: str = "answer"
    small_talk_intent: str | None = "small_talk"
    # Extra rules appended to the classifier prompt, one per line.
    guidance: tuple[str, ...] = ()
    # Optional per-intent overrides of the cost bucket, keyed by intent id.
    cost_overrides: Mapping[str, ActionCost] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [d.intent for d in self.destinations]
        if len(ids) != len(set(ids)):
            raise ValueError("Taxonomy destinations must have unique intent ids")
        if self.default_intent not in ids:
            raise ValueError(f"default_intent {self.default_intent!r} is not a destination")
        if self.small_talk_intent is not None and self.small_talk_intent not in ids:
            raise ValueError(f"small_talk_intent {self.small_talk_intent!r} is not a destination")

    def intents(self) -> tuple[str, ...]:
        return tuple(d.intent for d in self.destinations)

    def has(self, intent: str) -> bool:
        return intent in self.intents()

    def destination(self, intent: str) -> Destination:
        for d in self.destinations:
            if d.intent == intent:
                return d
        raise KeyError(intent)

    def cost(self, intent: str) -> ActionCost:
        return self.cost_overrides.get(intent, self.destination(intent).cost)


def example_taxonomy() -> Taxonomy:
    """A small neutral taxonomy for a research assistant. Replace it with your own."""
    return Taxonomy(
        destinations=(
            Destination(
                "answer",
                "Answer the question directly from retrieved sources.",
                ("what is the limitation period for a contract claim", "explain res judicata"),
            ),
            Destination(
                "small_talk",
                "Greeting, thanks or other conversation with no task.",
                ("hello", "thanks, that helps"),
            ),
            Destination(
                "summarize",
                "Summarize one document in scope.",
                ("summarize this opinion", "give me a short summary of the attached statute"),
                needs_documents=True,
            ),
            Destination(
                "compare",
                "Compare two or more documents in scope.",
                ("compare these two versions", "how do the attached rules differ"),
                cost=ActionCost.MODERATE,
                needs_documents=True,
                min_documents=2,
            ),
            Destination(
                "extract",
                "Extract structured fields from the documents in scope.",
                ("list every party and date in this filing", "pull out the defined terms"),
                cost=ActionCost.MODERATE,
                needs_documents=True,
            ),
            Destination(
                "find_authority",
                "Look up statutes, rules or cases on a topic.",
                ("find cases on adverse possession", "which statute governs notice periods"),
            ),
            Destination(
                "export_report",
                "Produce and store a long-form report from the conversation.",
                ("write this up as a report", "generate a research memo and save it"),
                cost=ActionCost.EXPENSIVE,
                requires=("can_export",),
            ),
        ),
        guidance=(
            "A request to rewrite or reformat a document in place is 'answer', not a tool.",
            "Only choose 'compare' when the user clearly wants a cross-document operation.",
        ),
    )
