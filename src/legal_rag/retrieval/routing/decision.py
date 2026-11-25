"""The confidence-tiered decision: turn a scored intent into an action.

A scored intent is mapped to one of four actions by confidence, with the execute threshold
asymmetric by the destination's action cost.
The safe default when uncertain is to answer the question and merely offer the tool,
never a silent handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from legal_rag.retrieval.routing.config import ActionCost, Taxonomy
from legal_rag.retrieval.routing.state import RouterState


class Action(str, Enum):
    """What the system should do with a routed intent."""

    EXECUTE = "execute"
    CONFIRM = "confirm"
    ANSWER_AND_OFFER = "answer_and_offer"
    ANSWER = "answer"


@dataclass(frozen=True)
class DecisionPolicy:
    """Confidence bands. Costly destinations must clear a higher execute bar."""

    execute_thresholds: dict[ActionCost, float] = field(
        default_factory=lambda: {
            ActionCost.REVERSIBLE: 0.55,
            ActionCost.MODERATE: 0.70,
            ActionCost.EXPENSIVE: 0.80,
            ActionCost.DESTRUCTIVE: 0.85,
        }
    )
    # Below execute but at or above this, a costly destination offers a confirm step.
    confirm_floor: float = 0.60
    # Below this, do not surface the tool at all.
    offer_floor: float = 0.35


DEFAULT_POLICY = DecisionPolicy()


@dataclass(frozen=True)
class RouteDecision:
    """The router's verdict for one turn."""

    intent: str
    confidence: float
    action: Action
    target_doc_ids: tuple[str, ...] = field(default_factory=tuple)
    secondary_intents: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    # The stage that produced the decision: "prefilter", "classifier" or "fallback".
    source: str = "classifier"
    precondition_note: str | None = None

    @property
    def executes(self) -> bool:
        return self.action in (Action.EXECUTE, Action.CONFIRM)


def threshold_for(
    taxonomy: Taxonomy, intent: str, policy: DecisionPolicy = DEFAULT_POLICY
) -> float:
    return policy.execute_thresholds[taxonomy.cost(intent)]


def _precondition_ok(
    taxonomy: Taxonomy, intent: str, state: RouterState
) -> tuple[bool, str | None]:
    """Hard preconditions the model can never override."""
    d = taxonomy.destination(intent)
    if d.needs_documents and not state.has_documents:
        return False, f"{intent} needs a document in scope; none is pinned"
    if d.min_documents and state.doc_count < d.min_documents:
        return False, f"{intent} needs >= {d.min_documents} documents; have {state.doc_count}"
    missing = [flag for flag in d.requires if flag not in state.flags]
    if missing:
        return False, f"{intent} needs state flags {', '.join(missing)}"
    return True, None


def decide(
    taxonomy: Taxonomy,
    intent: str,
    confidence: float,
    state: RouterState,
    *,
    secondary_intents: tuple[str, ...] = (),
    reason: str = "",
    source: str = "classifier",
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> RouteDecision:
    """Map a scored intent and state to an actionable decision."""
    confidence = max(0.0, min(1.0, confidence))
    default = taxonomy.default_intent
    if not taxonomy.has(intent):
        intent = default

    ok, note = _precondition_ok(taxonomy, intent, state)
    if not ok:
        return RouteDecision(
            intent=default,
            confidence=confidence,
            action=Action.ANSWER,
            secondary_intents=secondary_intents,
            reason=reason or f"degraded: {note}",
            source=source,
            precondition_note=note,
        )

    targets = (
        tuple(p.doc_id for p in state.pinned_docs)
        if taxonomy.destination(intent).needs_documents
        else ()
    )
    if intent == default:
        return RouteDecision(
            intent, confidence, Action.EXECUTE, targets, secondary_intents, reason, source
        )

    cost = taxonomy.cost(intent)
    if confidence >= policy.execute_thresholds[cost]:
        action = Action.EXECUTE
    elif cost >= ActionCost.EXPENSIVE and confidence >= policy.confirm_floor:
        action = Action.CONFIRM
    elif confidence >= policy.offer_floor:
        action = Action.ANSWER_AND_OFFER
    else:
        return RouteDecision(
            default,
            confidence,
            Action.ANSWER,
            (),
            secondary_intents,
            reason or "low confidence, safe default answer",
            source,
        )
    return RouteDecision(intent, confidence, action, targets, secondary_intents, reason, source)
