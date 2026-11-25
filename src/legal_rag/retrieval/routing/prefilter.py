"""Stage A: deterministic pre-filter for explicit, structural signals only.

Regex is used only for facts the request states outright: an empty turn, a pure greeting,
or an explicitly selected skill.
Anything intent-shaped falls through to the semantic and model stages.
"""

from __future__ import annotations

import re

from legal_rag.retrieval.routing.config import Taxonomy
from legal_rag.retrieval.routing.decision import Action, RouteDecision
from legal_rag.retrieval.routing.state import RouterState

# A pure greeting or acknowledgement with no task. Anchored and short so a greeting followed
# by a request does not match.
_PURE_GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|good (?:morning|afternoon|evening)|thanks?|thank you|"
    r"ty|thx|ok(?:ay)?|cool|great|got it|nice|perfect|bye|goodbye)"
    r"(?:\s+(?:there|all|everyone|team|folks|so much))?[\s!.,]*$",
    re.IGNORECASE,
)


def prefilter(
    instruction: str | None, state: RouterState, taxonomy: Taxonomy
) -> RouteDecision | None:
    """Return a decision for an explicit signal, or None to fall through."""
    text = (instruction or "").strip()
    small_talk = taxonomy.small_talk_intent or taxonomy.default_intent
    if not text:
        return RouteDecision(
            small_talk, 1.0, Action.ANSWER, reason="empty message", source="prefilter"
        )
    if state.skill_slug:
        return RouteDecision(
            taxonomy.default_intent,
            1.0,
            Action.ANSWER,
            reason=f"explicit skill: {state.skill_slug}",
            source="prefilter",
        )
    if len(text) <= 40 and _PURE_GREETING_RE.match(text):
        return RouteDecision(
            small_talk, 1.0, Action.ANSWER, reason="pure greeting", source="prefilter"
        )
    return None
