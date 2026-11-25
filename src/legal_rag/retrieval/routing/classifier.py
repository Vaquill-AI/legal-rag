"""Stage C: the model intent classifier over a candidate set.

The model reasons over the meaning of the instruction, given the current state, and returns a
structured verdict: primary intent, confidence and optional secondary intents.
The candidate set is narrowed by the shortlister in ``router.py`` so the prompt stays small.
The language model is injected through the ``LanguageModel`` protocol.
"""
# ruff: noqa: BLE001, ISC004, TRY004

from __future__ import annotations

import json
import logging

from legal_rag.providers.protocols import LanguageModel
from legal_rag.retrieval.routing.config import Taxonomy
from legal_rag.retrieval.routing.state import RouterState, RoutingInput
from legal_rag.types import Message

logger = logging.getLogger("legal_rag.retrieval.routing.classifier")

_BASE_RULES = (
    "Route on the user's actual intent, the governing verb and object of the request, "
    "not on incidental words. A long message that mentions a tool name in passing while "
    "asking a question is the default intent.",
    "The safe default is '{default}'. When the turn is a question, or you are unsure, choose "
    "'{default}' with a confidence that reflects your uncertainty. It is far better to answer "
    "than to fire the wrong tool.",
    "If the turn genuinely asks for two things, put the primary in 'intent' and the other in "
    "'secondary'.",
    "confidence is your calibrated certainty. Use >=0.85 only when the intent is unmistakable; "
    "use <0.5 when the turn is ambiguous or is really just a question.",
)


def build_catalog(taxonomy: Taxonomy, candidates: tuple[str, ...]) -> str:
    lines = []
    for intent in candidates:
        d = taxonomy.destination(intent)
        ex = "; ".join(d.examples[:3])
        lines.append(f'- "{d.intent}": {d.description}' + (f" e.g. {ex}" if ex else ""))
    return "\n".join(lines)


def build_state_note(state: RouterState) -> str:
    doc_desc = "none"
    if state.pinned_docs:
        names = ", ".join(p.filename or p.doc_id for p in state.pinned_docs[:5])
        doc_desc = f"{state.doc_count} pinned ({names})"
    flags = ", ".join(sorted(state.flags)) or "none"
    return (
        f"Documents in scope: {doc_desc}. "
        f"Capability flags: {flags}. "
        f"Follow-up turn: {'yes' if state.is_follow_up else 'no'}. "
        f"Country: {state.country_code}."
    )


def build_system_prompt(taxonomy: Taxonomy) -> str:
    rules = [r.format(default=taxonomy.default_intent) for r in _BASE_RULES]
    rules.extend(taxonomy.guidance)
    return (
        "You are the intent router for a legal research assistant. Decide what the user's "
        "latest turn is asking you to do, choosing from the capability catalog. Return STRICT "
        'JSON: {"intent": <id>, "confidence": <0..1>, "secondary": [<id>...], "reason": <short>}.'
        "\n\nRules:\n" + "\n".join(f"- {r}" for r in rules)
    )


def _coerce_intent(value: object, candidates: tuple[str, ...], taxonomy: Taxonomy) -> str | None:
    if not value:
        return None
    s = str(value).strip().lower()
    for intent in candidates:
        if intent.lower() == s:
            return intent
    # Accept any known intent even if it was shortlisted out; this is the safe direction.
    for intent in taxonomy.intents():
        if intent.lower() == s:
            return intent
    return None


async def classify(
    ri: RoutingInput,
    state: RouterState,
    candidates: tuple[str, ...],
    *,
    taxonomy: Taxonomy,
    llm: LanguageModel,
) -> tuple[str, float, tuple[str, ...], str]:
    """Return (intent, confidence, secondary_intents, reason).

    Falls back to the default intent at low confidence on any parse or model failure.
    It never raises, because the safe default is always to answer.
    """
    user = (
        f"Capability catalog (choose one intent id):\n{build_catalog(taxonomy, candidates)}\n\n"
        f"State: {build_state_note(state)}\n\n"
        f"User's latest turn:\n{ri.routing_text[:4000]}"
    )
    try:
        raw = await llm.generate(
            [Message("system", build_system_prompt(taxonomy)), Message("user", user)],
            max_tokens=400,
            temperature=0.0,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise ValueError("non-dict router output")
        intent = _coerce_intent(parsed.get("intent"), candidates, taxonomy)
        if intent is None:
            raise ValueError(f"unknown intent {parsed.get('intent')!r}")
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
        secondary = tuple(
            i
            for i in (
                _coerce_intent(s, candidates, taxonomy) for s in (parsed.get("secondary") or [])
            )
            if i is not None and i != intent
        )
        return intent, confidence, secondary, str(parsed.get("reason") or "")[:200]
    except Exception as exc:
        logger.warning("router_classify_failed error=%s", type(exc).__name__)
        return taxonomy.default_intent, 0.3, (), f"classifier error: {type(exc).__name__}"
