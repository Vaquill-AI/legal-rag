"""State-equivalent authorities for federal canonical doctrines.

When a question is scoped to a US state and reads as state-law, a federal canonical
injection (Celotex, Anderson, Fed. R. Civ. P. 56) can pin the wrong authority into a
state-court question. This module closes the resulting gap by mapping the doctrine ids that
``doctrine_detector`` produces to the per-state procedural rule that controls in that forum.

The map is data (``data/state_equivalents.json``), so adding a doctrine or a state needs no
code change. For doctrines or states not in the map the caller should stay silent, which is
strictly better than fabricating a state authority.

Citations are rule cites only: they are durable and unambiguous, while case-of-the-rule
selection is state-specific judgment that drifts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from legal_rag.types import RetrievedChunk


@dataclass(frozen=True)
class StateAuthority:
    """One state-law procedural authority for a federal doctrine.

    Mirrors the ``CanonicalStatute`` shape so the chunk builder can produce uniformly
    rendered sources.
    """

    citation: str  # short cite (e.g. "La. C.C.P. art. 966")
    title: str  # display label (e.g. "Louisiana summary judgment rule")
    text: str  # one-line description of the rule's operative role
    url: str | None = None  # deep link to the official text where stable


def _load_equivalents() -> dict[str, dict[str, list[StateAuthority]]]:
    raw = json.loads(
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/state_equivalents.json")
        .read_text(encoding="utf-8")
    )
    return {
        doctrine: {state: [StateAuthority(**a) for a in auths] for state, auths in by_state.items()}
        for doctrine, by_state in raw.items()
    }


STATE_EQUIVALENTS: dict[str, dict[str, list[StateAuthority]]] = _load_equivalents()


def get_state_equivalents(doctrine_id: str, us_state: str | None) -> list[StateAuthority]:
    """Return state-equivalent authorities for a federal doctrine + state.

    Returns an empty list if either the doctrine is not in the map or no
    entry exists for the requested state. Callers should treat empty as
    "stay silent" (the existing federal-canonical suppression is the
    correct fallback).
    """
    if not us_state:
        return []
    state_map = STATE_EQUIVALENTS.get(doctrine_id)
    if not state_map:
        return []
    return list(state_map.get(us_state.lower(), []))


def build_state_equivalent_chunks(
    doctrine_id: str,
    doctrine_name: str,
    us_state: str,
    query: str,
) -> list[RetrievedChunk]:
    """Build canonical-shape chunks for state equivalents of a federal doctrine.

    Returns an empty list if no state equivalents are registered for this doctrine and state.
    Each chunk carries ``is_canonical_authority=True`` and ``is_state_equivalent=True``.
    """
    authorities = get_state_equivalents(doctrine_id, us_state)
    if not authorities:
        return []

    chunks: list[RetrievedChunk] = []
    base_score = 0.99
    for i, a in enumerate(authorities):
        body = (
            f"{a.citation}\n"
            f"{a.title}\n\n"
            f"Controlling state authority for the {doctrine_name} in this matter: {a.text}\n\n"
            f"The question is scoped to {us_state.upper()}. It asks about: {query.strip()[:200]}"
        )
        meta: dict[str, Any] = {
            "source_type": "statute",
            "is_canonical_authority": True,
            "is_state_equivalent": True,
            "doctrine_id": doctrine_id,
            "us_state": us_state.lower(),
            "title": a.citation,
            "citation": a.citation,
            "citations": [a.citation],
            "url": a.url,
            "document_type": "state_statute",
            "document_id": f"state_equiv:{doctrine_id}:{us_state.lower()}:{i}",
        }
        chunks.append(
            RetrievedChunk(
                id=f"state_equiv_{doctrine_id}_{us_state.lower()}_{i}",
                text=body,
                score=base_score - 0.001 * i,
                metadata=meta,
            )
        )
    return chunks


__all__ = [
    "STATE_EQUIVALENTS",
    "StateAuthority",
    "build_state_equivalent_chunks",
    "get_state_equivalents",
]
