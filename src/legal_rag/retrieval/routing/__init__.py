"""Query routing: prefilter, shortlist, classify and a tiered decision.

``route`` is the entry point. It takes a ``RoutingInput`` and a ``RouterState``, an injected
``Taxonomy`` of destinations, and optionally an embedder and a language model.
It returns a ``RouteDecision`` and never executes a destination.
"""

from legal_rag.retrieval.routing.config import ActionCost, Destination, Taxonomy, example_taxonomy
from legal_rag.retrieval.routing.decision import (
    DEFAULT_POLICY,
    Action,
    DecisionPolicy,
    RouteDecision,
    decide,
)
from legal_rag.retrieval.routing.router import route, shortlist
from legal_rag.retrieval.routing.state import PinnedDoc, RouterState, RoutingInput

__all__ = [
    "DEFAULT_POLICY",
    "Action",
    "ActionCost",
    "DecisionPolicy",
    "Destination",
    "PinnedDoc",
    "RouteDecision",
    "RouterState",
    "RoutingInput",
    "Taxonomy",
    "decide",
    "example_taxonomy",
    "route",
    "shortlist",
]
