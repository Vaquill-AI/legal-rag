"""Citation data model returned by ``extract``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CitationKind(str, Enum):
    """The family a citation belongs to."""

    USC = "usc"
    CFR = "cfr"
    FEDERAL_RULE = "federal_rule"
    PUBLIC_LAW = "public_law"
    FEDERAL_REGISTER = "federal_register"
    STATE_STATUTE = "state_statute"
    CASE = "case"


@dataclass(frozen=True)
class Citation:
    """One citation found in a text.

    ``raw`` is the exact matched text, ``text[start:end] == raw``.
    ``normalized`` is a canonical rendering, so two spellings of the same cite compare equal.
    ``components`` carries parsed parts such as ``title``, ``section``, ``volume`` or ``reporter``.
    A match means the text has the SHAPE of a citation.
    It never means the cited authority exists.
    """

    raw: str
    kind: CitationKind
    normalized: str
    start: int
    end: int
    components: Mapping[str, Any] = field(default_factory=dict, compare=False)
