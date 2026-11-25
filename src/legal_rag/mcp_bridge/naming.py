"""Operation id to MCP tool name, and the rules that keep the map trustworthy.

Nothing in this module knows about a particular API, so the same rules serve any
OpenAPI document.

The map has to be INJECTIVE and it has to be checked, because the failure is
silent in the worst possible way: two operations collapsing onto one tool name
means the second registration wins and the first operation becomes unreachable
through MCP while remaining perfectly reachable over HTTP. Nothing errors, the
tool list still looks complete, and the only symptom is an agent that cannot do
one specific thing.
"""

from __future__ import annotations

import re
from typing import Final

#: What the MCP schema accepts for a tool name. Ours is a strict subset
#: (lowercase and underscores), but the check is written against the protocol's
#: rule rather than ours so a future naming change is caught by the protocol's
#: limit rather than by a client rejecting the tool list at runtime.
_LEGAL_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

#: The boundary inside a camelCase segment: a lowercase or digit followed by an
#: uppercase. `caseLaw` splits once, `list` not at all.
_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class ToolNameError(ValueError):
    """Raised at catalogue build time, never at request time."""


def tool_name(operation_id: str) -> str:
    """`statutes.getSection` becomes `statutes_get_section`.

    Dots are the operation id's own separator and are replaced rather than
    kept: they are legal in the protocol's pattern but several clients render a
    dotted name as a namespace and truncate the display at the first dot, which
    turns `cases.get` and `statutes.get` into two tools both labelled `get`.
    """
    if not operation_id:
        raise ToolNameError("operation id is empty")
    name = "_".join(
        _CAMEL_BOUNDARY.sub("_", segment).lower() for segment in operation_id.split(".")
    )
    if not _LEGAL_NAME.match(name):
        raise ToolNameError(
            f"{operation_id!r} produces the tool name {name!r}, which is not a "
            f"legal MCP tool name. Rename the operation rather than widening "
            f"this rule; the name is published to every client."
        )
    return name


def assert_injective(pairs: list[tuple[str, str]]) -> None:
    """Refuse a catalogue where two operations share a tool name.

    `pairs` is `(operation_id, tool_name)`. Raising here fails the call that
    builds the catalogue, so a collision is found at construction time and
    never by an agent that cannot reach an operation.
    """
    seen: dict[str, str] = {}
    for operation_id, name in pairs:
        if name in seen:
            raise ToolNameError(
                f"{operation_id!r} and {seen[name]!r} both map to the tool name "
                f"{name!r}. One of them would be silently unreachable through MCP."
            )
        seen[name] = operation_id
