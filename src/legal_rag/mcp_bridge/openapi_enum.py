"""Publish a closed vocabulary on a field whose type pydantic cannot express as an enum.

A filter typed `str | list[str] | None` (one value or several) whose accepted values are checked
by a validator rather than by a `Literal` gets no `enum` from pydantic. The common fix,
`json_schema_extra={"enum": [...]}`, merges the list into the TOP of the field schema, beside its
`anyOf`. JSON Schema applies a top-level `enum` to the whole value, so that schema says a list
(`["ca", "tx"]`) and `null` are both invalid: an array-only field such as `fields` admitted no
value at all. Every consumer of the document (documentation generators, generated SDKs, MCP
clients building tool schemas) then sees either a broken constraint or none.

`enum_values(values)` puts the list where it means what we accept: on every string branch and on
the items of every list branch. `null` stays valid, and so does a list of known values.
"""

from collections.abc import Callable, Iterable
from typing import Any


def _constrain(branch: dict[str, Any], values: list[Any]) -> None:
    if branch.get("type") == "string":
        branch["enum"] = values
        return
    items = branch.get("items")
    if branch.get("type") == "array" and isinstance(items, dict) and items.get("type") == "string":
        items["enum"] = values


def enum_values(values: Iterable[Any]) -> Callable[[dict[str, Any]], None]:
    """A `json_schema_extra` that publishes `values` on each string branch and list item."""
    published = list(values)

    def apply(schema: dict[str, Any]) -> None:
        branches = schema.get("anyOf")
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    _constrain(branch, published)
        else:
            _constrain(schema, published)

    return apply
