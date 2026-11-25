"""One OpenAPI operation to one MCP `inputSchema`, plus where each argument goes.

Deliberately a TRANSLATION rather than a re-description. The alternative is
hand-writing a tool schema per operation, which is a second copy of a contract
that already exists and drifts the first time somebody adds a field to a request
model. Everything here is derived, so a new field on a published model reaches
the tool the moment it reaches the OpenAPI document.

OpenAPI 3.1 is what makes this cheap: its schema objects ARE JSON Schema
2020-12, which is what MCP asks for. The only real work is relocating `$ref`
targets, because MCP tools carry no `components` section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

#: Where OpenAPI parks shared schemas, and where JSON Schema expects them.
_COMPONENTS_PREFIX: Final[str] = "#/components/schemas/"
_DEFS_PREFIX: Final[str] = "#/$defs/"

#: The single property a JSON request body is nested under.
#:
#: Flattening the body's fields alongside the path and query parameters reads
#: nicer and is wrong: `limit` is a query parameter on one operation and a body
#: field on another, and a flat merge would silently drop one of them. Nesting
#: costs an agent one level of indentation and cannot collide.
BODY_PROPERTY: Final[str] = "body"

#: Headers the bridge owns and must never accept as a tool argument. The
#: credential is set by the dispatcher from a value the caller supplies; letting
#: an argument carry one would let a tool call authenticate as somebody else.
RESERVED_HEADERS: Final[frozenset[str]] = frozenset(
    {"authorization", "proxy-authorization", "cookie", "host"}
)

_JSON: Final[str] = "application/json"
_TRANSLATED_LOCATIONS: Final[frozenset[str]] = frozenset({"path", "query", "header"})
_HEADER_WORD: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")


class SchemaTranslationError(ValueError):
    """Raised at build time when an operation cannot be described to a client."""


@dataclass(frozen=True, slots=True)
class Argument:
    """One tool argument, and the part of the HTTP request it becomes.

    Carried explicitly rather than re-derived by the dispatcher from the path
    template. A query parameter and a header are indistinguishable once they
    are keys in one flat object, and guessing wrong sends an idempotency key as
    `?Idempotency-Key=`, which the API ignores.
    """

    name: str
    location: str
    wire_name: str


def translate(
    operation: dict[str, Any], components: dict[str, Any], *, required: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], tuple[Argument, ...]]:
    """The tool input schema and its argument map, for one OpenAPI operation.

    `required` promotes arguments the document marks optional. It exists for
    the case where a requirement is enforced below the route and so cannot
    appear in the signature the document describes.
    """
    properties: dict[str, Any] = {}
    mandatory: set[str] = set()
    arguments: list[Argument] = []

    for parameter in operation.get("parameters") or ():
        wire_name = parameter.get("name")
        location = parameter.get("in")
        if not wire_name or location not in _TRANSLATED_LOCATIONS:
            continue
        if location == "header" and wire_name.lower() in RESERVED_HEADERS:
            raise SchemaTranslationError(
                f"{operation.get('operationId')!r} declares the {wire_name!r} header "
                f"as a parameter. This bridge owns that header; publishing it as a "
                f"tool argument would let a call present its own credential."
            )
        name = argument_name(wire_name, location)
        schema = dict(parameter.get("schema") or {})
        if description := parameter.get("description"):
            schema.setdefault("description", description)
        properties[name] = schema
        arguments.append(Argument(name=name, location=location, wire_name=wire_name))
        if parameter.get("required"):
            mandatory.add(name)

    body = _body_schema(operation)
    if body is not None:
        properties[BODY_PROPERTY] = body
        arguments.append(Argument(name=BODY_PROPERTY, location="body", wire_name=BODY_PROPERTY))
        if (operation.get("requestBody") or {}).get("required"):
            mandatory.add(BODY_PROPERTY)

    if unknown := sorted(required - set(properties)):
        raise SchemaTranslationError(
            f"{operation.get('operationId')!r} was asked to require {unknown}, which "
            f"it does not declare."
        )
    mandatory |= required

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        # Closed on purpose. An API that rejects unknown request keys answers
        # 422 over HTTP; letting the tool schema accept them would move that
        # refusal from the agent's turn, where it can correct itself, into a
        # failed HTTP call.
        "additionalProperties": False,
    }
    if mandatory:
        schema["required"] = sorted(mandatory)

    if defs := _collect_defs(schema, components):
        schema["$defs"] = defs
    return _relocated(schema), tuple(arguments)


def argument_name(wire_name: str, location: str) -> str:
    """`Idempotency-Key` becomes `idempotencyKey`; path and query names pass through.

    Path and query names are used exactly as the document spells them, because
    they also appear in the API's own documentation and a second spelling is a
    second thing to look up. Header names are not, and a JSON key containing a
    dash is legal but reads as a typo to a model.
    """
    if location != "header":
        return wire_name
    head, *rest = [word for word in _HEADER_WORD.split(wire_name) if word]
    return head.lower() + "".join(word[:1].upper() + word[1:].lower() for word in rest)


def _body_schema(operation: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON body's schema, or `None` when the operation takes no body."""
    content = ((operation.get("requestBody") or {}).get("content")) or {}
    if not content:
        return None
    if _JSON not in content:
        raise SchemaTranslationError(
            f"{operation.get('operationId')!r} accepts {sorted(content)} and not "
            f"{_JSON}. This bridge sends JSON; a multipart or binary upload has "
            f"to stay an HTTP call."
        )
    return dict(content[_JSON].get("schema") or {})


def _collect_defs(root: Any, components: dict[str, Any]) -> dict[str, Any]:
    """Every component schema reachable from `root`, transitively.

    Reachable rather than all of them: a large document can carry well over a
    hundred schemas, and a tool list repeating all of them on every tool is
    context an agent pays for on every session.
    """
    collected: dict[str, Any] = {}
    pending = _refs(root)
    while pending:
        name = pending.pop()
        if name in collected:
            continue
        if name not in components:
            raise SchemaTranslationError(
                f"schema {name!r} is referenced but absent from components. The "
                f"OpenAPI document is inconsistent; do not paper over it here."
            )
        collected[name] = components[name]
        pending |= _refs(components[name])
    return collected


def _refs(node: Any) -> set[str]:
    """Component schema names referenced anywhere beneath `node`."""
    found: set[str] = set()
    if isinstance(node, dict):
        target = node.get("$ref")
        if isinstance(target, str) and target.startswith(_COMPONENTS_PREFIX):
            found.add(target[len(_COMPONENTS_PREFIX) :])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for item in node:
            found |= _refs(item)
    return found


def _relocated(node: Any) -> Any:
    """Rewrite `#/components/schemas/X` to `#/$defs/X`, everywhere.

    A copy rather than an in-place edit. The OpenAPI document is usually built
    once and cached by whatever serves it, so mutating a nested schema here
    would corrupt the document the API's own documentation page serves.
    """
    if isinstance(node, dict):
        rewritten = {key: _relocated(value) for key, value in node.items()}
        target = rewritten.get("$ref")
        if isinstance(target, str) and target.startswith(_COMPONENTS_PREFIX):
            rewritten["$ref"] = _DEFS_PREFIX + target[len(_COMPONENTS_PREFIX) :]
        return rewritten
    if isinstance(node, list):
        return [_relocated(item) for item in node]
    return node
