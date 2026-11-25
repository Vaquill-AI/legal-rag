"""The tool catalogue, derived from a route declaration set and an OpenAPI document.

The caller supplies both halves and this module refuses to build unless they
agree in BOTH directions: a declared operation missing from the document, or a
documented operation nobody declared, is a disagreement about what the API is,
and guessing which side is right is how one of them silently stops being served.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from legal_rag.mcp_bridge.naming import assert_injective, tool_name
from legal_rag.mcp_bridge.schema import translate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from legal_rag.mcp_bridge.schema import Argument


class CatalogueError(ValueError):
    """Raised at build time. There is no request-time failure mode here."""


@dataclass(frozen=True, slots=True)
class RouteRef:
    """One declared operation, in the terms this bridge needs.

    Deliberately not the caller's own route type. Depending on an application's
    route dataclass would make this package importable from exactly one
    application, which is the thing it is trying not to be.
    """

    operation_id: str
    method: str
    path: str
    #: Published in the tool description so an agent can read why a call was
    #: refused rather than retrying it. Empty is legal for an unscoped API.
    scopes: tuple[str, ...] = ()
    #: Arguments to mark required even where the document marks them optional.
    #: For a requirement enforced below the route, the document says optional
    #: and the first call still fails; stating it here moves the refusal into
    #: the agent's own turn.
    required_arguments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything needed to advertise a tool and to make its HTTP call."""

    name: str
    operation_id: str
    method: str
    path: str
    title: str
    description: str
    input_schema: Mapping[str, Any] = field(repr=False)
    arguments: tuple[Argument, ...] = ()
    #: What a caller must hold to call this. Carried onto the spec, not left in
    #: the description text, because a listing layer may filter on it.
    scopes: tuple[str, ...] = ()

    @property
    def read_only(self) -> bool:
        """A GET changes nothing, which is the one hint clients act on.

        Several hosts use `readOnlyHint` to decide whether a call needs the
        user's confirmation, so getting it wrong in the permissive direction
        turns an approval prompt into a silent write.
        """
        return self.method == "GET"

    @property
    def destructive(self) -> bool:
        """Only a DELETE removes something the caller cannot re-derive."""
        return self.method == "DELETE"

    @property
    def idempotent(self) -> bool:
        """Repeating it lands the same state. Deliberately excludes POST.

        POST is where an API mints work, and the `Idempotency-Key` that makes
        a retry free is the CALLER's to supply. Claiming idempotency the caller
        has not asked for is how a retried tool call becomes a second job.
        """
        return self.method in {"GET", "PUT", "DELETE"}


def build_catalogue(
    *, openapi: Mapping[str, Any], routes: Sequence[RouteRef]
) -> tuple[ToolSpec, ...]:
    """One tool per declared route, or refuse."""
    documented = _by_operation_id(openapi)
    components = dict((openapi.get("components") or {}).get("schemas") or {})
    _assert_agreement({route.operation_id for route in routes}, set(documented))

    specs = tuple(
        _spec(route, *documented[route.operation_id], components=components)
        for route in sorted(routes, key=lambda r: r.operation_id)
    )
    assert_injective([(spec.operation_id, spec.name) for spec in specs])
    return specs


def _by_operation_id(
    openapi: Mapping[str, Any],
) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """`operationId` to `(METHOD, path, operation object)`."""
    found: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for path, item in (openapi.get("paths") or {}).items():
        for method, operation in item.items():
            operation_id = operation.get("operationId") if isinstance(operation, dict) else None
            if not operation_id:
                continue
            if operation_id in found:
                raise CatalogueError(f"operationId {operation_id!r} appears twice in the document")
            effective = {**operation, "parameters": _parameters(openapi, item, operation)}
            found[operation_id] = (method.upper(), path, effective)
    return found


def _parameters(
    openapi: Mapping[str, Any], item: Mapping[str, Any], operation: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The parameters an operation really takes: the path item's, then its own.

    OpenAPI lets a path item declare parameters shared by every method under it, and lets
    any parameter be a `$ref` into `components.parameters`. Both are resolved here, and an
    operation's own parameter replaces a shared one with the same name and location.
    A reference that cannot be resolved raises, because dropping it would leave a path
    placeholder such as `{id}` unfilled in the URL.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for source in (item.get("parameters"), operation.get("parameters")):
        for entry in source or ():
            parameter = _resolve_parameter(openapi, entry)
            merged[(parameter.get("in", ""), parameter.get("name", ""))] = parameter
    return list(merged.values())


def _resolve_parameter(openapi: Mapping[str, Any], entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise CatalogueError(f"a parameter must be an object, got {type(entry).__name__}")
    ref = entry.get("$ref")
    if ref is None:
        return entry
    prefix = "#/components/parameters/"
    components = (openapi.get("components") or {}).get("parameters") or {}
    target = components.get(ref[len(prefix) :]) if str(ref).startswith(prefix) else None
    if not isinstance(target, dict) or "$ref" in target:
        raise CatalogueError(f"parameter reference {ref!r} does not resolve to a parameter object")
    return target


def _assert_agreement(declared: set[str], documented: set[str]) -> None:
    """Both directions, because each has its own failure."""
    if undocumented := sorted(declared - documented):
        raise CatalogueError(
            f"declared but absent from the OpenAPI document: {undocumented}. "
            f"These operations would have no tool."
        )
    if undeclared := sorted(documented - declared):
        raise CatalogueError(
            f"documented but not declared: {undeclared}. A tool would be "
            f"published for an operation nothing governs."
        )


def _spec(
    route: RouteRef,
    method: str,
    path: str,
    operation: dict[str, Any],
    *,
    components: dict[str, Any],
) -> ToolSpec:
    if method != route.method.upper() or path != route.path:
        raise CatalogueError(
            f"{route.operation_id}: declared {route.method} {route.path}, "
            f"documented {method} {path}"
        )
    schema, arguments = translate(
        operation, components, required=frozenset(route.required_arguments)
    )
    return ToolSpec(
        name=tool_name(route.operation_id),
        operation_id=route.operation_id,
        method=method,
        path=path,
        title=operation.get("summary") or route.operation_id,
        description=_description(route, operation),
        input_schema=schema,
        arguments=arguments,
        scopes=tuple(sorted(route.scopes)),
    )


def _description(route: RouteRef, operation: dict[str, Any]) -> str:
    """The handler's own prose, plus the two facts an agent cannot infer.

    The scopes matter because a refusal is otherwise indistinguishable from a
    missing resource, and the operation id matters because it is the string a
    support ticket and the server logs both use.
    """
    parts = [operation.get("description") or operation.get("summary") or route.operation_id]
    if route.scopes:
        parts.append(f"Requires the scope {', '.join(sorted(route.scopes))}.")
    parts.append(f"Calls {route.method.upper()} {route.path} (operation {route.operation_id}).")
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
