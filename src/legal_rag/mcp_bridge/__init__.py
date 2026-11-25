"""Turn an OpenAPI document into MCP tools, without describing the API twice.

The bridge takes two things an API already has, its OpenAPI document and the list
of operations it declares, and describes one MCP tool per operation. It holds no
knowledge of any particular API: no credential format, no scope taxonomy, no
error classes. It does not run an MCP server. It supplies the three pieces a
server needs and leaves the transport to the caller.

    catalogue = build_catalogue(openapi=document, routes=[RouteRef(...), ...])
    spec = next(s for s in catalogue if s.name == "statutes_get_section")
    http = await call_operation(spec, {"code": "usc"}, client=client, credential="Bearer ...")
    result = to_tool_result(spec, http)      # result.is_error is True for a 4xx

The client is anything shaped like an `httpx.AsyncClient` (see `HttpClient`). The
`mcp` package and `httpx` are never imported by this package.

Two properties are worth knowing before extending it.

**Nothing is hand-written per operation.** Names, schemas, descriptions and the
read-only and destructive hints are all derived, and the catalogue refuses to
build when the declared route set and the OpenAPI document disagree in either
direction. Adding an operation to the API adds a tool; there is no second list.

**A tool call is the same HTTP request the API already serves.** The dispatcher
makes it against a caller-supplied client, so pointing that client at the
application's own ASGI callable means every tool call re-enters the real
middleware stack: the same authentication, the same scope enforcement, the same
error documents. No authorization logic is reimplemented here, which is the only
version of this that stays correct.
"""

from __future__ import annotations

from legal_rag.mcp_bridge.catalogue import CatalogueError, RouteRef, ToolSpec, build_catalogue
from legal_rag.mcp_bridge.dispatch import (
    HttpClient,
    HttpResponse,
    HttpResult,
    ToolCallError,
    call_operation,
)
from legal_rag.mcp_bridge.naming import ToolNameError, tool_name
from legal_rag.mcp_bridge.openapi_enum import enum_values
from legal_rag.mcp_bridge.result import ToolResult, to_tool_result
from legal_rag.mcp_bridge.schema import (
    Argument,
    SchemaTranslationError,
    argument_name,
    translate,
)

__all__ = [
    "Argument",
    "CatalogueError",
    "HttpClient",
    "HttpResponse",
    "HttpResult",
    "RouteRef",
    "SchemaTranslationError",
    "ToolCallError",
    "ToolNameError",
    "ToolResult",
    "ToolSpec",
    "argument_name",
    "build_catalogue",
    "call_operation",
    "enum_values",
    "to_tool_result",
    "tool_name",
    "translate",
]
