"""An HTTP answer becomes MCP tool content.

One assumption is baked in: errors are RFC 9457 problem documents. That is a
published standard rather than a house style, and a body that is not one still
renders, so the assumption degrades rather than breaks.

Two decisions here are worth stating because both look like omissions.

**A 4xx is `isError`, not an exception.** The MCP specification distinguishes a
protocol failure from a tool that ran and reported a problem, and a refusal by
the API is the second kind. Raising would deny the model the error document,
which is exactly the text that says which scope is missing or which id is wrong.

**A binary body is not inlined past a cap.** Base64 of a large document is
context the caller pays for on a response the model cannot read anyway.

The result is a plain `ToolResult` in the shape of an MCP `CallToolResult`, so
nothing here imports the `mcp` package. `ToolResult.as_dict()` is the wire form
and `ToolResult.to_mcp()` builds the SDK's own type, importing it lazily.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from legal_rag.mcp_bridge.catalogue import ToolSpec
    from legal_rag.mcp_bridge.dispatch import HttpResult

#: Above this, a binary body is described rather than carried. Chosen to sit
#: under the payload most clients will hold comfortably in one message while
#: still admitting an ordinary contract.
MAX_INLINE_BYTES: Final[int] = 1_048_576

_JSON_TYPES: Final[frozenset[str]] = frozenset({"application/json", "application/problem+json"})
_TEXT_PREFIX: Final[str] = "text/"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool call returns to the model, shaped like an MCP `CallToolResult`."""

    #: Content blocks in their MCP wire form: `{"type": "text", "text": ...}` or an
    #: embedded blob resource.
    content: tuple[dict[str, Any], ...]
    is_error: bool = False
    #: The parsed JSON body when it is an object. A list body is legal JSON but
    #: not legal structured content, so it stays in `content` only.
    structured_content: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The MCP `CallToolResult` wire shape, with its camelCase keys."""
        wire: dict[str, Any] = {"content": [dict(block) for block in self.content]}
        if self.structured_content is not None:
            wire["structuredContent"] = self.structured_content
        wire["isError"] = self.is_error
        return wire

    def to_mcp(self) -> Any:
        """The same result as the MCP SDK's `CallToolResult`.

        Imports `mcp` here and nowhere else, so the package is only needed by a
        caller that wants the SDK type.
        """
        from mcp import types

        return types.CallToolResult.model_validate(self.as_dict())


def to_tool_result(spec: ToolSpec, result: HttpResult) -> ToolResult:
    """Render one HTTP answer for the model."""
    if result.truncated:
        # Refused by the DISPATCHER before the bytes were read, so there is
        # nothing here to measure.
        size = f"{result.declared_length} bytes" if result.declared_length else "its size"
        return _text(
            f"{result.status}, and the response was not transferred because "
            f"{size} exceeds what a tool call will carry. Call "
            f"{spec.method} {spec.path} over HTTP to fetch it.",
            is_error=result.status >= 400,
        )
    if result.status == 204 or not result.body:
        return _text(f"{result.status}, no content.", is_error=result.status >= 400)
    if result.media_type in _JSON_TYPES:
        return _json_result(result)
    if result.media_type.startswith(_TEXT_PREFIX):
        return _text(result.body.decode("utf-8", "replace"), is_error=result.status >= 400)
    return _binary_result(spec, result)


def _text(text: str, *, is_error: bool, structured: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        content=({"type": "text", "text": text},),
        is_error=is_error,
        structured_content=structured,
    )


def _json_result(result: HttpResult) -> ToolResult:
    try:
        payload = json.loads(result.body)
    except ValueError:
        return _text(
            f"{result.status}, unparseable JSON: {result.body.decode('utf-8', 'replace')[:2000]}",
            is_error=True,
        )

    failed = result.status >= 400
    text = _problem_text(payload) if failed else json.dumps(payload, indent=2, ensure_ascii=False)
    # Only an object can be structured content, and a list-valued body is
    # legal JSON. Sending a list here fails the client's own schema check,
    # which reads as a broken tool rather than an unusual body.
    return _text(
        text,
        is_error=failed,
        structured=payload if isinstance(payload, dict) else None,
    )


def _problem_text(payload: Any) -> str:
    """The problem document as prose, so the model does not have to parse it.

    Falls through to the raw JSON for a body that is not a problem document,
    because inventing a shape for it would hide whatever the API did say.
    """
    if not isinstance(payload, dict) or "title" not in payload:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    lines = [f"{payload.get('status', '')} {payload['title']}".strip()]
    if detail := payload.get("detail"):
        lines.append(str(detail))
    known = {"title", "detail", "status", "instance", "type", "requestId"}
    for key in sorted(set(payload) - known):
        lines.append(f"{key}: {json.dumps(payload[key], ensure_ascii=False)}")
    if kind := payload.get("type"):
        lines.append(f"See {kind}")
    return "\n".join(lines)


def _binary_result(spec: ToolSpec, result: HttpResult) -> ToolResult:
    """A blob under the cap, a description above it."""
    media_type = result.media_type or "application/octet-stream"
    if len(result.body) > MAX_INLINE_BYTES:
        return _text(
            f"{result.status}, {len(result.body)} bytes of {media_type}, which is "
            f"over the {MAX_INLINE_BYTES} byte inline limit and was not "
            f"transferred. Call {spec.method} {spec.path} over HTTP to fetch it.",
            is_error=result.status >= 400,
        )
    return ToolResult(
        content=(
            {
                "type": "resource",
                "resource": {
                    "uri": f"mcp+http://{spec.name}",
                    "mimeType": media_type,
                    "blob": base64.b64encode(result.body).decode("ascii"),
                },
            },
        ),
        is_error=result.status >= 400,
    )
