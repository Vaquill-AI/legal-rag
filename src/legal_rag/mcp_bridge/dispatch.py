"""Turn one tool call into one HTTP request against the API being bridged.

The caller injects the client, which is what lets an application point this at
its own ASGI callable with no socket, and lets a test point it at a stub. The
client is described by the small `HttpClient` protocol below; an
`httpx.AsyncClient` satisfies it, and importing httpx is never required here.

**The credential is set here and nowhere else.** It comes from the caller, never
from a tool argument, and `RESERVED_HEADERS` in `schema.py` is what stops an
operation from publishing an argument that could reach the same header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol
from urllib.parse import quote

from legal_rag.mcp_bridge.schema import RESERVED_HEADERS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from legal_rag.mcp_bridge.catalogue import ToolSpec


class HttpResponse(Protocol):
    """The part of a streamed response the dispatcher reads."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def aiter_bytes(self) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class HttpClient(Protocol):
    """The httpx-style client the dispatcher calls: build a request, then send it streamed."""

    def build_request(
        self,
        method: str,
        url: str,
        *,
        params: list[tuple[str, str]] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any: ...

    async def send(self, request: Any, *, stream: bool = False) -> HttpResponse: ...


class ToolCallError(ValueError):
    """A tool call this bridge refuses to turn into a request.

    Separate from an HTTP failure on purpose: this is the argument set being
    wrong before anything is sent, which is the agent's to fix, whereas a 4xx
    is the API's answer and carries its own published error document.
    """


#: The most a tool call will pull into memory. Above this the body is refused
#: without being read: the bridge may run in the same process as the bridged
#: application, so a single call must not be able to spend an arbitrary file
#: size in RSS. Comfortably above any JSON an API normally produces and far
#: below what a download route can serve.
MAX_RESPONSE_BYTES: Final[int] = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HttpResult:
    """What the bridged API answered, before it becomes tool content."""

    status: int
    media_type: str
    body: bytes
    #: `Content-Length` when the response declared one, for a refusal that can
    #: say how big the thing actually was.
    declared_length: int | None = None
    #: True when the body was refused unread rather than carried.
    truncated: bool = False


async def call_operation(
    spec: ToolSpec,
    arguments: dict[str, Any],
    *,
    client: HttpClient,
    credential: str,
    forwarded_headers: dict[str, str] | None = None,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> HttpResult:
    """Make the call the tool describes, and return the raw answer."""
    if unknown := sorted(set(arguments) - {argument.name for argument in spec.arguments}):
        raise ToolCallError(
            f"{spec.name} does not accept {unknown}. Accepted arguments: "
            f"{sorted(argument.name for argument in spec.arguments)}."
        )

    path_values: dict[str, str] = {}
    query: list[tuple[str, str]] = []
    # Reserved names are stripped case-insensitively BEFORE the credential is
    # set. A dict holding both `authorization` and `Authorization` is two
    # header lines on the wire, and which one the API reads is not ours to
    # decide: a forwarded value would silently become the credential.
    headers: dict[str, str] = {
        name: value
        for name, value in (forwarded_headers or {}).items()
        if name.lower() not in RESERVED_HEADERS
    }
    body: Any = None

    for argument in spec.arguments:
        value = arguments.get(argument.name)
        if value is None:
            continue
        if argument.location == "path":
            path_values[argument.wire_name] = str(value)
        elif argument.location == "query":
            query.extend(_query_pairs(argument.wire_name, value))
        elif argument.location == "header":
            headers[argument.wire_name] = str(value)
        else:
            body = value

    headers["Authorization"] = credential
    request = client.build_request(
        spec.method,
        _fill(spec, path_values),
        params=query or None,
        json=body,
        headers=headers,
    )
    # STREAMED so the size can be refused before the bytes are resident.
    # `client.request()` reads the whole body first, and an ASGI transport
    # buffers every chunk into one `bytes`, so a cap applied to the returned
    # value has already lost: a download route serves whatever the upload path
    # accepted, and one tool call would spend the full file size in RSS to
    # produce a refusal.
    response = await client.send(request, stream=True)
    try:
        media_type = (_header(response.headers, "content-type") or "").split(";")[0].strip()
        declared = _declared_length(_header(response.headers, "content-length"))
        if declared is not None and declared > max_bytes:
            return HttpResult(
                status=response.status_code,
                media_type=media_type,
                body=b"",
                declared_length=declared,
                truncated=True,
            )
        body_bytes = await _read_capped(response, max_bytes)
    finally:
        await response.aclose()

    if body_bytes is None:
        # No `Content-Length` and it ran past the cap anyway (a chunked or
        # streamed response). Refused on the same terms rather than trusted.
        return HttpResult(
            status=response.status_code,
            media_type=media_type,
            body=b"",
            declared_length=None,
            truncated=True,
        )
    return HttpResult(status=response.status_code, media_type=media_type, body=body_bytes)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Read one header without caring about case, whatever mapping the client returned.

    `httpx.Headers` already ignores case, but the injected client protocol only promises a
    `Mapping`, and a plain dict keeps the case the server sent.
    """
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _declared_length(raw: str | None) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


async def _read_capped(response: HttpResponse, max_bytes: int) -> bytes | None:
    """The body, or `None` once it exceeds the cap.

    Stops pulling chunks at the limit rather than reading to the end and
    measuring, which is the whole point: an unbounded read is what the cap was
    supposed to prevent.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _fill(spec: ToolSpec, values: dict[str, str]) -> str:
    """The concrete path, or a refusal naming what is missing.

    Percent-encoding with `safe=""` so a value containing a slash cannot invent
    a path segment and reach an operation the caller did not name.
    """
    path = spec.path
    for name in (argument.wire_name for argument in spec.arguments if argument.location == "path"):
        if name not in values:
            raise ToolCallError(f"{spec.name} requires {name!r}, which was not supplied.")
        path = path.replace("{" + name + "}", quote(values[name], safe=""))
    return path


def _query_pairs(name: str, value: Any) -> list[tuple[str, str]]:
    """Repeat the key for a list, the usual spelling of a multi-value filter."""
    if isinstance(value, (list, tuple)):
        return [(name, _scalar(item)) for item in value if item is not None]
    return [(name, _scalar(value))]


def _scalar(value: Any) -> str:
    """`True` has to reach the wire as `true`, not as Python's `True`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
