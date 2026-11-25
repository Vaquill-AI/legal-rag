# mcp_bridge

`legal_rag.mcp_bridge` turns an OpenAPI document into an MCP tool catalogue.
A tool call becomes exactly one HTTP request through a client you inject, and the HTTP response becomes a tool result.
It does not run an MCP server and knows nothing about any particular API: you supply the transport and the credential.

## Install

The bridge uses only the standard library and imports neither `mcp` nor `httpx`.
The `mcp` extra is needed only for `ToolResult.to_mcp()`, which imports the SDK lazily.

```bash
pip install .          # from a checkout of the repository
pip install ".[mcp]"   # adds the mcp SDK, only for ToolResult.to_mcp()
```

## Pipeline

```text
openapi (dict) + routes (list of RouteRef)
  -> build_catalogue(openapi=..., routes=...)                   tuple of ToolSpec
  -> ToolSpec                                                   name, input_schema, arguments, read_only
  -> call_operation(spec, args, client=..., credential=...)     one HTTP request
  -> HttpResult                                                 status, media_type, body, truncated
  -> to_tool_result(spec, http)
  -> ToolResult                                                 content, is_error, structured_content
       .as_dict()   MCP wire form ("isError")
       .to_mcp()    the SDK's CallToolResult (needs the mcp extra)
```

## Quick start

This runs as written against a fake client: one operation, a 200 and a 404.

```python
import asyncio
import json

from legal_rag.mcp_bridge import RouteRef, build_catalogue, call_operation, to_tool_result

OPENAPI = {
    "paths": {
        "/notes/{noteId}": {
            "get": {
                "operationId": "notes.getNote",
                "parameters": [
                    {"name": "noteId", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "includeHistory", "in": "query", "schema": {"type": "boolean"}},
                ],
            }
        }
    }
}
ROUTES = [RouteRef("notes.getNote", "GET", "/notes/{noteId}", scopes=("notes:read",))]


class Response:
    def __init__(self, status, payload):
        self.status_code, self.headers = status, {"content-type": "application/json"}
        self.body = json.dumps(payload).encode()

    async def aiter_bytes(self):
        yield self.body

    async def aclose(self):
        pass


class Client:
    def build_request(self, method, url, *, params=None, json=None, headers=None):
        print(method, url, params, headers["Authorization"])
        return url

    async def send(self, request, *, stream=False):
        if request.endswith("/missing"):
            return Response(404, {"title": "Not Found", "detail": "No such note."})
        return Response(200, {"id": request.rsplit("/", 1)[1]})


async def main():
    (spec,) = build_catalogue(openapi=OPENAPI, routes=ROUTES)
    print(spec.name, spec.read_only, spec.scopes, sorted(spec.input_schema["properties"]))
    for note_id in ("n1", "missing"):
        args = {"noteId": note_id, "includeHistory": True}
        raw = await call_operation(spec, args, client=Client(), credential="Bearer demo")
        result = to_tool_result(spec, raw)
        print(raw.status, result.is_error, result.content[0]["text"].replace("\n", " / "))


asyncio.run(main())
```

```text
notes_get_note True ('notes:read',) ['includeHistory', 'noteId']
GET /notes/n1 [('includeHistory', 'true')] Bearer demo
200 False { /   "id": "n1" / }
GET /notes/missing [('includeHistory', 'true')] Bearer demo
404 True Not Found / No such note.
```

A 4xx is a tool result with `is_error=True`, not an exception, so the model sees the error document.
`build_catalogue` refuses to build unless the declared `RouteRef` set and the document agree in both directions.

## The HTTP client you inject

`call_operation` accepts anything shaped like this protocol (`dispatch.py`).

```python
class HttpClient(Protocol):
    def build_request(
        self, method: str, url: str, *,
        params: list[tuple[str, str]] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any: ...

    async def send(self, request: Any, *, stream: bool = False) -> HttpResponse: ...
```

The response needs `status_code`, `headers`, `aiter_bytes()` and `async aclose()`.
The dispatcher always calls `send(request, stream=True)` and always closes the response.
`url` is the path with path arguments percent-encoded and no host, so the client supplies the base URL.
`headers` always carries `Authorization` set to the `credential` argument, never to a tool argument.
The dispatcher reads `content-type` and `content-length` without regard to case, so a plain-dict `headers` may use any spelling.

The next snippet is NOT run in this document, and this document does not claim `httpx` is installed (the bridge does not depend on it).
It relies on `httpx.AsyncClient` having the two methods above, as the module docstring states.

```python
import httpx

from legal_rag.mcp_bridge import ToolSpec, call_operation, to_tool_result


def make_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, timeout=30.0)


def make_in_process_client(asgi_app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi_app), base_url="http://app.test")


async def run_tool(spec: ToolSpec, arguments: dict, client: httpx.AsyncClient, token: str):
    http = await call_operation(spec, arguments, client=client, credential=f"Bearer {token}")
    return to_tool_result(spec, http)
```

## Naming, schema and enum rules

| Concern | Rule |
| --- | --- |
| Tool name (`naming.py`) | Split the operation id on `.`, split camelCase where a lowercase letter or digit meets an uppercase one, lowercase, join with `_`. The result must match `^[a-zA-Z0-9_-]{1,128}$` or `ToolNameError` is raised. |
| Collisions | Two operation ids that map to one tool name (`a.getB` and `a.get_b`) raise `ToolNameError` at build time. |
| Argument names (`schema.py`) | Path and query names are used as the document spells them. Header names become lowerCamel (`Idempotency-Key` becomes `idempotencyKey`). |
| Request body | A JSON body is nested under one `body` argument, never flattened. Any other content type raises `SchemaTranslationError`. |
| Schema | `#/components/schemas/X` becomes `#/$defs/X`, only reachable components are copied, and a missing component raises. The schema is closed (`additionalProperties: false`) and `required` is sorted. |
| Reserved headers | A header parameter named `authorization`, `proxy-authorization`, `cookie` or `host` raises `SchemaTranslationError`. The same names are dropped from `forwarded_headers`. |
| `required_arguments` | A `RouteRef` field that promotes an optional argument to required. An undeclared name raises. |
| Hints | `read_only` is GET, `destructive` is DELETE, `idempotent` is GET, PUT and DELETE (never POST). |
| `enum_values` (`openapi_enum.py`) | A `json_schema_extra` that puts the enum on each string branch of an `anyOf` and on the items of each string-array branch, never at the top level. `null` stays valid. |

Parameters declared on the path item are merged into every operation under it, and `{"$ref": "#/components/parameters/Name"}` entries are resolved.
An operation's own parameter replaces a shared one with the same name and location.
A reference that does not resolve raises `CatalogueError`, because dropping it would leave a placeholder such as `{id}` unfilled in the URL.

```python
import json

from legal_rag.mcp_bridge import CatalogueError, RouteRef, build_catalogue, enum_values, tool_name

print(tool_name("statutes.getSection"), tool_name("cases.search"))

JOB = {"$ref": "#/components/schemas/Job"}
OPENAPI = {
    "paths": {"/jobs": {"post": {
        "operationId": "jobs.create",
        "parameters": [
            {"name": "Idempotency-Key", "in": "header", "schema": {"type": "string"}},
            {"name": "dryRun", "in": "query", "schema": {"type": "boolean"}},
        ],
        "requestBody": {"required": True, "content": {"application/json": {"schema": JOB}}},
    }}},
    "components": {"schemas": {"Job": {"type": "object"}}},
}
routes = [RouteRef("jobs.create", "POST", "/jobs", required_arguments=("idempotencyKey",))]
(spec,) = build_catalogue(openapi=OPENAPI, routes=routes)
print(json.dumps(spec.input_schema))
print([(a.name, a.location, a.wire_name) for a in spec.arguments])
print(spec.read_only, spec.destructive, spec.idempotent)

schema = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}]}
enum_values(["ca", "tx"])(schema)
print(json.dumps(schema))

try:
    build_catalogue(openapi=OPENAPI, routes=[])
except CatalogueError as error:
    print("refused:", str(error)[:62])
```

## Result mapping

| HTTP answer | Tool result |
| --- | --- |
| 2xx JSON object | Text is the indented JSON and `structured_content` is the object. A list body is text only. |
| Status 400 or above, JSON | `is_error=True`. A problem document (has `title`) becomes prose: `status title`, `detail`, extra members as `key: json`, then `See <type>`. Other JSON is shown as is. |
| Unparseable JSON | `is_error=True`, even on a 200. |
| 204 or empty body | `<status>, no content.`, with `is_error` true from status 400 up. |
| `text/*` | Decoded text, `is_error` true from status 400 up. |
| Binary up to `MAX_INLINE_BYTES` (1,048,576) | One `resource` block: URI `mcp+http://<tool name>`, the media type and a base64 `blob`. |
| Binary above that | Described, not inlined: size, media type and `Call <METHOD> <path> over HTTP to fetch it.` |
| Body above `max_bytes` in `call_operation` (default 8 MiB) | Refused unread when `Content-Length` is over the cap, or the read stops at the cap. `HttpResult.truncated` is true and the body is empty. The text says the size exceeds what a tool call will carry. |

The last two rows keep `is_error` true from status 400 up, so a 5xx with an oversized body is still flagged, and a 200 that was too large to carry is not an error.

```python
from legal_rag.mcp_bridge import HttpResult, RouteRef, build_catalogue, to_tool_result
from legal_rag.mcp_bridge.result import MAX_INLINE_BYTES

OPENAPI = {"paths": {"/report": {"get": {"operationId": "reports.get"}}}}
(spec,) = build_catalogue(openapi=OPENAPI, routes=[RouteRef("reports.get", "GET", "/report")])

pdf = "application/pdf"
cases = {
    "small blob": HttpResult(200, pdf, b"%PDF-1.7"),
    "large blob": HttpResult(200, pdf, b"x" * (MAX_INLINE_BYTES + 1)),
    "refused unread": HttpResult(200, pdf, b"", declared_length=9_000_000, truncated=True),
    "empty 403": HttpResult(403, "application/json", b""),
    "bad json 200": HttpResult(200, "application/json", b"{oops"),
}
for label, http in cases.items():
    result = to_tool_result(spec, http)
    block = result.content[0]
    print(f"{label:15} is_error={result.is_error!s:5} {block.get('text', block['type'])[:60]!r}")

problem = b'{"title": "Not Found", "status": 404}'
wire = to_tool_result(spec, HttpResult(404, "application/problem+json", problem))
print(wire.as_dict())
try:
    sdk = wire.to_mcp()
except ImportError:
    print("mcp is not installed; as_dict() is unaffected")
else:
    print([n for n in ("isError", "is_error") if hasattr(sdk, n)], sdk.model_dump(by_alias=True)["isError"])
```

## MCP SDK attribute drift

`ToolResult` is a plain dataclass with `is_error`, and it never imports `mcp`.
`ToolResult.as_dict()` always emits the MCP wire keys `content`, `structuredContent` (when set) and `isError`.
`ToolResult.to_mcp()` validates that dict into the SDK's `CallToolResult`.
The SDK spells the attribute `isError` in 1.x and `is_error` in 2.x (per the note in `tests/mcp_bridge/test_mcp_bridge_result.py`).
The last line of the snippet above ran against `mcp` 2.3.0 and found only `is_error`.
The stable form is the wire form: read `as_dict()["isError"]` or `sdk.model_dump(by_alias=True)["isError"]`, not the attribute.

## Tests

The tests live in `tests/mcp_bridge/`.
`support.py` holds an inline OpenAPI document and a recording fake client.
`test_mcp_bridge_catalogue.py`, `test_mcp_bridge_dispatch.py` and `test_mcp_bridge_result.py` cover the three stages.
`test_mcp_bridge_hygiene.py` scans the source for forbidden strings, checks that only the standard library is imported (apart from the lazy `mcp` import), and checks that importing the bridge loads neither `mcp` nor `httpx`.
A runnable end-to-end example is `examples/mcp_bridge_basic.py`.
