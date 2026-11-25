"""Turn an OpenAPI document into MCP tools and call one, with no server and no network.

Shows: build_catalogue (one tool per declared operation), call_operation (one tool call becomes
one HTTP request on an injected client) and to_tool_result (a 2xx is text content, a 4xx is
isError). The client is a fake that prints the request; an httpx.AsyncClient fits the same slot.
Needs: nothing beyond the base install. The bridge never imports `mcp` or `httpx`.

Run: python examples/mcp_bridge_basic.py
"""

import asyncio
import json

from legal_rag.mcp_bridge import RouteRef, build_catalogue, call_operation, to_tool_result

PATH = "/us/statutes/{code}/sections/{section}"
PARAMS = [
    {"name": n, "in": "path", "required": True, "schema": {"type": "string"}}
    for n in ("code", "section")
]
GET = {"operationId": "statutes.getSection", "summary": "Look up a section", "parameters": PARAMS}
BODY = {"required": True, "content": {"application/json": {"schema": {"type": "object"}}}}
POST = {"operationId": "cases.search", "summary": "Search cases", "requestBody": BODY}
OPENAPI = {"paths": {PATH: {"get": GET}, "/us/cases/search": {"post": POST}}}
ROUTES = [
    RouteRef("statutes.getSection", "GET", PATH, scopes=("statutes:read",)),
    RouteRef("cases.search", "POST", "/us/cases/search"),
]


class FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code, self._body = status, json.dumps(payload).encode()
        self.headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield self._body

    async def aclose(self) -> None: ...


class FakeClient:
    """Prints the request it is asked to build and answers 404 for section 9999."""

    def build_request(self, method, url, *, params=None, json=None, headers=None):
        print(f"  HTTP {method} {url} params={params} body={json}")
        return url

    async def send(self, request, *, stream=False):
        if request.endswith("/9999"):
            return FakeResponse(404, {"title": "Not Found", "detail": "No section 9999."})
        return FakeResponse(200, {"citation": "26 U.S.C. 501", "heading": "Exemption from tax"})


async def main() -> None:
    tools = {spec.name: spec for spec in build_catalogue(openapi=OPENAPI, routes=ROUTES)}
    spec = tools["statutes_get_section"]
    print("tools:", sorted(tools), "| read_only:", spec.read_only)
    for section in ("501", "9999"):
        print(f"tool call: section {section}")
        args = {"code": "usc", "section": section}
        raw = await call_operation(spec, args, client=FakeClient(), credential="Bearer demo")
        result = to_tool_result(spec, raw)
        print(f"  is_error={result.is_error}", result.content[0]["text"].replace("\n", " | "))


asyncio.run(main())
