"""A small inline OpenAPI document and a fake httpx-style client that records requests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from legal_rag.mcp_bridge import RouteRef, ToolSpec, build_catalogue

SECTION_PATH = "/us/statutes/{code}/sections/{section}"

OPENAPI: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"title": "Legal data", "version": "1"},
    "paths": {
        SECTION_PATH: {
            "get": {
                "operationId": "statutes.getSection",
                "summary": "Look up a statute section",
                "description": "Returns one section of a code, such as 26 U.S.C. 501.",
                "parameters": [
                    {"name": "code", "in": "path", "required": True, "schema": {"type": "string"}},
                    {
                        "name": "section",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "effectiveDate",
                        "in": "query",
                        "required": False,
                        "description": "Return the text in force on this date.",
                        "schema": {"type": "string", "format": "date"},
                    },
                    {
                        "name": "includeHistory",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean"},
                    },
                    {
                        "name": "X-Request-Id",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                ],
            }
        },
        "/us/cases/search": {
            "post": {
                "operationId": "cases.search",
                "summary": "Search case law",
                "parameters": [
                    {
                        "name": "court",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "array", "items": {"type": "string"}},
                    },
                    {
                        "name": "Idempotency-Key",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CaseSearchRequest"}
                        }
                    },
                },
            }
        },
        "/us/cases/saved/{savedId}": {
            "delete": {
                "operationId": "savedCases.delete",
                "summary": "Remove a saved case",
                "parameters": [
                    {
                        "name": "savedId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            }
        },
    },
    "components": {
        "schemas": {
            "CaseSearchRequest": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "level": {"$ref": "#/components/schemas/CourtLevel"},
                },
                "required": ["query"],
            },
            "CourtLevel": {"type": "string", "enum": ["trial", "appellate", "supreme"]},
            "NeverReferenced": {"type": "object"},
        }
    },
}

ROUTES: list[RouteRef] = [
    RouteRef("statutes.getSection", "GET", SECTION_PATH, scopes=("statutes:read",)),
    RouteRef("cases.search", "POST", "/us/cases/search", scopes=("cases:read", "cases:search")),
    RouteRef("savedCases.delete", "DELETE", "/us/cases/saved/{savedId}"),
]


def catalogue() -> dict[str, ToolSpec]:
    """The catalogue for the inline document, keyed by tool name."""
    return {spec.name: spec for spec in build_catalogue(openapi=OPENAPI, routes=ROUTES)}


@dataclass
class RecordedRequest:
    method: str
    url: str
    params: list[tuple[str, str]] | None
    json: Any
    headers: dict[str, str]


@dataclass
class FakeResponse:
    status_code: int
    headers: dict[str, str]
    chunks: list[bytes]
    closed: bool = False
    chunks_read: int = 0

    async def aiter_bytes(self) -> Any:
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeClient:
    """Records each request and answers with one canned response."""

    status: int = 200
    body: bytes = b"{}"
    content_type: str = "application/json"
    declare_length: bool = True
    chunks: list[bytes] | None = None
    requests: list[RecordedRequest] = field(default_factory=list)
    responses: list[FakeResponse] = field(default_factory=list)

    @classmethod
    def answering_json(cls, payload: Any, *, status: int = 200, content_type: str | None = None):
        return cls(
            status=status,
            body=json.dumps(payload).encode(),
            content_type=content_type or "application/json",
        )

    def build_request(
        self,
        method: str,
        url: str,
        *,
        params: list[tuple[str, str]] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> RecordedRequest:
        request = RecordedRequest(method, url, params, json, dict(headers or {}))
        self.requests.append(request)
        return request

    async def send(self, request: RecordedRequest, *, stream: bool = False) -> FakeResponse:
        assert stream is True, "the dispatcher must stream so it can refuse an oversized body"
        headers = {"content-type": f"{self.content_type}; charset=utf-8"}
        if self.declare_length:
            headers["content-length"] = str(len(self.body))
        chunks = self.chunks if self.chunks is not None else [self.body]
        response = FakeResponse(self.status, headers, chunks)
        self.responses.append(response)
        return response
