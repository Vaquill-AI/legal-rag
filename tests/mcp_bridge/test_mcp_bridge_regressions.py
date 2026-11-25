"""Regressions found while documenting the bridge: shared parameters, error status, header case."""

from __future__ import annotations

import copy

import pytest

from legal_rag.mcp_bridge import (
    CatalogueError,
    HttpResult,
    RouteRef,
    build_catalogue,
    call_operation,
    to_tool_result,
)

from .support import OPENAPI, ROUTES, FakeClient, FakeResponse, RecordedRequest, catalogue

SECTION = catalogue()["statutes_get_section"]


def _document_with_shared_parameters() -> dict:
    """A path item that declares `code` once for every method, plus a `$ref` parameter."""
    document = copy.deepcopy(OPENAPI)
    path = "/us/codes/{code}/outline"
    document["components"] = {
        **document.get("components", {}),
        "parameters": {
            "Depth": {
                "name": "depth",
                "in": "query",
                "required": False,
                "schema": {"type": "integer"},
            }
        },
    }
    document["paths"][path] = {
        "parameters": [
            {"name": "code", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "depth", "in": "query", "required": False, "schema": {"type": "string"}},
        ],
        "get": {
            "operationId": "codes.outline",
            "summary": "Outline of a code",
            "parameters": [{"$ref": "#/components/parameters/Depth"}],
        },
    }
    return document


def test_path_item_and_ref_parameters_reach_the_tool_schema():
    routes = [*ROUTES, RouteRef("codes.outline", "GET", "/us/codes/{code}/outline")]
    specs = {
        s.name: s
        for s in build_catalogue(openapi=_document_with_shared_parameters(), routes=routes)
    }
    properties = specs["codes_outline"].input_schema["properties"]

    assert "code" in properties
    assert "code" in specs["codes_outline"].input_schema["required"]
    # The operation's own `$ref` replaces the shared `depth`, so it is the integer one.
    assert properties["depth"]["type"] == "integer"


async def test_a_shared_path_parameter_is_filled_into_the_url():
    routes = [*ROUTES, RouteRef("codes.outline", "GET", "/us/codes/{code}/outline")]
    specs = {
        s.name: s
        for s in build_catalogue(openapi=_document_with_shared_parameters(), routes=routes)
    }
    client = FakeClient.answering_json({"ok": True})

    await call_operation(
        specs["codes_outline"], {"code": "usc", "depth": 2}, client=client, credential="Bearer k"
    )

    (request,) = client.requests
    assert request.url == "/us/codes/usc/outline"
    assert request.params == [("depth", "2")]


def test_a_reference_that_does_not_resolve_is_refused_not_dropped():
    document = _document_with_shared_parameters()
    document["paths"]["/us/codes/{code}/outline"]["get"]["parameters"] = [
        {"$ref": "#/components/parameters/Missing"}
    ]
    routes = [*ROUTES, RouteRef("codes.outline", "GET", "/us/codes/{code}/outline")]
    with pytest.raises(CatalogueError, match="does not resolve"):
        build_catalogue(openapi=document, routes=routes)


@pytest.mark.parametrize("status", [404, 500])
def test_an_error_whose_body_was_not_transferred_is_still_an_error(status):
    oversized = HttpResult(
        status=status,
        media_type="application/json",
        body=b"",
        declared_length=9_000_000,
        truncated=True,
    )
    assert to_tool_result(SECTION, oversized).is_error is True


def test_a_successful_answer_that_was_too_large_is_not_an_error():
    oversized = HttpResult(
        status=200,
        media_type="application/pdf",
        body=b"",
        declared_length=9_000_000,
        truncated=True,
    )
    assert to_tool_result(SECTION, oversized).is_error is False


class _TitleCaseClient(FakeClient):
    """Answers with headers spelled the way many servers send them."""

    async def send(self, request: RecordedRequest, *, stream: bool = False) -> FakeResponse:
        response = await super().send(request, stream=stream)
        response.headers = {
            "Content-Type": response.headers["content-type"],
            **(
                {"Content-Length": response.headers["content-length"]}
                if "content-length" in response.headers
                else {}
            ),
        }
        return response


async def test_response_headers_are_read_without_regard_to_case():
    client = _TitleCaseClient.answering_json({"text": "Exemption from tax"})
    result = await call_operation(
        SECTION, {"code": "usc", "section": "501"}, client=client, credential="Bearer k"
    )
    assert result.media_type == "application/json"


async def test_a_declared_length_in_a_title_case_header_still_triggers_the_size_cap():
    client = _TitleCaseClient(body=b"x" * 64, content_type="application/pdf", declare_length=True)
    result = await call_operation(
        SECTION,
        {"code": "usc", "section": "501"},
        client=client,
        credential="Bearer k",
        max_bytes=8,
    )
    assert result.truncated is True
    assert result.declared_length == 64
