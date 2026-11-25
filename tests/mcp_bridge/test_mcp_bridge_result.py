"""Result mapping: 4xx is isError, 2xx is text content, and nothing imports `mcp` eagerly."""

from __future__ import annotations

import base64
import json

import pytest

from legal_rag.mcp_bridge import HttpResult, enum_values, to_tool_result
from legal_rag.mcp_bridge.result import MAX_INLINE_BYTES

from .support import catalogue

SPEC = catalogue()["statutes_get_section"]


def _http(status: int, media_type: str, body: bytes, **extra) -> HttpResult:
    return HttpResult(status=status, media_type=media_type, body=body, **extra)


def _json(status: int, payload, media_type: str = "application/json") -> HttpResult:
    return _http(status, media_type, json.dumps(payload).encode())


def test_a_2xx_json_object_is_text_content_with_structured_content():
    payload = {"citation": "26 U.S.C. 501", "text": "Exemption from tax"}
    result = to_tool_result(SPEC, _json(200, payload))

    assert result.is_error is False
    assert result.structured_content == payload
    (block,) = result.content
    assert block["type"] == "text"
    assert json.loads(block["text"]) == payload
    assert result.as_dict() == {
        "content": [block],
        "structuredContent": payload,
        "isError": False,
    }


def test_a_list_valued_body_is_text_only_with_no_structured_content():
    result = to_tool_result(SPEC, _json(200, [{"id": 1}, {"id": 2}]))
    assert result.is_error is False
    assert result.structured_content is None
    assert "structuredContent" not in result.as_dict()
    assert json.loads(result.content[0]["text"]) == [{"id": 1}, {"id": 2}]


def test_a_4xx_problem_document_is_an_error_result_not_an_exception():
    problem = {
        "type": "https://example.test/problems/not-found",
        "title": "Not Found",
        "status": 404,
        "detail": "No section 999 in code usc.",
        "requestId": "r-1",
        "scope": "statutes:read",
    }
    result = to_tool_result(SPEC, _json(404, problem, "application/problem+json"))

    assert result.is_error is True
    assert result.content[0]["text"].splitlines() == [
        "404 Not Found",
        "No section 999 in code usc.",
        'scope: "statutes:read"',
        "See https://example.test/problems/not-found",
    ]
    assert result.structured_content == problem


@pytest.mark.parametrize("status", [400, 403, 404, 422, 429, 500, 503])
def test_every_error_status_is_marked_is_error(status):
    assert to_tool_result(SPEC, _json(status, {"title": "Refused"})).is_error is True


def test_an_error_body_that_is_not_a_problem_document_is_shown_as_json():
    result = to_tool_result(SPEC, _json(400, {"error": "bad request"}))
    assert result.is_error is True
    assert json.loads(result.content[0]["text"]) == {"error": "bad request"}


def test_unparseable_json_is_an_error_even_on_a_2xx():
    result = to_tool_result(SPEC, _http(200, "application/json", b"{not json"))
    assert result.is_error is True
    assert result.content[0]["text"] == "200, unparseable JSON: {not json"


def test_empty_bodies_report_no_content_and_follow_the_status():
    ok = to_tool_result(SPEC, _http(204, "", b""))
    assert (ok.content[0]["text"], ok.is_error) == ("204, no content.", False)
    refused = to_tool_result(SPEC, _http(403, "application/json", b""))
    assert (refused.content[0]["text"], refused.is_error) == ("403, no content.", True)


def test_text_bodies_are_passed_through_and_follow_the_status():
    ok = to_tool_result(SPEC, _http(200, "text/plain", "Section 501 §".encode()))
    assert (ok.content[0]["text"], ok.is_error) == ("Section 501 §", False)
    failed = to_tool_result(SPEC, _http(502, "text/html", b"<h1>bad gateway</h1>"))
    assert failed.is_error is True


def test_a_small_binary_body_is_an_embedded_blob():
    pdf = b"%PDF-1.7 statute"
    result = to_tool_result(SPEC, _http(200, "application/pdf", pdf))
    assert result.is_error is False
    (block,) = result.content
    assert block["type"] == "resource"
    assert block["resource"] == {
        "uri": "mcp+http://statutes_get_section",
        "mimeType": "application/pdf",
        "blob": base64.b64encode(pdf).decode("ascii"),
    }
    assert to_tool_result(SPEC, _http(500, "application/pdf", pdf)).is_error is True


def test_a_large_binary_body_is_described_not_inlined():
    body = b"x" * (MAX_INLINE_BYTES + 1)
    result = to_tool_result(SPEC, _http(200, "application/pdf", body))
    assert result.is_error is False
    text = result.content[0]["text"]
    assert f"{len(body)} bytes of application/pdf" in text
    assert "Call GET /us/statutes/{code}/sections/{section} over HTTP" in text


def test_a_truncated_answer_says_how_big_it_was_and_how_to_fetch_it():
    sized = to_tool_result(
        SPEC, _http(200, "application/pdf", b"", declared_length=9_000_000, truncated=True)
    )
    assert sized.is_error is False
    assert "9000000 bytes exceeds" in sized.content[0]["text"]
    unsized = to_tool_result(SPEC, _http(200, "", b"", truncated=True))
    assert "its size exceeds" in unsized.content[0]["text"]


def test_to_mcp_builds_the_sdk_type_when_the_package_is_installed():
    types = pytest.importorskip("mcp.types")
    result = to_tool_result(SPEC, _json(404, {"title": "Not Found", "status": 404}))
    mcp_result = result.to_mcp()
    assert isinstance(mcp_result, types.CallToolResult)
    # The SDK spells the attribute `isError` in 1.x and `is_error` in 2.x. The wire form is stable.
    wire = mcp_result.model_dump(by_alias=True, exclude_none=True)
    assert wire["isError"] is True
    assert wire["content"][0]["text"] == result.content[0]["text"]


def test_enum_values_constrains_every_string_branch_and_list_item_but_not_null():
    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ]
    }
    enum_values(["ca", "tx"])(schema)
    assert schema == {
        "anyOf": [
            {"type": "string", "enum": ["ca", "tx"]},
            {"type": "array", "items": {"type": "string", "enum": ["ca", "tx"]}},
            {"type": "null"},
        ]
    }
    plain = {"type": "string"}
    enum_values(("a",))(plain)
    assert plain == {"type": "string", "enum": ["a"]}
