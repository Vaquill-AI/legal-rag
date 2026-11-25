"""Dispatch: one tool call becomes one recorded HTTP request."""

from __future__ import annotations

import pytest

from legal_rag.mcp_bridge import ToolCallError, call_operation

from .support import FakeClient, catalogue

CREDENTIAL = "Bearer test-key"


async def test_get_builds_the_url_query_headers_and_credential():
    client = FakeClient.answering_json({"text": "..."})
    arguments = {
        "code": "usc",
        "section": "26/501",
        "effectiveDate": "2024-01-01",
        "includeHistory": True,
        "xRequestId": "req-1",
    }
    result = await call_operation(
        catalogue()["statutes_get_section"], arguments, client=client, credential=CREDENTIAL
    )

    (request,) = client.requests
    assert request.method == "GET"
    # The slash inside a path value is encoded, so it cannot invent a path segment.
    assert request.url == "/us/statutes/usc/sections/26%2F501"
    assert request.params == [("effectiveDate", "2024-01-01"), ("includeHistory", "true")]
    assert request.json is None
    assert request.headers == {"X-Request-Id": "req-1", "Authorization": CREDENTIAL}
    assert (result.status, result.media_type, result.body) == (
        200,
        "application/json",
        b'{"text": "..."}',
    )


async def test_post_sends_the_body_repeats_list_query_keys_and_maps_headers():
    client = FakeClient.answering_json({"results": []})
    body = {"query": "adverse possession", "level": "appellate"}
    arguments = {"body": body, "court": ["ca9", "ca2"], "idempotencyKey": "abc-123"}
    await call_operation(
        catalogue()["cases_search"], arguments, client=client, credential=CREDENTIAL
    )

    (request,) = client.requests
    assert request.method == "POST"
    assert request.url == "/us/cases/search"
    assert request.params == [("court", "ca9"), ("court", "ca2")]
    assert request.json == body
    assert request.headers == {"Idempotency-Key": "abc-123", "Authorization": CREDENTIAL}


async def test_delete_percent_encodes_the_path_value():
    client = FakeClient(status=204, body=b"")
    await call_operation(
        catalogue()["saved_cases_delete"],
        {"savedId": "a b/c"},
        client=client,
        credential=CREDENTIAL,
    )
    assert client.requests[0].method == "DELETE"
    assert client.requests[0].url == "/us/cases/saved/a%20b%2Fc"
    assert client.requests[0].params is None


async def test_none_values_are_skipped_and_list_items_that_are_none_are_dropped():
    client = FakeClient()
    await call_operation(
        catalogue()["cases_search"],
        {"body": {"query": "x"}, "court": ["ca9", None], "idempotencyKey": None},
        client=client,
        credential=CREDENTIAL,
    )
    assert client.requests[0].params == [("court", "ca9")]
    assert "Idempotency-Key" not in client.requests[0].headers


async def test_the_credential_always_wins_over_a_forwarded_authorization_header():
    client = FakeClient()
    forwarded = {
        "authorization": "Bearer stolen",
        "Cookie": "session=1",
        "HOST": "evil.example",
        "X-Trace": "t-1",
    }
    await call_operation(
        catalogue()["saved_cases_delete"],
        {"savedId": "1"},
        client=client,
        credential=CREDENTIAL,
        forwarded_headers=forwarded,
    )
    assert client.requests[0].headers == {"X-Trace": "t-1", "Authorization": CREDENTIAL}


async def test_an_unknown_argument_is_refused_before_anything_is_sent():
    client = FakeClient()
    with pytest.raises(ToolCallError, match="does not accept"):
        await call_operation(
            catalogue()["saved_cases_delete"],
            {"savedId": "1", "force": True},
            client=client,
            credential=CREDENTIAL,
        )
    assert client.requests == []


async def test_a_missing_path_argument_is_refused_before_anything_is_sent():
    client = FakeClient()
    with pytest.raises(ToolCallError, match="requires 'section'"):
        await call_operation(
            catalogue()["statutes_get_section"],
            {"code": "usc"},
            client=client,
            credential=CREDENTIAL,
        )
    assert client.requests == []


async def test_the_response_is_always_closed():
    client = FakeClient()
    await call_operation(
        catalogue()["saved_cases_delete"], {"savedId": "1"}, client=client, credential=CREDENTIAL
    )
    assert client.responses[0].closed is True


async def test_a_declared_length_over_the_cap_is_refused_without_reading_the_body():
    client = FakeClient(body=b"x" * 50)
    result = await call_operation(
        catalogue()["saved_cases_delete"],
        {"savedId": "1"},
        client=client,
        credential=CREDENTIAL,
        max_bytes=10,
    )
    assert result.truncated is True
    assert result.body == b""
    assert result.declared_length == 50
    assert client.responses[0].chunks_read == 0
    assert client.responses[0].closed is True


async def test_an_undeclared_length_that_runs_past_the_cap_stops_reading():
    client = FakeClient(declare_length=False, chunks=[b"x" * 6, b"x" * 6, b"x" * 6])
    result = await call_operation(
        catalogue()["saved_cases_delete"],
        {"savedId": "1"},
        client=client,
        credential=CREDENTIAL,
        max_bytes=10,
    )
    assert (result.truncated, result.body, result.declared_length) == (True, b"", None)
    assert client.responses[0].chunks_read == 2
    assert client.responses[0].closed is True


async def test_a_body_exactly_at_the_cap_is_carried():
    client = FakeClient(body=b"x" * 10)
    result = await call_operation(
        catalogue()["saved_cases_delete"],
        {"savedId": "1"},
        client=client,
        credential=CREDENTIAL,
        max_bytes=10,
    )
    assert result.truncated is False
    assert result.body == b"x" * 10
