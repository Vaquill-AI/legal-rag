"""Accumulators, semantic cache, payload validation and SSE events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from legal_rag.providers.fakes import FakeCache, FakeEmbedder
from legal_rag.retrieval.accumulators.neighbor import NeighborAccumulator, _fit_to_token_budget
from legal_rag.retrieval.caching.payload_validator import (
    PayloadField,
    PayloadSchema,
    validate_payload,
)
from legal_rag.retrieval.caching.semantic import SemanticCache, SemanticCacheConfig
from legal_rag.retrieval.streaming.events import SSEEventType, chunk_event, heartbeat_event
from legal_rag.types import RetrievedChunk


def run(coro):
    return asyncio.run(coro)


class _Fetcher:
    def __init__(self, chunks: Sequence[RetrievedChunk]):
        self.by_id = {c.id: c for c in chunks}
        self.scopes: list[str | None] = []

    async def fetch_by_ids(self, chunk_ids, *, scope=None):
        self.scopes.append(scope)
        return [self.by_id[i] for i in chunk_ids if i in self.by_id]


def _chain():
    a = RetrievedChunk("c1", "A", 0.1, {"next_chunk_id": "c2", "document_id": "d"})
    b = RetrievedChunk(
        "c2", "B", 0.9, {"prev_chunk_id": "c1", "next_chunk_id": "c3", "document_id": "d"}
    )
    c = RetrievedChunk("c3", "C", 0.1, {"prev_chunk_id": "c2", "document_id": "d"})
    return a, b, c


def test_neighbors_are_merged_in_reading_order_and_scope_is_forwarded():
    a, b, c = _chain()
    fetcher = _Fetcher([a, c])
    result = run(NeighborAccumulator(fetcher).accumulate("what happened", [b], scope="tenant-1"))
    assert result.chunks[0].full_content == "A\n\nB\n\nC"
    assert result.neighbor_count == 2
    assert fetcher.scopes == ["tenant-1"]
    assert "[Source 1]" in result.combined_context


def test_fetch_failure_degrades_to_the_original_chunk():
    _, b, _ = _chain()

    class Broken:
        async def fetch_by_ids(self, chunk_ids, *, scope=None):
            raise RuntimeError("down")

    result = run(NeighborAccumulator(Broken()).accumulate("q", [b]))
    assert result.chunks[0].full_content == "B"


def test_budget_keeps_highest_scored_chunk():
    class C:
        def __init__(self, score):
            self.score = score

    chunks = [C(0.1), C(0.9), C(0.5)]
    kept, dropped = _fit_to_token_budget(chunks, [10, 10, 10], 10)
    assert [c.score for c in kept] == [0.9]
    assert dropped == 2


def test_semantic_cache_hits_on_the_same_query_and_isolates_scopes():
    backing = FakeCache()
    cache = SemanticCache(FakeEmbedder(), backing, SemanticCacheConfig(similarity_threshold=0.999))
    assert run(cache.set("what is fraud", {"answer": "x"}, scope="a"))
    assert run(cache.get("what is fraud", scope="a")) == {"answer": "x"}
    assert run(cache.get("what is fraud", scope="b")) is None
    assert run(cache.get("an unrelated question", scope="a")) is None
    assert len(backing.data) == 1
    assert run(cache.invalidate("a"))
    assert backing.data == {}
    assert run(cache.get("what is fraud", scope="a")) is None
    assert cache.get_stats()["hits"] == 1


def test_payload_validation_reports_missing_and_wrong_types():
    schema = PayloadSchema(
        fields=(PayloadField("query", (str,)), PayloadField("limit", (int,), required=False))
    )
    assert validate_payload({"query": "q"}, schema).is_valid
    bad = validate_payload({"limit": "ten"}, schema)
    assert bad.missing == ["query"]
    assert bad.wrong_type == ["limit"]


def test_sse_events_are_framed_and_carry_the_payload():
    chunk = chunk_event("hello")
    assert chunk.startswith("event: chunk\ndata: ")
    assert chunk.endswith("\n\n")
    assert json.loads(chunk.split("data: ", 1)[1]) == {"type": "chunk", "content": "hello"}
    beat = heartbeat_event()
    assert beat.startswith("event: heartbeat\ndata: ")
    assert json.loads(beat.split("data: ", 1)[1])["type"] == "heartbeat"
    assert "draft_handoff" not in {e.value for e in SSEEventType}
