"""Reciprocal Rank Fusion (RRF) for merging ranked lists.

RRF scores a document as the sum over lists of ``1 / (k + rank)``.
It needs no score calibration, so it suits merging lists whose raw scores live on different scales,
for example a user-document search and a reference-corpus search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from legal_rag.types import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one.

    Each list is taken as already ordered best first.
    A chunk that appears in several lists accumulates score from each.
    The first copy seen is kept, with its ``score`` replaced by the fused score.
    Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    order: dict[str, int] = {}
    for chunks in ranked_lists:
        for rank, chunk in enumerate(chunks, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            if chunk.id not in first_seen:
                first_seen[chunk.id] = chunk
                order[chunk.id] = len(order)
    ranked_ids = sorted(scores, key=lambda cid: (-scores[cid], order[cid]))
    fused = [replace(first_seen[cid], score=scores[cid]) for cid in ranked_ids]
    return fused[:top_k] if top_k else fused


def rrf_merge_chunks(
    chunks: Sequence[RetrievedChunk],
    *,
    k: int = DEFAULT_RRF_K,
    source_of: Callable[[RetrievedChunk], str] | None = None,
) -> list[RetrievedChunk]:
    """Merge one flat list of chunks that came from several sources.

    Chunks are grouped by source (``metadata["source"]`` unless ``source_of`` says otherwise),
    each group is ranked by its own raw score, and the groups are fused with RRF.
    """
    if not chunks:
        return []
    pick = source_of or (lambda c: str(c.metadata.get("source", "unknown")))
    groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        groups.setdefault(pick(chunk), []).append(chunk)
    ranked_lists = [sorted(group, key=lambda c: c.score, reverse=True) for group in groups.values()]
    return reciprocal_rank_fusion(ranked_lists, k=k)
