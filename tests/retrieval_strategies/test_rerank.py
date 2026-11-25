import asyncio
import time

import pytest

from legal_rag.providers.fakes import FakeReranker
from legal_rag.retrieval.rerank import (
    AuthorityConfig,
    AuthorityReranker,
    ConfidenceScorer,
    CourtLevel,
    CoverageConfig,
    CrossEncoderReranker,
    DiversityReorderer,
    LostInMiddleReorderer,
    RateLimitWaitExceeded,
    RerankRateLimited,
    RerankRateLimiter,
    ScoreBasedReranker,
    apply_court_boost,
    apply_coverage_notice,
    dedupe_sources,
    get_court_level,
)
from legal_rag.retrieval.rerank.rate_limit import estimate_tokens, parse_retry_after
from legal_rag.retrieval.rerank.source_dedup import (
    build_chunk_to_post_dedup_remap,
    normalise_case_name,
    normalise_citation,
    stabilize_pinned_source_order,
)
from legal_rag.types import RerankResult, RetrievedChunk, UsageEvent


def chunk(cid: str, text: str, score: float = 0.5, **meta) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, score=score, metadata=dict(meta))


# ---- score based and cross encoder ----


async def test_score_based_boosts_term_overlap_and_does_not_mutate():
    a = chunk("a", "negligence duty of care elements", 0.5)
    b = chunk("b", "unrelated tax text", 0.52)
    out = await ScoreBasedReranker(top_k=2).rerank("duty of care", [b, a])
    assert [c.id for c in out.chunks] == ["a", "b"]
    assert a.score == 0.5 and not out.degraded
    assert (await ScoreBasedReranker().rerank("q", [])).chunks == []


async def test_cross_encoder_reorders_with_provider_scores_and_meters():
    events: list[UsageEvent] = []
    reranker = CrossEncoderReranker(
        FakeReranker(), top_k=2, usage_hook=events.append, provider_name="fake"
    )
    chunks = [chunk("a", "tax law"), chunk("b", "duty of care in negligence"), chunk("c", "care")]
    out = await reranker.rerank("duty of care", chunks)
    assert out.chunks[0].id == "b" and len(out.chunks) == 2 and not out.degraded
    assert events and events[0].provider == "fake"


class ExplodingReranker:
    async def rerank(self, query, documents, *, top_k=None):
        raise RuntimeError("provider down")


class LimitedReranker:
    async def rerank(self, query, documents, *, top_k=None):
        raise RerankRateLimited(None)


async def test_cross_encoder_degrades_visibly_on_failure():
    chunks = [chunk("a", "alpha duty", 0.9), chunk("b", "beta", 0.1)]
    out = await CrossEncoderReranker(ExplodingReranker()).rerank("duty", chunks)
    assert out.degraded and out.degraded_reason == "error:RuntimeError"
    assert out.chunks[0].id == "a"


async def test_cross_encoder_rate_limited_without_time_degrades():
    limiter = RerankRateLimiter(tokens_per_minute=6_000_000)
    reranker = CrossEncoderReranker(
        LimitedReranker(), rate_limiter=limiter, default_wait_seconds=0.5
    )
    out = await reranker.rerank("q", [chunk("a", "alpha")], deadline=time.monotonic() + 0.2)
    assert out.degraded and out.degraded_reason == "rate_limited"


async def test_rate_limiter_acquire_pause_and_deadline():
    limiter = RerankRateLimiter(tokens_per_minute=600_000)
    reserved = await limiter.acquire(1000, time.monotonic() + 1)
    assert reserved == 1000
    await limiter.settle(reserved, 500)
    await limiter.refund(0)
    assert await limiter.pause_after_429(0.3) == 0.3
    with pytest.raises(RateLimitWaitExceeded):
        await limiter.acquire(10, time.monotonic() + 0.05)
    await asyncio.sleep(0.31)
    assert await limiter.acquire(10, time.monotonic() + 1) == 10


def test_rate_limit_helpers():
    assert parse_retry_after("2") == 2.0
    assert parse_retry_after("soon") is None and parse_retry_after(None) is None
    assert estimate_tokens("query", ["doc"] * 3) > estimate_tokens("query", ["doc"])


# ---- authority ----


def test_court_level_and_boost_use_injected_config():
    assert get_court_level("U.S. Supreme Court") is CourtLevel.HIGHEST_COURT
    assert get_court_level("Ninth Circuit Court of Appeals") is CourtLevel.APPELLATE_COURT
    assert get_court_level("") is CourtLevel.LOWER_COURT
    cfg = AuthorityConfig(
        court_levels={"Regional Panel": CourtLevel.TRIBUNAL},
        boost_by_level={CourtLevel.TRIBUNAL: 2.0},
    )
    assert get_court_level("regional panel", cfg) is CourtLevel.TRIBUNAL
    assert apply_court_boost(chunk("a", "t", court="Regional Panel"), 0.5, cfg) == 1.0
    assert apply_court_boost(chunk("a", "t"), 0.5, cfg) == 0.5
    assert apply_court_boost(chunk("a", "t", court="Supreme Court"), 1.0) == 1.5


async def test_authority_reranker_prefers_binding_and_local_authority():
    cfg = AuthorityConfig(jurisdiction_aliases={"California": ["ca", "calif"]})
    low = chunk("low", "x", 0.6, court="District Court", jurisdiction="Texas")
    high = chunk("high", "x", 0.6, court="Supreme Court", jurisdiction="Texas", decision_year=2024)
    local = chunk(
        "local", "x", 0.6, court="Court of Appeals", jurisdiction="CA", decision_year=2020
    )
    reranker = AuthorityReranker(top_k=3, config=cfg)
    out = await reranker.rerank(
        "q", [low, local, high], target_jurisdiction="California", current_year=2025
    )
    assert out.chunks[-1].id == "low"
    assert out.chunks[0].metadata["authority_score"] >= out.chunks[1].metadata["authority_score"]
    assert reranker.jurisdictions_match("ca", "California") and not reranker.jurisdictions_match(
        "", "x"
    )


async def test_authority_penalises_overruled_authority():
    reranker = AuthorityReranker()
    live = chunk("live", "x", 0.5, court="District Court")
    dead = chunk("dead", "x", 0.5, court="District Court", is_overruled=True)
    assert reranker.authority_score(dead, None, 2025) < reranker.authority_score(live, None, 2025)


# ---- reorderers, confidence ----


def test_lost_in_middle_puts_best_at_the_end():
    chunks = [chunk(str(i), f"text {i}", score=1 - i / 10) for i in range(6)]
    result = LostInMiddleReorderer().reorder(chunks)
    assert result.reordering_applied
    assert result.reordered_chunks[-1].id == "0"
    assert result.reordered_chunks[0].id == "1"
    assert sorted(c.id for c in result.reordered_chunks) == ["0", "1", "2", "3", "4", "5"]
    short = LostInMiddleReorderer().reorder(chunks[:2])
    assert not short.reordering_applied


def test_diversity_reorderer_spreads_near_duplicates():
    chunks = [
        chunk("a", "negligence duty of care breach", 0.9),
        chunk("b", "negligence duty of care breach causation", 0.88),
        chunk("c", "contract formation offer acceptance", 0.8),
    ]
    result = DiversityReorderer({"lambda_param": 0.3}).reorder(chunks)
    assert result.reordered_chunks[0].id == "a"
    assert result.reordered_chunks[1].id == "c"


def test_confidence_scorer_levels():
    scorer = ConfidenceScorer()
    assert scorer.score([], "q").requires_review
    strong = [chunk(str(i), f"statute text {i}", 0.6) for i in range(5)]
    weak = [chunk(str(i), f"text {i}", 0.05) for i in range(5)]
    assert scorer.score(strong, "q").overall_score > scorer.score(weak, "q").overall_score
    assert "overall_score" in scorer.score(strong, "q").to_dict()


# ---- source dedup and coverage ----


def test_dedupe_merges_same_case_and_keeps_longest_excerpt():
    sources = [
        {"case_name": "Acme Inc. v. Doe", "citation": "123 F.3d 45 (2000)", "content": "short"},
        {
            "case_name": "Acme v Doe",
            "citation": "123 F.3d 45",
            "content": "a much longer excerpt",
            "url": "u",
        },
        {"case_name": "Acme v. Doe", "content": "no citation entry"},
        {"title": "Other", "document_id": "d9", "content": "x"},
    ]
    merged = dedupe_sources(sources)
    assert len(merged) == 2
    assert merged[0]["content"] == "a much longer excerpt" and merged[0]["url"] == "u"
    remap = build_chunk_to_post_dedup_remap(sources, merged)
    assert remap == [1, 1, 1, 2]
    assert dedupe_sources(merged) == merged


def test_source_dedup_normalisers_and_pinning():
    assert normalise_case_name("Acme Inc. v. Doe") == normalise_case_name("acme vs doe")
    assert normalise_citation("603 U.S. 369 (2024)") == normalise_citation("603 US 369")
    sources = [{"document_id": "b"}, {"document_id": "z"}, {"document_id": "a"}]
    assert stabilize_pinned_source_order(sources, ["a", "b"]) == [
        {"document_id": "a"},
        {"document_id": "b"},
        {"document_id": "z"},
    ]
    assert stabilize_pinned_source_order(sources, None) is sources


def test_coverage_notice_fires_only_when_own_documents_are_missing():
    ref = [{"source_type": "statute"}, {"metadata": {"source_type": "case"}}]
    base = "context"
    out, applied = apply_coverage_notice(
        base, ref, own_docs_requested=True, query="what does my contract say"
    )
    assert applied and out.startswith("[RETRIEVAL NOTICE")
    assert apply_coverage_notice(base, ref, own_docs_requested=True, query="what is estoppel") == (
        base,
        False,
    )
    assert apply_coverage_notice(base, ref, own_docs_requested=True, docs_pinned=True)[1]
    with_docs = ref + [{"source_type": "upload"}]
    assert apply_coverage_notice(base, with_docs, own_docs_requested=True, docs_pinned=True) == (
        base,
        False,
    )
    assert apply_coverage_notice(base, [], own_docs_requested=True, docs_pinned=True) == (
        base,
        False,
    )
    custom = CoverageConfig(
        own_source_types=frozenset({"matter_file"}), reference_source_types=frozenset({"statute"})
    )
    assert apply_coverage_notice(
        base, ref, own_docs_requested=True, docs_pinned=True, config=custom
    )[1]


def test_cross_encoder_results_use_protocol_types():
    assert RerankResult(index=0, score=1.0).index == 0
