"""Regressions found while documenting the pipeline: tier key case, silent analyze failure, exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag import pipeline
from legal_rag.pipeline import (
    DEGRADED_WARNINGS,
    WARNING_ANALYZE_FAILED,
    WARNING_TIER_RESOLUTION_FAILED,
    analyze_stage,
    available_tiers,
    exact_cache_key,
    get_tier,
    load_tiers,
    semantic_scope_key,
)
from legal_rag.pipeline.types import QueryContext


def _default_tier_data() -> dict:
    path = Path(pipeline.tiers.__file__).with_name("tiers.default.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_tier_key_written_with_capitals_can_be_listed_and_selected():
    data = _default_tier_data()
    data["tiers"] = {"Deep": data["tiers"]["deep"]}

    assert available_tiers(data) == ("deep",)
    assert get_tier("Deep", data).name == "deep"
    assert get_tier("deep", data).name == "deep"


def test_two_tier_keys_that_differ_only_in_case_are_refused():
    data = _default_tier_data()
    data["tiers"] = {"deep": data["tiers"]["deep"], "DEEP": data["tiers"]["deep"]}

    with pytest.raises(ValueError, match="defined twice"):
        load_tiers(data)


async def test_a_failing_analyze_stage_hands_the_context_back_with_a_warning():
    ctx = QueryContext(query="what notice must a landlord give")

    result = await analyze_stage(ctx, analyzer=object())

    assert result is ctx
    assert WARNING_ANALYZE_FAILED in result.warnings
    assert WARNING_ANALYZE_FAILED in DEGRADED_WARNINGS


def test_every_name_in_all_exists_and_the_cache_key_helpers_are_public():
    missing = [name for name in pipeline.__all__ if not hasattr(pipeline, name)]
    assert not missing
    assert callable(exact_cache_key)
    assert callable(semantic_scope_key)
    assert WARNING_TIER_RESOLUTION_FAILED == "tier_resolution_failed"
