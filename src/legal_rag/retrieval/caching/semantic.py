"""Semantic cache: reuse an answer when a new query is close enough to a cached one.

Entries live in an injected ``Cache`` as one JSON blob per scope, so a scope can only ever see
its own entries. ``scope`` is an opaque string chosen by the caller (a tenant, a project, a
user), or ``None`` for a shared scope. Results must be JSON serializable. Invalidation deletes
the scope's blob through ``Cache.delete``.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from legal_rag.providers.protocols import Cache, Embedder
from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("caching.semantic")


@dataclass
class SemanticCacheConfig:
    """Configuration for the semantic cache."""

    similarity_threshold: float = 0.85
    ttl: int = 3600
    max_entries_per_scope: int = 200
    namespace: str = "semantic_cache"
    require_validated: bool = False


@dataclass
class ValidationMetadata:
    """
    Validation metadata stored with cached results.

    Quick Win 3: Store validation results alongside cached responses so a hit on a
    verified entry skips re-validation, removing that LLM call. (The "40-60%
    faster validation" once quoted here was never measured.)

    Based on:
    - AWS Semantic Cache with Verified Answers (2024)
    - Research showing curated+verified cache prevents hallucinations
    """

    validation_passed: bool = True
    validation_score: float = 1.0  # 0.0-1.0 confidence score
    hallucination_flags: list[str] | None = None  # Any detected issues
    sources_used: list[int] | None = None  # Source IDs referenced
    sub_question_coverage: float = 1.0  # Coverage from deterministic validation
    source_coverage: float = 1.0  # Source coverage from deterministic validation
    recommendation: str = "proceed"  # Validation recommendation
    validated_at: float | None = None  # Timestamp of validation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "validation_passed": self.validation_passed,
            "validation_score": self.validation_score,
            "hallucination_flags": self.hallucination_flags or [],
            "sources_used": self.sources_used or [],
            "sub_question_coverage": self.sub_question_coverage,
            "source_coverage": self.source_coverage,
            "recommendation": self.recommendation,
            "validated_at": self.validated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationMetadata:
        """Create from dictionary."""
        return cls(
            validation_passed=data.get("validation_passed", True),
            validation_score=data.get("validation_score", 1.0),
            hallucination_flags=data.get("hallucination_flags"),
            sources_used=data.get("sources_used"),
            sub_question_coverage=data.get("sub_question_coverage", 1.0),
            source_coverage=data.get("source_coverage", 1.0),
            recommendation=data.get("recommendation", "proceed"),
            validated_at=data.get("validated_at"),
        )


@dataclass
class CachedResult:
    """A cached query result with its embedding and validation status."""

    query: str
    query_embedding: list[float]
    result: Any
    timestamp: float
    scope: str | None = None
    hit_count: int = 0
    validation: ValidationMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_embedding": self.query_embedding,
            "result": self.result,
            "timestamp": self.timestamp,
            "scope": self.scope,
            "hit_count": self.hit_count,
            "validation": self.validation.to_dict() if self.validation else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedResult:
        validation = data.get("validation")
        return cls(
            query=data["query"],
            query_embedding=list(data["query_embedding"]),
            result=data["result"],
            timestamp=float(data["timestamp"]),
            scope=data.get("scope"),
            hit_count=int(data.get("hit_count", 0)),
            validation=ValidationMetadata.from_dict(validation) if validation else None,
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, 0.0 when either has no magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """Embedding-similarity cache over an injected key-value ``Cache``."""

    def __init__(self, embedder: Embedder, cache: Cache, config: SemanticCacheConfig | None = None):
        self.embedder = embedder
        self.cache = cache
        self.config = config or SemanticCacheConfig()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}

    def _make_key(self, scope: str | None) -> str:
        return f"{self.config.namespace}:{scope or 'shared'}"

    async def _embed(self, query: str) -> list[float]:
        vectors = await self.embedder.embed_documents([query])
        return [float(x) for x in vectors[0]]

    async def _load(self, scope: str | None) -> list[CachedResult]:
        raw = await self.cache.get(self._make_key(scope))
        if not raw:
            return []
        now = time.time()
        entries = [CachedResult.from_dict(d) for d in json.loads(raw.decode("utf-8"))]
        return [e for e in entries if now - e.timestamp <= self.config.ttl]

    async def _save(self, scope: str | None, entries: list[CachedResult]) -> None:
        blob = json.dumps([e.to_dict() for e in entries]).encode("utf-8")
        await self.cache.set(self._make_key(scope), blob, ttl_seconds=self.config.ttl)

    async def get_entry(self, query: str, scope: str | None = None) -> CachedResult | None:
        """Best cached entry at or above the similarity threshold, or ``None``. Fails open."""
        try:
            entries = await self._load(scope)
            if not entries:
                self.stats["misses"] += 1
                return None
            vector = await self._embed(query)
            best: CachedResult | None = None
            best_sim = 0.0
            for entry in entries:
                if entry.scope != scope:
                    continue
                if self.config.require_validated and not (
                    entry.validation and entry.validation.validation_passed
                ):
                    continue
                sim = cosine_similarity(vector, entry.query_embedding)
                if sim > best_sim:
                    best, best_sim = entry, sim
            if best is None or best_sim < self.config.similarity_threshold:
                self.stats["misses"] += 1
                return None
            best.hit_count += 1
            await self._save(scope, entries)
            self.stats["hits"] += 1
            logger.info("semantic_cache_hit", similarity=round(best_sim, 4), hits=best.hit_count)
            return best
        except Exception as exc:
            self.stats["errors"] += 1
            logger.warning("semantic_cache_get_error", error=str(exc))
            return None

    async def get(self, query: str, scope: str | None = None) -> Any | None:
        """The cached result for a similar query, or ``None``."""
        entry = await self.get_entry(query, scope)
        return entry.result if entry else None

    async def set(
        self,
        query: str,
        result: Any,
        scope: str | None = None,
        validation: ValidationMetadata | None = None,
    ) -> bool:
        """Cache a result. Returns False (and logs) on any failure."""
        try:
            entries = await self._load(scope)
            entries.append(
                CachedResult(
                    query=query,
                    query_embedding=await self._embed(query),
                    result=result,
                    timestamp=time.time(),
                    scope=scope,
                    validation=validation,
                )
            )
            entries = entries[-self.config.max_entries_per_scope :]
            await self._save(scope, entries)
            self.stats["sets"] += 1
            return True
        except Exception as exc:
            self.stats["errors"] += 1
            logger.warning("semantic_cache_set_error", error=str(exc))
            return False

    async def invalidate(self, scope: str | None = None) -> bool:
        """Drop every entry in a scope."""
        try:
            await self.cache.delete(self._make_key(scope))
            return True
        except Exception as exc:
            logger.warning("semantic_cache_invalidate_error", error=str(exc))
            return False

    def get_stats(self) -> dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        rate = (self.stats["hits"] / total * 100) if total else 0.0
        return {**self.stats, "hit_rate": f"{rate:.1f}%", "total_queries": total}


def create_validation_metadata_from_results(
    deterministic_result=None,
    llm_result=None,
    sources_used: list[int] | None = None,
) -> ValidationMetadata:
    """
    Create ValidationMetadata from validation results.

    Quick Win 3 helper: Combines deterministic and LLM validation
    results into a single ValidationMetadata for caching.

    Args:
        deterministic_result: DeterministicValidationResult from quick validation
        llm_result: ValidationResult from LLM-based validation
        sources_used: List of source IDs used in the response

    Returns:
        ValidationMetadata for caching

    """
    # Determine overall validation status (both must pass if both present)
    det_passed = deterministic_result.is_valid if deterministic_result else True
    llm_passed = llm_result.is_valid if llm_result else True
    validation_passed = det_passed and llm_passed

    # Use LLM confidence if available, else deterministic, else default
    if llm_result:
        validation_score = llm_result.confidence_score
        recommendation = llm_result.recommendation
        hallucination_flags = llm_result.missing_support if not llm_result.is_valid else None
    elif deterministic_result:
        validation_score = deterministic_result.confidence
        recommendation = deterministic_result.recommendation
        hallucination_flags = (
            deterministic_result.missing_sub_questions[:3]
            if not deterministic_result.is_valid
            else None
        )
    else:
        validation_score = 1.0
        recommendation = "proceed"
        hallucination_flags = None

    # Get coverage metrics from deterministic validation
    sub_question_coverage = (
        deterministic_result.sub_question_coverage if deterministic_result else 1.0
    )
    source_coverage = deterministic_result.source_coverage if deterministic_result else 1.0

    return ValidationMetadata(
        validation_passed=validation_passed,
        validation_score=validation_score,
        hallucination_flags=hallucination_flags,
        sources_used=sources_used,
        sub_question_coverage=sub_question_coverage,
        source_coverage=source_coverage,
        recommendation=recommendation,
        validated_at=time.time(),
    )
