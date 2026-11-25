"""Court-authority weighting: a rank multiplier and a jurisdiction-aware reranker.

Higher courts bind lower ones, so a passage from a higher court should usually outrank an
equally relevant passage from a lower one. The mechanism is generic. The court tables are
data: build an ``AuthorityConfig`` for your jurisdiction and inject it. The default config is
a small keyword-based US-style hierarchy, meant as a starting point.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from enum import IntEnum
from typing import Any

from legal_rag.retrieval.rerank._log import get_logger
from legal_rag.retrieval.rerank.base import BaseReranker, RerankOutcome
from legal_rag.types import RetrievedChunk

logger = get_logger("retrieval.rerank.authority")


class CourtLevel(IntEnum):
    """Authority levels. A higher value is a higher authority."""

    LOWER_COURT = 1
    TRIBUNAL = 2
    TRIAL_COURT = 3
    APPELLATE_COURT = 4
    HIGHEST_COURT = 5


DEFAULT_KEYWORD_LEVELS: tuple[tuple[str, CourtLevel], ...] = (
    ("supreme court", CourtLevel.HIGHEST_COURT),
    ("court of appeals", CourtLevel.APPELLATE_COURT),
    ("court of appeal", CourtLevel.APPELLATE_COURT),
    ("circuit", CourtLevel.APPELLATE_COURT),
    ("appellate", CourtLevel.APPELLATE_COURT),
    ("district court", CourtLevel.TRIAL_COURT),
    ("trial court", CourtLevel.TRIAL_COURT),
    ("superior court", CourtLevel.TRIAL_COURT),
    ("tribunal", CourtLevel.TRIBUNAL),
    ("board", CourtLevel.TRIBUNAL),
    ("commission", CourtLevel.TRIBUNAL),
    ("magistrate", CourtLevel.LOWER_COURT),
)

DEFAULT_BOOST_BY_LEVEL: Mapping[CourtLevel, float] = {
    CourtLevel.HIGHEST_COURT: 1.5,
    CourtLevel.APPELLATE_COURT: 1.2,
    CourtLevel.TRIAL_COURT: 1.0,
    CourtLevel.TRIBUNAL: 1.0,
    CourtLevel.LOWER_COURT: 1.0,
}

DEFAULT_WEIGHT_BY_LEVEL: Mapping[CourtLevel, float] = {
    CourtLevel.HIGHEST_COURT: 2.0,
    CourtLevel.APPELLATE_COURT: 1.5,
    CourtLevel.TRIAL_COURT: 1.0,
    CourtLevel.TRIBUNAL: 0.8,
    CourtLevel.LOWER_COURT: 0.5,
}


@dataclass
class AuthorityConfig:
    """Court tables and bonus sizes. Everything jurisdiction-specific lives here."""

    court_levels: Mapping[str, CourtLevel] = field(default_factory=dict)
    """Exact court name (case-insensitive) to level. Checked before ``keyword_levels``."""
    keyword_levels: Sequence[tuple[str, CourtLevel]] = DEFAULT_KEYWORD_LEVELS
    """Substring (lower case) to level. The first match wins, so list specific phrases first."""
    default_level: CourtLevel = CourtLevel.LOWER_COURT
    boost_by_level: Mapping[CourtLevel, float] = field(
        default_factory=lambda: dict(DEFAULT_BOOST_BY_LEVEL)
    )
    """Multipliers used by ``apply_court_boost``."""
    weight_by_level: Mapping[CourtLevel, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHT_BY_LEVEL)
    )
    """Authority weights used by ``AuthorityReranker``."""
    jurisdiction_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    """Canonical jurisdiction name to alternative names that should count as the same place."""
    binding_levels: frozenset[CourtLevel] = frozenset({CourtLevel.HIGHEST_COURT})
    """Levels whose decisions bind everywhere. Others bind only within their own jurisdiction."""
    same_jurisdiction_bonus: float = 0.3
    binding_precedent_bonus: float = 0.5
    local_appellate_factor: float = 0.8
    """Share of the binding bonus given to appellate courts inside the target jurisdiction."""
    recency_bonus: float = 0.1
    recency_decay_years: int = 10
    landmark_case_bonus: float = 0.2
    overruled_penalty: float = 0.8
    jurisdiction_weight: float = 0.3
    """Share of the final score from authority, the rest being the retrieval score."""
    unknown_weight: float = 0.5


def get_court_level(court_name: str, config: AuthorityConfig | None = None) -> CourtLevel:
    """Resolve a court name to a level: exact match, then keyword match, then the default."""
    cfg = config or AuthorityConfig()
    if not court_name:
        return cfg.default_level
    lowered = court_name.strip().lower()
    for name, level in cfg.court_levels.items():
        if name.strip().lower() == lowered:
            return level
    for keyword, level in cfg.keyword_levels:
        if keyword in lowered:
            return level
    return cfg.default_level


def apply_court_boost(
    chunk: RetrievedChunk | Mapping[str, Any],
    base_score: float,
    config: AuthorityConfig | None = None,
    metadata_key: str = "court",
) -> float:
    """Multiply ``base_score`` by the boost for the chunk's court. No court means no change."""
    cfg = config or AuthorityConfig()
    metadata = (
        chunk.get("metadata", chunk)
        if isinstance(chunk, Mapping)
        else getattr(chunk, "metadata", {})
    )
    court = (metadata or {}).get(metadata_key, "")
    if not court:
        return base_score
    return base_score * cfg.boost_by_level.get(get_court_level(str(court), cfg), 1.0)


class AuthorityReranker(BaseReranker):
    """Blend retrieval score with a court-authority score built from chunk metadata.

    Reads ``court``, ``jurisdiction``, ``decision_year``, ``is_landmark`` and ``is_overruled``
    from ``chunk.metadata``. Missing fields contribute nothing.
    """

    def __init__(self, top_k: int = 5, config: AuthorityConfig | None = None) -> None:
        super().__init__(top_k=top_k)
        self.config = config or AuthorityConfig()

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        target_jurisdiction: str | None = None,
        current_year: int | None = None,
        **kwargs: object,
    ) -> RerankOutcome:
        start = time.monotonic()
        k = top_k or self.top_k
        if not chunks:
            return RerankOutcome(query=query)
        year = current_year or date.today().year
        w = self.config.jurisdiction_weight
        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            authority = self.authority_score(chunk, target_jurisdiction, year)
            combined = chunk.score * (1 - w) + authority * w
            scored.append(
                replace(
                    chunk,
                    score=combined,
                    metadata={
                        **chunk.metadata,
                        "authority_score": authority,
                        "original_score": chunk.score,
                        "target_jurisdiction": target_jurisdiction,
                    },
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        kept = scored[:k]
        return RerankOutcome(
            query=query,
            chunks=kept,
            original_count=len(chunks),
            final_count=len(kept),
            processing_time_ms=(time.monotonic() - start) * 1000,
        )

    def authority_score(
        self, chunk: RetrievedChunk, target_jurisdiction: str | None, current_year: int
    ) -> float:
        """Authority score between 0 and 1 for one chunk."""
        cfg = self.config
        meta = chunk.metadata or {}
        level = get_court_level(str(meta.get("court", "")), cfg)
        known = bool(meta.get("court"))
        weight = cfg.weight_by_level.get(level, cfg.unknown_weight) if known else cfg.unknown_weight
        score = 0.5 * weight
        same_place = bool(
            target_jurisdiction
            and self.jurisdictions_match(str(meta.get("jurisdiction", "")), target_jurisdiction)
        )
        if same_place:
            score += cfg.same_jurisdiction_bonus
        if known and level in cfg.binding_levels:
            score += cfg.binding_precedent_bonus
        elif known and level is CourtLevel.APPELLATE_COURT and same_place:
            score += cfg.binding_precedent_bonus * cfg.local_appellate_factor
        year = meta.get("decision_year")
        if isinstance(year, int) and year <= current_year:
            factor = max(0.0, 1 - (current_year - year) / cfg.recency_decay_years)
            score += cfg.recency_bonus * factor
        if meta.get("is_landmark"):
            score += cfg.landmark_case_bonus
        if meta.get("is_overruled"):
            score *= cfg.overruled_penalty
        return min(max(score, 0.0), 1.0)

    def jurisdictions_match(self, a: str, b: str) -> bool:
        """True when two jurisdiction names are equal or listed as aliases of each other."""
        if not a or not b:
            return False
        a, b = a.lower(), b.lower()
        if a == b:
            return True
        for canonical, aliases in self.config.jurisdiction_aliases.items():
            names = {canonical.lower(), *(x.lower() for x in aliases)}
            if a in names and b in names:
                return True
        return False
