"""Quality tiers as data.

A tier is a named bundle of retrieval depth, reranking, expansion and complexity caps.
The built-in table lives in ``tiers.default.json`` (tiers ``standard`` and ``deep``), so a tuning
change is a data change. ``load_tiers`` accepts a file or mapping of your own in the same shape.

The tier data names no model, provider, price or plan. Which tier a request gets is a caller
decision: supply a ``TierResolver`` (see ``legal_rag.pipeline.protocols``) that maps a request to
a tier name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

DEFAULT_TIER = "standard"
_DATA_FILE = Path(__file__).resolve().parent / "tiers.default.json"

KNOWN_STRATEGIES: frozenset[str] = frozenset(
    {"hybrid", "self_rag", "crag", "hyde", "multi_query", "multi_hop"}
)
COMPLEXITY_LEVELS: tuple[str, ...] = ("trivial", "simple", "moderate", "complex")
MAX_HOP_ITERATIONS = 5

INFORMATIONAL_FLAGS: frozenset[str] = frozenset(
    {
        "use_self_rag",
        "use_hallucination_detection",
        "use_iterative_retrieval",
        "use_temporal_tracking",
        "use_loop_detection",
        "use_evaluation_metrics",
    }
)
"""Flags no stage reads directly. They describe the tier, and ``retrieval_strategy`` and the
verification step act on them. Treat them as labels, not switches."""


def _reject_unknown(kind: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {kind} keys: {unknown}")


@dataclass(frozen=True)
class TierFeatures:
    """Feature flags for a tier. Every flag defaults to off except the four baseline ones."""

    use_hybrid_retrieval: bool = True
    use_context_budget: bool = True
    use_query_cache: bool = True
    use_preserve_citations: bool = True

    use_reranking: bool = False
    use_query_condensation: bool = False
    use_anaphora_resolution: bool = False
    use_entity_extraction: bool = False
    use_neighbor_accumulation: bool = False
    use_rrf_fusion: bool = False
    use_self_rag: bool = False
    use_crag: bool = False
    use_hallucination_detection: bool = False
    use_multi_signal_confidence: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    use_query_expansion: bool = False
    use_evaluation_metrics: bool = False
    use_lost_in_middle_reorder: bool = False
    use_diversity_enforcement: bool = False
    use_multi_hop: bool = False
    use_iterative_retrieval: bool = False
    use_temporal_tracking: bool = False
    use_query_decomposer: bool = False
    use_loop_detection: bool = False
    enable_web_search: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TierFeatures:
        _reject_unknown("feature", data, {f.name for f in fields(cls)})
        for key, value in data.items():
            if not isinstance(value, bool):
                # Bad data is a data error, so ValueError, like every other tier check.
                raise ValueError(  # noqa: TRY004
                    f"Feature flag {key!r} must be a boolean, got {value!r}"
                )
        return cls(**dict(data))


def _freeze_caps(kind: str, data: Mapping[str, Any]) -> Mapping[str, int | None]:
    _reject_unknown(f"{kind} cap", data, set(COMPLEXITY_LEVELS))
    caps: dict[str, int | None] = {}
    for level, cap in data.items():
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError(f"{kind} cap for {level!r} must be null or a non-negative integer")
        caps[level] = cap
    return MappingProxyType(caps)


@dataclass(frozen=True)
class ComplexityCaps:
    """Per-complexity limits and floors that scale a tier down for simple queries.

    A cap of ``None`` means uncapped: use the tier's own number.
    A cap of ``0`` on a ``trivial`` query means the stage should not run at all.
    An unknown or missing complexity level is treated as uncapped.
    The effective value is ``min(tier_value, cap)``, then a floor may raise it again.
    """

    retrieval: Mapping[str, int | None] = field(default_factory=dict)
    rerank: Mapping[str, int | None] = field(default_factory=dict)
    neighbor: Mapping[str, int | None] = field(default_factory=dict)
    case_law_retrieval_floor: int = 0
    case_law_rerank_floor: int = 0
    single_document_rerank_floor: int = 0
    case_law_task_types: tuple[str, ...] = ()

    def retrieval_cap(self, complexity: str) -> int | None:
        return self.retrieval.get(complexity)

    def rerank_cap(self, complexity: str) -> int | None:
        return self.rerank.get(complexity)

    def neighbor_cap(self, complexity: str) -> int | None:
        return self.neighbor.get(complexity)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ComplexityCaps:
        allowed = {f.name for f in fields(cls)}
        _reject_unknown("complexity", data, allowed)
        kwargs: dict[str, Any] = {}
        for table in ("retrieval", "rerank", "neighbor"):
            if table in data:
                kwargs[table] = _freeze_caps(table, data[table])
        for floor in (
            "case_law_retrieval_floor",
            "case_law_rerank_floor",
            "single_document_rerank_floor",
        ):
            if floor in data:
                kwargs[floor] = data[floor]
        if "case_law_task_types" in data:
            kwargs["case_law_task_types"] = tuple(data["case_law_task_types"])
        return cls(**kwargs)


@dataclass(frozen=True)
class TierConfig:
    """Everything a stage needs to know about one tier."""

    name: str
    description: str
    retrieval_strategy: str
    top_k_retrieval: int
    top_k_rerank: int
    rerank_input_cap: int
    rerank_min_chunks: int
    rerank_score_threshold: float
    neighbor_chunks: int
    max_multi_hop_iterations: int
    max_chunks_per_document: int
    target_latency_ms: int
    quality_target: float
    features: TierFeatures
    complexity: ComplexityCaps

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> TierConfig:
        """Build and validate a tier from its JSON shape. Unknown keys raise ``ValueError``."""
        scalar_names = {f.name for f in fields(cls)} - {"name", "features", "complexity"}
        _reject_unknown("tier", data, scalar_names | {"features", "complexity"})
        missing = sorted(scalar_names - set(data))
        if missing:
            raise ValueError(f"Tier {name!r} is missing keys: {missing}")
        scalars = {key: data[key] for key in scalar_names}
        config = cls(
            name=name,
            features=TierFeatures.from_mapping(data.get("features", {})),
            complexity=ComplexityCaps.from_mapping(data.get("complexity", {})),
            **scalars,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ``ValueError`` when the tier is internally inconsistent."""
        if self.retrieval_strategy not in KNOWN_STRATEGIES:
            raise ValueError(
                f"Tier {self.name!r}: unknown retrieval_strategy {self.retrieval_strategy!r}"
            )
        if self.retrieval_strategy == "multi_hop" and not self.features.use_multi_hop:
            raise ValueError(f"Tier {self.name!r}: multi_hop strategy needs use_multi_hop")
        if self.top_k_retrieval <= 0 or self.top_k_rerank <= 0:
            raise ValueError(f"Tier {self.name!r}: top_k values must be positive")
        if self.top_k_rerank > self.top_k_retrieval:
            raise ValueError(f"Tier {self.name!r}: top_k_rerank exceeds top_k_retrieval")
        if self.rerank_input_cap < 0:
            raise ValueError(f"Tier {self.name!r}: rerank_input_cap must be 0 (no cap) or more")
        if 0 < self.rerank_input_cap < self.top_k_rerank:
            raise ValueError(f"Tier {self.name!r}: rerank_input_cap is below top_k_rerank")
        if self.rerank_min_chunks < 0 or self.neighbor_chunks < 0:
            raise ValueError(f"Tier {self.name!r}: chunk counts must not be negative")
        if not 0.0 <= self.rerank_score_threshold <= 1.0:
            raise ValueError(f"Tier {self.name!r}: rerank_score_threshold must be in [0, 1]")
        if not 1 <= self.max_multi_hop_iterations <= MAX_HOP_ITERATIONS:
            raise ValueError(
                f"Tier {self.name!r}: max_multi_hop_iterations must be 1 to {MAX_HOP_ITERATIONS}"
            )
        if self.max_chunks_per_document <= 0:
            raise ValueError(f"Tier {self.name!r}: max_chunks_per_document must be positive")
        if not 0.0 < self.quality_target <= 1.0:
            raise ValueError(f"Tier {self.name!r}: quality_target must be in (0, 1]")
        floors = (
            self.complexity.case_law_retrieval_floor,
            self.complexity.case_law_rerank_floor,
            self.complexity.single_document_rerank_floor,
        )
        if any(floor < 0 for floor in floors):
            raise ValueError(f"Tier {self.name!r}: complexity floors must not be negative")


def load_tiers(source: Path | str | Mapping[str, Any] | None = None) -> Mapping[str, TierConfig]:
    """Load and validate a tier table.

    ``source`` is a path to a JSON file, an already parsed mapping, or ``None`` for the built-in
    table. The shape is ``{"tiers": {name: {...}}}``. Unknown keys raise ``ValueError``.
    """
    if source is None:
        return _default_tiers()
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        with Path(source).open(encoding="utf-8") as handle:
            raw = json.load(handle)
    _reject_unknown("top-level", raw, {"tiers"})
    tiers_raw = raw.get("tiers")
    if not isinstance(tiers_raw, Mapping) or not tiers_raw:
        raise ValueError("Tier data must hold a non-empty 'tiers' mapping")
    tiers: dict[str, TierConfig] = {}
    for raw_name, body in tiers_raw.items():
        # `get_tier` looks names up in lower case, so a key kept as written ("Deep") would be
        # listed by `available_tiers` yet impossible to select.
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError("A tier name must not be empty")
        if name in tiers:
            raise ValueError(f"Tier {name!r} is defined twice (tier names ignore case)")
        tiers[name] = TierConfig.from_mapping(name, body)
    return MappingProxyType(tiers)


@lru_cache(maxsize=1)
def _default_tiers() -> Mapping[str, TierConfig]:
    return load_tiers(_DATA_FILE)


def available_tiers(source: Path | str | Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Tier names in the table, in file order."""
    return tuple(load_tiers(source))


def get_tier(
    name: str | None,
    source: Path | str | Mapping[str, Any] | None = None,
) -> TierConfig:
    """Look a tier up by name. An unknown name, including a retired one, raises ``ValueError``."""
    tiers = load_tiers(source)
    key = (name or "").strip().lower()
    try:
        return tiers[key]
    except KeyError:
        raise ValueError(f"Unknown tier {name!r}. Available tiers: {sorted(tiers)}") from None
