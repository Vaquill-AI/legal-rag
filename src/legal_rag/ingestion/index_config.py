"""Vector index configuration as plain objects.

Describes a disk-first hybrid collection: one dense vector plus one sparse (BM25-style)
vector, INT8 scalar quantization of the dense vectors, an on-disk HNSW graph and a list of
payload indexes.
The objects hold no hostnames and no collection names, and they import no vector-store
client.
``HybridCollectionConfig.to_create_kwargs`` and ``PayloadIndexSpec.to_kwargs`` return plain
dictionaries in the shape the Qdrant REST and client APIs accept, so you can pass them to
your own client when you create the collection.

Scoping fields (for example a tenant, workspace or project key) are ordinary payload
indexes.
Mark the field you filter on in every query with ``is_scope_key=True`` so the store can
co-locate points that share a value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

IndexKind = Literal["keyword", "integer", "float", "datetime", "uuid", "bool", "text"]
Distance = Literal["Cosine", "Dot", "Euclid", "Manhattan"]
Tokenizer = Literal["word", "whitespace", "prefix", "multilingual"]


@dataclass(frozen=True)
class QuantizationConfig:
    """INT8 scalar quantization of dense vectors."""

    quantile: float = 0.99
    always_ram: bool = False  # disk-first: quantized vectors stay on disk too

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "scalar": {"type": "int8", "quantile": self.quantile, "always_ram": self.always_ram}
        }


@dataclass(frozen=True)
class HnswSettings:
    """HNSW graph parameters."""

    m: int = 16
    ef_construct: int = 128
    full_scan_threshold: int = 10_000
    on_disk: bool = True
    max_indexing_threads: int = 0  # 0 = use all cores

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "full_scan_threshold": self.full_scan_threshold,
            "on_disk": self.on_disk,
            "max_indexing_threads": self.max_indexing_threads,
        }


@dataclass(frozen=True)
class OptimizerSettings:
    """Segment optimizer parameters.

    Larger ``max_segment_size`` means fewer segments, so a query merges fewer HNSW graphs.
    """

    indexing_threshold: int = 20_000  # delay HNSW build until this many points
    memmap_threshold: int = 200_000
    flush_interval_sec: int = 5
    max_segment_size: int = 500_000
    default_segment_number: int = 4

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "indexing_threshold": self.indexing_threshold,
            "memmap_threshold": self.memmap_threshold,
            "flush_interval_sec": self.flush_interval_sec,
            "max_segment_size": self.max_segment_size,
            "default_segment_number": self.default_segment_number,
        }


@dataclass(frozen=True)
class SparseSettings:
    """Sparse (lexical) vector index parameters."""

    on_disk: bool = True
    full_scan_threshold: int = 5_000
    idf: bool = True  # server-side IDF weighting, for BM25-style ranking

    def to_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": {"on_disk": self.on_disk, "full_scan_threshold": self.full_scan_threshold}
        }
        if self.idf:
            out["modifier"] = "idf"
        return out


@dataclass(frozen=True)
class TextIndexOptions:
    """Options for a full-text payload index."""

    tokenizer: Tokenizer = "word"
    min_token_len: int = 2
    max_token_len: int = 40
    lowercase: bool = True


@dataclass(frozen=True)
class PayloadIndexSpec:
    """One payload index.

    Args:
        field: Payload field name.
        kind: Index type.
        is_scope_key: The field is a partition key that nearly every query filters on.
        text: Options when ``kind`` is ``"text"``.
    """

    field: str
    kind: IndexKind = "keyword"
    is_scope_key: bool = False
    text: TextIndexOptions | None = None

    def to_kwargs(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.kind}
        if self.kind == "keyword" and self.is_scope_key:
            schema["is_tenant"] = True
        if self.kind == "text":
            options = self.text or TextIndexOptions()
            schema.update(
                tokenizer=options.tokenizer,
                min_token_len=options.min_token_len,
                max_token_len=options.max_token_len,
                lowercase=options.lowercase,
            )
        return {"field_name": self.field, "field_schema": schema}


@dataclass(frozen=True)
class HybridCollectionConfig:
    """Disk-first hybrid collection layout.

    Args:
        dense_dimension: Size of the dense embedding. Must match your embedder.
        distance: Dense similarity metric.
        dense_name: Name of the dense vector in the collection.
        sparse_name: Name of the sparse vector in the collection.
        shard_number: Shards, for a distributed deployment.
        replication_factor: Replicas per shard.
        write_consistency_factor: Replicas that must acknowledge a write.
        payload_indexes: Payload indexes to create after the collection exists.
    """

    dense_dimension: int
    distance: Distance = "Cosine"
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    hnsw: HnswSettings = field(default_factory=HnswSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    sparse: SparseSettings = field(default_factory=SparseSettings)
    on_disk_payload: bool = True
    shard_number: int = 1
    replication_factor: int = 1
    write_consistency_factor: int = 1
    payload_indexes: Sequence[PayloadIndexSpec] = ()

    def __post_init__(self) -> None:
        if self.dense_dimension <= 0:
            raise ValueError("dense_dimension must be positive")
        if self.write_consistency_factor > self.replication_factor:
            raise ValueError("write_consistency_factor cannot exceed replication_factor")

    def to_create_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for a create-collection call (no collection name included)."""
        return {
            "vectors_config": {
                self.dense_name: {
                    "size": self.dense_dimension,
                    "distance": self.distance,
                    "on_disk": True,
                    "quantization_config": self.quantization.to_kwargs(),
                }
            },
            "sparse_vectors_config": {self.sparse_name: self.sparse.to_kwargs()},
            "hnsw_config": self.hnsw.to_kwargs(),
            "on_disk_payload": self.on_disk_payload,
            "optimizers_config": self.optimizer.to_kwargs(),
            "shard_number": self.shard_number,
            "replication_factor": self.replication_factor,
            "write_consistency_factor": self.write_consistency_factor,
        }

    def index_kwargs(self) -> list[dict[str, Any]]:
        """One keyword-argument mapping per payload index to create."""
        return [spec.to_kwargs() for spec in self.payload_indexes]


def missing_indexes(
    config: HybridCollectionConfig, existing_fields: Sequence[str] | Mapping[str, Any]
) -> list[PayloadIndexSpec]:
    """Return the configured payload indexes that are not yet present on a collection.

    ``existing_fields`` is the set of already-indexed payload field names, for example the
    keys of a collection's payload schema.
    """
    present = set(existing_fields)
    return [spec for spec in config.payload_indexes if spec.field not in present]
