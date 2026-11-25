"""Semantic caching and payload validation."""

from legal_rag.retrieval.caching.payload_validator import (
    PayloadField,
    PayloadSchema,
    PayloadValidationError,
    PayloadValidationResult,
    assert_valid_payload,
    validate_payload,
)
from legal_rag.retrieval.caching.semantic import (
    CachedResult,
    SemanticCache,
    SemanticCacheConfig,
    ValidationMetadata,
    cosine_similarity,
    create_validation_metadata_from_results,
)

__all__ = [
    "CachedResult",
    "PayloadField",
    "PayloadSchema",
    "PayloadValidationError",
    "PayloadValidationResult",
    "SemanticCache",
    "SemanticCacheConfig",
    "ValidationMetadata",
    "assert_valid_payload",
    "cosine_similarity",
    "create_validation_metadata_from_results",
    "validate_payload",
]
