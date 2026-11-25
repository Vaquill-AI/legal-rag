"""Small helpers shared by the retrieval strategies."""

from __future__ import annotations

import hashlib

from legal_rag.providers.usage import emit_usage, estimate_tokens


def fingerprint(text: str, n: int = 12) -> str:
    """Short SHA-256 fingerprint, for correlating a query in logs without storing its text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def safe_error_fields(error: Exception) -> dict[str, str]:
    """Error metadata that is safe to log: the type and a truncated message."""
    return {"error_type": type(error).__name__, "error_message": str(error)[:160]}


__all__ = ["emit_usage", "estimate_tokens", "fingerprint", "safe_error_fields"]
