"""Usage metering helpers: call an optional ``UsageHook`` without ever breaking the caller."""

from __future__ import annotations

import math
from typing import Any

from legal_rag.providers.protocols import UsageHook
from legal_rag.types import UsageEvent


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about four characters per token), good enough for metering."""
    return max(1, math.ceil(len(text) / 4))


def emit_usage(
    hook: UsageHook | None,
    operation: str,
    *,
    provider: str | None = None,
    input_text: str = "",
    output_text: str = "",
    **extra: Any,
) -> None:
    """Call the optional usage hook with one event. A failing hook never breaks the pipeline.

    Token counts are estimated from the text lengths and left ``None`` for an empty text.
    ``provider`` names the backend that served the call. Anything else goes in ``extra``.
    """
    if hook is None:
        return
    try:
        hook(
            UsageEvent(
                operation=operation,
                provider=provider,
                input_tokens=estimate_tokens(input_text) if input_text else None,
                output_tokens=estimate_tokens(output_text) if output_text else None,
                extra=extra,
            )
        )
    except Exception:  # noqa: BLE001 - metering must never break the pipeline
        return


__all__ = ["emit_usage", "estimate_tokens"]
