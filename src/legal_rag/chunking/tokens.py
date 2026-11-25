"""Token counting for chunk sizing.

Uses ``tiktoken`` when it can load an encoding.
When it cannot (not installed, or no network to fetch the encoding file),
a whitespace-based approximation is used so chunking still works.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

DEFAULT_ENCODING = "o200k_base"

_PIECE_RE = re.compile(r"\s*\S+|\s+")


class TextEncoding(Protocol):
    """The subset of a tiktoken encoding that the chunkers use."""

    def encode(self, text: str) -> list[int] | list[str]: ...

    def decode(self, tokens: list) -> str: ...  # type: ignore[type-arg]


class ApproximateEncoding:
    """Whitespace-piece encoding used when tiktoken is unavailable."""

    name = "approximate"

    def encode(self, text: str) -> list[str]:
        return _PIECE_RE.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


@lru_cache(maxsize=8)
def get_encoding(name: str = DEFAULT_ENCODING) -> TextEncoding:
    """Return a cached encoding by tiktoken name, falling back to an approximation."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - degrade gracefully by design
        return ApproximateEncoding()


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Count tokens in ``text``."""
    if not text:
        return 0
    return len(get_encoding(encoding).encode(text))
