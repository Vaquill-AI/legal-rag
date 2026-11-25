"""Default chunking parameters.

Chunkers accept explicit sizes; these values apply when none are given.
Nothing here reads the environment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk sizing in tokens."""

    chunk_size: int = 512
    chunk_overlap: int = 100
    min_chunk_size: int = 50


_DEFAULT = ChunkingConfig()


def get_chunking_config() -> ChunkingConfig:
    """Return the default chunking configuration."""
    return _DEFAULT
