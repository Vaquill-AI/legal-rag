"""Concrete provider adapters that implement the protocols in ``legal_rag.providers.protocols``."""

from legal_rag.providers.adapters.voyage import VoyageEmbedder, VoyageReranker

__all__ = ["VoyageEmbedder", "VoyageReranker"]
