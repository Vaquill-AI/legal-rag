"""legal-rag: retrieval techniques for legal AI."""

import logging

from legal_rag.types import Message, RerankResult, RetrievedChunk, UsageEvent

# A library installs no handlers of its own. Attach your own to the "legal_rag" logger to see its records.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1.dev0"

__all__ = ["Message", "RerankResult", "RetrievedChunk", "UsageEvent", "__version__"]
