"""One logger factory for every pipeline module, so records share a namespace and field style."""

from legal_rag.retrieval.rerank._log import KeywordLogger, get_logger

__all__ = ["KeywordLogger", "get_logger"]
