"""
Multilingual support.

Provides:
- Script-based language detection with legal vocabulary hints
- A pipeline that routes texts to an injected embedder
- Cross-lingual search over language-agnostic embeddings

Language and vocabulary data live in ``legal_rag/chunking/data/languages.json``.
"""

from legal_rag.chunking.multilingual.cross_lingual import (
    CrossLingualSearcher,
    get_cross_lingual_searcher,
)
from legal_rag.chunking.multilingual.language_detector import (
    LanguageDetector,
    LanguageInfo,
    get_language_detector,
)
from legal_rag.chunking.multilingual.pipeline import (
    EmbedderType,
    MultilingualConfig,
    MultilingualEmbedding,
    MultilingualPipeline,
)

__all__ = [
    # Cross-Lingual Search
    "CrossLingualSearcher",
    "EmbedderType",
    # Language Detection
    "LanguageDetector",
    "LanguageInfo",
    "MultilingualConfig",
    "MultilingualEmbedding",
    # Unified Pipeline
    "MultilingualPipeline",
    "get_cross_lingual_searcher",
    "get_language_detector",
]
