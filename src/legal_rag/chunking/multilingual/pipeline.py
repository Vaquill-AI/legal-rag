"""
Multilingual embedding pipeline.

Detects the language of each text and routes it to an embedder.
Embedders are injected through the ``Embedder`` protocol, so the pipeline works
with any provider. One multilingual embedder is enough for every language; a
second embedder can be supplied for cross-lingual work.

Routing:
- Every language goes to the default embedder unless ``LANGUAGE_EMBEDDER_MAP``
  (or the config) says otherwise.
- ``EmbedderType.CROSS_LINGUAL`` selects the cross-lingual embedder.
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from legal_rag.chunking._logging import get_logger
from legal_rag.chunking.multilingual.cross_lingual import (
    CrossLingualSearcher,
    SearchDocument,
)
from legal_rag.chunking.multilingual.language_detector import (
    LanguageDetector,
    LanguageInfo,
    get_language_detector,
)
from legal_rag.providers.protocols import Embedder

logger = get_logger("pipeline")


class EmbedderType(str, Enum):
    """Which injected embedder handles a text."""

    DEFAULT = "default"  # The main multilingual embedder
    CROSS_LINGUAL = "cross_lingual"  # The embedder used for cross-lingual search
    HYBRID = "hybrid"  # Reserved for combined approaches


@dataclass
class MultilingualConfig:
    """Configuration for the multilingual pipeline."""

    # Embedding configuration
    default_embedder: EmbedderType = EmbedderType.DEFAULT
    cross_lingual_embedder: EmbedderType = EmbedderType.CROSS_LINGUAL

    # Language detection
    auto_detect_language: bool = True
    fallback_language: str = "en"

    # Search configuration
    enable_cross_lingual: bool = True
    cross_lingual_threshold: float = 0.5

    # Performance
    cache_embeddings: bool = True
    batch_size: int = 32


@dataclass
class MultilingualEmbedding:
    """Result of multilingual embedding."""

    text: str
    language: str
    language_info: LanguageInfo
    embedding: list[float]
    embedder_used: EmbedderType
    dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_preview": (self.text[:100] + "..." if len(self.text) > 100 else self.text),
            "language": self.language,
            "language_name": self.language_info.primary_language_name,
            "embedder_used": self.embedder_used.value,
            "dimension": self.dimension,
            "is_code_mixed": self.language_info.is_code_mixed,
            "is_transliterated": self.language_info.is_transliterated,
        }


class MultilingualPipeline:
    """
    Unified pipeline for multilingual embedding and cross-lingual search.

    Example:
        pipeline = MultilingualPipeline(embedder=my_embedder)
        result = await pipeline.embed("El tribunal dicto sentencia")
        print(result.language, result.dimension)
    """

    # Per-language routing overrides. Empty by default: every language uses the
    # configured default embedder.
    LANGUAGE_EMBEDDER_MAP: ClassVar[dict[str, EmbedderType]] = {}

    def __init__(
        self,
        embedder: Embedder,
        config: MultilingualConfig | None = None,
        cross_lingual_embedder: Embedder | None = None,
        language_detector: LanguageDetector | None = None,
    ):
        """
        Args:
            embedder: Main multilingual embedder.
            config: Pipeline configuration.
            cross_lingual_embedder: Embedder for cross-lingual work. Defaults to ``embedder``.
            language_detector: Detector to use. Defaults to the shared detector.
        """
        self.config = config or MultilingualConfig()
        self._embedder = embedder
        self._cross_embedder = cross_lingual_embedder or embedder
        self._language_detector = language_detector or get_language_detector()
        self._cross_lingual: CrossLingualSearcher | None = None
        self._initialized = False
        self._embedding_cache: dict[str, list[float]] = {}

    async def initialize(self) -> bool:
        """
        Prepare the cross-lingual searcher when it is enabled.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        try:
            if self.config.enable_cross_lingual:
                self._cross_lingual = CrossLingualSearcher(embedder=self._cross_embedder)
                await self._cross_lingual.initialize()

            self._initialized = True
            logger.info(
                "multilingual_pipeline_initialized",
                cross_lingual_ready=(
                    self._cross_lingual.is_initialized() if self._cross_lingual else False
                ),
            )
            return True

        except Exception as e:  # noqa: BLE001 - degrade gracefully by design
            logger.error("multilingual_pipeline_initialization_failed", error=str(e))
            return False

    async def embed(
        self,
        text: str,
        force_embedder: EmbedderType | None = None,
    ) -> MultilingualEmbedding:
        """
        Generate an embedding for ``text`` with automatic language detection.

        Args:
            text: Text to embed
            force_embedder: Force a specific embedder (bypasses routing)

        Returns:
            MultilingualEmbedding with vector and metadata
        """
        if not self._initialized:
            await self.initialize()

        lang_info = self._language_detector.detect(text)
        language = lang_info.primary_language
        embedder_type = force_embedder or self._get_embedder_for_language(language, lang_info)

        cache_key = f"{embedder_type.value}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        if self.config.cache_embeddings and cache_key in self._embedding_cache:
            embedding = self._embedding_cache[cache_key]
        else:
            embedding = await self._generate_embedding(text, embedder_type)
            if self.config.cache_embeddings:
                self._embedding_cache[cache_key] = embedding

        logger.debug(
            "text_embedded",
            language=language,
            embedder=embedder_type.value,
            dimension=len(embedding),
        )

        return MultilingualEmbedding(
            text=text,
            language=language,
            language_info=lang_info,
            embedding=embedding,
            embedder_used=embedder_type,
            dimension=len(embedding),
        )

    async def embed_batch(
        self,
        texts: list[str],
        force_embedder: EmbedderType | None = None,
    ) -> list[MultilingualEmbedding]:
        """
        Generate embeddings for several texts, batching by embedder.

        Args:
            texts: Texts to embed
            force_embedder: Force a specific embedder for all texts

        Returns:
            Results in the same order as ``texts``.
        """
        if not self._initialized:
            await self.initialize()

        if force_embedder:
            return list(await asyncio.gather(*(self.embed(t, force_embedder) for t in texts)))

        groups: dict[EmbedderType, list[tuple[int, str, LanguageInfo]]] = {}
        for i, text in enumerate(texts):
            lang_info = self._language_detector.detect(text)
            embedder_type = self._get_embedder_for_language(lang_info.primary_language, lang_info)
            groups.setdefault(embedder_type, []).append((i, text, lang_info))

        results: list[MultilingualEmbedding | None] = [None] * len(texts)
        for embedder_type, items in groups.items():
            indices, group_texts, lang_infos = zip(*items, strict=False)
            embeddings = await self._batch_embed(list(group_texts), embedder_type)
            for idx, text, lang_info, embedding in zip(
                indices, group_texts, lang_infos, embeddings, strict=False
            ):
                results[idx] = MultilingualEmbedding(
                    text=text,
                    language=lang_info.primary_language,
                    language_info=lang_info,
                    embedding=embedding,
                    embedder_used=embedder_type,
                    dimension=len(embedding),
                )

        return [r for r in results if r is not None]

    def _get_embedder_for_language(
        self,
        language: str,
        lang_info: LanguageInfo,
    ) -> EmbedderType:
        """Pick the embedder for a language."""
        return self.LANGUAGE_EMBEDDER_MAP.get(language, self.config.default_embedder)

    async def _generate_embedding(self, text: str, embedder_type: EmbedderType) -> list[float]:
        """Generate one embedding with the selected embedder."""
        return (await self._batch_embed([text], embedder_type))[0]

    async def _batch_embed(
        self,
        texts: list[str],
        embedder_type: EmbedderType,
    ) -> list[list[float]]:
        """Embed texts with the selected embedder."""
        if embedder_type == EmbedderType.CROSS_LINGUAL:
            return await self._cross_embedder.embed_documents(texts)
        return await self._embedder.embed_documents(texts)

    async def cross_lingual_search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search documents across languages.

        Args:
            query: Search query (any language)
            documents: Documents with ``text`` and optional ``language``, ``id``, ``metadata``
            top_k: Number of results

        Returns:
            Ranked search results
        """
        if not self.config.enable_cross_lingual:
            raise ValueError("Cross-lingual search is disabled in config")

        if not self._initialized:
            await self.initialize()

        search_docs = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            lang = doc.get("language") or self._language_detector.detect(text).primary_language
            search_docs.append(
                SearchDocument(
                    id=doc.get("id", str(i)),
                    text=text,
                    language=lang,
                    metadata=doc.get("metadata", {}),
                )
            )

        if self._cross_lingual is None:
            raise RuntimeError("Cross-lingual searcher not initialized")
        await self._cross_lingual.index_documents(search_docs)
        result = await self._cross_lingual.search(
            query,
            top_k=top_k,
            min_score=self.config.cross_lingual_threshold,
        )
        return result.results

    def get_supported_languages(self) -> dict[str, dict[str, Any]]:
        """Describe the languages the detector knows and the embedder each routes to."""
        return {
            code: {
                "name": str(row["name"]),
                "embedder": self.LANGUAGE_EMBEDDER_MAP.get(
                    code, self.config.default_embedder
                ).value,
            }
            for code, row in self._language_detector.LANGUAGES.items()
        }

    def detect_language(self, text: str) -> LanguageInfo:
        """Detect the language of ``text``."""
        return self._language_detector.detect(text)

    def is_initialized(self) -> bool:
        """Check if the pipeline is initialized."""
        return self._initialized

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
