"""
Cross-lingual search.

Enables searching across languages: a query in one language can retrieve
documents written in another.

Embeddings come from one of two places:
- An injected ``Embedder`` (see ``legal_rag.providers.protocols``), preferably a
  multilingual model. This is the default path and needs no extra packages.
- A local sentence-transformers model (LaBSE by default), loaded lazily when no
  embedder is injected. This path needs the optional ``sentence-transformers``
  package.

References:
- LaBSE: https://arxiv.org/abs/2007.01852
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, ClassVar

from legal_rag.chunking._logging import get_logger
from legal_rag.providers.protocols import Embedder

logger = get_logger("cross_lingual")


@dataclass
class CrossLingualConfig:
    """Configuration for cross-lingual search."""

    model_name: str = "sentence-transformers/LaBSE"
    max_length: int = 256  # LaBSE optimized for shorter sequences
    batch_size: int = 32
    normalize_embeddings: bool = True
    similarity_threshold: float = 0.5
    device: str = "auto"
    cache_dir: str | None = None


@dataclass
class CrossLingualResult:
    """Result of cross-lingual search."""

    query: str
    query_language: str
    results: list[dict[str, Any]]
    languages_searched: list[str]
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_language": self.query_language,
            "result_count": len(self.results),
            "languages_searched": self.languages_searched,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "results": self.results,
        }


@dataclass
class SearchDocument:
    """Document for cross-lingual search."""

    id: str
    text: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


class CrossLingualSearcher:
    """
    Cross-lingual semantic search over language-agnostic embeddings.

    Enables language-agnostic search where queries in one language
    can find documents in any other supported language.

    Example:
        searcher = CrossLingualSearcher(embedder=my_multilingual_embedder)
        await searcher.initialize()

        # Index documents in multiple languages
        docs = [
            SearchDocument(id="1", text="Murder is punishable under the statute", language="en"),
            SearchDocument(id="2", text="El homicidio es punible segun la ley", language="es"),
        ]
        await searcher.index_documents(docs)

        # Search in English, find documents in any indexed language
        results = await searcher.search("What is the punishment for murder?")
    """

    # Languages with display names (LaBSE covers 109 languages; extend as needed)
    SUPPORTED_LANGUAGES: ClassVar[dict[str, str]] = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ar": "Arabic",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "kn": "Kannada",
        "ml": "Malayalam",
        "gu": "Gujarati",
        "mr": "Marathi",
        "bn": "Bengali",
        "pa": "Punjabi",
        "or": "Odia",
        "ur": "Urdu",
        "ne": "Nepali",
        "as": "Assamese",
    }

    def __init__(
        self,
        config: CrossLingualConfig | None = None,
        embedder: Embedder | None = None,
    ):
        """
        Initialize cross-lingual searcher.

        Args:
            config: Optional configuration
            embedder: Optional multilingual embedder. When given, no local model is loaded.
        """
        self.config = config or CrossLingualConfig()
        self._embedder = embedder
        self._model = None
        self._device = None
        self._initialized = False
        self._document_store: dict[str, SearchDocument] = {}
        self._embeddings_cache: dict[str, list[float]] = {}

    async def initialize(self) -> bool:
        """
        Prepare the embedder, loading the local LaBSE model only when none was injected.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        if self._embedder is not None:
            self._initialized = True
            return True

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            self._initialized = True

            logger.info(
                "cross_lingual_searcher_initialized",
                model=self.config.model_name,
                device=str(self._device),
                dimension=768,
            )
            return True

        except Exception as e:  # noqa: BLE001 - degrade gracefully by design
            logger.error(
                "cross_lingual_initialization_failed",
                error=str(e),
            )
            return False

    def _load_model(self):
        """Load LaBSE model (blocking operation)."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self.config.device

            self._model = SentenceTransformer(
                self.config.model_name,
                device=self._device,
                cache_folder=self.config.cache_dir,
            )

        except ImportError:
            logger.warning(
                "labse_dependencies_missing",
                message="Install sentence-transformers: pip install sentence-transformers",
            )
            raise

    async def embed(
        self,
        texts: str | list[str],
    ) -> list[list[float]]:
        """
        Generate language-agnostic embeddings.

        Args:
            texts: Text(s) to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        if isinstance(texts, str):
            texts = [texts]

        if self._embedder is not None:
            return await self._embedder.embed_documents(texts)

        loop = asyncio.get_event_loop()

        def _encode():
            if self._model is None:
                raise RuntimeError("Model not initialized. Call initialize() first.")
            return self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            ).tolist()

        return await loop.run_in_executor(None, _encode)

    async def index_documents(
        self,
        documents: list[SearchDocument],
    ) -> int:
        """
        Index documents for cross-lingual search.

        Args:
            documents: Documents to index

        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0

        # Generate embeddings for documents without them
        texts_to_embed = []
        docs_to_embed = []

        for doc in documents:
            if doc.embedding is None:
                texts_to_embed.append(doc.text)
                docs_to_embed.append(doc)

        if texts_to_embed:
            embeddings = await self.embed(texts_to_embed)
            for doc, embedding in zip(docs_to_embed, embeddings, strict=False):
                doc.embedding = embedding

        # Store documents
        for doc in documents:
            self._document_store[doc.id] = doc
            if doc.embedding is not None:
                self._embeddings_cache[doc.id] = doc.embedding

        logger.info(
            "documents_indexed",
            count=len(documents),
            languages=list({d.language for d in documents}),
        )

        return len(documents)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        language_filter: list[str] | None = None,
        min_score: float | None = None,
    ) -> CrossLingualResult:
        """
        Search documents across languages.

        Args:
            query: Search query (any language)
            top_k: Number of results to return
            language_filter: Optional filter for specific languages
            min_score: Minimum similarity score

        Returns:
            CrossLingualResult with ranked results
        """
        import time

        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Detect query language
        from legal_rag.chunking.multilingual.language_detector import detect_language

        query_lang_info = detect_language(query)

        # Embed query
        query_embedding = (await self.embed(query))[0]

        # Calculate similarities with all documents
        results = []
        languages_found = set()

        if min_score is None:
            min_score = self.config.similarity_threshold

        for doc in self._document_store.values():
            # Apply language filter if specified
            if language_filter and doc.language not in language_filter:
                continue

            # Skip documents without embeddings
            if doc.embedding is None:
                continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, doc.embedding)

            if score >= min_score:
                results.append(
                    {
                        "id": doc.id,
                        "text": (doc.text[:500] + "..." if len(doc.text) > 500 else doc.text),
                        "language": doc.language,
                        "score": round(score, 4),
                        "metadata": doc.metadata,
                        "is_cross_lingual": doc.language != query_lang_info.primary_language,
                    }
                )
                languages_found.add(doc.language)

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "cross_lingual_search_complete",
            query_language=query_lang_info.primary_language,
            results_count=len(results),
            languages_found=list(languages_found),
            processing_time_ms=round(processing_time, 2),
        )

        return CrossLingualResult(
            query=query,
            query_language=query_lang_info.primary_language,
            results=results,
            languages_searched=list(languages_found),
            processing_time_ms=processing_time,
        )

    async def find_translations(
        self,
        text: str,
        target_languages: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Find semantically similar text in other languages.

        Useful for finding parallel content or translations.

        Args:
            text: Source text
            target_languages: Languages to search (default: all except source)
            top_k: Results per language

        Returns:
            Dictionary mapping language codes to similar texts
        """
        from legal_rag.chunking.multilingual.language_detector import detect_language

        source_lang_info = detect_language(text)
        source_lang = source_lang_info.primary_language

        if target_languages is None:
            target_languages = [lang for lang in self.SUPPORTED_LANGUAGES if lang != source_lang]

        # Embed source text
        source_embedding = (await self.embed(text))[0]

        results_by_language = {}

        for target_lang in target_languages:
            results = []

            for doc in self._document_store.values():
                if doc.language != target_lang:
                    continue

                # Skip documents without embeddings
                if doc.embedding is None:
                    continue

                score = self._cosine_similarity(source_embedding, doc.embedding)

                if score >= self.config.similarity_threshold:
                    results.append(
                        {
                            "id": doc.id,
                            "text": (doc.text[:300] + "..." if len(doc.text) > 300 else doc.text),
                            "score": round(score, 4),
                        }
                    )

            # Sort and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            if results:
                results_by_language[target_lang] = results[:top_k]

        return results_by_language

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_document(self, doc_id: str) -> SearchDocument | None:
        """Get document by ID."""
        return self._document_store.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """Remove document from index."""
        if doc_id in self._document_store:
            del self._document_store[doc_id]
            if doc_id in self._embeddings_cache:
                del self._embeddings_cache[doc_id]
            return True
        return False

    def clear_index(self):
        """Clear all indexed documents."""
        self._document_store.clear()
        self._embeddings_cache.clear()

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._document_store)

    @property
    def dimension(self) -> int:
        """Embedding dimension (768 for LaBSE)."""
        return 768

    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized


# Singleton instance
_default_searcher: CrossLingualSearcher | None = None


def get_cross_lingual_searcher() -> CrossLingualSearcher:
    """Get or create default cross-lingual searcher instance."""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = CrossLingualSearcher()
    return _default_searcher


async def cross_lingual_search(
    query: str,
    documents: list[SearchDocument],
    top_k: int = 10,
) -> CrossLingualResult:
    """
    Convenience function for cross-lingual search.

    Args:
        query: Search query
        documents: Documents to search
        top_k: Number of results

    Returns:
        CrossLingualResult
    """
    searcher = get_cross_lingual_searcher()
    await searcher.index_documents(documents)
    return await searcher.search(query, top_k=top_k)
