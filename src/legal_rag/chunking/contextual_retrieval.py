"""
Contextual retrieval.

Generates a short, retrieval-oriented context for each chunk with a language model
and prepends it to the chunk before embedding. Situating a chunk within its document
helps with cross-references and defined terms.

Technique reference: https://www.anthropic.com/news/contextual-retrieval

The language model and the cache are injected through the protocols in
``legal_rag.providers.protocols``. Nothing here reads settings or the environment.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from legal_rag.chunking._logging import get_logger
from legal_rag.chunking.base import Chunk
from legal_rag.providers.protocols import Cache, LanguageModel
from legal_rag.types import Message

logger = get_logger("contextual_retrieval")

_DATA_FILE = Path(__file__).resolve().parent / "data" / "contextual_retrieval.json"


def _load_prompt_data() -> dict[str, Any]:
    with _DATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


_PROMPT_DATA = _load_prompt_data()

# Characters of the document shown to the model (about 6K tokens).
DOCUMENT_SAMPLE_CHARS = 25000

DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 3600


def _content_hash(text: str) -> str:
    """Stable short hash of text, used to key the context cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_context_prompt(chunk_text: str, document_text: str) -> str:
    """Build the prompt that asks for a chunk's retrieval context."""
    functions = " | ".join(_PROMPT_DATA["legal_functions"])
    signals = "\n".join(f"- For {kind}: {text}" for kind, text in _PROMPT_DATA["signals"].items())
    return f"""<document>
{document_text[:DOCUMENT_SAMPLE_CHARS]}
</document>

<chunk_to_contextualize>
{chunk_text}
</chunk_to_contextualize>

<task>Generate retrieval-optimized context for this legal document chunk.</task>

<context_requirements>
<required_elements>
1. Document section/article location with hierarchy path (e.g., "Article 5, Clause 5.2(a)")
2. Defined terms referenced in the chunk (with their document-specific meanings)
3. Cross-references to other sections this chunk relates to
4. Legal function: classify as one of [{functions}]
</required_elements>

<legal_signals>
{signals}
</legal_signals>
</context_requirements>

<output_format>Maximum 2-3 sentences. Start with section location, then legal function, then key terms/cross-references. No preamble.</output_format>"""


class ContextualRetrievalEnhancer:
    """
    Adds model-generated context to chunks.

    Example:
        enhancer = ContextualRetrievalEnhancer(llm=my_model, cache=my_cache)
        texts = await enhancer.enhance_chunks(chunks, full_document, document_id)
    """

    def __init__(
        self,
        llm: LanguageModel,
        cache: Cache | None = None,
        *,
        max_tokens: int = 100,
        max_concurrent: int = 5,
        cache_namespace: str = "contextual",
        cache_ttl_seconds: int | None = DEFAULT_CACHE_TTL_SECONDS,
    ):
        """
        Args:
            llm: Language model used to generate the context.
            cache: Optional cache for generated contexts.
            max_tokens: Max tokens for each generated context.
            max_concurrent: Concurrent model calls.
            cache_namespace: Prefix for cache keys. Change it when the model or prompt changes.
            cache_ttl_seconds: Cache expiry passed to the cache.
        """
        self.llm = llm
        self.cache = cache
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.cache_namespace = cache_namespace
        self.cache_ttl_seconds = cache_ttl_seconds

    def _get_cache_key(self, document_hash: str, chunk_hash: str) -> str:
        """
        Generate the cache key for a contextualized chunk.

        The key uses the prompt version and content hashes of the document and the chunk,
        never the chunk id, so re-chunking the same document hits the cache.
        """
        return f"{self.cache_namespace}:v{_PROMPT_DATA['version']}:{document_hash}:{chunk_hash}"

    async def _generate_context_for_chunk(self, chunk: Chunk, full_document: str) -> str:
        """
        Generate succinct context for one chunk.

        Falls back to a section label when the model call fails.
        """
        prompt = build_context_prompt(chunk.content, full_document)
        try:
            reply = await self.llm.generate(
                [Message(role="user", content=prompt)],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            return reply.strip()
        except Exception as e:  # noqa: BLE001 - degrade gracefully by design
            logger.error("context_generation_failed", error=str(e), chunk_id=chunk.id)
            return f"[Section: {chunk.metadata.get('section', 'Unknown')}]"

    async def enhance_chunks(
        self,
        chunks: list[Chunk],
        full_document: str,
        document_id: str,
    ) -> list[str]:
        """
        Generate contextualized versions of all chunks in a document.

        Args:
            chunks: Chunks to enhance
            full_document: Full document text
            document_id: Document id, used in log fields

        Returns:
            Contextualized chunk texts, in the order of ``chunks``.
        """
        doc_hash = _content_hash(full_document)
        results: list[str | None] = [None] * len(chunks)
        misses: list[int] = []

        for i, chunk in enumerate(chunks):
            cached = None
            if self.cache is not None:
                cached = await self.cache.get(
                    self._get_cache_key(doc_hash, _content_hash(chunk.content))
                )
            if cached:
                results[i] = f"{cached.decode('utf-8')}\n\n{chunk.content}"
            else:
                misses.append(i)

        logger.debug(
            "contextual_cache_check",
            document_id=document_id,
            total_chunks=len(chunks),
            cache_hits=len(chunks) - len(misses),
            cache_misses=len(misses),
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate(idx: int) -> tuple[int, str]:
            async with semaphore:
                return idx, await self._generate_context_for_chunk(chunks[idx], full_document)

        generated = await asyncio.gather(*(generate(i) for i in misses), return_exceptions=True)
        for item in generated:
            if isinstance(item, BaseException):
                logger.error("context_generation_error", error=str(item))
                continue
            idx, context = item
            results[idx] = f"{context}\n\n{chunks[idx].content}"
            if self.cache is not None:
                await self.cache.set(
                    self._get_cache_key(doc_hash, _content_hash(chunks[idx].content)),
                    context.encode("utf-8"),
                    ttl_seconds=self.cache_ttl_seconds,
                )

        final: list[str] = []
        for i, text in enumerate(results):
            if text is None:
                logger.warning("context_fallback", chunk_index=i, document_id=document_id)
                final.append(chunks[i].content)
            else:
                final.append(text)
        return final


class SimplifiedContextEnhancer:
    """
    Rule-based context enhancement without model calls.

    Uses pattern matching to add document context:
    - Document title and type
    - Section hierarchy
    - Defined terms (extracted via regex)

    """

    def enhance_chunks(
        self,
        chunks: list[Chunk],
        document_context: dict,
        section_paths: list[str],
        defined_terms: list[str],
    ) -> list[str]:
        """
        Add rule-based context to chunks.

        Args:
            chunks: Chunks to enhance
            document_context: Document metadata
            section_paths: Section hierarchy per chunk
            defined_terms: Defined terms from document

        Returns:
            List of contextualized chunk texts
        """
        contextualized = []

        for i, chunk in enumerate(chunks):
            context_parts = []

            # Document metadata
            if document_context.get("title"):
                context_parts.append(f"[Document: {document_context['title']}]")

            if document_context.get("document_type"):
                context_parts.append(f"[Type: {document_context['document_type']}]")

            # Section hierarchy
            if i < len(section_paths) and section_paths[i]:
                context_parts.append(f"[Section: {section_paths[i]}]")

            # Defined terms (if chunk references them)
            chunk_text_lower = chunk.content.lower()
            referenced_terms = [term for term in defined_terms if term.lower() in chunk_text_lower]

            if referenced_terms:
                terms_str = ", ".join(referenced_terms[:3])  # Max 3 terms
                context_parts.append(f"[References: {terms_str}]")

            # Build contextualized chunk
            if context_parts:
                context = "\n".join(context_parts)
                contextualized.append(f"{context}\n\n{chunk.content}")
            else:
                contextualized.append(chunk.content)

        return contextualized
