"""HyDE query transform for conversations. A thin re-export of the strategies implementation.

The implementation lives in ``legal_rag.retrieval.strategies.hyde``, which holds the one HyDE
code path: ``HyDEGenerator`` writes the hypothetical document and ``HyDETransformer`` gates the
call and builds the text to embed. This module keeps the names that earlier releases exposed
here so existing imports keep working.

``HyDEResult`` here is the transformer's result (``original_query``, ``hypothetical_document``,
``combined_query``, ``processing_time_ms``, ``used_hyde``). It is NOT the
``legal_rag.retrieval.strategies.HyDEResult`` that ``HyDEGenerator.generate`` returns. New code
should use the unambiguous name ``HyDETransformResult``.
"""

from __future__ import annotations

from legal_rag.retrieval.strategies.hyde import HyDETransformer, HyDETransformResult
from legal_rag.retrieval.strategies.hyde_prompts import (
    HYDE_GENERAL_SYSTEM_PROMPT,
    HYDE_LEGAL_SYSTEM_PROMPT,
)

HyDEResult = HyDETransformResult
"""Historic name of ``HyDETransformResult``. Not the result type of ``HyDEGenerator``."""

__all__ = [
    "HYDE_GENERAL_SYSTEM_PROMPT",
    "HYDE_LEGAL_SYSTEM_PROMPT",
    "HyDEResult",
    "HyDETransformResult",
    "HyDETransformer",
]
