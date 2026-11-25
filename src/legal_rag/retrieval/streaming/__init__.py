"""
SSE streaming module for RAG pipeline.

Provides production-grade Server-Sent Events infrastructure for real-time streaming of:
- Thinking steps (visible AI reasoning)
- Source documents found (with rich legal metadata)
- LLM response chunks
- Factual verification results
- Completion and error events
- Heartbeats for connection stability
- Event IDs for reconnection support

Industry Best Practices Implemented:
- Event IDs for reconnection (MDN SSE specification)
- Sequence numbers for ordering
- Keep-alive heartbeats (Shopify Engineering)
- Retry directives (SSE standard)
- Rich legal metadata (Westlaw/LexisNexis patterns)
- Resilient error handling (FastAPI AI SDK pattern)

Example usage:
    >>> from legal_rag.retrieval.streaming import (
    ...     thinking_event,
    ...     sources_event,
    ...     chunk_event,
    ...     ThinkingStep,
    ...     ThinkingStepType,
    ...     get_stream_context,
    ... )
    >>>
    >>> # Create stream context for event tracking
    >>> context = get_stream_context()
    >>>
    >>> # Create a thinking step with event ID
    >>> step = ThinkingStep(
    ...     step_type=ThinkingStepType.SEARCHING_DOCUMENTS,
    ...     message="Searching through matter documents...",
    ...     progress=0.2
    ... )
    >>>
    >>> # Convert to SSE format with event ID and sequence
    >>> sse_data = thinking_event(step, context)
    >>> # yields: 'id: <uuid>:1\\nevent: thinking\\ndata: {...}\\n\\n'
"""

from legal_rag.retrieval.streaming.events import (
    # Constants
    DEFAULT_RETRY_MS,
    HEARTBEAT_INTERVAL_SECONDS,
    # Models & Classes
    SSEEvent,
    # Enums
    SSEEventType,
    SSEStreamContext,
    ThinkingStep,
    ThinkingStepType,
    # Event helper functions
    chunk_event,
    # Stream context management
    cleanup_stream_context,
    done_event,
    error_event,
    get_stream_context,
    heartbeat_event,
    response_replace_event,
    retry_directive,
    sources_event,
    stream_init_event,
    suggestion_prompt_event,
    thinking_complete_event,
    thinking_event,
    verification_event,
)
from legal_rag.retrieval.streaming.response import (
    SSE_HEADERS,
    ChatEventStream,
    coerce_client_message_id,
)

__all__ = [
    "DEFAULT_RETRY_MS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "SSE_HEADERS",
    "ChatEventStream",
    "SSEEvent",
    "SSEEventType",
    "SSEStreamContext",
    "ThinkingStep",
    "ThinkingStepType",
    "chunk_event",
    "cleanup_stream_context",
    "coerce_client_message_id",
    "done_event",
    "error_event",
    "get_stream_context",
    "heartbeat_event",
    "response_replace_event",
    "retry_directive",
    "sources_event",
    "stream_init_event",
    "suggestion_prompt_event",
    "thinking_complete_event",
    "thinking_event",
    "verification_event",
]
