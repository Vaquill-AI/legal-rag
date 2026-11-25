# SSE Streaming Events Module

Server-Sent Events infrastructure for real-time RAG pipeline transparency.

## Overview

This module provides SSE event types and helper functions for streaming:
- **Thinking steps** - Visible AI reasoning process
- **Sources** - Documents found during retrieval
- **Chunks** - LLM response tokens
- **Verification** - Factual grounding scores
- **Done/Error** - Completion and error events

## Usage

### Basic Example

```python
from legal_rag.retrieval.streaming import (
    thinking_event,
    sources_event,
    chunk_event,
    verification_event,
    done_event,
    ThinkingStep,
    ThinkingStepType,
)


async def stream_chat_response(query: str, scope: str):
    """Stream chat response with thinking steps."""

    # Step 1: Analyzing query
    yield thinking_event(
        ThinkingStep(
            step_type=ThinkingStepType.ANALYZING_QUERY,
            message="Understanding your question...",
            progress=0.1,
        )
    )

    # Step 2: Searching documents
    yield thinking_event(
        ThinkingStep(
            step_type=ThinkingStepType.SEARCHING_DOCUMENTS,
            message="Searching through matter documents...",
            progress=0.2,
        )
    )

    # Perform retrieval
    chunks = await retriever.retrieve(query, scope)

    # Step 3: Found sources
    yield thinking_event(
        ThinkingStep(
            step_type=ThinkingStepType.FOUND_SOURCES,
            message=f"Found {len(chunks)} relevant sections",
            details={"count": len(chunks)},
            progress=0.4,
        )
    )
    yield sources_event([c.to_dict() for c in chunks])

    # Step 4: Generating response
    yield thinking_event(
        ThinkingStep(
            step_type=ThinkingStepType.GENERATING_RESPONSE,
            message="Generating response based on sources...",
            progress=0.6,
        )
    )

    # Stream LLM response
    full_response = ""
    async for token in llm.stream(query, chunks):
        full_response += token
        yield chunk_event(token)

    # Step 5: Verify facts
    yield thinking_event(
        ThinkingStep(
            step_type=ThinkingStepType.VERIFYING_FACTS,
            message="Verifying response against sources...",
            progress=0.9,
        )
    )

    verification = await hallucination_checker.check(full_response, chunks)
    yield verification_event(verification.score, verification.unsupported_claims)

    # Done
    message_id = await save_message(full_response, chunks)
    yield done_event(message_id, [c.to_dict() for c in chunks])
```

## Event Types

### ThinkingStepType Enum

```python
class ThinkingStepType(str, Enum):
    ANALYZING_QUERY = "analyzing_query"
    SEARCHING_DOCUMENTS = "searching_documents"
    FOUND_SOURCES = "found_sources"
    ANALYZING_SOURCES = "analyzing_sources"
    CHECKING_RELEVANCE = "checking_relevance"
    GENERATING_RESPONSE = "generating_response"
    VERIFYING_FACTS = "verifying_facts"
```

### Helper Functions

#### `thinking_event(step: ThinkingStep) -> str`

Create a thinking step event.

**Parameters:**
- `step`: ThinkingStep instance with:
  - `step_type`: ThinkingStepType
  - `message`: Human-readable description
  - `details`: Optional metadata dict
  - `progress`: Optional 0.0-1.0 progress indicator

**Returns:** SSE-formatted string

**Example:**
```python
step = ThinkingStep(
    step_type=ThinkingStepType.SEARCHING_DOCUMENTS,
    message="Searching through 15 documents...",
    details={"count": 15, "collection": "legal_docs"},
    progress=0.3,
)
yield thinking_event(step)
# Output: data: {"type": "thinking", "step": "searching_documents", ...}\n\n
```

#### `sources_event(sources: List[Dict]) -> str`

Create a sources found event.

**Parameters:**
- `sources`: List of source dicts with `document_id`, `filename`, `score`, `text`

**Returns:** SSE-formatted string with preview truncated to 200 chars

**Example:**
```python
sources = [
    {"document_id": "doc1", "filename": "contract.pdf", "score": 0.95, "text": "..."},
    {"document_id": "doc2", "filename": "memo.pdf", "score": 0.88, "text": "..."},
]
yield sources_event(sources)
# Output: data: {"type": "sources", "count": 2, "sources": [...]}\n\n
```

#### `chunk_event(content: str) -> str`

Create a streaming chunk event.

**Parameters:**
- `content`: Text chunk from LLM

**Returns:** SSE-formatted string

**Example:**
```python
async for token in llm.stream(...):
    yield chunk_event(token)
# Output: data: {"type": "chunk", "content": "According to "}\n\n
```

#### `verification_event(score: float, unsupported_claims: List[str]) -> str`

Create a verification result event.

**Parameters:**
- `score`: Factual score 0.0-1.0
- `unsupported_claims`: List of claims not supported by sources

**Returns:** SSE-formatted string with `verified=True` if score >= 0.8

**Example:**
```python
yield verification_event(0.92, [])
# Output: data: {"type": "verification", "factual_score": 0.92, "verified": true, ...}\n\n
```

#### `done_event(message_id: str, sources: List[Dict]) -> str`

Create a completion event.

**Parameters:**
- `message_id`: ID of generated message
- `sources`: List of source dicts used

**Returns:** SSE-formatted string

**Example:**
```python
yield done_event("msg-12345", sources)
# Output: data: {"type": "done", "message_id": "msg-12345", ...}\n\n
```

#### `error_event(message: str, code: str = "UNKNOWN") -> str`

Create an error event.

**Parameters:**
- `message`: Human-readable error message
- `code`: Error code (e.g., "EMB_2005", "RAG_1001")

**Returns:** SSE-formatted string

**Example:**
```python
yield error_event("Embedding service unavailable", "EMB_2005")
# Output: data: {"type": "error", "message": "...", "code": "EMB_2005"}\n\n
```

## Edge Cases

The module handles edge cases gracefully:

- **Empty sources**: Returns `{"count": 0, "sources": []}`
- **Empty chunks**: Returns `{"content": ""}`
- **None values**: Replaced with empty defaults
- **Missing fields**: Returns partial data without crashing
- **Score out of range**: Clamped to 0.0-1.0
- **Serialization errors**: Falls back to error event

## Integration with FastAPI

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response with SSE events."""

    async def event_generator():
        try:
            async for event in stream_chat_response(query=request.query, scope=request.scope):
                yield event
        except Exception as e:
            yield error_event(str(e), "CHAT_ERROR")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

## Frontend Consumption

```typescript
const eventSource = new EventSource('/api/chat/stream');

eventSource.addEventListener('message', (e) => {
  const data = JSON.parse(e.data);

  switch (data.type) {
    case 'thinking':
      updateThinkingSteps(data);
      break;
    case 'sources':
      displaySources(data.sources);
      break;
    case 'chunk':
      appendToMessage(data.content);
      break;
    case 'verification':
      showVerificationBadge(data.factual_score, data.verified);
      break;
    case 'done':
      finishMessage(data.message_id);
      break;
    case 'error':
      handleError(data.message, data.code);
      break;
  }
});
```

## Testing

Run the test suite:

```bash
python test_sse_events.py
```

All tests should pass with output showing validation of:
- All 7 ThinkingStepType values
- Empty sources/chunks handling
- Score verification thresholds
- Preview truncation
- SSE format validation
