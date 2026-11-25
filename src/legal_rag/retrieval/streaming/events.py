"""
SSE (Server-Sent Events) event types and helper functions for RAG streaming.

This module provides production-grade infrastructure for streaming thinking steps,
sources, chunks, and verification results to the frontend during chat responses.

Based on the "Stream of Thought" UX pattern used by Perplexity, Claude, and Harvey AI.

Industry Best Practices Implemented:
- Event IDs for reconnection support (MDN SSE specification)
- Sequence numbers for ordering
- Keep-alive heartbeats for connection stability (Shopify Engineering)
- Retry directives for client reconnection timing (SSE standard)
- Rich legal metadata for sources (Westlaw/LexisNexis patterns)
- Resilient error handling (FastAPI AI SDK pattern)
- Named event types for semantic clarity (SSE best practice)

References:
- https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- https://shopify.engineering/server-sent-events-data-streaming
- https://www.speakeasy.com/openapi/content/server-sent-events
"""
# ruff: noqa: BLE001, PERF102

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("rag.streaming.events")


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Default retry interval for client reconnection (milliseconds)
DEFAULT_RETRY_MS = 3000

# Heartbeat interval recommendation (seconds)
HEARTBEAT_INTERVAL_SECONDS = 15


# =============================================================================
# ENUMS
# =============================================================================


class ThinkingStepType(str, Enum):
    """
    Enumeration of thinking step types for visible AI reasoning.

    These steps represent the different stages of the RAG pipeline
    and provide transparency into what the AI is doing.
    """

    ANALYZING_QUERY = "analyzing_query"
    SEARCHING_DOCUMENTS = "searching_documents"
    SEARCHING_WEB = "searching_web"
    FOUND_SOURCES = "found_sources"
    ANALYZING_SOURCES = "analyzing_sources"
    CHECKING_RELEVANCE = "checking_relevance"
    GENERATING_RESPONSE = "generating_response"
    VERIFYING_FACTS = "verifying_facts"

    # Multi-hop retrieval steps (CoRAG-style iterative reasoning)
    # These provide transparency into the iterative retrieval process
    MULTI_HOP_STARTED = "multi_hop_started"  # Starting multi-hop retrieval
    MULTI_HOP_ITERATION = "multi_hop_iteration"  # Each hop in the chain
    MULTI_HOP_GAP_ANALYSIS = "multi_hop_gap_analysis"  # Analyzing what info is missing
    MULTI_HOP_COMPLETE = "multi_hop_complete"  # Multi-hop retrieval finished

    # Deep Research Mode specific steps (Exa Research API)
    # These provide richer feedback during the 30-120s research process
    DEEP_RESEARCH_STARTING = "deep_research_starting"  # Initial setup
    DEEP_RESEARCH_SEARCHING = "deep_research_searching"  # Actively searching web
    DEEP_RESEARCH_ANALYZING = "deep_research_analyzing"  # Reading and analyzing sources
    DEEP_RESEARCH_SYNTHESIZING = "deep_research_synthesizing"  # Creating report
    DEEP_RESEARCH_COMPLETE = "deep_research_complete"  # Research finished

    # Exa Research API real-time event types (from ?events=true streaming)
    # These map directly to Exa's event types for live progress display

    # Agentic tool calling steps (LLM-driven tool use)
    CALLING_TOOL = "calling_tool"  # LLM requested a tool call

    # Contract review / redline steps (inline CR pipeline in chat). Kept OUT of
    # the frontend's RAG-flavored STEP_REASONING map so each phase shows its own
    # clean label ("Analyzing clauses against playbook") with no "your query"
    # subtext.
    CONTRACT_REVIEW = "contract_review"
    TOOL_FETCHING_URL = "tool_fetching_url"  # fetch_url tool in progress
    TOOL_RESOLVING_CITATION = "tool_resolving_citation"  # resolve_citation in progress
    TOOL_CALLING_MCP = "tool_calling_mcp"  # MCP server tool in progress
    TOOL_RESULT_RECEIVED = "tool_result_received"  # Tool completed, result injected

    # Skill activation (curated playbooks loaded on demand)
    LOADING_SKILL = "loading_skill"  # Skill playbook being loaded from DB

    # Autonomous capability auto-enable (pre-flight resolution)
    AUTO_ENABLED = "auto_enabled"  # System autonomously enabled a capability


class SSEEventType(str, Enum):
    """
    Standard SSE event types used in the streaming protocol.

    Following common AI streaming patterns.
    """

    THINKING = "thinking"
    THINKING_COMPLETE = "thinking_complete"
    SOURCES = "sources"
    CHUNK = "chunk"
    VERIFICATION = "verification"
    DONE = "done"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STREAM_INIT = "stream_init"
    SUGGESTION_PROMPT = "suggestion_prompt"  # Interactive yes/no before LLM runs
    RESPONSE_REPLACE = "response_replace"  # Clear streamed content before regeneration


# =============================================================================
# MODELS
# =============================================================================


class ThinkingStep(BaseModel):
    """
    Represents a single thinking step in the AI reasoning process.

    Attributes:
        step_type: The type of thinking step being performed
        message: Human-readable description of what's happening
        details: Optional additional metadata (e.g., {"count": 10})
        progress: Optional progress indicator from 0.0 to 1.0
    """

    step_type: ThinkingStepType = Field(..., description="Type of thinking step")
    message: str = Field(..., description="Human-readable description")
    details: dict[str, Any] | None = Field(None, description="Additional metadata")
    progress: float | None = Field(
        None, ge=0.0, le=1.0, description="Progress indicator (0.0 to 1.0)"
    )


@dataclass
class SSEStreamContext:
    """
    Maintains state for an SSE stream session.

    Provides event IDs and sequence numbers for reconnection support
    as per MDN SSE specification.

    Attributes:
        stream_id: Unique identifier for this stream session
        sequence: Auto-incrementing sequence number for event ordering
        created_at: Unix timestamp of stream creation
    """

    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence: int = field(default=0)
    created_at: float = field(default_factory=time.time)

    def next_event_id(self) -> str:
        """Generate next event ID with sequence number."""
        self.sequence += 1
        return f"{self.stream_id}:{self.sequence}"

    def get_sequence(self) -> int:
        """Get current sequence number without incrementing."""
        return self.sequence


class SSEEvent(BaseModel):
    """
    Production-grade SSE event structure.

    Follows the SSE specification with additions for:
    - Event IDs (for reconnection via Last-Event-ID header)
    - Sequence numbers (for ordering)
    - Named event types (for semantic event filtering)
    - Retry directives (for client reconnection timing)

    Attributes:
        event_type: Type of event (thinking, sources, chunk, etc.)
        data: Event payload as a dictionary
        event_id: Optional unique event ID for reconnection
        sequence: Optional sequence number for ordering
        retry_ms: Optional client retry interval in milliseconds
    """

    event_type: str = Field(..., description="Event type identifier")
    data: dict[str, Any] = Field(..., description="Event payload")
    event_id: str | None = Field(None, description="Unique event ID for reconnection")
    sequence: int | None = Field(None, description="Sequence number for ordering")
    retry_ms: int | None = Field(None, description="Client retry interval (ms)")

    def to_sse(self) -> str:
        """
        Convert event to SSE format following the specification.

        Format:
            id: <event_id>           (optional, for reconnection)
            event: <event_type>      (optional, for named events)
            retry: <milliseconds>    (optional, for reconnection timing)
            data: <json_payload>     (required)

            (empty line to end event)

        Returns:
            Properly formatted SSE string
        """
        try:
            lines = []

            # Add event ID for reconnection support (MDN best practice)
            if self.event_id:
                lines.append(f"id: {self.event_id}")

            # Add named event type for semantic filtering (SSE best practice)
            # This allows EventSource.addEventListener('sources', ...) on client
            lines.append(f"event: {self.event_type}")

            # Add retry directive for first event or errors (SSE standard)
            if self.retry_ms:
                lines.append(f"retry: {self.retry_ms}")

            # Build payload with type and sequence for compatibility
            payload = {
                "type": self.event_type,
                **self.data,
            }

            # Add sequence number if present
            if self.sequence is not None:
                payload["sequence_number"] = self.sequence

            lines.append(f"data: {json.dumps(payload)}")

            # End with empty line (SSE specification)
            lines.append("")
            lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(
                "sse_serialization_failed",
                error=str(e),
                event_type=self.event_type,
                event_id=self.event_id,
            )
            # Fallback to minimal error event
            return 'event: error\ndata: {"type": "error", "message": "Serialization failed"}\n\n'


# =============================================================================
# STREAM CONTEXT FACTORY
# =============================================================================


_stream_contexts: dict[str, SSEStreamContext] = {}


def get_stream_context(stream_id: str | None = None) -> SSEStreamContext:
    """
    Get or create a stream context for event tracking.

    Args:
        stream_id: Optional existing stream ID to retrieve context for

    Returns:
        SSEStreamContext for the stream
    """
    if stream_id and stream_id in _stream_contexts:
        return _stream_contexts[stream_id]

    context = SSEStreamContext(stream_id=stream_id or str(uuid.uuid4()))
    _stream_contexts[context.stream_id] = context
    return context


def cleanup_stream_context(stream_id: str) -> None:
    """
    Clean up stream context after stream completion.

    Args:
        stream_id: Stream ID to clean up
    """
    _stream_contexts.pop(stream_id, None)


# =============================================================================
# SOURCE DEDUPLICATION
# =============================================================================


def _get_dedup_key(source: dict) -> str:
    """
    Generate a deduplication key for a source.

    Key priority (most specific to least specific):
    1. document_id + section_title (preserves different sections as separate)
    2. document_id + section_number (fallback for numbered sections)
    3. document_id + page_number (fallback when no section info)
    4. document_id only (last resort for web sources, etc.)

    This ensures:
    - Same document, different sections -> separate sources (different keys)
    - Same document, same section, different pages -> merged (same key)
    - Same document, same page, no section -> merged (same key)

    Args:
        source: Source dictionary with metadata

    Returns:
        Deduplication key string
    """
    doc_id = source.get("document_id") or source.get("chunk_id") or "unknown"
    metadata = source.get("metadata") or {}

    # Priority 1: section_title (most semantically meaningful for legal docs)
    section_title = source.get("section_title") or metadata.get("section_title")
    if section_title:
        # Normalize: lowercase, strip whitespace for consistent matching
        normalized_section = str(section_title).lower().strip()
        return f"{doc_id}::section::{normalized_section}"

    # Priority 2: section_number (common in legal documents)
    section_number = source.get("section_number") or metadata.get("section_number")
    if section_number:
        return f"{doc_id}::section_num::{section_number}"

    # Priority 3: page_number (fallback when no section info)
    # This prevents merging content from completely different pages
    page = source.get("page_start") or source.get("page_number") or metadata.get("page_number")
    if page is not None:
        return f"{doc_id}::page::{page}"

    # Priority 4: document_id only (web sources, sources without page info)
    return doc_id


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    """
    Deduplicate sources by document + section/page, merging related chunks.

    Uses a hierarchical deduplication key to balance between:
    - Avoiding duplicate entries (same content shown multiple times)
    - Preserving distinct content (different sections of same document)

    Example transformations:
    BEFORE (duplicates):
    - "Contract.pdf" Section 5 "Definitions" (page 2, score 0.95)
    - "Contract.pdf" Section 5 "Definitions" (page 2, score 0.88)  <- MERGED
    - "Contract.pdf" Section 12 "Termination" (page 8, score 0.82) <- KEPT SEPARATE

    AFTER:
    - "Contract.pdf" Section 5 "Definitions" (page 2, score 0.95, 2 chunks)
    - "Contract.pdf" Section 12 "Termination" (page 8, score 0.82)

    Deduplication key priority:
    1. document_id + section_title (different sections stay separate)
    2. document_id + section_number (fallback)
    3. document_id + page_number (fallback when no section)
    4. document_id only (web sources, etc.)

    Merge strategy for grouped chunks:
    - page_start: min of all chunks
    - page_end: max of all chunks
    - score: max of all chunks (most relevant)
    - text: keep highest-scored chunk's text (best excerpt)
    - char_start/char_end: keep from highest-scored chunk (for PDF highlighting)
    - chunk_ids: collect all for internal tracking

    Args:
        sources: List of source dictionaries from RAG pipeline

    Returns:
        Deduplicated list of sources with merged metadata
    """
    if not sources:
        return sources

    # Group by deduplication key
    doc_groups: dict[str, list[dict]] = {}
    for source in sources:
        dedup_key = _get_dedup_key(source)
        if dedup_key not in doc_groups:
            doc_groups[dedup_key] = []
        doc_groups[dedup_key].append(source)

    # Merge each group
    deduplicated = []
    for _dedup_key, group in doc_groups.items():
        if len(group) == 1:
            # No duplicates - pass through unchanged
            deduplicated.append(group[0])
            continue

        # Sort by score descending to keep best one as base
        group.sort(key=lambda x: x.get("score") or 0, reverse=True)
        merged = group[0].copy()  # Start with highest-scored chunk (best text, char ranges)

        # Collect all chunk_ids for internal tracking
        all_chunk_ids = [s.get("chunk_id") for s in group if s.get("chunk_id")]
        if all_chunk_ids:
            merged["chunk_ids"] = all_chunk_ids
            # Keep first chunk_id as primary (for React key compatibility)
            merged["chunk_id"] = all_chunk_ids[0]

        # P3: preserve the per-chunk excerpts that were collapsed into this
        # source so the UI can render expandable sub-citations. Without this
        # the merge silently throws away N-1 chunks' text/char ranges/scores,
        # and users have no way to see what grounding the LLM actually had.
        # Order matches the score-descending sort above (best first).
        merged["merged_chunks"] = [
            {
                "chunk_id": s.get("chunk_id"),
                "text": s.get("text", ""),
                "score": s.get("score"),
                "page_number": s.get("page_number") or (s.get("metadata") or {}).get("page_number"),
                "page_start": s.get("page_start"),
                "page_end": s.get("page_end"),
                "char_start": s.get("char_start"),
                "char_end": s.get("char_end"),
                "section_title": s.get("section_title")
                or (s.get("metadata") or {}).get("section_title"),
                "section_number": s.get("section_number")
                or (s.get("metadata") or {}).get("section_number"),
            }
            for s in group
        ]

        # Merge page ranges across all chunks in group
        page_starts = []
        page_ends = []
        for s in group:
            ps = s.get("page_start") or s.get("page_number")
            pe = s.get("page_end") or s.get("page_number")
            if ps is not None:
                page_starts.append(ps)
            if pe is not None:
                page_ends.append(pe)

        if page_starts:
            merged["page_start"] = min(page_starts)
            merged["page_number"] = min(page_starts)  # For single-page display fallback
        if page_ends:
            merged["page_end"] = max(page_ends)

        # NOTE: char_start/char_end are kept from the highest-scored chunk (group[0])
        # We don't merge char ranges because:
        # 1. Merged chunks may span multiple pages (char ranges become meaningless)
        # 2. The highest-scored chunk has the most relevant excerpt for highlighting

        # Keep max score (most relevant)
        scores = [s.get("score") for s in group if s.get("score") is not None]
        if scores:
            merged["score"] = max(scores)

        # Mark as consolidated for frontend awareness
        merged["chunks_consolidated"] = len(group)

        deduplicated.append(merged)

    # Do NOT sort by score here. The answer's [N] citations are numbered in the
    # pipeline's source order (reorder / U-shape), and done_event emits sources in
    # that SAME order. Re-sorting the streaming panel by score desynced it from
    # the [N] numbering, so a click on [1] mid-stream opened the wrong card.
    # Preserve the incoming (first-occurrence) order so every emission of the
    # source list shares one order. (pipeline bug-fix #8)

    logger.debug(
        "sources_deduplicated",
        original_count=len(sources),
        deduplicated_count=len(deduplicated),
        reduction=len(sources) - len(deduplicated),
    )

    return deduplicated


# =============================================================================
# EVENT HELPER FUNCTIONS
# =============================================================================


def thinking_event(step: ThinkingStep, context: SSEStreamContext | None = None) -> str:
    """
    Create a thinking step SSE event.

    Args:
        step: ThinkingStep instance with step details
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted string for frontend consumption

    Example:
        >>> step = ThinkingStep(
        ...     step_type=ThinkingStepType.SEARCHING_DOCUMENTS,
        ...     message="Searching through matter documents...",
        ...     progress=0.2
        ... )
        >>> thinking_event(step)
        'event: thinking\\ndata: {"type": "thinking", "step": "searching_documents", ...}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.THINKING.value,
            data={
                "step": step.step_type.value,
                "message": step.message,
                "details": step.details,
                "progress": step.progress,
            },
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("thinking_event_failed", error=str(e), step_type=step.step_type)
        return error_event(f"Failed to create thinking event: {e!s}", "THINKING_EVENT_ERROR")


def thinking_complete_event(context: SSEStreamContext | None = None) -> str:
    """
    Create a thinking complete SSE event.

    This event signals that all thinking/reasoning steps are done
    and the answer streaming is about to begin. Frontend should
    collapse/close the thinking steps UI when this is received.

    Args:
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted string signaling thinking is complete

    Example:
        >>> thinking_complete_event()
        'event: thinking_complete\\ndata: {"type": "thinking_complete"}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.THINKING_COMPLETE.value,
            data={},
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("thinking_complete_event_failed", error=str(e))
        return error_event(
            f"Failed to create thinking_complete event: {e!s}", "THINKING_COMPLETE_EVENT_ERROR"
        )


def sources_event(
    sources: list[dict],
    context: SSEStreamContext | None = None,
    alternative_sources: list[dict] | None = None,
    confidence: float | None = None,
) -> str:
    """
    Create a sources found SSE event with rich legal metadata.

    Following Westlaw/LexisNexis patterns for legal AI, includes:
    - Document identification (ID, filename, chunk)
    - Relevancy scoring
    - Legal metadata (court, jurisdiction, case type)
    - Citation information
    - Temporal metadata (year, decision date)
    - Alternative sources for contrastive citation explanations

    Args:
        sources: List of source dictionaries with metadata (selected sources)
        context: Optional stream context for event IDs
        alternative_sources: Optional list of alternative sources for contrastive
            citation explanations ("why this and not that")

    Returns:
        SSE-formatted string with source information

    Example:
        >>> sources = [
        ...     {
        ...         "document_id": "doc1",
        ...         "filename": "contract.pdf",
        ...         "score": 0.95,
        ...         "text": "...",
        ...         "metadata": {"court": "Supreme Court", "year": 2023}
        ...     }
        ... ]
        >>> sources_event(sources)
        'event: sources\\ndata: {"type": "sources", "count": 1, ...}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        # Handle empty sources gracefully.
        #
        # NOTE the deliberate absence of a "confidence" key. Nothing was
        # retrieved, so there is no confidence to report, and inventing one here
        # would be a lie about grounding. Do NOT "fix the shape" by defaulting it.
        #
        # The consumer must treat a missing confidence as UNKNOWN. The frontend
        # used to do `event.confidence ?? 0.8` and persisted that placeholder to
        # chat_messages.metadata.confidence: 276 answers over 60 days, every one
        # with zero sources, every one reading exactly 0.800 across 17 different
        # rag tiers, while genuinely grounded answers scored anywhere from 0.29
        # to 0.857. An ungrounded answer therefore outranked almost every
        # grounded one. See A4 in docs/CHAT_QUALITY_AUDIT_2026_08_19.md.
        if not sources:
            return SSEEvent(
                event_type=SSEEventType.SOURCES.value,
                data={"count": 0, "sources": [], "has_sources": False},
                event_id=event_id,
                sequence=sequence,
            ).to_sse()

        # Deduplicate sources by document_id to prevent duplicate entries in sidepanel
        # Multiple chunks from the same document are merged into a single source
        sources = _deduplicate_sources(sources)

        # Extract sources with full legal metadata for frontend consumption
        # Following Westlaw/LexisNexis patterns for legal AI
        formatted_sources = []
        for idx, s in enumerate(sources):
            metadata = s.get("metadata") or {}

            # Debug: Log source data to trace pdf_url, is_shared, and PDF highlighting fields
            logger.debug(
                "sources_event_source_data",
                source_index=idx,
                document_id=s.get("document_id"),
                # PDF highlighting fields (CRITICAL for inline viewer)
                page_start=s.get("page_start"),
                page_end=s.get("page_end"),
                char_start=s.get("char_start"),
                char_end=s.get("char_end"),
                page_number=s.get("page_number"),
                metadata_page_start=metadata.get("page_start"),
                metadata_char_start=metadata.get("char_start"),
                # File URL fields
                has_top_level_file_url=bool(s.get("file_url")),
                top_level_file_url=s.get("file_url"),
                has_top_level_pdf_url=bool(s.get("pdf_url")),
                top_level_pdf_url=s.get("pdf_url"),
                has_top_level_is_shared=bool(s.get("is_shared")),
                top_level_is_shared=s.get("is_shared"),
                has_metadata_file_url=bool(metadata.get("file_url")),
                metadata_file_url=metadata.get("file_url"),
                has_metadata_pdf_url=bool(metadata.get("pdf_url")),
                metadata_pdf_url=metadata.get("pdf_url"),
                has_metadata_is_shared=bool(metadata.get("is_shared")),
                metadata_is_shared=metadata.get("is_shared"),
                is_corpus=metadata.get("is_corpus", False),
            )

            formatted_source = {
                # Core identifiers
                "document_id": s.get("document_id"),
                "chunk_id": s.get("chunk_id"),
                # The card's human label. Web + some statute sources leave `title`
                # empty and carry the label in `filename`; when even that is empty
                # (e.g. a web result whose title came back as ""), fall back to the
                # domain/url so the card never renders a blank chip.
                "filename": (
                    s.get("filename")
                    or metadata.get("title")
                    or s.get("domain")
                    or metadata.get("domain")
                    or s.get("url")
                    or metadata.get("url")
                ),
                # Relevancy scoring (like Vincent AI relevancy ranking)
                "score": s.get("score"),
                # Full text for inline display
                "text": s.get("text", ""),
                # Section information (for document navigation)
                "section_title": s.get("section_title") or metadata.get("section_title"),
                "section_number": s.get("section_number") or metadata.get("section_number"),
                # Page/char position fields use `is not None` checks instead of `or`
                # because 0 is a valid value (char_start=0 for first chunk) but falsy
                "page_number": s.get("page_number")
                if s.get("page_number") is not None
                else metadata.get("page_number"),
                # Page range for multi-page chunks (e.g., "42-44")
                "page_start": s.get("page_start")
                if s.get("page_start") is not None
                else metadata.get("page_start"),
                "page_end": s.get("page_end")
                if s.get("page_end") is not None
                else metadata.get("page_end"),
                # Character positions for precise PDF highlighting (PyMuPDF)
                "char_start": s.get("char_start")
                if s.get("char_start") is not None
                else metadata.get("char_start"),
                "char_end": s.get("char_end")
                if s.get("char_end") is not None
                else metadata.get("char_end"),
                # Legal citation metadata (Westlaw/LexisNexis pattern)
                "title": metadata.get("title"),
                "citation": metadata.get("citation"),
                "case_number": metadata.get("case_number"),
                # Court information (for authority ranking)
                "court": metadata.get("court"),
                "court_type": metadata.get("court_type"),
                "court_level": metadata.get("court_level"),
                "jurisdiction": metadata.get("jurisdiction"),
                "bench_strength": metadata.get("bench_strength"),
                # Temporal metadata (for currency assessment)
                "year": metadata.get("year"),
                "decision_date": metadata.get("decision_date"),
                "disposition": metadata.get("disposition"),
                # Party information
                "petitioner": metadata.get("petitioner"),
                "respondent": metadata.get("respondent"),
                "judges": metadata.get("judges"),
                # Legal references (for KeyCite/Shepard's style features)
                "acts_referenced": metadata.get("acts_referenced"),
                "sections_cited": metadata.get("sections_cited"),
                "legal_domain": metadata.get("legal_domain"),
                # Authority indicators (like Westlaw's citation analysis)
                "is_landmark": metadata.get("is_landmark"),
                "cited_by_count": metadata.get("cited_by_count"),
                # File access - user docs use file_url, corpus uses pdf_url
                "file_url": s.get("file_url") or metadata.get("file_url"),
                "pdf_url": s.get("pdf_url") or metadata.get("pdf_url"),
                # Source tracking - check top-level first (pipeline.py sets it there), then metadata
                "is_shared": s.get("is_shared") or metadata.get("is_shared"),
                # Source type for frontend viewer routing (web, document, etc.)
                "source_type": s.get("source_type") or metadata.get("source_type"),
                # External source fields
                "external_url": s.get("external_url") or metadata.get("external_url"),
                "url": s.get("url") or metadata.get("url"),
                "domain": s.get("domain") or metadata.get("domain"),
                "court_citation_string": s.get("court_citation_string")
                or metadata.get("court_citation_string"),
                # Document type classification (case_law, act, constitution, etc.)
                "document_type": metadata.get("document_type"),
                # Act/legislation fields (populated for acts, None for case law)
                "act_id": metadata.get("act_id"),
                "act_name": metadata.get("act_name"),
                "chapter": metadata.get("chapter"),
                "jurisdiction_scope": metadata.get("jurisdiction"),
                "category": metadata.get("category"),
                "state": metadata.get("state"),
                "provision_type": metadata.get("provision_type"),
                "act_status": metadata.get("act_status"),
                "enactment_date": metadata.get("enactment_date"),
                # Deduplication metadata - indicates multiple chunks were merged
                "chunks_consolidated": s.get("chunks_consolidated"),
                "chunk_ids": s.get("chunk_ids"),  # All original chunk_ids if consolidated
                # P3: per-chunk excerpts preserved through dedup so the UI can
                # render expandable sub-citations (text + page + char range
                # for each merged chunk). Absent when no merge happened.
                "merged_chunks": s.get("merged_chunks"),
                # Full metadata passthrough for extensibility
                "metadata": metadata,
            }

            # Remove None values for cleaner JSON (reduces payload size)
            formatted_source = {k: v for k, v in formatted_source.items() if v is not None}
            formatted_sources.append(formatted_source)

        # Format alternative sources for contrastive citation explanations
        formatted_alternatives = []
        if alternative_sources:
            for s in alternative_sources:
                metadata = s.get("metadata") or {}
                alt_source = {
                    "document_id": s.get("document_id"),
                    "chunk_id": s.get("chunk_id"),
                    "filename": s.get("filename"),
                    "score": s.get("score"),
                    "text": s.get("text", ""),
                    "section_title": s.get("section_title") or metadata.get("section_title"),
                    "page_number": s.get("page_number") or metadata.get("page_number"),
                    "selected": False,  # Mark as not selected (alternative)
                }
                # Remove None values
                alt_source = {k: v for k, v in alt_source.items() if v is not None}
                formatted_alternatives.append(alt_source)

        # Aggregate confidence. Prefer the pipeline's HONEST multi-signal
        # confidence (grounding / consistency / NLI over the retrieved set) when
        # the caller passes it, so the sources panel shows the same number the
        # rest of the system uses. Only fall back to the max reranker score when
        # it is unavailable - and max-score alone OVERSTATES (one strong chunk
        # is not a well-grounded answer), which is why we no longer prefer it.
        if confidence is not None:
            confidence = round(float(confidence), 4)
        else:
            scores = [s.get("score") for s in formatted_sources if s.get("score") is not None]
            confidence = round(max(scores), 4) if scores else 0.0

        event_data = {
            "count": len(formatted_sources),
            "sources": formatted_sources,
            "has_sources": len(formatted_sources) > 0,
            "confidence": confidence,
        }

        # Include alternative sources if present
        if formatted_alternatives:
            event_data["alternative_sources"] = formatted_alternatives

        return SSEEvent(
            event_type=SSEEventType.SOURCES.value,
            data=event_data,
            event_id=event_id,
            sequence=sequence,
        ).to_sse()

    except Exception as e:
        logger.error(
            "sources_event_failed", error=str(e), sources_count=len(sources) if sources else 0
        )
        return error_event(f"Failed to create sources event: {e!s}", "SOURCES_EVENT_ERROR")


def chunk_event(content: str, context: SSEStreamContext | None = None) -> str:
    """
    Create a streaming chunk SSE event.

    Args:
        content: Text chunk from LLM streaming response
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted string with chunk content

    Example:
        >>> chunk_event("According to the ")
        'event: chunk\\ndata: {"type": "chunk", "content": "According to the "}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        # Handle empty content gracefully
        if content is None:
            content = ""

        return SSEEvent(
            event_type=SSEEventType.CHUNK.value,
            data={"content": content},
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("chunk_event_failed", error=str(e))
        return error_event(f"Failed to create chunk event: {e!s}", "CHUNK_EVENT_ERROR")


def verification_event(
    score: float, unsupported_claims: list[str], context: SSEStreamContext | None = None
) -> str:
    """
    Create a verification result SSE event.

    Args:
        score: Factual verification score (0.0 to 1.0)
        unsupported_claims: List of claims not supported by sources
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted string with verification results

    Example:
        >>> verification_event(0.92, [])
        'event: verification\\ndata: {"type": "verification", "factual_score": 0.92, ...}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        # Handle missing unsupported_claims
        if unsupported_claims is None:
            unsupported_claims = []

        # Ensure score is in valid range
        verified_score = max(0.0, min(1.0, score))

        return SSEEvent(
            event_type=SSEEventType.VERIFICATION.value,
            data={
                "factual_score": verified_score,
                "unsupported_claims": unsupported_claims,
                "verified": verified_score >= 0.8,
            },
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("verification_event_failed", error=str(e), score=score)
        return error_event(
            f"Failed to create verification event: {e!s}", "VERIFICATION_EVENT_ERROR"
        )


def response_replace_event(
    reason: str = "regeneration",
    context: SSEStreamContext | None = None,
) -> str:
    """
    Signal the frontend to clear accumulated content before regeneration.

    Emitted when the refinement actor decides to regenerate the response.
    The frontend should set accumulatedContent = "" so the regenerated
    chunks replace (not append to) the original response.
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.RESPONSE_REPLACE.value,
            data={"reason": reason},
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("response_replace_event_failed", error=str(e))
        return ""


def done_event(
    message_id: str,
    sources: list[dict],
    context: SSEStreamContext | None = None,
    query_topic: str | None = None,
    tool_cost_paise: int = 0,
    suggested_actions: list[dict[str, str]] | None = None,
    suggested_capabilities: list[dict[str, str]] | None = None,
    corrected_content: str | None = None,
    request_context: dict | None = None,
) -> str:
    """
    Create a completion SSE event.

    This event is always emitted when a stream completes, regardless of outcome
    (following the common response.done pattern). Clients should check the
    status/error fields to determine actual outcome.

    Args:
        message_id: ID of the generated message
        sources: List of source dictionaries used in response
        context: Optional stream context for event IDs
        query_topic: Classified legal task type for analytics

    Returns:
        SSE-formatted string indicating completion

    Example:
        >>> done_event("msg-123", [{"document_id": "doc1"}])
        'event: done\\ndata: {"type": "done", "message_id": "msg-123", ...}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        # Handle missing sources
        if sources is None:
            sources = []

        data: dict[str, Any] = {
            "message_id": message_id,
            "sources": sources,
            "status": "completed",
        }
        if query_topic:
            data["query_topic"] = query_topic
        if tool_cost_paise > 0:
            data["tool_cost_paise"] = tool_cost_paise
        if suggested_actions:
            data["suggested_actions"] = suggested_actions
        if suggested_capabilities:
            data["suggested_capabilities"] = suggested_capabilities
        # Citation correction: when the remapper fixed LLM citation numbers,
        # include the corrected full response so the frontend can replace
        # the streamed text with correctly attributed citations.
        if corrected_content:
            data["corrected_content"] = corrected_content
        # Bug 5: surface request audit context so the frontend can roundtrip
        # it back as chat_messages.metadata.request_context on POST /chats.
        if request_context:
            data["request_context"] = request_context

        return SSEEvent(
            event_type=SSEEventType.DONE.value,
            data=data,
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("done_event_failed", error=str(e), message_id=message_id)
        return error_event(f"Failed to create done event: {e!s}", "DONE_EVENT_ERROR")


def suggestion_prompt_event(
    suggestions: list[dict[str, str]],
    context: SSEStreamContext | None = None,
) -> str:
    """
    Emit an interactive suggestion prompt BEFORE LLM generation.

    Sent when a capability that cannot be auto-resolved (agent mode, deep mode)
    is OFF but would benefit the query. The frontend renders a yes/no modal.
    If the user clicks "Yes", the frontend re-sends the query with the toggle
    enabled. The current stream continues with best-effort regardless.

    Args:
        suggestions: List of dicts, each with:
            - id: capability key (e.g. "agentic_mode")
            - label: Display text (e.g. "Enable Agent Mode")
            - description: Why this would help
            - setting: Frontend Zustand store key to toggle
        context: SSE stream context for sequencing
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.SUGGESTION_PROMPT.value,
            data={"suggestions": suggestions},
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("suggestion_prompt_event_failed", error=str(e))
        return ""  # Non-critical - stream continues without prompt


def error_event(
    message: str,
    code: str = "UNKNOWN",
    recoverable: bool = False,
    context: SSEStreamContext | None = None,
) -> str:
    """
    Create an error SSE event.

    Following the resilient streaming pattern from FastAPI AI SDK,
    errors include a 'recoverable' flag to indicate if the stream
    can continue after this error.

    Args:
        message: Human-readable error message
        code: Error code for categorization
        recoverable: Whether the stream can continue after this error
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted string with error information

    Example:
        >>> error_event("Embedding service unavailable", "EMB_2005", recoverable=True)
        'event: error\\ndata: {"type": "error", "message": "...", "code": "EMB_2005", "recoverable": true}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.ERROR.value,
            data={
                "message": message,
                "code": code,
                "recoverable": recoverable,
                # Surface the stream id so a client-side error (and any record the
                # client keeps for a dropped stream) can be correlated back to
                # server logs and error trackers, which key on this id.
                "stream_id": context.stream_id if context else None,
            },
            event_id=event_id,
            sequence=sequence,
            retry_ms=DEFAULT_RETRY_MS,  # Include retry for errors
        ).to_sse()
    except Exception as e:
        # Last resort fallback - manual construction
        logger.error("error_event_failed", error=str(e), original_message=message)
        return 'event: error\ndata: {"type": "error", "message": "Critical error in event system", "code": "SYSTEM_ERROR", "recoverable": false}\n\n'


def heartbeat_event(context: SSEStreamContext | None = None) -> str:
    """
    Create a heartbeat/keep-alive SSE event.

    Per Shopify Engineering best practices, periodic heartbeats prevent
    network timeouts on idle connections. This can be sent as a comment
    (prefixed with ':') or as a named event.

    Args:
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted heartbeat string

    Example:
        >>> heartbeat_event()
        'event: heartbeat\\ndata: {"type": "heartbeat", "timestamp": 1704067200}\\n\\n'
    """
    try:
        event_id = context.next_event_id() if context else None
        sequence = context.get_sequence() if context else None

        return SSEEvent(
            event_type=SSEEventType.HEARTBEAT.value,
            data={"timestamp": int(time.time())},
            event_id=event_id,
            sequence=sequence,
        ).to_sse()
    except Exception as e:
        logger.error("heartbeat_event_failed", error=str(e))
        # Fallback to SSE comment format (simpler, always works)
        return f": heartbeat {int(time.time())}\n\n"


def retry_directive(retry_ms: int = DEFAULT_RETRY_MS) -> str:
    """
    Create an SSE retry directive.

    Per SSE specification, this tells the client how long to wait
    before attempting to reconnect after a connection drop.

    Args:
        retry_ms: Retry interval in milliseconds

    Returns:
        SSE retry directive string

    Example:
        >>> retry_directive(5000)
        'retry: 5000\\n\\n'
    """
    return f"retry: {retry_ms}\n\n"


def stream_init_event(stream_id: str, context: SSEStreamContext | None = None) -> str:
    """
    Create a stream initialization event.

    Sent at the start of a stream to establish the stream ID and
    set initial retry timing. This allows clients to track streams
    and reconnect properly.

    Args:
        stream_id: Unique identifier for this stream
        context: Optional stream context for event IDs

    Returns:
        SSE-formatted stream init event
    """
    try:
        event_id = context.next_event_id() if context else None

        return SSEEvent(
            event_type=SSEEventType.STREAM_INIT.value,
            data={
                "stream_id": stream_id,
                "protocol_version": "1.0",
                "features": ["thinking", "sources", "verification", "reconnection"],
            },
            event_id=event_id,
            sequence=0,
            retry_ms=DEFAULT_RETRY_MS,
        ).to_sse()
    except Exception as e:
        logger.error("stream_init_event_failed", error=str(e), stream_id=stream_id)
        return error_event(f"Failed to create stream init event: {e!s}", "STREAM_INIT_ERROR")
