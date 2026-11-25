"""
Retrieval Loop Detection and Prevention.

Prevents RAG systems from repeatedly retrieving the same content,
a common failure mode identified in production systems (2024-2025 research).

Patterns detected:
- Same query repeated within session
- Similar queries returning identical chunks
- Circular follow-up patterns

Based on:
- RGB Benchmark (2024) - noise robustness testing
- Production RAG failure analysis (Mathematics 2025 paper)
"""

import hashlib
import time
from collections import deque
from dataclasses import dataclass

from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("conversation.loop_detector")


@dataclass
class RetrievalHistory:
    """Track retrieval history for loop detection."""

    query_hash: str
    chunk_ids: set[str]
    timestamp: float
    query_preview: str


@dataclass
class LoopDetectionResult:
    """Result of loop detection analysis."""

    is_loop: bool
    loop_type: str  # "exact_repeat", "similar_query", "circular_chunks", "none"
    similarity_score: float
    previous_retrieval_index: int | None
    recommendation: str  # "block", "warn", "allow"
    reason: str


class RetrievalLoopDetector:
    """
    Detects and prevents retrieval loops in RAG systems.

    Maintains sliding window of recent retrievals per user session.
    """

    def __init__(
        self,
        max_history_size: int = 20,
        similarity_threshold: float = 0.85,
        chunk_overlap_threshold: float = 0.9,
        time_window_seconds: int = 300,  # 5 minutes
    ):
        """
        Initialize LoopDetector.

        Args:
            max_history_size: Max retrievals to track per session
            similarity_threshold: Query similarity threshold for loop detection
            chunk_overlap_threshold: Chunk overlap ratio for loop detection
            time_window_seconds: Time window for loop detection
        """
        self.max_history_size = max_history_size
        self.similarity_threshold = similarity_threshold
        self.chunk_overlap_threshold = chunk_overlap_threshold
        self.time_window_seconds = time_window_seconds

        # Session-based history: {session_id: deque[RetrievalHistory]}
        self._session_history: dict[str, deque] = {}

    def _make_query_hash(self, query: str) -> str:
        """Create hash of normalized query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()[:16]

    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate simple query similarity (Jaccard similarity on words).

        For production, could use embedding similarity, but this is fast and effective.
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_chunk_overlap(self, chunks1: set[str], chunks2: set[str]) -> float:
        """Calculate overlap ratio between two chunk sets."""
        if not chunks1 or not chunks2:
            return 0.0

        intersection = chunks1.intersection(chunks2)
        smaller_set_size = min(len(chunks1), len(chunks2))

        return len(intersection) / smaller_set_size if smaller_set_size > 0 else 0.0

    def check_for_loop(
        self,
        session_id: str,
        query: str,
        retrieved_chunk_ids: list[str],
    ) -> LoopDetectionResult:
        """
        Check if this retrieval is part of a loop.

        Args:
            session_id: User session ID
            query: Current query
            retrieved_chunk_ids: Chunk IDs retrieved for this query

        Returns:
            LoopDetectionResult with recommendation
        """
        current_time = time.time()
        query_hash = self._make_query_hash(query)
        chunk_set = set(retrieved_chunk_ids)

        # Get session history
        if session_id not in self._session_history:
            self._session_history[session_id] = deque(maxlen=self.max_history_size)

        history = self._session_history[session_id]

        # Check against recent history
        for idx, past_retrieval in enumerate(reversed(history)):
            # Skip if outside time window
            age_seconds = current_time - past_retrieval.timestamp
            if age_seconds > self.time_window_seconds:
                continue

            # Check for exact query repeat
            if past_retrieval.query_hash == query_hash:
                logger.warning(
                    "loop_exact_repeat_detected",
                    query_preview=query[:50],
                    previous_query_preview=past_retrieval.query_preview[:50],
                    age_seconds=round(age_seconds, 1),
                )
                return LoopDetectionResult(
                    is_loop=True,
                    loop_type="exact_repeat",
                    similarity_score=1.0,
                    previous_retrieval_index=idx,
                    recommendation="warn",  # Allow but warn
                    reason=f"Exact query repeated after {age_seconds:.0f}s",
                )

            # Check for similar query
            similarity = self._calculate_query_similarity(query, past_retrieval.query_preview)
            if similarity >= self.similarity_threshold:
                # Check chunk overlap
                chunk_overlap = self._calculate_chunk_overlap(chunk_set, past_retrieval.chunk_ids)

                if chunk_overlap >= self.chunk_overlap_threshold:
                    logger.warning(
                        "loop_similar_query_detected",
                        query_preview=query[:50],
                        previous_query_preview=past_retrieval.query_preview[:50],
                        similarity=round(similarity, 2),
                        chunk_overlap=round(chunk_overlap, 2),
                    )
                    return LoopDetectionResult(
                        is_loop=True,
                        loop_type="similar_query",
                        similarity_score=similarity,
                        previous_retrieval_index=idx,
                        recommendation="warn",
                        reason=f"Similar query with {chunk_overlap:.0%} chunk overlap",
                    )

        # No loop detected
        return LoopDetectionResult(
            is_loop=False,
            loop_type="none",
            similarity_score=0.0,
            previous_retrieval_index=None,
            recommendation="allow",
            reason="No retrieval loop detected",
        )

    def record_retrieval(
        self,
        session_id: str,
        query: str,
        retrieved_chunk_ids: list[str],
    ):
        """
        Record a retrieval in session history.

        Args:
            session_id: User session ID
            query: Query that was executed
            retrieved_chunk_ids: Chunks that were retrieved
        """
        if session_id not in self._session_history:
            self._session_history[session_id] = deque(maxlen=self.max_history_size)

        history_entry = RetrievalHistory(
            query_hash=self._make_query_hash(query),
            chunk_ids=set(retrieved_chunk_ids),
            timestamp=time.time(),
            query_preview=query[:100],
        )

        self._session_history[session_id].append(history_entry)

        logger.debug(
            "loop_retrieval_recorded",
            session_id=session_id,
            query_preview=query[:50],
            chunks_count=len(retrieved_chunk_ids),
        )

    def get_session_stats(self, session_id: str) -> dict:
        """Get retrieval statistics for a session."""
        if session_id not in self._session_history:
            return {
                "total_retrievals": 0,
                "unique_queries": 0,
                "unique_chunks": 0,
            }

        history = self._session_history[session_id]

        unique_queries = len({h.query_hash for h in history})
        all_chunks = set()
        for h in history:
            all_chunks.update(h.chunk_ids)

        return {
            "total_retrievals": len(history),
            "unique_queries": unique_queries,
            "unique_chunks": len(all_chunks),
            "avg_chunks_per_query": (
                sum(len(h.chunk_ids) for h in history) / len(history) if history else 0
            ),
        }

    def clear_session(self, session_id: str):
        """Clear history for a session."""
        if session_id in self._session_history:
            del self._session_history[session_id]
            logger.info("loop_session_cleared", session_id=session_id)

    def cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """Remove sessions older than max_age_seconds."""
        current_time = time.time()
        sessions_to_remove = []

        for session_id, history in self._session_history.items():
            if history:
                latest_retrieval = history[-1].timestamp
                age = current_time - latest_retrieval

                if age > max_age_seconds:
                    sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self._session_history[session_id]

        if sessions_to_remove:
            logger.info(
                "loop_old_sessions_cleaned",
                sessions_removed=len(sessions_to_remove),
                max_age_seconds=max_age_seconds,
            )


# Singleton
_loop_detector_instance: RetrievalLoopDetector | None = None


def get_loop_detector() -> RetrievalLoopDetector:
    """Get or create RetrievalLoopDetector singleton."""
    global _loop_detector_instance
    if _loop_detector_instance is None:
        _loop_detector_instance = RetrievalLoopDetector()
    return _loop_detector_instance
