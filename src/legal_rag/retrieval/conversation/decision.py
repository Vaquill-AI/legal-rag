"""
Retrieval Decision System.

Determines whether new retrieval is needed or if previous context can be reused.
Based on SELF-multi-RAG research (arXiv:2409.15515).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("conversation.decision")


class RetrievalAction(Enum):
    """Possible retrieval actions."""

    RETRIEVE = "retrieve"  # Perform new retrieval
    REUSE_CONTEXT = "reuse_context"  # Reuse previous context
    SKIP = "skip"  # No retrieval needed (general knowledge)


@dataclass
class RetrievalDecision:
    """Decision about retrieval strategy."""

    action: RetrievalAction
    reason: str
    confidence: float
    use_previous_sources: bool = False
    filter_to_documents: list[str] | None = None  # Limit retrieval to these docs

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "use_previous_sources": self.use_previous_sources,
            "filter_to_documents": self.filter_to_documents,
        }


class RetrievalDecisionMaker:
    """
    Decides whether to retrieve new documents or reuse context.

    Decision matrix:
    - New topic -> Full retrieval
    - Follow-up on same topic -> Retrieval with topic filter
    - Clarification request -> Reuse previous context
    - Elaboration request -> Expanded retrieval
    - Reference to previous -> Use cached context
    """

    # Patterns indicating no retrieval needed
    NO_RETRIEVAL_PATTERNS: ClassVar[set[str]] = {
        # Clarifications
        "can you clarify",
        "what do you mean",
        "explain that",
        "in simpler terms",
        "rephrase",
        "in other words",
        # Meta questions
        "why did you say",
        "how did you know",
        "where did you get",
        # Formatting requests
        "format as",
        "make it",
        "can you list",
        "bullet points",
    }

    # Patterns indicating same-topic follow-up
    SAME_TOPIC_PATTERNS: ClassVar[set[str]] = {
        "tell me more",
        "what else",
        "anything else",
        "continue",
        "go on",
        "more about that",
        "what about",
        "how about",
    }

    # Patterns indicating topic shift
    TOPIC_SHIFT_PATTERNS: ClassVar[set[str]] = {
        "now",
        "let's talk about",
        "switch to",
        "different topic",
        "moving on",
        "another question",
        "forget that",
        "never mind",
        "new question",
    }

    def __init__(self):
        """Initialize RetrievalDecisionMaker."""

    def decide(
        self,
        query: str,
        chat_history: list[dict],
        previous_sources: list[str] | None = None,
        previous_confidence: float = 0.0,
    ) -> RetrievalDecision:
        """
        Decide retrieval strategy based on query and context.

        Args:
            query: Current query
            chat_history: Conversation history
            previous_sources: Document IDs from previous retrieval
            previous_confidence: Confidence from previous response

        Returns:
            RetrievalDecision with recommended action
        """
        query_lower = query.lower().strip()

        # No history = always retrieve
        if not chat_history:
            return RetrievalDecision(
                action=RetrievalAction.RETRIEVE,
                reason="First query in conversation",
                confidence=1.0,
            )

        # Check for no-retrieval patterns (clarification/formatting)
        for pattern in self.NO_RETRIEVAL_PATTERNS:
            if pattern in query_lower:
                return RetrievalDecision(
                    action=RetrievalAction.REUSE_CONTEXT,
                    reason=f"Clarification/formatting request: '{pattern}'",
                    confidence=0.9,
                    use_previous_sources=True,
                )

        # Check for topic shift
        for pattern in self.TOPIC_SHIFT_PATTERNS:
            if pattern in query_lower:
                return RetrievalDecision(
                    action=RetrievalAction.RETRIEVE,
                    reason=f"Topic shift detected: '{pattern}'",
                    confidence=0.85,
                )

        # Check for same-topic follow-up
        for pattern in self.SAME_TOPIC_PATTERNS:
            if pattern in query_lower:
                # If we have previous sources, limit retrieval to those docs
                if previous_sources and previous_confidence > 0.6:
                    return RetrievalDecision(
                        action=RetrievalAction.RETRIEVE,
                        reason=f"Same-topic follow-up: '{pattern}'",
                        confidence=0.8,
                        filter_to_documents=previous_sources,
                    )
                else:
                    return RetrievalDecision(
                        action=RetrievalAction.RETRIEVE,
                        reason="Follow-up but no previous sources to filter",
                        confidence=0.75,
                    )

        # Check query length - very short queries likely need previous context
        if len(query.split()) <= 3 and previous_sources and previous_confidence > 0.7:
            return RetrievalDecision(
                action=RetrievalAction.RETRIEVE,
                reason="Short query, filtering to previous sources",
                confidence=0.7,
                filter_to_documents=previous_sources,
            )

        # Default: perform full retrieval
        return RetrievalDecision(
            action=RetrievalAction.RETRIEVE,
            reason="Standard query requiring retrieval",
            confidence=0.8,
        )

    def should_skip_rag(self, query: str) -> bool:
        """
        Check if query is likely general knowledge (no RAG needed).

        Examples:
        - "What time is it?"
        - "What's 2+2?"
        - "Who is the president?"
        """
        general_patterns = [
            "what time",
            "what's the time",
            "current time",
            "what day",
            "what's the date",
            "today's date",
            "who is the president",
            "who is the prime minister",
            "what is the capital of",
            "calculate",
            "compute",
            "what's",
            "what is",
        ]

        query_lower = query.lower()

        # Very short math-like queries
        if (
            len(query.split()) <= 4
            and any(c.isdigit() for c in query)
            and ("+" in query or "-" in query or "*" in query or "/" in query)
        ):
            return True

        # General knowledge patterns
        for pattern in general_patterns:
            if pattern in query_lower:
                # But not if it seems document-specific
                document_indicators = ["document", "file", "uploaded", "my", "our"]
                if not any(ind in query_lower for ind in document_indicators):
                    return True

        return False


# Singleton instance
_decision_maker: RetrievalDecisionMaker | None = None


def get_retrieval_decision_maker() -> RetrievalDecisionMaker:
    """Get or create RetrievalDecisionMaker singleton."""
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = RetrievalDecisionMaker()
    return _decision_maker
