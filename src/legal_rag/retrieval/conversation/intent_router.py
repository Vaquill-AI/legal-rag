"""Intent router: a pre-classification layer for user intent detection.

Separates the user's instruction from pasted content and classifies intent with a lightweight
call to an injected language model. This keeps retrieval from firing on text-processing tasks
where the user pastes text and asks for editing, formatting or summarization.

Architecture:
    1. Rule-based instruction extraction (structural, no model call). Detects instruction versus
       pasted content by blank-line boundaries, structural markers and length ratios. It is
       position-agnostic: the instruction can come before or after the paste.
    2. Model-based intent classification. Only fires when a paste is detected (60+ words).
       Classifies the instruction into text_processing, research, hybrid or conversational.
       Fail-open: defaults to research on any error.

Usage:
    router = IntentRouter(llm)
    result = await router.route("Proofread this brief:\\n\\n[2000-word legal text]")
    # result.user_intent == UserIntent.TEXT_PROCESSING
    # result.should_retrieve == False
"""
# ruff: noqa: BLE001

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from legal_rag.providers.protocols import LanguageModel
from legal_rag.retrieval.routing.structured_log import get_logger
from legal_rag.types import Message

logger = get_logger("conversation.intent_router")


class UserIntent(Enum):
    """High-level user intent classification."""

    RESEARCH = "research"  # User needs new information from corpus (trigger RAG)
    TEXT_PROCESSING = "text_processing"  # User wants LLM to transform their pasted text (skip RAG)
    HYBRID = "hybrid"  # Needs both: user provides text AND needs corpus search
    CONVERSATIONAL = "conversational"  # Greetings, meta (deferred to existing analyzer)


@dataclass
class IntentRoutingResult:
    """Result of intent-based routing analysis."""

    user_intent: UserIntent
    instruction_text: str  # The extracted instruction portion
    content_text: str | None  # The pasted/provided content (None if no paste detected)
    should_retrieve: bool  # Final routing decision
    confidence: float  # 0.0-1.0 how confident we are in the classification
    routing_reason: str  # Human-readable reason for the decision
    instruction_position: str  # "before", "after", "only"
    content_word_count: int  # Size of pasted content (0 if none)
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Step A: Rule-based instruction extraction
# ---------------------------------------------------------------------------

# Structural boundary patterns that separate instruction from content
_COLON_TERMINATED_LINE = re.compile(r"^.{5,80}:\s*$", re.MULTILINE)


def _word_count(text: str) -> int:
    """Fast word count."""
    return len(text.split())


def _detect_blank_line_boundary(message: str) -> tuple[str, str | None, str]:
    """
    Split on double-newline boundary when one side is dramatically shorter.

    Handles both instruction-before and instruction-after patterns:
    - "Rewrite this in plain English\\n\\n[long pasted text]"
    - "[long pasted text]\\n\\nSummarize this for me"
    """
    parts = re.split(r"\n\s*\n", message, maxsplit=1)
    if len(parts) != 2:
        return message, None, "only"

    first_words = _word_count(parts[0])
    second_words = _word_count(parts[1])

    # First part is instruction, second is content
    if first_words < 40 and second_words > 50 and second_words / max(first_words, 1) > 2.5:
        return parts[0].strip(), parts[1].strip(), "before"

    # Second part is instruction, first is content
    if second_words < 40 and first_words > 50 and first_words / max(second_words, 1) > 2.5:
        return parts[1].strip(), parts[0].strip(), "after"

    return message, None, "only"


def _detect_structural_boundary(message: str) -> tuple[str, str | None, str]:
    """
    Detect structural markers: colon-terminated short lines, ---, code fences.
    """
    lines = message.strip().split("\n")
    if len(lines) < 2:
        return message, None, "only"

    # Check first line for colon-terminated instruction
    first_line = lines[0].strip()
    if first_line.endswith(":") and _word_count(first_line) < 30:
        rest = "\n".join(lines[1:]).strip()
        if _word_count(rest) > 50:
            return first_line, rest, "before"

    # Check for --- separator
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^-{3,}$", stripped) or re.match(r"^={3,}$", stripped):
            before = "\n".join(lines[:i]).strip()
            after = "\n".join(lines[i + 1 :]).strip()
            before_words = _word_count(before)
            after_words = _word_count(after)

            if before_words < 40 and after_words > 50:
                return before, after, "before"
            if after_words < 40 and before_words > 50:
                return after, before, "after"

    return message, None, "only"


def _detect_by_length_ratio(message: str) -> tuple[str, str | None, str]:
    """
    Split message into instruction + content using length ratio of first/last lines
    vs the rest of the message.
    """
    lines = message.strip().split("\n")
    if len(lines) < 3:
        return message, None, "only"

    total_words = _word_count(message)
    if total_words < 80:
        return message, None, "only"

    # Try instruction-first: first 1-2 lines are short instruction
    first_line_words = _word_count(lines[0])
    rest = "\n".join(lines[1:])
    rest_words = _word_count(rest)

    if first_line_words < 30 and rest_words > 50 and rest_words / max(first_line_words, 1) > 3:
        return lines[0].strip(), rest.strip(), "before"

    # Try instruction-last: last 1-2 lines are short instruction
    last_line_words = _word_count(lines[-1])
    preceding = "\n".join(lines[:-1])
    preceding_words = _word_count(preceding)

    if (
        last_line_words < 30
        and preceding_words > 50
        and preceding_words / max(last_line_words, 1) > 3
    ):
        return lines[-1].strip(), preceding.strip(), "after"

    return message, None, "only"


def extract_instruction_and_content(message: str) -> tuple[str, str | None, str]:
    """
    Extract instruction and pasted content from a user message.

    Tries three strategies in order:
    1. Blank-line boundary (most common: instruction\\n\\ncontent)
    2. Structural boundary (colon-terminated line, ---, ===)
    3. Length-ratio heuristic (first/last lines vs rest)

    Returns:
        (instruction_text, content_text_or_none, position)
        position: "before" | "after" | "only"
    """
    # 1. Blank-line boundary
    instruction, content, position = _detect_blank_line_boundary(message)
    if content is not None:
        return instruction, content, position

    # 2. Structural boundary
    instruction, content, position = _detect_structural_boundary(message)
    if content is not None:
        return instruction, content, position

    # 3. Length-ratio heuristic
    instruction, content, position = _detect_by_length_ratio(message)
    if content is not None:
        return instruction, content, position

    # No paste detected
    return message, None, "only"


# ---------------------------------------------------------------------------
# Step B: LLM-based intent classification
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """\
You classify user instructions into intent categories.

The user has pasted a long block of text along with a short instruction. You are given \
ONLY the instruction portion (the pasted text is not shown to you). Based on the instruction, \
classify what the user wants.

## Intent Categories

### "text_processing"
The user wants the AI to transform, edit, format, proofread, summarize, translate, rewrite, \
restructure, or otherwise operate ON their pasted text. The AI should work directly on the \
text the user provided. No external search or document retrieval is needed.

Examples:
- "Rewrite this in plain English"
- "Proofread this brief"
- "Add page numbers to these references"
- "Summarize this contract"
- "Make bullet points from this"
- "Fix the grammar and punctuation"
- "Convert these citations to OSCOLA format"
- "Translate this to Hindi"
- "Make this more readable"
- "Draft a response to this email"
- "Extract the key terms from this agreement"
- "Kindly correct the grammatical errors"
- "Format this as a table"
- "Shorten this to 500 words"
- "Remove the legal jargon"
- "Number the paragraphs"
- "Identify the parties and obligations in this"
- "Create an executive summary"

### "research"
The user wants to search for NEW information that is NOT in their pasted text. They need \
the system to look up cases, statutes, legal principles, or analysis from a document corpus \
or knowledge base. The pasted text provides context but the user needs external information.

Examples:
- "What does the law say about this?"
- "Is this clause enforceable?"
- "What are the relevant cases?"
- "Explain the legal implications of this"
- "What is the penalty for this offense?"
- "Which statute covers this?"
- "Analyze this under contract law"

### "hybrid"
The user provides text AND explicitly needs external information ADDED to it. They want \
the AI to both process their text AND search for external sources to enrich it.

Examples:
- "Add case citations to support these arguments"
- "Verify if these case references are correct"
- "Find supporting case law for each point in my draft"
- "Check these citations against the original judgments"
- "Add relevant statutory references to this memo"
- "Find similar clauses from other contracts"
- "Compare this with standard market terms"

### "conversational"
Greetings, thanks, acknowledgments, or meta-questions about a previous response. \
Not a substantive task.

Examples:
- "Thanks for that"
- "Can you explain your last answer differently?"
- "Hello"
- "Got it, what else can you help with?"

## Decision Rules
- If the instruction asks the AI to DO something TO the text (edit, format, rewrite, summarize, etc.) -> "text_processing"
- If the instruction asks for information NOT in the text (look up, find, what does the law say) -> "research"
- If the instruction asks to ADD external information TO the text (add citations, verify references) -> "hybrid"
- When in doubt between "text_processing" and "research", consider: does the user need anything beyond what they already pasted? If no -> "text_processing". If yes -> "research" or "hybrid".
- Set confidence to 0.9+ when the intent is very clear, 0.7-0.9 when fairly clear, below 0.7 when ambiguous."""


class IntentClassification(BaseModel):
    """Structured output of the intent classification call."""

    intent: str = Field(description="One of: text_processing, research, hybrid, conversational")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)


async def _classify_intent_with_llm(
    instruction: str, llm: LanguageModel, max_tokens: int = 512
) -> tuple[str, float]:
    """Classify instruction intent with structured output from the injected model.

    Returns ``(intent_string, confidence)``. Fail-open defaults to ``("research", 0.3)``.
    """
    try:
        result = await llm.generate_structured(
            [Message("system", _CLASSIFIER_SYSTEM_PROMPT), Message("user", instruction)],
            IntentClassification,
            max_tokens=max_tokens,
        )
        valid_intents = {"text_processing", "research", "hybrid", "conversational"}
        if result.intent not in valid_intents:
            logger.warning("intent_classifier_invalid_intent", intent=result.intent)
            return "research", 0.3
        return result.intent, result.confidence
    except Exception as e:
        logger.warning("intent_classifier_error", error=str(e), error_type=type(e).__name__)
        return "research", 0.3


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------

# Minimum content word count to consider as "pasted text"
MIN_PASTE_WORDS = 60

_INTENT_TO_ENUM = {
    "text_processing": UserIntent.TEXT_PROCESSING,
    "research": UserIntent.RESEARCH,
    "hybrid": UserIntent.HYBRID,
    "conversational": UserIntent.CONVERSATIONAL,
}


class IntentRouter:
    """
    Pre-classifies user intent before query analysis.

    Two-stage approach:
    1. Rule-based instruction extraction (structural, <0.5ms)
    2. LLM-based intent classification (only when a paste is detected)

    Performance: <1ms for short messages (no LLM call), ~100-200ms for long messages
    with detected pasted content.
    """

    def __init__(self, llm: LanguageModel | None = None):
        """``llm`` is optional: without it the router stays rule-based and fails open."""
        self.llm = llm

    async def route(self, message: str) -> IntentRoutingResult:
        """
        Determine user intent and extract instruction/content boundaries.

        Decision tree:
        1. If message < MIN_PASTE_WORDS -> skip router, defer to existing analyzer
        2. Extract instruction + content
        3. If no content detected -> defer to existing analyzer
        4. Classify instruction with LLM
        5. Route based on LLM classification
        """
        start_time = time.time()

        total_words = _word_count(message)

        # Normal queries (under paste threshold): skip router, existing analyzer handles these
        if total_words < MIN_PASTE_WORDS:
            return IntentRoutingResult(
                user_intent=UserIntent.RESEARCH,
                instruction_text=message,
                content_text=None,
                should_retrieve=True,
                confidence=0.3,
                routing_reason="normal_query_below_paste_threshold",
                instruction_position="only",
                content_word_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Step A: Extract instruction and content
        instruction, content, position = extract_instruction_and_content(message)
        content_words = _word_count(content) if content else 0

        # No significant paste detected
        if content is None or content_words < MIN_PASTE_WORDS:
            return IntentRoutingResult(
                user_intent=UserIntent.RESEARCH,
                instruction_text=message,
                content_text=None,
                should_retrieve=True,
                confidence=0.3,
                routing_reason="no_paste_detected",
                instruction_position="only",
                content_word_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Step B: Classify instruction with LLM
        intent_str, llm_confidence = await _classify_intent_with_llm(instruction, self.llm)
        user_intent = _INTENT_TO_ENUM.get(intent_str, UserIntent.RESEARCH)

        # Routing decision
        if user_intent == UserIntent.TEXT_PROCESSING and llm_confidence >= 0.7:
            should_retrieve = False
            routing_reason = "text_processing_intent_with_pasted_content"
        elif user_intent == UserIntent.HYBRID and llm_confidence >= 0.7:
            should_retrieve = True
            routing_reason = "hybrid_intent_needs_retrieval_and_processing"
        elif user_intent == UserIntent.CONVERSATIONAL and llm_confidence >= 0.7:
            should_retrieve = False
            routing_reason = "conversational_intent"
        else:
            # RESEARCH or low confidence - default to retrieval
            should_retrieve = True
            routing_reason = (
                "research_intent"
                if user_intent == UserIntent.RESEARCH
                else f"low_confidence_{llm_confidence:.2f}_defaulting_to_research"
            )

        processing_time = (time.time() - start_time) * 1000

        logger.debug(
            "intent_router_classified",
            user_intent=user_intent.value,
            should_retrieve=should_retrieve,
            confidence=llm_confidence,
            reason=routing_reason,
            instruction_preview=instruction[:80],
            content_word_count=content_words,
            instruction_position=position,
            processing_time_ms=round(processing_time, 2),
        )

        return IntentRoutingResult(
            user_intent=user_intent,
            instruction_text=instruction,
            content_text=content,
            should_retrieve=should_retrieve,
            confidence=llm_confidence,
            routing_reason=routing_reason,
            instruction_position=position,
            content_word_count=content_words,
            processing_time_ms=processing_time,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
