"""
Anaphora Resolution for Conversational RAG.

Resolves pronouns and demonstratives (it, this, that, they, etc.)
to their explicit referents from conversation history.

This is a lightweight component that works alongside QueryCondenser.
"""

import re
from typing import ClassVar

from legal_rag.retrieval.conversation.text_tokens import is_spaceless_script
from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("conversation.anaphora")


class AnaphoraResolver:
    """
    Resolves pronouns to explicit entities from conversation context.

    Uses rule-based detection + entity extraction.
    Falls back to LLM resolution for complex cases (via QueryCondenser).
    """

    # Pronouns that typically need resolution
    PRONOUNS: ClassVar[set[str]] = {
        "it",
        "its",
        "it's",
        "this",
        "that",
        "these",
        "those",
        "they",
        "them",
        "their",
        "theirs",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
    }

    # Demonstratives and references
    REFERENCES: ClassVar[set[str]] = {
        "the above",
        "the following",
        "the previous",
        "the former",
        "the latter",
        "the same",
        "such",
    }

    def __init__(self):
        """Initialize AnaphoraResolver."""

    def has_anaphora(self, query: str) -> bool:
        """
        Check if query contains pronouns or references that need resolution.

        Both signals below are ASCII-only: `PRONOUNS` is matched against
        whitespace-split tokens and `REFERENCES` against English phrases. Neither
        can fire on a script written without spaces, so a Chinese follow-up like
        "expand the instances in that answer above" scored False and its
        references were never resolved. Absence of an English pronoun is not
        evidence of no anaphora, so for spaceless scripts we assume resolution is
        needed and let the LLM condenser decide -- the safe direction, since a
        false positive costs one condense pass and a false negative silently
        drops the user's referent.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check for pronouns
        if query_words & self.PRONOUNS:
            return True

        # Check for references
        if any(ref in query_lower for ref in self.REFERENCES):
            return True

        return is_spaceless_script(query)

    def extract_entities(self, text: str) -> list[str]:
        """
        Extract potential entities from text.

        Looks for:
        - Case numbers (1239/2023)
        - Section references (Section 420)
        - Proper nouns (capitalized words)
        - Quoted terms
        """
        entities = []

        # Case numbers
        case_patterns = [
            r"\d+/\d{4}",  # 1239/2023 (year-numbered docket)
            r"\d{4}\s+\w+\s+\d+",  # 2024 WL 123
            r"Case\s+No\.?\s*[\w\d/-]+",  # Case No. XYZ
        ]
        for pattern in case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)

        # Section/Article references
        legal_patterns = [
            r"Section\s+\d+[\w]*(?:\s+\w+)?",  # Section 1983
            r"Article\s+\d+[\w]*",  # Article 21
            r"Rule\s+\d+[\w]*",  # Rule 37
            r"Order\s+\w+\s+Rule\s+\d+",  # Order XXI Rule 37
        ]
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)

        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)
        quoted_single = re.findall(r"'([^']+)'", text)
        entities.extend(quoted_single)

        # Proper nouns (consecutive capitalized words)
        proper_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        for noun in proper_nouns:
            # Filter out common words
            if noun.lower() not in {
                "the",
                "this",
                "that",
                "what",
                "how",
                "why",
                "when",
                "where",
                "user",
                "assistant",
                "i",
            }:
                entities.append(noun)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity and entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def get_most_recent_entity(
        self, chat_history: list[dict], entity_type: str | None = None
    ) -> str | None:
        """
        Get the most recently mentioned entity from chat history.

        Args:
            chat_history: Conversation history
            entity_type: Optional type filter ("case", "section", "person", etc.)

        Returns:
            Most recent matching entity or None
        """
        if not chat_history:
            return None

        # Go through history in reverse
        for turn in reversed(chat_history):
            content = ""
            if "content" in turn:
                content = turn["content"]
            elif "user" in turn:
                content = turn["user"] + " " + turn.get("assistant", "")

            entities = self.extract_entities(content)

            if not entities:
                continue

            # Filter by type if specified
            if entity_type:
                if entity_type == "case":
                    filtered = [e for e in entities if re.search(r"\d+/\d{4}", e)]
                elif entity_type == "section":
                    filtered = [
                        e for e in entities if re.search(r"Section|Article|Rule", e, re.IGNORECASE)
                    ]
                else:
                    filtered = entities

                if filtered:
                    return filtered[0]
            else:
                return entities[0]

        return None

    def simple_resolve(self, query: str, chat_history: list[dict]) -> str:
        """
        Simple pronoun resolution using most recent entity.

        For complex cases, use QueryCondenser which uses LLM.

        Args:
            query: Query with potential pronouns
            chat_history: Conversation history

        Returns:
            Query with pronouns replaced (best effort)
        """
        if not self.has_anaphora(query) or not chat_history:
            return query

        # Get most recent entity
        recent_entity = self.get_most_recent_entity(chat_history)

        if not recent_entity:
            return query

        # Simple replacement for common patterns
        resolved = query

        # Replace "it" with entity (careful with "it's" = "it is")
        resolved = re.sub(
            r"\bit\b(?!\s+is|\s+was|\s+has)",
            recent_entity,
            resolved,
            flags=re.IGNORECASE,
        )

        # Replace "this" and "that" when used as pronouns
        resolved = re.sub(
            r"\b(this|that)\b(?!\s+\w+)",  # Not followed by noun
            recent_entity,
            resolved,
            flags=re.IGNORECASE,
        )

        logger.debug(
            "anaphora_resolved",
            original=query,
            resolved=resolved,
            entity=recent_entity,
        )

        return resolved

    def detect_ambiguity(self, query: str, chat_history: list[dict]) -> dict | None:
        """
        Detect if pronoun resolution would be ambiguous.

        Returns disambiguation info if multiple valid entities exist.
        """
        if not self.has_anaphora(query) or not chat_history:
            return None

        # Collect recent entities
        recent_entities = []
        for turn in chat_history[-3:]:  # Last 3 turns
            content = ""
            if "content" in turn:
                content = turn["content"]
            elif "user" in turn:
                content = turn["user"] + " " + turn.get("assistant", "")

            entities = self.extract_entities(content)
            recent_entities.extend(entities)

        # Deduplicate
        unique_entities = list(dict.fromkeys(recent_entities))

        if len(unique_entities) > 2:
            return {
                "ambiguous": True,
                "candidates": unique_entities[:5],
                "message": f"Which are you referring to: {', '.join(unique_entities[:3])}?",
            }

        return None


# Singleton instance
_resolver_instance: AnaphoraResolver | None = None


def get_anaphora_resolver() -> AnaphoraResolver:
    """Get or create AnaphoraResolver singleton."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = AnaphoraResolver()
    return _resolver_instance
