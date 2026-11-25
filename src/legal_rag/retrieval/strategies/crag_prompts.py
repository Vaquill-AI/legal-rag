"""Answer-prompt templates for mixed-source generation.

Used when an answer draws on two kinds of context: primary sources the caller supplied
and supplementary sources found elsewhere. Chunks are plain dictionaries with ``content``
and optional ``source`` (primary) or ``title`` and ``url`` (supplementary).
"""

from __future__ import annotations

from enum import StrEnum


class PromptVariant(StrEnum):
    """Source-attribution styles."""

    EXPLICIT_LABELS = "explicit_labels"
    SECTION_HEADERS = "section_headers"
    INLINE_CITATIONS = "inline_citations"


MIXED_SOURCE_SYSTEM_PROMPTS: dict[PromptVariant, str] = {
    PromptVariant.EXPLICIT_LABELS: """<system>
<role>Legal research assistant answering from two kinds of sources.</role>

<context_sources>
1. Primary sources: documents supplied for this question. Chunks are marked [PRIMARY].
2. Supplementary sources: outside material that fills gaps. Chunks are marked [SUPPLEMENTARY].
</context_sources>

<guidelines>
- Prioritize [PRIMARY] sources.
- Use [SUPPLEMENTARY] sources to fill gaps or give general context.
- Make clear in the answer which kind of source a statement comes from.
- If the primary sources are thin on a point, say so.
- Never state anything that neither kind of source supports.
</guidelines>
</system>""",
    PromptVariant.SECTION_HEADERS: """<system>
<role>Legal research assistant combining primary and supplementary sources.</role>

<instructions>
Context arrives in two sections: "PRIMARY SOURCES" and "SUPPLEMENTARY SOURCES".
1. Address what the primary sources say first.
2. Add supplementary findings only where needed.
3. Keep the two clearly distinguished in the answer.
If the primary sources are incomplete, say so and explain what the supplementary sources add.
</instructions>
</system>""",
    PromptVariant.INLINE_CITATIONS: """<system>
<role>Legal assistant giving sourced answers.</role>

<citation_format>
Cite inline: (primary:1), (primary:2) for primary sources and (supp:1), (supp:2) for supplementary ones.
</citation_format>

<priority>Primary sources take precedence. Cite supplementary sources only where the primary sources lack the information.</priority>
</system>""",
}

PARTIAL_KNOWLEDGE_ADDENDUM = """
<partial_knowledge_note>
The primary sources have limited information on this topic.
The answer draws mainly on supplementary sources. Say so plainly in the response.
</partial_knowledge_note>
"""

_SEPARATOR = "\n\n" + "-" * 40 + "\n\n"


def format_chunk_with_label(
    content: str,
    source_type: str,
    source_name: str | None = None,
    variant: PromptVariant = PromptVariant.EXPLICIT_LABELS,
) -> str:
    """Format one chunk for ``variant``. ``source_type`` is ``"primary"`` or ``"supplementary"``."""
    if variant is PromptVariant.EXPLICIT_LABELS:
        label = "[PRIMARY]" if source_type == "primary" else "[SUPPLEMENTARY]"
        return f"{label} Source: {source_name}\n{content}" if source_name else f"{label}\n{content}"
    if variant is PromptVariant.SECTION_HEADERS and source_name:
        return f"**Source: {source_name}**\n{content}"
    return content


def format_mixed_context_for_prompt(
    primary_chunks: list[dict],
    supplementary_chunks: list[dict],
    variant: PromptVariant = PromptVariant.EXPLICIT_LABELS,
) -> str:
    """Join both chunk lists into one context block laid out for ``variant``."""

    def supp_name(chunk: dict) -> str | None:
        return chunk.get("title") or chunk.get("url")

    if variant is PromptVariant.SECTION_HEADERS:
        sections = []
        if primary_chunks:
            body = "\n\n---\n\n".join(
                format_chunk_with_label(c.get("content", ""), "primary", c.get("source"), variant)
                for c in primary_chunks
            )
            sections.append("## PRIMARY SOURCES:\n\n" + body)
        if supplementary_chunks:
            body = "\n\n---\n\n".join(
                format_chunk_with_label(
                    c.get("content", ""), "supplementary", supp_name(c), variant
                )
                for c in supplementary_chunks
            )
            sections.append("## SUPPLEMENTARY SOURCES:\n\n" + body)
        return _SEPARATOR.join(sections)

    if variant is PromptVariant.INLINE_CITATIONS:
        sections = [
            f"[primary:{i}] {c.get('source', 'Document')}\n{c.get('content', '')}"
            for i, c in enumerate(primary_chunks, 1)
        ]
        sections += [
            f"[supp:{i}] {c.get('title', 'Source')}\n{c.get('content', '')}"
            for i, c in enumerate(supplementary_chunks, 1)
        ]
        return _SEPARATOR.join(sections)

    sections = [
        format_chunk_with_label(c.get("content", ""), "primary", c.get("source"), variant)
        for c in primary_chunks
    ]
    sections += [
        format_chunk_with_label(c.get("content", ""), "supplementary", supp_name(c), variant)
        for c in supplementary_chunks
    ]
    return _SEPARATOR.join(sections)


def get_system_prompt(variant: PromptVariant = PromptVariant.EXPLICIT_LABELS) -> str:
    """System prompt for ``variant``."""
    return MIXED_SOURCE_SYSTEM_PROMPTS[variant]


def get_prompt_with_partial_knowledge_note(base_prompt: str, has_partial_knowledge: bool) -> str:
    """Append the partial-knowledge note when the primary sources are thin."""
    return base_prompt + PARTIAL_KNOWLEDGE_ADDENDUM if has_partial_knowledge else base_prompt
