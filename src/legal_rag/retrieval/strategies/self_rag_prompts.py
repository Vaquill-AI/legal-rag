"""Prompt data for the Self-RAG controller (retrieval routing and answer critique)."""

from __future__ import annotations

RETRIEVAL_ROUTER_PROMPT = """<system>
<role>Retrieval router for a legal research assistant.</role>

<decision_criteria>
<RETRIEVE required_when="The query needs authoritative sources">
- Specific statutory sections or provisions
- Case citations or precedents
- Legal definitions or interpretations
- Procedural requirements or deadlines
- Jurisdiction-specific rules
- Contract clause analysis
- Regulatory compliance questions
</RETRIEVE>

<SKIP when="The query is answerable without document lookup">
- Greetings, acknowledgments, chitchat
- Arithmetic
- General knowledge that is not legal
- Clarification of the previous response
- Opinion or preference questions
</SKIP>

<UNCERTAIN when="The case is borderline">
- The query could be legal but lacks specificity
- The intent is ambiguous
</UNCERTAIN>
</decision_criteria>

<count_extraction>
Set requested_count only when the user explicitly asks for a number of results:
- "give me 10 cases" -> 10
- "list 5 examples" -> 5
Leave it null when the number is a year ("cases from 1995"), a section ("Section 1983"),
an article ("Article III") or when no count was requested.
</count_extraction>
</system>"""

ANSWER_CRITIQUE_PROMPT = """<system>
<role>Answer quality evaluator for retrieval-augmented legal responses.</role>

<evaluation_criteria>
<GOOD verdict="The answer is acceptable">
- Directly addresses the query with relevant analysis
- Claims are supported by the provided context
- Appropriate citations or source references are included
- No material omissions for a complete response
</GOOD>

<NEEDS_IMPROVEMENT verdict="The answer has fixable deficiencies">
- Misses key information that exists in the context
- Incomplete analysis of relevant provisions
- Citation gaps where sources are available
- Structural or clarity issues
- Minor inaccuracies that the context can correct
</NEEDS_IMPROVEMENT>

<NEEDS_MORE_CONTEXT verdict="The source material is insufficient">
- The query needs information that is not in the context
- The context is tangential to the query
- Critical provisions are missing
- An accurate answer is impossible without more retrieval
</NEEDS_MORE_CONTEXT>
</evaluation_criteria>
</system>"""

DEFAULT_RETRIEVE_KEYWORDS: tuple[str, ...] = (
    "section",
    "article",
    "case",
    "law",
    "statute",
    "code",
    "explain",
    "what is",
    "how",
    "why",
    "when",
    "where",
    "penalty",
    "procedure",
    "definition",
)

DEFAULT_SKIP_KEYWORDS: tuple[str, ...] = (
    "calculate",
    "compute",
    "math",
    "simple",
    "hello",
    "hi",
    "thank",
    "okay",
    "yes",
    "no",
)
