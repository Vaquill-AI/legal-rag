"""Prompt templates for multi-query expansion.

Templates use ``{query}`` and ``{num_queries}`` placeholders, filled with ``str.replace``.
Pass your own through ``MultiQueryConfig.prompt_template`` to change jurisdiction or facets.
"""

from __future__ import annotations

LEGAL_FACET_PROMPT = """<system>
<role>Legal query expansion specialist for multi-query retrieval.</role>

<objective>
Generate {num_queries} semantically diverse search query variations to maximize retrieval coverage over a legal corpus.
Each variation should capture a different facet of the legal question.
</objective>

<query_to_expand>{query}</query_to_expand>

<expansion_strategy>
<facet type="factual">Facts, parties and circumstances</facet>
<facet type="statutory">Specific statutes, sections, regulations and rules</facet>
<facet type="procedural">Procedural requirements, deadlines and jurisdiction</facet>
<facet type="remedial">Remedies, relief, penalties and consequences</facet>
<facet type="precedential">Case law and judicial interpretation</facet>
</expansion_strategy>

<rules>
<rule priority="critical">Preserve verbatim: docket numbers, citations, section references, dates and amounts</rule>
<rule priority="high">Use the legal terminology of the jurisdiction the question is about</rule>
<rule priority="high">Each variation must target a distinctly different retrieval aspect</rule>
<rule priority="medium">Use synonyms and alternative legal phrasings</rule>
<rule priority="medium">Keep the original intent while varying the search angle</rule>
</rules>

<output_format>One query per line. No numbering, bullets or explanations.</output_format>
</system>

Variations:"""

GENERAL_MULTI_QUERY_PROMPT = """<system>
<role>Query expansion specialist for multi-query document retrieval.</role>

<objective>Generate {num_queries} semantically diverse search query variations to maximize retrieval coverage.</objective>

<query_to_expand>{query}</query_to_expand>

<rules>
<rule>Each variation should target a different semantic aspect of the question</rule>
<rule>Use synonyms, alternative phrasings and related concepts</rule>
<rule>Preserve specific identifiers, numbers and technical terms</rule>
<rule>Maintain the original intent across all variations</rule>
</rules>

<output_format>One query per line. No numbering or prefixes.</output_format>
</system>

Variations:"""

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a legal-research question into {num_queries} alternative phrasings "
    "that preserve the legal meaning but use different vocabulary, so a hybrid dense and "
    "sparse retriever can find passages the original wording would miss.\n\n"
    "Guidelines:\n"
    "- Each rewrite is a self-contained question of 8 to 30 words.\n"
    "- Vary doctrine and case names with synonyms where appropriate.\n"
    "- Add relevant statute citations the user did not use when they clearly apply.\n"
    "- Do not change the jurisdiction (keep federal questions federal).\n"
    "- Do not introduce new sub-issues. Stay on the same legal point."
)
