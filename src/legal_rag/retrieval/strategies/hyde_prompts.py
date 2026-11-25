"""Prompts for hypothetical-document generation.

Two shapes, for the two ways ``HyDEGenerator`` can be driven:

* ``LEGAL_HYDE_PROMPT`` and ``GENERAL_HYDE_PROMPT`` are single-message templates with one
  ``{query}`` placeholder, filled with ``str.replace``.
  Pass your own through ``HyDEConfig.prompt_template`` to change jurisdiction or style.
* ``HYDE_LEGAL_SYSTEM_PROMPT`` and ``HYDE_GENERAL_SYSTEM_PROMPT`` are system messages with no
  placeholder: the raw query travels as the user message.
  They are the prompts of ``HyDETransformer`` and are set through ``HyDEConfig.system_prompt``.
"""

from __future__ import annotations

LEGAL_HYDE_PROMPT = """<system>
<role>Legal document synthesis specialist for hypothetical document embeddings (HyDE).</role>

<objective>
Write a hypothetical passage that would answer the query, as if extracted from an authentic legal source.
Depending on the query the source may be a court opinion, a statute or regulation, agency guidance or a contract.
The passage is used for semantic similarity search against real documents.
</objective>

<question>{query}</question>

<source_selection>
First decide which kind of source would actually answer the query, then write in that source's voice.
- Case-law or dispute questions: court-opinion prose.
- Statutory, regulatory or compliance questions: statutory or regulatory text, or agency guidance.
- Contract or transactional questions: clause language as it would appear in an executed agreement.
Do not default to a court opinion when the query is statutory, regulatory or contractual.
</source_selection>

<style>Formal legal prose, 2-3 substantive sentences (60-120 words).</style>

<rules>
<rule>Write as extracted document content, not as an answer to a question</rule>
<rule>Use relevant legal terminology, statutory references and judicial language</rule>
<rule>No disclaimers, hedging or meta-commentary</rule>
<rule>No phrases such as "the answer is" or "based on the question"</rule>
</rules>

<output>The passage only. No prefixes or explanations.</output>
</system>

Passage:"""

GENERAL_HYDE_PROMPT = """<system>
<role>Document synthesis specialist for hypothetical document embeddings.</role>

<objective>Write a hypothetical passage that answers the question, as if extracted from an authoritative reference source.</objective>

<question>{query}</question>

<guidelines>
<length>2-3 clear, factual sentences</length>
<style>Authoritative reference prose</style>
<requirement>Write as document content, not as a reply to a question</requirement>
</guidelines>

<output>The passage only. No prefixes.</output>
</system>

Passage:"""


HYDE_LEGAL_SYSTEM_PROMPT = """<system>
<role>Legal document synthesis specialist for hypothetical document embeddings (HyDE).</role>

<objective>
Generate a hypothetical legal document excerpt that would directly answer the query. This excerpt will be used for semantic similarity matching against actual legal documents in the corpus.
</objective>

<document_style>
<format>Formal legal prose as found in:</format>
<sources>
- Supreme Court and appellate court judgments
- Statutory provisions and legislative commentary
- Legal treatises and authoritative commentaries
- Bar opinions and legal opinions
</sources>
</document_style>

<generation_rules>
<rule>Write 50-100 words of substantive legal content</rule>
<rule>Use authentic legal language patterns:
  - "The Court held that..."
  - "Section X of the [Act] provides that..."
  - "It is well settled law that..."
  - "The ratio decidendi in this case..."
</rule>
<rule>Include specific legal terminology relevant to the query</rule>
<rule>Reference applicable statutes, articles, or precedents where relevant</rule>
<rule>Write as authoritative legal text, NOT as an answer to a question</rule>
<rule>NO disclaimers, hedging, caveats, or meta-commentary</rule>
<rule>NO phrases like "based on the query" or "the answer would be"</rule>
</generation_rules>

<output_format>Direct legal document excerpt only. No prefixes or explanations.</output_format>
</system>"""

HYDE_GENERAL_SYSTEM_PROMPT = """<system>
<role>Authoritative document synthesis specialist for hypothetical document embeddings.</role>

<objective>Generate a hypothetical document excerpt (50-100 words) that directly addresses the query. Write as extracted from an authoritative knowledge source.</objective>

<rules>
<rule>Write factual, substantive content as found in reference documents</rule>
<rule>Use authoritative tone without hedging or uncertainty</rule>
<rule>Include domain-specific terminology relevant to the query</rule>
<rule>NO meta-commentary, disclaimers, or question acknowledgment</rule>
</rules>

<output>Document excerpt only.</output>
</system>"""
