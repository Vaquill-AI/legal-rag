"""Relevance-grading rubric for the CRAG corrector, as data.

The template uses ``{query}`` and ``{chunk}`` placeholders, filled with ``str.replace``.
Replace it to grade against a different rubric or domain.
"""

from __future__ import annotations

RELEVANCE_PROMPT = """<system>
<role>Chunk relevance evaluator for corrective retrieval (CRAG) over legal text.</role>

<task>Score how relevant this chunk is to the query on a 0.0 to 1.0 scale.</task>

<query>{query}</query>

<chunk>{chunk}</chunk>

<scoring_rubric>
<score value="0.9-1.0">Directly answers the core question with specific, applicable content</score>
<score value="0.7-0.8">Highly relevant statutes, precedents or principles</score>
<score value="0.5-0.6">Partially relevant: related concepts, not directly applicable</score>
<score value="0.3-0.4">Tangential: same domain, different specific issue</score>
<score value="0.1-0.2">Marginal: general content with minimal connection</score>
<score value="0.0">Irrelevant: no meaningful connection to the query</score>
</scoring_rubric>

<evaluation_factors>
<factor>Does the chunk contain provisions, case law or principles directly applicable to the query?</factor>
<factor>Does it address the specific issue, jurisdiction or remedy sought?</factor>
<factor>Would a practitioner find it useful for answering the query?</factor>
</evaluation_factors>

<output>A single number between 0.0 and 1.0.</output>
</system>"""
