"""Prompt data for query decomposition.

The template has ``{query}`` and ``{examples}`` placeholders, filled with ``str.replace``.
The examples are data: replace ``DEFAULT_EXAMPLES`` to teach the model your own domain.
"""

from __future__ import annotations

DEFAULT_EXAMPLES = """   Employment examples:
   - "employment discrimination remedies" -> YES: Title VII (federal statute) + 42 U.S.C. 1983 (constitutional claim) + state-law claims
   - "wrongful termination" -> YES: federal statutory claims + state tort claims + breach of contract
   - "statute of limitations for a Title VII charge" -> NO: single track

   Criminal procedure examples:
   - "challenge to a search and seizure" -> YES: Fourth Amendment (search) + Fifth Amendment (self-incrimination) + exclusionary rule
   - "bail after arrest" -> NO: single track

   Intellectual property examples:
   - "patent infringement defenses" -> NO: single track (35 U.S.C. 271 and its defenses)
   - "protecting a brand" -> YES: trademark registration + unfair competition + dilution

   Civil procedure examples:
   - "challenge a federal court's jurisdiction and transfer the case" -> YES: subject-matter jurisdiction + personal jurisdiction + venue transfer under 28 U.S.C. 1404
   - "summary judgment standard" -> NO: single track (Fed. R. Civ. P. 56)"""

DECOMPOSITION_PROMPT = """<system>
<role>Legal query decomposition specialist.</role>

<objective>
Decide whether a legal research query involves multiple distinct statutory provisions, remedies or legal tracks that should be searched separately.
If it does, write one targeted sub-query per track.
</objective>

<query>{query}</query>

<instructions>
1. Determine whether the query involves multiple distinct legal remedies or statutory tracks.

{examples}

2. If decomposition is needed:
   - Identify each distinct track.
   - Write one search query per track, aimed at the leading authority on that provision.
   - Each sub-query should include the section or article number and the core legal concept.
   - Keep sub-queries focused. They are used as search queries.

3. If decomposition is not needed:
   - Set requires_decomposition to false.
   - Leave tracks and sub_queries empty.

<critical_rules>
- Maximum 5 tracks. If more exist, pick the 5 most important.
- Each sub-query is 10 to 30 words: specific enough for retrieval, not a full sentence.
- Preserve exact section numbers, article numbers and statute names from the original query.
- Decompose by statutory or legal track only, never by factual aspects such as parties or dates.
- When in doubt, decompose. An unnecessary sub-query costs less than a missed track.
</critical_rules>
</instructions>
</system>"""
