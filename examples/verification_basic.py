"""Check an answer's citations: extract them, flag a fabricated case, resolve each one.

Shows: citations.extract (finds citation-shaped text), detect_fabricated_cases (flags cases
that appear in no retrieved source) and a CaseResolver (an in-memory stand-in for a real
case-law index) as a second opinion. A citation match means the SHAPE is right, not that
the case exists.
Needs: nothing beyond the base install (the `verification` extra has no dependencies).
No API keys, no network.

Run: python examples/verification_basic.py
"""

import asyncio
import logging
from typing import ClassVar

from legal_rag.verification import CaseResolver, CitationKind, ResolvedCase, extract
from legal_rag.verification.grounding import detect_fabricated_cases

ANSWER = (
    "Courts may strike down statutes that conflict with the Constitution, see "
    "Marbury v. Madison, 5 U.S. 137 (1803). The Court later extended this reasoning to "
    "state agencies in Hartwell v. Blackstone Holdings, 482 U.S. 911 (1987). "
    "Standing is governed by 28 U.S.C. § 1331."
)

# The library logs a warning per flag. Keep this demo output clean.
logging.getLogger("legal_rag").addHandler(logging.NullHandler())

# What retrieval actually returned: only Marbury.
SOURCES = [{"case_name": "Marbury v. Madison", "citation": "5 U.S. 137"}]


class InMemoryResolver:
    """A CaseResolver backed by a dict. Swap in your own case-law index or API client."""

    KNOWN: ClassVar[dict[str, ResolvedCase]] = {
        "5 U.S. 137": ResolvedCase("Marbury v. Madison", "5 U.S. 137", "U.S.", 1803)
    }

    async def resolve(self, query: str) -> ResolvedCase | None:
        return self.KNOWN.get(query)


async def main() -> None:
    print("Citations found in the answer:")
    citations = extract(ANSWER)
    for c in citations:
        print(f"  {c.kind.value:<5} {c.raw!r} -> {c.normalized}")

    print("\nCases that appear in no retrieved source:")
    flagged = detect_fabricated_cases(ANSWER, SOURCES)
    for f in flagged:
        print(f"  FLAGGED {f.case_name}, {f.citation}")
    if not flagged:
        print("  none")

    resolver: CaseResolver = InMemoryResolver()
    print("\nResolver lookup of every case citation:")
    for c in citations:
        if c.kind is CitationKind.CASE:
            case = await resolver.resolve(c.normalized)
            verdict = f"found: {case.case_name} ({case.year})" if case else "NOT FOUND"
            print(f"  {c.normalized}: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
