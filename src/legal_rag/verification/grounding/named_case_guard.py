"""Named-case guard.

When a query looks up a specific court case (by name or by reporter citation) and retrieval surfaced no source that matches it, a generated answer should not characterize a holding it did not retrieve.
Inventing a holding for a case that does not exist is a serious failure mode for legal AI.

The guard is deliberately high precision and fires only when ALL of these hold:

1. The query names a case ("X v. Y") or carries a reporter citation.
2. The query reads as a lookup of that specific case (a bare case name, a holding or facts cue, or a name paired with a citation), not a broad topical question that mentions a case in passing.
3. No retrieved source matches the case name (both parties present) or any of the cited reporter citations.

Because of (2) it never blocks a statutory or topical query.
``evaluate_named_case_guard`` is synchronous and uses only the retrieved sources.
``evaluate_named_case_guard_with_resolver`` additionally asks an injected ``CaseResolver`` before declining.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.verification.grounding.case_names import is_version_party, parse_case_name_query
from legal_rag.verification.resolver import CaseResolver

# A reporter citation such as "512 U.S. 44", "123 F.3d 456", "5 Cal. 4th 1",
# "999 So. 2d 12", "42 N.E.2d 7". Volume, reporter, page.
_REPORTER_CITE = re.compile(
    r"\b\d{1,4}\s+"
    r"(?:"
    r"U\.?\s?S\.?|S\.?\s?Ct\.?|L\.?\s?Ed\.?(?:\s?2d)?|"
    r"F\.?\s?Supp\.?(?:\s?[23]d)?|F\.?\s?App'?x|F\.?\s?(?:2d|3d|4th)|F\.?|"
    r"Cal\.?\s?Rptr\.?(?:\s?[23]d)?|Cal\.?(?:\s?App\.?)?(?:\s?[2-5](?:d|th))?|"
    r"N\.?\s?[EWY]\.?(?:\s?[23]d)?|So\.?(?:\s?[23]d)?|P\.?(?:\s?[23]d)?|A\.?(?:\s?[23]d)?"
    r")"
    r"\s+\d{1,4}\b",
    re.IGNORECASE,
)

# "Marbury v. Madison" anywhere in the text: capitalized party tokens on
# each side of "v.". Kept tight (each party is 1 to 4 capitalized-ish tokens) so
# it does not swallow surrounding sentence text.
_CASE_NAME_ANY = re.compile(
    r"\b([A-Z][\w.\-']+(?:\s+(?:of\s+|the\s+|&\s+)?[A-Z][\w.\-']+){0,3})"
    r"\s+v\.?s?\.?\s+"
    r"([A-Z][\w.\-']+(?:\s+(?:of\s+|the\s+|&\s+)?[A-Z][\w.\-']+){0,3})"
)

# Signals the user wants THIS specific case (a lookup), not a broad topic.
_HOLDING_LOOKUP_CUE = re.compile(
    r"\b(hold|holding|held|decid|ruled|the ruling|opinion|majority|dissent|"
    r"reasoning|facts of|stand[s]? for|summari[sz]e (?:this|that|the) case|"
    r"what (?:did|was) .*(?:court|case|decision)|in (?:this|that) case)\b",
    re.IGNORECASE,
)

# Generic party tokens that are not distinctive enough to confirm a match on
# their own (a source mentioning "state" or "united" does not confirm the case).
_STOP = frozenset(
    {
        "the",
        "of",
        "and",
        "co",
        "inc",
        "llc",
        "corp",
        "corporation",
        "company",
        "et",
        "al",
        "v",
        "vs",
        "versus",
        "state",
        "united",
        "states",
        "board",
        "dept",
        "department",
        "in",
        "re",
        "ex",
        "rel",
        "city",
        "county",
        "county's",
        "commissioner",
        "commission",
        "commr",
        "director",
        "secretary",
    }
)


@dataclass(frozen=True)
class NamedCaseGuardResult:
    """Outcome of the guard. ``should_decline`` is the only load-bearing field."""

    should_decline: bool
    case_label: str | None = None
    citations: tuple[str, ...] = ()
    matched_source_index: int | None = None  # 1-indexed source that matched, if any
    message: str | None = None


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


def _distinctive_tokens(name: str) -> set[str]:
    return {t for t in _norm(name).split() if len(t) > 2 and t not in _STOP}


def _norm_cite(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (c or "").lower())


def _source_haystack(source: dict) -> str:
    if not isinstance(source, dict):
        return ""
    fields = (
        "title",
        "case_name",
        "citation",
        "primary_citation",
        "label",
        "short_citation",
        "filename",
        "display_label",
    )
    return _norm(" ".join(str(source.get(k) or "") for k in fields))


def _extract_case_name(query: str) -> tuple[str, str] | None:
    """Return (party_a, party_b) for a case-name query, bare or embedded."""
    bare = parse_case_name_query(query)
    if bare is not None:
        return (bare.party_a, bare.party_b)
    m = _CASE_NAME_ANY.search(query)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        # Same version-label guard as parse_case_name_query: this second path matches
        # "X v. Y" embedded anywhere, so "compare Draft v2 v. Final" must not count.
        if is_version_party(a) or is_version_party(b):
            return None
        return (a, b)
    return None


def _has_lookup_intent(
    query: str, name: tuple[str, str] | None, citations: tuple[str, ...]
) -> bool:
    # A bare case name is itself a lookup.
    if parse_case_name_query(query) is not None:
        return True
    # A case name paired with a specific reporter citation is a specific-case ref.
    if name and citations:
        return True
    # Otherwise require an explicit holding/facts cue.
    return bool(_HOLDING_LOOKUP_CUE.search(query))


def _source_matches_name(source: dict, name: tuple[str, str]) -> bool:
    hay = _source_haystack(source)
    if not hay.strip():
        return False
    a = _distinctive_tokens(name[0])
    b = _distinctive_tokens(name[1])
    # Require at least one distinctive token from EACH party in the same source.
    if not a or not b:
        # One side has only generic tokens; fall back to any-side match so we do
        # not over-decline (better to answer than to wrongly refuse).
        toks = a or b
        return any(t in hay for t in toks) if toks else True
    a_hit = any(t in hay for t in a)
    b_hit = any(t in hay for t in b)
    return a_hit and b_hit


def _source_matches_citation(source: dict, citations: tuple[str, ...]) -> bool:
    if not citations:
        return False
    hay = _norm_cite(_source_haystack(source))
    return any(_norm_cite(c) in hay for c in citations if _norm_cite(c))


def _build_decline(label: str) -> str:
    return (
        f"No retrieved source matches {label}, so no description of what it holds is given. "
        "Supply the full citation (reporter, volume, court or year) to retry the lookup, "
        "or restate the legal issue to search for controlling authority on that point."
    )


def evaluate_named_case_guard(query: str, sources: list[dict]) -> NamedCaseGuardResult:
    """Decide whether to decline a case lookup that retrieval could not verify.

    Args:
        query: the user's question (the raw natural-language message).
        sources: the retrieved sources reaching generation. Each is a dict that
            may carry ``title`` / ``case_name`` / ``citation`` / ``source_type``.

    Returns:
        NamedCaseGuardResult with ``should_decline`` True only when a specific
        case lookup found no matching source.
    """
    if not query or not query.strip():
        return NamedCaseGuardResult(should_decline=False)

    name = _extract_case_name(query)
    citations = tuple(dict.fromkeys(m.group(0).strip() for m in _REPORTER_CITE.finditer(query)))

    if name is None and not citations:
        return NamedCaseGuardResult(should_decline=False)

    if not _has_lookup_intent(query, name, citations):
        return NamedCaseGuardResult(should_decline=False)

    for i, s in enumerate(sources or []):
        if not isinstance(s, dict):
            continue
        if (name and _source_matches_name(s, name)) or _source_matches_citation(s, citations):
            return NamedCaseGuardResult(
                should_decline=False,
                citations=citations,
                matched_source_index=i + 1,
            )

    if name is not None:
        label = f"a case called {name[0]} v. {name[1]}"
    else:
        label = f"the case at {citations[0]}"
    return NamedCaseGuardResult(
        should_decline=True,
        case_label=label,
        citations=citations,
        message=_build_decline(label),
    )


# Words that sit next to a case name in a question but are not part of it.
_PROBE_PROSE = frozenset(
    {
        "summarize", "summarise", "explain", "describe", "discuss", "tell", "find", "show",
        "give", "brief", "analyze", "analyse", "what", "who", "how", "why", "does", "did",
        "is", "was", "see", "cf", "in", "on", "about", "case", "decide", "decided", "hold",
        "held", "holding", "court", "me", "please", "and", "for", "from", "with", "still",
    }
)  # fmt: skip
_NAME_ABBREV = frozenset({"inc", "corp", "co", "ltd", "llc", "jr", "sr", "no", "dept", "assn"})
# Used only when the query has no capitalized caption (for example all lowercase input).
_CASE_NAME_LOOSE = re.compile(
    r"((?:\S+\s+){0,2}\S+)\s+vs?\.?\s+(\S+(?:\s+\S+){0,2})",
    re.IGNORECASE,
)


def _ends_sentence(token: str) -> bool:
    """True for a token that closes a sentence, not an abbreviation such as ``Inc.`` or ``U.S.``."""
    if not token.endswith((".", "?", "!")):
        return False
    bare = token.rstrip(".?!")
    return "." not in bare and len(bare) > 1 and bare.lower() not in _NAME_ABBREV


def _extract_case_name_as_written(query: str) -> str | None:
    """The ``A v. B`` name for a resolver probe, as written in the query and trimmed to the name.

    Case and spelling are kept, and neighbouring prose words, the sentence
    around the name and trailing punctuation are cut off.
    A name typed in all lowercase carries no capital letters to find its edges,
    so there it is cut to at most three words on each side.
    """
    if _extract_case_name(query) is None:
        return None
    m = _CASE_NAME_ANY.search(query) or _CASE_NAME_LOOSE.search(query)
    if m is None:
        return None
    left: list[str] = []
    for tok in reversed(m.group(1).split()):
        if _ends_sentence(tok):
            break
        left.insert(0, tok)
    while len(left) > 1 and left[0].lower().strip(".,") in _PROBE_PROSE:
        left.pop(0)
    right: list[str] = []
    for tok in m.group(2).split():
        if right and tok.lower().strip(".,?!") in _PROBE_PROSE:
            break
        right.append(tok)
        if tok.endswith((",", ";", ":")) or _ends_sentence(tok):
            break
    if not left or not right:
        return None
    last = right[-1]
    right[-1] = last.rstrip(",;:?!") if not _ends_sentence(last) else last.rstrip(".?!")
    return f"{' '.join(left).strip(',;:')} v. {' '.join(right)}"


async def evaluate_named_case_guard_with_resolver(
    query: str, sources: list[dict], resolver: CaseResolver
) -> NamedCaseGuardResult:
    """Like ``evaluate_named_case_guard``, but ask ``resolver`` before declining.

    A case missing from the retrieved sources can still exist.
    When the resolver confirms it, the result does not decline.
    The resolver is probed with each reporter citation verbatim, then with the
    case name as written in the query (``A v. B``, trimmed to the name).
    """
    result = evaluate_named_case_guard(query, sources)
    if not result.should_decline:
        return result
    probes = [*result.citations]
    name = _extract_case_name_as_written(query)
    if name is not None:
        probes.append(name)
    for probe in probes:
        if await resolver.resolve(probe) is not None:
            return NamedCaseGuardResult(should_decline=False, citations=result.citations)
    return result
