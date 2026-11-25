"""Detect fabricated case citations in an LLM answer.

A bracketed-index citation validator (``citations.validator``) catches
[N] index hallucinations (out-of-range or quote-mismatch). It does NOT catch
the more dangerous failure mode where the LLM writes a CASE NAME plus a
reporter citation IN PROSE without a bracketed source index, and the case
either doesn't exist or its citation is wrong.

Example failure: an answer says "the Eleventh Circuit held in *Smith v. Jones*,
123 F.4th 456 (11th Cir. 2024), that ..." but no such case appears in the
retrieved sources and there is no Smith v. Jones at 123 F.4th 456. The user
sees an authoritative-looking citation and may act on it.

This module:

1. Extracts every "Case Name v. Other Name, <reporter cite>" pattern from
   the answer text.
2. Normalises each candidate to a stable key.
3. Cross-references against the retrieved source list (case_name + citation
   normalised the same way).
4. Returns the candidates that do NOT appear in the source list.

Detection is intentionally conservative: it skips bare case-name mentions
without a reporter cite (those are too ambiguous to flag), Latin shortforms
(*id.*, *supra*), and statute references. False positives are worse than
false negatives in legal AI: the goal is to surface high-confidence
fabrications, not flag every name.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib import resources

logger = logging.getLogger(__name__)


# Reporter abbreviations recognised in the cite tail. Order matters: longer
# names first so 'F. Supp. 2d' doesn't match as 'F.' alone.
_REPORTER_RE = (
    r"(?:F\.\s*Supp\.?\s*(?:2d|3d)?|"
    r"F\.\s*(?:2d|3d|4th)?|"
    r"U\.\s*S\.|"
    r"S\.\s*Ct\.|"
    r"L\.\s*Ed\.?\s*2d|"
    r"L\.\s*Ed\.|"
    r"Cal\.\s*(?:Rptr\.\s*(?:2d|3d)?|App\.\s*(?:2d|3d|4th|5th)?|\d?(?:d|st|nd|rd|th)?)?|"
    r"Cal\.|"
    r"N\.\s*Y\.\s*(?:2d|3d)?|"
    r"N\.\s*E\.\s*(?:2d|3d)?|"
    r"N\.\s*W\.\s*(?:2d|3d)?|"
    r"P\.\s*(?:2d|3d)?|"
    r"A\.\s*(?:2d|3d)?|"
    r"S\.\s*W\.\s*(?:2d|3d)?|"
    r"So\.\s*(?:2d|3d)?|"
    r"Tex\.\s*(?:S\.\s*W\.\s*(?:2d|3d)?)?|"
    r"Pa\.\s*Super\.?|"
    r"Wash\.\s*App\.?|"
    r"Wash\.\s*(?:2d)?)"
)

# Matches "Case Name v. Other Name, <volume> <reporter> <page>"
# Conservative: requires a proper "v." between two name fragments.
# Allows things like "Bell Atlantic Corp. v. Twombly" and
# "United States v. Smith".
_CASE_CITE_PATTERN = re.compile(
    # Group 1: case name (lower priority: stop at the citation comma)
    r"\b([A-Z][A-Za-z0-9'&\.\-]+"  # first party (must start uppercase)
    r"(?:\s+[A-Za-z0-9'&\.\-]+){0,5}"  # rest of first party (up to 6 words)
    r"\s+v\.?\s+"  # v.
    r"[A-Z][A-Za-z0-9'&\.\-]+"  # second party
    r"(?:\s+[A-Za-z0-9'&\.\-]+){0,5})"  # rest of second party
    r",\s*"
    r"(\d+\s+" + _REPORTER_RE + r"\s+\d+\w*)"  # volume + reporter + page
    r"(?:\s*\((?:[^)]{0,30})?\d{4}\))?",  # optional (court? year)
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FlaggedCase:
    """A case-name + reporter citation found in the answer that was not
    located in the retrieved sources."""

    case_name: str
    citation: str
    surrounding_text: str  # 80-char snippet around the citation


# Common prose preambles that show up before a case name when the regex
# captures too eagerly (e.g., "the court in Smith v. Jones" -> we want
# just "Smith v. Jones"). Stripped from the leading edge of the captured
# first-party token. The "v" separator is preserved.
_LEADING_PREFIX_WORDS = {
    "see",
    "see also",
    "see e.g.",
    "see, e.g.",
    "see generally",
    "see contra",
    "accord",
    "but see",
    "cf.",
    "compare",
    "but cf",
    "the court in",
    "the court's decision in",
    "the supreme court in",
    "the panel in",
    "the supreme court's decision in",
    "court in",
    "in",
    "and",
    "or",
    "or in",
    "and in",
    "as in",
    "again",
    "then in",
    "for example,",
    "for example",
    "e.g.,",
    "e.g.",
    "i.e.,",
    "i.e.",
    "again,",
    "again in",
    "the panel's decision in",
    "the eleventh circuit in",
    "the ninth circuit in",
    "the second circuit in",
    "the fifth circuit in",
    "the third circuit in",
    "the seventh circuit in",
    "the d.c. circuit in",
    "the federal circuit in",
    "the first circuit in",
    "the fourth circuit in",
    "the sixth circuit in",
    "the eighth circuit in",
    "the tenth circuit in",
    "the dc circuit in",
}


def _strip_leading_prose(name: str) -> str:
    """Trim common prose prefixes from a captured first-party name."""
    n = name.strip()
    lower = n.lower()
    changed = True
    while changed:
        changed = False
        for prefix in sorted(_LEADING_PREFIX_WORDS, key=len, reverse=True):
            if prefix == "in" and lower.startswith("in re "):
                continue  # "In re Gault" is a party name, not prose
            if lower.startswith(prefix + " "):
                n = n[len(prefix) :].lstrip(" ,.")
                lower = n.lower()
                changed = True
                break
    return n


# Words that open a sentence or a citation signal and are never the first word
# of a party name. Only stripped from the LEADING edge of a captured name.
# "in" is handled separately because "In re Gault" is a party name.
_SIGNAL_WORDS = frozenset(
    {
        "see",
        "also",
        "accord",
        "cf",
        "compare",
        "contra",
        "but",
        "generally",
        "e.g",
        "i.e",
        "in",
        "and",
        "or",
        "as",
        "again",
        "then",
        "thus",
        "therefore",
        "however",
        "moreover",
        "accordingly",
        "similarly",
        "under",
    }
)

# Lowercase words that legitimately sit INSIDE a party name ("Doe ex rel. Roe",
# "In re Gault", "Board of Education"). A lowercase word outside this set in the
# leading edge ("held", "court", "in", "relied") is prose.
_NAME_CONNECTORS = frozenset(
    {
        "of",
        "the",
        "and",
        "&",
        "for",
        "ex",
        "rel",
        "re",
        "parte",
        "de",
        "del",
        "la",
        "du",
        "von",
        "van",
        "der",
        "den",
        "al",
        "et",
    }
)

_VERSUS_TOKENS = frozenset({"v", "v.", "vs", "vs."})


def _is_prose_token(tok: str) -> bool:
    """True for a lowercase word that cannot be part of a party name."""
    bare = tok.rstrip(".").lower()
    if bare in _NAME_CONNECTORS:
        return False
    # A name-like token carries an uppercase letter ("eBay", "NLRB") or a digit.
    return not any(c.isupper() or c.isdigit() for c in tok)


def _extract_case_name(raw: str) -> str:
    """Trim leading prose and signal words from a captured case name.

    ``The court held in Smith v. Jones`` -> ``Smith v. Jones``.
    Party names such as ``In re Gault``, ``Ex parte Young``,
    ``United States v. Nixon`` and ``Matter of Acme v. Beta`` are kept intact.
    When trimming would leave no first party, the capture is returned unchanged,
    so this can only shorten a name, never invent or drop the "v." half.
    """
    text = raw.strip()
    spans = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    versus = next((i for i, (_, t) in enumerate(spans) if t.lower() in _VERSUS_TOKENS), None)
    if versus is None or versus == 0:
        return text
    head = [t for _, t in spans[:versus]]

    # 1. Everything up to and including the last lowercase prose word is prose.
    start = 0
    for i, tok in enumerate(head):
        if _is_prose_token(tok):
            start = i + 1

    # 1b. A sentence that ended just before a signal word ("...the Constitution. See Marbury")
    #     belongs to the previous sentence. Only a signal word counts as the proof of a
    #     boundary, so abbreviations inside names ("Acme Inc. Holdings v. Beta") are safe.
    for i in range(len(head) - 1):
        if head[i].endswith((".", "?", "!")) and head[i + 1].rstrip(".").lower() in _SIGNAL_WORDS:
            start = max(start, i + 1)

    # 2. Strip leading signal words ("See", "In", "Under", "However").
    while start < len(head):
        bare = head[start].rstrip(".").lower()
        if bare not in _SIGNAL_WORDS:
            break
        if bare == "in" and start + 1 < len(head) and head[start + 1].lower() == "re":
            break  # "In re Gault"
        start += 1

    if start >= len(head):
        return text
    return text[spans[start][0] :]


def _norm_name(s: str) -> str:
    """Normalise a case name to a stable comparison key.

    Same shape as a typical case-name normaliser (kept local to
    avoid import cycle with the dedup module).
    """
    s = _strip_leading_prose(s or "")
    s = s.lower()
    s = re.sub(r"\bv\.s?\b\.?", "v", s)
    s = re.sub(
        r"\b(inc|llc|llp|corp|corporation|co|enterprises|enters|ltd|ass'?n|association)\.?\b",
        "",
        s,
    )
    s = re.sub(r"[,.;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_cite(s: str) -> str:
    """Normalise a reporter citation. '123 F. 3d 456' -> '123 f3d 456'."""
    s = (s or "").lower()
    s = re.sub(r"\(\d{4}\)", "", s)
    s = re.sub(r"\bu\.\s*s\.\b", "us", s)
    s = re.sub(r"\bf\.\s*(\d)(?:d|st|nd|rd|th)\b", r"f\1d", s)
    s = re.sub(r"\bs\.\s*ct\.\b", "sct", s)
    s = re.sub(r"[,.;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _collect_source_index(sources: list) -> tuple[set[str], set[str]]:
    """Build the set of (normalised-name+cite) and (normalised-name) keys
    we'll treat as 'present in retrieved sources'."""
    nc_keys: set[str] = set()
    name_keys: set[str] = set()
    for s in sources or []:
        if isinstance(s, dict):
            name = (
                s.get("case_name") or s.get("caseName") or s.get("title") or s.get("filename") or ""
            )
            cite = s.get("citation") or s.get("primary_citation") or s.get("citations") or ""
            # citations may be a list
            if isinstance(cite, list):
                cite_list = cite
            else:
                cite_list = [cite] if cite else []
        else:
            name = (
                getattr(s, "case_name", None)
                or getattr(s, "title", None)
                or getattr(s, "filename", None)
                or ""
            )
            cite_list = [getattr(s, "citation", None) or ""]
        nm = _norm_name(name)
        if nm:
            name_keys.add(nm)
        for c in cite_list:
            cn = _norm_cite(str(c or ""))
            if nm and cn:
                nc_keys.add(f"{nm}|{cn}")
    return nc_keys, name_keys


def detect_fabricated_cases(
    answer: str, sources: list, *, snippet_chars: int = 80
) -> list[FlaggedCase]:
    """Return cases the LLM cited that don't appear in the retrieved sources.

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    sources:
        The retrieved-source list (dict or object). Must carry ``case_name``/``title`` and ``citation``.
    snippet_chars:
        Half-width of the surrounding-text snippet stored on each flag (for
        downstream display / logging).

    A case is flagged only when BOTH of these are true:
    1. The (normalised name + citation) pair is missing from sources.
    2. The normalised name alone is also missing.

    Rationale: if the name matches but the citation differs, the LLM may have
    cited a parallel reporter, so we don't flag that as fabricated. We flag
    only when the case name itself has no presence at all in retrieved
    sources.

    An empty list is ambiguous when ``sources`` carries no case names (for
    example an empty list): nothing could be checked, which is not the same as
    "nothing fabricated". The return type stays ``list[FlaggedCase]``, so that
    case is signalled by one ``WARNING`` log record per call from this module's
    logger. Callers must test for empty sources themselves.
    """
    if not answer:
        return []
    nc_keys, name_keys = _collect_source_index(sources)
    if not name_keys:
        # No sources at all: we do not flag every cite in the answer, but the
        # empty result must not read as a clean check, so say so once per call.
        logger.warning(
            "fabricated_case_check_skipped: no sources to check against, "
            "an empty result means nothing was checked"
        )
        return []

    flags: list[FlaggedCase] = []
    seen: set[str] = set()
    for m in _CASE_CITE_PATTERN.finditer(answer):
        captured = m.group(1)
        raw_name = _extract_case_name(captured)
        raw_cite = m.group(2)
        nm = _norm_name(raw_name)
        ct = _norm_cite(raw_cite)
        if not nm:
            continue
        if nm in name_keys:
            continue  # case name appears in sources: give benefit of doubt
        if f"{nm}|{ct}" in nc_keys:
            continue  # exact combo in sources
        # A stripped lead word may be part of a real party name ("Under Armour
        # Inc. v. X"): if the untrimmed capture is a known source, do not flag.
        if raw_name != captured.strip():
            full = _norm_name(captured)
            if full in name_keys or f"{full}|{ct}" in nc_keys:
                continue
        key = f"{nm}|{ct}"
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - snippet_chars)
        end = min(len(answer), m.end() + snippet_chars)
        flags.append(
            FlaggedCase(
                case_name=raw_name.strip(),
                citation=raw_cite.strip(),
                surrounding_text=answer[start:end].replace("\n", " "),
            )
        )

    if flags:
        logger.warning(
            "fabricated_case_detected",
            extra={
                "count": len(flags),
                "first": flags[0].case_name + ", " + flags[0].citation,
            },
        )
    return flags


@dataclass(frozen=True)
class StateMismatch:
    """A citation whose reporter or state-code prefix does not match the expected state."""

    citation: str
    # The state inferred from the citation. A regional reporter names several
    # states at once and this carries all of them, slash-joined ("al/fl/la/ms"),
    # because naming one of the four would be a claim the citation does not
    # support. `detected_states` is the machine-readable form.
    detected_state: str
    expected_state: str  # the state the text is supposed to be governed by
    surrounding_text: str
    detected_states: frozenset[str] = frozenset()


def _load_state_forms() -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    """Load the state citation form tables from the bundled data file.

    ``state_prefixes`` holds prefixes that identify one state (a state code, a state
    reporter).
    ``regional_reporters`` holds reporters that cover a SET of states.
    Federal reporters are intentionally absent: they do not betray a state.
    """
    raw = json.loads(
        resources.files("legal_rag.verification.citations")
        .joinpath("data/state_citation_forms.json")
        .read_text(encoding="utf-8")
    )
    return (
        dict(raw["state_prefixes"]),
        {k: frozenset(v) for k, v in raw["regional_reporters"].items()},
    )


_STATE_CITATION_PREFIXES, _REGIONAL_REPORTER_STATES = _load_state_forms()

# One lookup for both kinds, sorted longest-first so "cal. civ. code" matches
# before "cal.". A state-specific prefix is just the one-element case.
_SORTED_STATE_PREFIXES: list[tuple[str, frozenset[str]]] = sorted(
    (
        [(p, frozenset({s})) for p, s in _STATE_CITATION_PREFIXES.items()]
        + list(_REGIONAL_REPORTER_STATES.items())
    ),
    key=lambda kv: len(kv[0]),
    reverse=True,
)


def detect_state_mismatches(
    answer: str,
    expected_state: str | None,
    *,
    snippet_chars: int = 80,
) -> list[StateMismatch]:
    """Find citations whose state-distinctive prefix does not match ``expected_state``.

    Returns an empty list when ``expected_state`` is falsy, equals
    ``'federal'``, or is not a recognized state code -- the caller is
    responsible for invoking only when a single governing state is known.

    Federal reporters (F., F.2d, F.3d, F.4th, U.S., S. Ct., L. Ed.) are
    never flagged -- federal authority is admissible in any state.
    """
    if not answer or not expected_state:
        return []
    expected = expected_state.strip().lower()
    if expected == "federal" or len(expected) != 2:
        return []

    lower_answer = answer.lower()
    flags: list[StateMismatch] = []
    seen: set[int] = set()

    for prefix, prefix_states in _SORTED_STATE_PREFIXES:
        # A regional reporter that publishes the pinned state is not a
        # mismatch: it is where that state's decisions actually appear.
        if expected in prefix_states:
            continue
        detected = "/".join(sorted(prefix_states))
        start = 0
        while True:
            idx = lower_answer.find(prefix, start)
            if idx == -1:
                break
            # Require a word boundary on the left so 'cal.' inside a
            # bigger word ('typical.') is not flagged. The right side
            # ends with '.' or a digit so it's already self-delimiting.
            left_ok = idx == 0 or not lower_answer[idx - 1].isalnum()
            # One flag per position: prefixes are sorted longest first, so the most
            # specific prefix wins and its shorter forms ('cal.') are not repeated.
            if left_ok and idx not in seen:
                seen.add(idx)
                ctx_start = max(0, idx - snippet_chars)
                ctx_end = min(len(answer), idx + len(prefix) + snippet_chars)
                flags.append(
                    StateMismatch(
                        citation=answer[idx : idx + len(prefix)],
                        detected_state=detected,
                        expected_state=expected,
                        surrounding_text=answer[ctx_start:ctx_end].replace("\n", " "),
                        detected_states=prefix_states,
                    )
                )
            start = idx + len(prefix)

    if flags:
        logger.warning(
            "state_citation_mismatch_detected",
            extra={
                "count": len(flags),
                "expected": expected,
                "first": flags[0].citation,
            },
        )
    return flags


__all__ = [
    "FlaggedCase",
    "StateMismatch",
    "detect_fabricated_cases",
    "detect_state_mismatches",
]
