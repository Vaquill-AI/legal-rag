"""Split a chat message into the user's instruction and any pasted body.

A router that matches over the whole message is steered by words the user never wrote when a
short instruction is followed by a large pasted third-party body. Any pasted transcript or
contract is a bag of legal keywords that will satisfy some rule. The fix is to route on the
instruction and treat the body as payload.

Design commitments:

- Structural, not lexical. Detection keys on message shape (turn markers, email headers,
  recitals, clause numbering), never on topic vocabulary.
- Precision over recall. A long question must never be read as a paste. Every uncertain case
  returns ``(text, None)``.
- ``body is None`` means the instruction is the original string, so a caller that routes on
  the instruction behaves exactly as it did on the whole message.
"""

from __future__ import annotations

import re

_TRANSCRIPT_MARKER_RE = re.compile(
    r"^\s*(?:\[\d{1,2}:\d{2}(?::\d{2})?\]"  # [0:00:01]
    r"|\d{1,2}:\d{2}(?::\d{2})?\s"  # 0:00:01 at line start
    r"|(?:speaker|provider|interviewer|witness|attorney|counsel|the\s+court|q|a|mr|ms|dr)\.?\s*[:.]"
    r"|[A-Z][a-zA-Z]+\s*:)",  # "Name:" turn label
    re.IGNORECASE | re.MULTILINE,
)

_CONTRACT_STRUCTURE_RE = re.compile(
    r"\bWHEREAS\b|\bNOW,?\s+THEREFORE\b|\bIN\s+WITNESS\s+WHEREOF\b"
    r"|\bthis\s+(?:agreement|contract|deed|lease|addendum|amendment|assignment)\b"
    r"|\bshall\s+mean\b|\bhereinafter\b|\bhereunder\b|\bthe\s+parties\s+(?:agree|hereto)\b"
    r"|\bsection\s+\d+\.\d+\b|\barticle\s+(?:[IVXLC]+|\d+)\b"
    r"|\(\s*[\"“][^\"”]{2,40}[\"”]\s*\)"  # a defined term, e.g. ("Agreement")
    r"|\bexecuted\s+as\s+of\b|\beffective\s+date\b|\bgoverning\s+law\b",
    re.IGNORECASE,
)


def looks_like_transcript(text: str | None) -> bool:
    """True when the text carries several transcript turn markers (timestamps or
    speaker labels)."""
    return len(_TRANSCRIPT_MARKER_RE.findall(text or "")) >= 4


def has_contract_structure(text: str | None) -> bool:
    """True when the text shows contract signals (recitals, defined terms, clause
    numbering, execution language)."""
    return bool(_CONTRACT_STRUCTURE_RE.search(text or ""))


# A pasted body has to be big enough to be worth protecting the router from.
# Below this it is a quote, and splitting would strip context the router needs.
_MIN_BODY_CHARS = 400
_MIN_BODY_LINES = 4
# An instruction is a request, not a document. Generous (rewrite rules run to a
# paragraph) because the BODY's structure is what actually decides the split.
_MAX_INSTRUCTION_CHARS = 1500

# Line-anchored body openers. Anchoring to line starts is what keeps a QUESTION
# from splitting: "Does Section 2.1 of this agreement..." carries contract
# vocabulary mid-sentence but never opens a line as a document does.
_TIMESTAMP_TURN = r"\[\d{1,2}:\d{2}(?::\d{2})?\]|\d{1,2}:\d{2}(?::\d{2})?\s+\S"
_SPEAKER_TURN = (
    r"(?:speaker|interviewer|witness|attorney|counsel|the\s+court|deponent"
    r"|mr|ms|mrs|dr|q|a)\.?\s*[:.]\s"
)
_EMAIL_HEADER = r"(?:from|to|sent|subject|cc|bcc|date)\s*:\s+\S"
_RECITAL = r"WHEREAS\b|NOW,?\s+THEREFORE\b|IN\s+WITNESS\s+WHEREOF\b"
_HEADING = r"(?:article|section)\s+(?:[IVXLC]+|\d+)\b|\d+(?:\.\d+)*\.?\s+[A-Z]"

_BODY_START_RE = re.compile(
    rf"^[ \t]*(?:{_TIMESTAMP_TURN}|{_SPEAKER_TURN}|{_EMAIL_HEADER}|{_RECITAL}|{_HEADING})",
    re.IGNORECASE | re.MULTILINE,
)
# A named turn label ("Ms. Reported:", "Dad:"). Case-sensitive on purpose: a
# capitalized name opening a line is a turn, "note:" in prose is not.
_NAMED_TURN_RE = re.compile(r"^[ \t]*[A-Z][A-Za-z.'’\- ]{0,30}:\s+\S", re.MULTILINE)
# An ALL-CAPS document title, e.g. "MUTUAL NONDISCLOSURE AGREEMENT".
_CAPS_TITLE_RE = re.compile(r"^[ \t]*[A-Z][A-Z0-9 \-'&,.]{8,}$", re.MULTILINE)
_EMAIL_HEADER_RE = re.compile(rf"^[ \t]*{_EMAIL_HEADER}", re.IGNORECASE | re.MULTILINE)


def _body_start(text: str) -> int | None:
    """Index of the earliest line that opens a pasted body, if any."""
    starts = [
        m.start()
        for m in (
            _BODY_START_RE.search(text),
            _NAMED_TURN_RE.search(text),
            _CAPS_TITLE_RE.search(text),
        )
        if m is not None
    ]
    return min(starts) if starts else None


def _is_real_body(body: str) -> bool:
    """True when `body` is structurally a pasted document, not more prose.

    Requires BOTH bulk (length and line count) and a corroborating structure, so
    a few wrapped lines of prose cannot qualify.
    """
    if len(body.strip()) < _MIN_BODY_CHARS:
        return False
    if len(body.strip().splitlines()) < _MIN_BODY_LINES:
        return False
    return (
        looks_like_transcript(body)
        or has_contract_structure(body)
        or len(_EMAIL_HEADER_RE.findall(body)) >= 2
    )


def split_message(text: str | None) -> tuple[str, str | None]:
    """Return ``(instruction, body)``; ``body`` is None unless a paste is certain.

    ``(text, None)`` is the conservative answer and the default: it preserves
    today's whole-message routing exactly.
    """
    if not text or not text.strip():
        return (text or "", None)

    start = _body_start(text)
    if start is None:
        return (text, None)

    instruction = text[:start].strip()
    body = text[start:]

    # No instruction to route ON (a bare paste): keep today's behavior rather
    # than handing the cascade an empty string.
    if not instruction:
        return (text, None)
    # An "instruction" this long is likely prose we mis-cut, not a request.
    if len(instruction) > _MAX_INSTRUCTION_CHARS:
        return (text, None)
    if not _is_real_body(body):
        return (text, None)

    return (instruction, body)
