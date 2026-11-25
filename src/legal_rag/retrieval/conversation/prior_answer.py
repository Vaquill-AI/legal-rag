"""Re-ground follow-ups that refer to the assistant's own previous answer.

"Expand every instance listed under section X of the answer above" is a question about a
message that was already written. Chat history normally only feeds a query rewrite, and the
rewritten string then drives a fresh search, so nothing re-pins the previous answer or the
sources it was built from.

The fix is to stop searching for something that is already in hand. When a follow-up refers to
the previous answer, that answer is pinned into the context verbatim, so the model expands what
it actually said instead of retrieving a fresh guess.
"""

from __future__ import annotations

import re
from typing import Any

# Reference cues in the scripts we actually see. Kept explicit rather than
# LLM-detected because this runs on every turn and must be cheap; the condenser's
# own `query_type` provides the language-agnostic signal alongside it.
_SELF_REF_RE = re.compile(
    # English
    r"\b(?:the |your |that |this )?(?:previous|prior|last|above|earlier|foregoing)\s+"
    r"(?:answer|response|reply|message|list|summary|table|output)\b"
    r"|\b(?:answer|response|list|summary|table)\s+above\b"
    r"|\byou (?:just )?(?:listed|mentioned|said|gave|wrote|provided)\b"
    r"|\bin your (?:last|previous|earlier)\b"
    # Chinese: 上面/上述/前面/以上 (above), 之前 (earlier), 你的回答 (your answer),
    # 那个答案/这个答案 (that/this answer)
    r"|上面|上述|前面|以上|之前|你的回答|那个答案|这个答案|刚才"
    # Japanese / Korean
    r"|上記|前の回答|위의|이전 답변",
    re.IGNORECASE,
)

# Condenser classifications that mean "this turn depends on what came before".
# `follow_up` is deliberately excluded: it is broad enough to fire on most turns,
# and pinning a long prior answer on every one of them wastes context budget.
_REFERENTIAL_QUERY_TYPES = frozenset({"reference", "elaboration"})

# A prior answer can run to 15k+ chars. Cap it so pinning cannot crowd out the
# retrieved sources, and trim from the END (the head carries the structure the
# user is referring to).
MAX_PRIOR_ANSWER_CHARS = 16_000


def last_assistant_message(chat_history: list[dict[str, Any]] | None) -> str | None:
    """The most recent assistant turn's text, or None."""
    for turn in reversed(chat_history or []):
        if (turn.get("role") or "").lower() == "assistant":
            content = (turn.get("content") or "").strip()
            if content:
                return content
    return None


# A reformat asks us to re-present what we just said in a different shape. It is
# a transformation of our own output, so re-retrieving to satisfy one cannot beat
# simply re-reading it -- and on a long document it does measurably worse.
#
# Prod (chat f880214b): after a 52-date chronology table, "just dates and what was
# the event only two column" carried no chronology cue, so the document-task
# router did not re-classify it and it fell through to generic top-k. Recall
# dropped to 12 dates, a 77% loss, on what the user reasonably expected to be a
# re-render of the same table.
_REFORMAT_RE = re.compile(
    r"\b(?:make|put|show|give|display|render|turn|convert|redo|present)\s+(?:it|that|this|them)\b"
    r"|\b(?:reformat|reword|rephrase|reorganize|restructure|condense|shorten|expand)\b"
    r"|\bsame\s+(?:thing|list|table|answer|but|again)\b"
    r"|\b(?:just|only)\s+(?:the\s+)?\w+\s+(?:and|,)?\s*\w*\s*(?:column|columns|rows?)\b"
    r"|\b(?:two|three|2|3)\s+columns?\b"
    # Bare pronouns only. In "summarize this contract", "this" is a DETERMINER
    # introducing a new subject, not a reference to our last answer -- that is a
    # fresh request and must keep its own retrieval.
    r"|\b(?:rewrite|translate|summari[sz]e)\s+(?:it|them)\b",
    re.IGNORECASE,
)
# Reformat instructions are terse. Anything longer is carrying new subject matter,
# which means it is a fresh question that merely mentions a format ("make a table
# of every state's statute of limitations") and must NOT pin the prior answer.
_MAX_REFORMAT_CHARS = 90


def is_reformat_request(query: str | None) -> bool:
    """True when the turn asks to re-present the previous answer differently."""
    if not query:
        return False
    q = query.strip()
    if len(q) > _MAX_REFORMAT_CHARS:
        return False
    return bool(_REFORMAT_RE.search(q))


def references_prior_answer(query: str | None, *, query_type: str | None = None) -> bool:
    """True when the turn is asking about the assistant's own previous answer."""
    if query_type and query_type.lower() in _REFERENTIAL_QUERY_TYPES:
        return True
    if _SELF_REF_RE.search(query or ""):
        return True
    return is_reformat_request(query)


def build_prior_answer_block(answer: str, *, max_chars: int = MAX_PRIOR_ANSWER_CHARS) -> str:
    """Wrap the prior answer as pinned context the model must treat as the subject."""
    body = answer.strip()
    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars].rstrip()
    notice = (
        "[PREVIOUS ANSWER - THE SUBJECT OF THIS QUESTION]\n"
        "The user is asking about the answer below, which you wrote earlier in this "
        "conversation. Work from THIS text: expand, translate, reformat or explain "
        "what it actually says. Do not substitute different material for it, and do "
        "not introduce cases or sources that do not appear in it unless the user "
        "asked for new ones.\n"
    )
    if truncated:
        notice += "(Truncated; the earlier part of the answer is shown.)\n"
    return f"{notice}<previous_answer>\n{body}\n</previous_answer>\n\n"


def apply_prior_answer_context(
    context: str,
    query: str | None,
    chat_history: list[dict[str, Any]] | None,
    *,
    query_type: str | None = None,
    max_chars: int = MAX_PRIOR_ANSWER_CHARS,
) -> tuple[str, bool]:
    """Pin the previous answer into context when the turn refers to it.

    Returns `(context, applied)`. Pure, so the decision is unit-testable without
    a pipeline run.
    """
    if not references_prior_answer(query, query_type=query_type):
        return context, False
    answer = last_assistant_message(chat_history)
    if not answer:
        return context, False
    return f"{build_prior_answer_block(answer, max_chars=max_chars)}{context}", True
