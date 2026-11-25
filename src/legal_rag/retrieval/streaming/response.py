"""What one chat turn returns: its SSE frames plus the headers they ride on.

Sibling to `events.py`, which builds the individual frames. This module holds
the envelope around a whole turn's worth of them.

It lives here rather than in `app/api/v1/streaming.py` because the envelope is
not an API concern: the point of `ChatEventStream` is that a turn can be
consumed by something that is not an HTTP request. Keeping it in the route
module meant every consumer had to import from a 14,000-line file to build one.
"""

import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Final

__all__ = ["SSE_HEADERS", "ChatEventStream", "coerce_client_message_id"]

# NOTE: deliberately no `from __future__ import annotations` here. The original
# definition site had none, so `ChatEventStream.__annotations__` held real type
# objects rather than strings, and `dataclasses.fields(...)[0].type` returned the
# `AsyncGenerator` class. Postponing annotations would silently turn that into a
# string, which is a behavior change smuggled into a move.

#: The response headers every SSE path in the chat surface has always sent.
#: Named once so the route is the only place that knows this surface is SSE
#: at all.
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@dataclass(frozen=True, slots=True)
class ChatEventStream:
    """One chat turn, as the event stream it produces plus its headers.

    `stream_chat_events` returns this instead of a `StreamingResponse`, and the
    difference is what makes the whole capability callable from somewhere that
    is not an HTTP request. A `StreamingResponse` can only be returned to
    Starlette; an async iterator of SSE frames can be consumed by a Celery
    worker, which is what `app/services/chat_answer_service.py` does to produce
    a finished answer for the Workspace API.

    The headers ride along rather than being rebuilt by the route because two of
    the twenty-four paths add `X-Conversation-ID` and the rest do not, and
    flattening that difference here would be a behaviour change smuggled into a
    structural one.
    """

    events: AsyncGenerator[str, None]
    headers: dict[str, str]


def coerce_client_message_id(client_id: str | None) -> str:
    """Use the frontend-supplied assistant-message id when it is a valid UUID,
    else mint a fresh one. Adopting the client's id keeps the verification /
    hallucination_metrics ``message_id`` equal to ``chat_messages.id`` so quality
    data joins back to the conversation. Validating guards against a malformed or
    abusive client value polluting the metrics tables."""
    if client_id:
        try:
            return str(uuid.UUID(str(client_id)))
        except (ValueError, AttributeError, TypeError):
            pass
    return str(uuid.uuid4())
