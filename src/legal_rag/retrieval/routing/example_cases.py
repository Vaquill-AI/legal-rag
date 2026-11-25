"""A small neutral labeled set for ``example_taxonomy``.

It shows how to build a case set: each case carries the state it must be judged in, because
many intents are only valid in certain states.
"""

from __future__ import annotations

from legal_rag.retrieval.routing.eval import Case
from legal_rag.retrieval.routing.state import PinnedDoc, RouterState

_NO_DOCS = RouterState()
_ONE_DOC = RouterState(pinned_docs=(PinnedDoc("d0", "opinion.pdf"),))
_TWO_DOCS = RouterState(pinned_docs=(PinnedDoc("d0", "v1.pdf"), PinnedDoc("d1", "v2.pdf")))
_CAN_EXPORT = RouterState(flags=frozenset({"can_export"}))

EXAMPLE_CASES: tuple[Case, ...] = (
    Case(
        "What is the statute of limitations for breach of contract?", _NO_DOCS, "answer", tag="qa"
    ),
    Case(
        "Explain the difference between void and voidable contracts", _NO_DOCS, "answer", tag="qa"
    ),
    Case("hello", _NO_DOCS, "small_talk", tag="chat"),
    Case("Summarize this opinion", _ONE_DOC, "summarize", tag="doc"),
    Case("Compare these two versions", _TWO_DOCS, "compare", tag="doc"),
    Case("Compare these two versions", _ONE_DOC, "answer", tag="state", accept=("summarize",)),
    Case("Pull out every party and date", _ONE_DOC, "extract", tag="doc"),
    Case(
        "Find cases on adverse possession",
        _NO_DOCS,
        "find_authority",
        accept=("answer",),
        tag="research",
    ),
    Case("Write this up as a research memo and save it", _CAN_EXPORT, "export_report", tag="tool"),
    Case("Write this up as a research memo and save it", _NO_DOCS, "answer", tag="state"),
)
