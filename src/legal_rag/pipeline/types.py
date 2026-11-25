"""Pipeline data types: request, per-query context, context budget, answer scope and usage ledger.

Nothing here carries a tenant, user or billing key.
Isolation is the caller's job: the request holds an opaque ``filters`` mapping, and the caller's
retriever and chunk fetcher apply the tenant filter themselves.
Anything deployment specific goes in a free-form ``extra`` mapping.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from legal_rag.providers.protocols import UsageHook
from legal_rag.types import Message, RetrievedChunk, UsageEvent

# ---------------------------------------------------------------------------
# Vector types shared by the protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector as parallel index and value lists."""

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("SparseVector indices and values must have the same length")


@dataclass
class VectorPoint:
    """One point to write to a vector index: an id, a dense vector, optional sparse, a payload."""

    id: str
    vector: list[float]
    sparse: SparseVector | None = None
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunk metadata keys shared by more than one stage
# ---------------------------------------------------------------------------

PINNED_KEY = "is_pinned"
"""Chunk metadata flag: the chunk was pinned (for example by an exact lookup).

A pinned chunk skips the reranker, is exempt from the relevance gate and the diversity caps, and
is reserved first when the context budget is filled.
Every stage and plug-in reads and writes this one name.
"""

PINNED_BY_KEY = "pinned_by"
"""Chunk metadata key naming what pinned the chunk, for diagnostics. Not read for decisions."""


# ---------------------------------------------------------------------------
# Request and per-query context
# ---------------------------------------------------------------------------


@dataclass
class PipelineRequest:
    """What a caller asks the query pipeline.

    ``filters`` is opaque to the pipeline and passed to the retriever as given.
    ``tier`` may be ``None``, in which case a caller-supplied ``TierResolver`` picks one.
    ``extra`` is free-form and is NOT part of the answer scope, so it must never change what the
    answer is. Put every answer-changing input in ``filters`` or one of the scope fields.
    """

    query: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    tier: str | None = None
    retrieval_strategy: str | None = None
    chat_history: Sequence[Mapping[str, Any] | Message] = field(default_factory=tuple)
    collection: str | None = None
    pinned_ids: Sequence[str] = field(default_factory=tuple)
    scope_lock_ids: Sequence[str] = field(default_factory=tuple)
    boost_ids: Sequence[str] = field(default_factory=tuple)
    enable_shared_corpus: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# Request fields that are deliberately not part of the answer scope.
# ``query`` is keyed by the cache itself, ``extra`` must not change the answer.
ANSWER_SCOPE_EXEMPT_REQUEST_FIELDS: frozenset[str] = frozenset({"query", "extra"})


@dataclass
class QueryContext:
    """The state one query carries through the stages.

    A stage is ``async def stage(ctx: QueryContext) -> QueryContext``.
    Scope fields:

    - ``pinned_ids``: the user pinned these documents, so search is restricted to them.
    - ``scope_lock_ids``: search is restricted to these case or document ids.
    - ``boost_ids``: fetch these ids and merge them with the normal results. Not a lock.
    - ``enable_shared_corpus``: whether the shared reference corpus is searched.
    - ``collection``: name of the collection to search instead of the default one.
    """

    query: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    tier: str = "standard"
    retrieval_strategy: str = "hybrid"
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)
    context_chunks: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    retrieval_warning: str | None = None
    pinned_ids: list[str] = field(default_factory=list)
    scope_lock_ids: list[str] = field(default_factory=list)
    boost_ids: list[str] = field(default_factory=list)
    enable_shared_corpus: bool = True
    collection: str | None = None
    chat_history: list[Mapping[str, Any] | Message] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls,
        request: PipelineRequest,
        *,
        tier: str,
        retrieval_strategy: str,
    ) -> QueryContext:
        """Start a context from a request, with the tier and strategy already resolved."""
        return cls(
            query=request.query,
            filters=request.filters,
            tier=tier,
            retrieval_strategy=retrieval_strategy,
            pinned_ids=list(request.pinned_ids),
            scope_lock_ids=list(request.scope_lock_ids),
            boost_ids=list(request.boost_ids),
            enable_shared_corpus=request.enable_shared_corpus,
            collection=request.collection,
            chat_history=list(request.chat_history),
            extra=dict(request.extra),
        )

    def add_warning(self, message: str) -> None:
        """Record a degraded-result warning once."""
        if message not in self.warnings:
            self.warnings.append(message)

    def answer_scope(self) -> AnswerScope:
        """The answer scope of this context, for a cache key."""
        return AnswerScope.build(
            tier=self.tier,
            retrieval_strategy=self.retrieval_strategy,
            enable_shared_corpus=self.enable_shared_corpus,
            collection=self.collection,
            pinned_ids=self.pinned_ids,
            scope_lock_ids=self.scope_lock_ids,
            boost_ids=self.boost_ids,
            filters=self.filters,
            chat_history=self.chat_history,
        )


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBudget:
    """Token and chunk limits for the context handed to the answering model."""

    total_tokens: int = 48000
    reserve_tokens: int = 1000
    max_chunks: int = 60
    max_chunk_tokens: int = 2000

    def __post_init__(self) -> None:
        if self.total_tokens <= 0 or self.max_chunks <= 0 or self.max_chunk_tokens <= 0:
            raise ValueError("ContextBudget limits must be positive")
        if not 0 <= self.reserve_tokens < self.total_tokens:
            raise ValueError("ContextBudget reserve_tokens must be in [0, total_tokens)")

    @property
    def available_tokens(self) -> int:
        """Tokens left for retrieved context after the reserve for prompts and instructions."""
        return self.total_tokens - self.reserve_tokens


# ---------------------------------------------------------------------------
# Answer scope
# ---------------------------------------------------------------------------

# Every input that can change the answer. A filter that does not reach the cache key is a
# wrong answer served from cache, so this list is closed: a missing or unknown name raises.
ANSWER_SCOPE_FIELDS: tuple[str, ...] = (
    "tier",
    "retrieval_strategy",
    "enable_shared_corpus",
    "collection",
    "pinned_ids",
    "scope_lock_ids",
    "boost_ids",
    "filters",
    "chat_history",
)
_BOOL_FIELDS = frozenset({"enable_shared_corpus"})
_ID_SET_FIELDS = frozenset({"pinned_ids", "scope_lock_ids", "boost_ids"})
# A follow-up such as "what about section 5?" is condensed against the recent turns, so the
# same literal text is a different question in a different conversation.
_CHAT_HISTORY_TURNS = 6
_ABSENT: Any = object()


def _canonical(value: Any) -> Any:
    """Normalize a filter value. Returns ``_ABSENT`` for None, blank strings and empty containers.

    ``False`` and ``0`` are kept: for an arbitrary filter they can differ from an unset filter.
    List order is kept too, so two orders miss the cache instead of risking a wrong answer.
    """
    if value is None:
        return _ABSENT
    if isinstance(value, str):
        text = value.strip()
        return text if text else _ABSENT
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value, key=str):
            item = _canonical(value[key])
            if item is not _ABSENT:
                out[str(key)] = item
        return out or _ABSENT
    if isinstance(value, set | frozenset):
        items = [_canonical(v) for v in value]
        items = sorted((i for i in items if i is not _ABSENT), key=lambda v: json.dumps(v))
        return items or _ABSENT
    if isinstance(value, Iterable) and not isinstance(value, bytes | bytearray):
        items = [_canonical(v) for v in value]
        items = [i for i in items if i is not _ABSENT]
        return items or _ABSENT
    return str(value)


def _history_digest(history: Sequence[Mapping[str, Any] | Message] | None) -> str | None:
    """Short digest of the recent turns, or ``None`` when there is no history."""
    if not history:
        return None
    turns: list[list[str]] = []
    for turn in list(history)[-_CHAT_HISTORY_TURNS:]:
        if isinstance(turn, Message):
            turns.append([turn.role, turn.content])
        elif isinstance(turn, Mapping):
            turns.append([str(turn.get("role") or ""), str(turn.get("content") or "")])
    if not turns:
        return None
    blob = json.dumps(turns, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class AnswerScope:
    """The canonical set of answer-changing inputs, for building a cache key.

    Build it with ``AnswerScope.build`` (or ``build_answer_scope``).
    Every name in ``ANSWER_SCOPE_FIELDS`` must be passed, ``None`` is fine, and no other name may
    be. ``None``, ``False``, blank strings and empty containers all normalize to "absent", so a
    request that leaves a field unset and one that sends its default share a key, and nothing
    else does.
    """

    values: Mapping[str, Any]

    @classmethod
    def build(cls, **inputs: Any) -> AnswerScope:
        """Build a scope from keyword inputs.

        Raises:
            TypeError: a registered field is missing or an unknown field is given.
        """
        missing = [name for name in ANSWER_SCOPE_FIELDS if name not in inputs]
        unknown = sorted(set(inputs) - set(ANSWER_SCOPE_FIELDS))
        if missing or unknown:
            raise TypeError(f"AnswerScope.build: missing={missing} unknown={unknown}")

        scope: dict[str, Any] = {}
        for name in ANSWER_SCOPE_FIELDS:
            value = inputs[name]
            if name in _BOOL_FIELDS:
                if value:
                    scope[name] = True
            elif name in _ID_SET_FIELDS:
                ids = sorted({str(v).strip() for v in (value or ()) if str(v).strip()})
                if ids:
                    scope[name] = ids
            elif name == "chat_history":
                digest = _history_digest(value)
                if digest:
                    scope[name] = digest
            else:
                item = _canonical(value)
                if item is not _ABSENT:
                    scope[name] = item
        return cls(values=scope)

    @classmethod
    def from_request(
        cls,
        request: PipelineRequest,
        *,
        tier: str,
        retrieval_strategy: str,
    ) -> AnswerScope:
        """Scope of a request, using the already resolved tier and strategy."""
        return cls.build(
            tier=tier,
            retrieval_strategy=retrieval_strategy,
            enable_shared_corpus=request.enable_shared_corpus,
            collection=request.collection,
            pinned_ids=request.pinned_ids,
            scope_lock_ids=request.scope_lock_ids,
            boost_ids=request.boost_ids,
            filters=request.filters,
            chat_history=request.chat_history,
        )

    def to_dict(self) -> dict[str, Any]:
        """A plain copy of the normalized values."""
        return dict(self.values)

    def key(self) -> str:
        """Stable short digest of the normalized values."""
        blob = json.dumps(self.values, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def build_answer_scope(**inputs: Any) -> AnswerScope:
    """Shorthand for ``AnswerScope.build``."""
    return AnswerScope.build(**inputs)


def request_scope_field_names() -> frozenset[str]:
    """Request fields that must be registered in ``ANSWER_SCOPE_FIELDS``.

    A test uses this to fail when a new ``PipelineRequest`` field is added without deciding
    whether it changes the answer.
    """
    return frozenset(f.name for f in fields(PipelineRequest)) - ANSWER_SCOPE_EXEMPT_REQUEST_FIELDS


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


class UsageLedger:
    """Collects ``UsageEvent`` objects emitted during one query or ingest run.

    The ledger is itself a ``UsageHook``: pass it wherever a hook is accepted.
    It never prices anything. Pricing is the caller's job, applied to the events it collects.
    """

    def __init__(self, downstream: UsageHook | None = None) -> None:
        self._events: list[UsageEvent] = []
        self._lock = threading.Lock()
        self._downstream = downstream

    def __call__(self, event: UsageEvent) -> None:
        self.record(event)

    def record(self, event: UsageEvent) -> None:
        """Store an event, then forward it to the downstream hook when one was given."""
        with self._lock:
            self._events.append(event)
        if self._downstream is not None:
            self._downstream(event)

    @property
    def events(self) -> list[UsageEvent]:
        """A snapshot copy of the collected events."""
        with self._lock:
            return list(self._events)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens or 0 for e in self.events)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens or 0 for e in self.events)

    def counts_by_operation(self) -> dict[str, int]:
        """How many events each operation emitted."""
        return dict(Counter(e.operation for e in self.events))

    def counts_by_provider(self) -> dict[str, int]:
        """How many events each provider emitted. Events without a provider are skipped."""
        return dict(Counter(e.provider for e in self.events if e.provider))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
