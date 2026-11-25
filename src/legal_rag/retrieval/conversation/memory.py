"""Conversation memory over an injected cache.

Sessions are stored as JSON under a key built from a caller-chosen ``scope`` string and a
session id. The scope is opaque: it can be a user, a tenant, a project or empty. The cache is
any object satisfying the ``Cache`` protocol, so nothing here depends on a particular backend.

The ``Cache`` protocol has get, set and delete but no scan or expire. This module therefore:

- clears a session by deleting its key;
- keeps a small per-scope index key to list sessions;
- extends a session's lifetime by re-writing it.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from legal_rag.providers.protocols import Cache
from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("conversation.memory")


@dataclass
class ConversationTurn:
    """Single conversation turn."""

    user: str
    assistant: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        return cls(
            user=data.get("user", ""),
            assistant=data.get("assistant", ""),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            metadata=data.get("metadata", {}),
        )

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to message list format."""
        return [
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


@dataclass
class ConversationSession:
    """Complete conversation session."""

    scope: str
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationSession:
        return cls(
            scope=data.get("scope", ""),
            session_id=data.get("session_id", ""),
            turns=[ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            summary=data.get("summary"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
        )

    def to_chat_history(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Convert to chat history format for LLM."""
        messages = []
        recent_turns = self.turns[-max_turns:]
        for turn in recent_turns:
            messages.extend(turn.to_messages())
        return messages


class ConversationManager:
    """Manages conversation state in an injected cache.

    Features:
    - Scope and session isolation
    - Automatic TTL (24 hours by default)
    - Turn-count truncation
    - Graceful degradation when the cache is unavailable or fails
    """

    DEFAULT_TTL = timedelta(hours=24)
    DEFAULT_MAX_TURNS = 50
    _MAX_INDEX = 200

    def __init__(
        self,
        cache: Cache | None = None,
        ttl: timedelta | None = None,
        max_turns: int | None = None,
    ):
        """
        Args:
            cache: Any ``Cache`` implementation. ``None`` turns every call into a no-op.
            ttl: Session lifetime (default: 24 hours).
            max_turns: Max turns kept per session.
        """
        self._cache = cache
        self.ttl = ttl or self.DEFAULT_TTL
        self.max_turns = max_turns or self.DEFAULT_MAX_TURNS

    @property
    def _ttl_seconds(self) -> int:
        return int(self.ttl.total_seconds())

    def _make_key(self, scope: str, session_id: str) -> str:
        return f"conv:{scope}:{session_id}"

    def _index_key(self, scope: str) -> str:
        return f"conv-index:{scope}"

    async def _load(self, scope: str, session_id: str) -> ConversationSession | None:
        if self._cache is None:
            return None
        raw = await self._cache.get(self._make_key(scope, session_id))
        if not raw:
            return None
        return ConversationSession.from_dict(json.loads(raw))

    async def _store(self, session: ConversationSession) -> None:
        assert self._cache is not None
        await self._cache.set(
            self._make_key(session.scope, session.session_id),
            json.dumps(session.to_dict()).encode("utf-8"),
            ttl_seconds=self._ttl_seconds,
        )

    async def _index_add(self, scope: str, session_id: str) -> None:
        assert self._cache is not None
        key = self._index_key(scope)
        raw = await self._cache.get(key)
        ids: list[str] = json.loads(raw) if raw else []
        if session_id in ids:
            ids.remove(session_id)
        ids.append(session_id)
        await self._cache.set(
            key, json.dumps(ids[-self._MAX_INDEX :]).encode("utf-8"), ttl_seconds=self._ttl_seconds
        )

    async def add_turn(
        self,
        scope: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a conversation turn. Returns True on success."""
        if self._cache is None:
            logger.warning("memory_add_turn_no_cache")
            return False
        try:
            turn = ConversationTurn(
                user=user_message, assistant=assistant_message, metadata=metadata or {}
            )
            session = await self._load(scope, session_id) or ConversationSession(
                scope=scope, session_id=session_id
            )
            session.turns.append(turn)
            session.updated_at = datetime.now(UTC).isoformat()
            if len(session.turns) > self.max_turns:
                session.turns = session.turns[-self.max_turns :]
            await self._store(session)
            await self._index_add(scope, session_id)
            logger.debug("memory_turn_added", session_id=session_id, turns_count=len(session.turns))
            return True
        except Exception as e:
            logger.error("memory_add_turn_failed", error=str(e))
            return False

    async def get_history(
        self,
        scope: str,
        session_id: str,
        max_turns: int = 10,
        format: str = "messages",
    ) -> list[dict]:
        """Get history as ``[{role, content}]`` ("messages") or ``[{user, assistant}]`` ("turns")."""
        try:
            session = await self._load(scope, session_id)
            if session is None:
                return []
            if format == "messages":
                return session.to_chat_history(max_turns)
            return [t.to_dict() for t in session.turns[-max_turns:]]
        except Exception as e:
            logger.error("memory_get_history_failed", error=str(e))
            return []

    async def get_session(self, scope: str, session_id: str) -> ConversationSession | None:
        """Get the full conversation session."""
        try:
            return await self._load(scope, session_id)
        except Exception as e:
            logger.error("memory_get_session_failed", error=str(e))
            return None

    async def clear_session(self, scope: str, session_id: str) -> bool:
        """Clear a session by deleting its key and dropping it from the scope index."""
        if self._cache is None:
            return False
        try:
            await self._cache.delete(self._make_key(scope, session_id))
            key = self._index_key(scope)
            raw = await self._cache.get(key)
            ids: list[str] = [i for i in (json.loads(raw) if raw else []) if i != session_id]
            await self._cache.set(
                key, json.dumps(ids).encode("utf-8"), ttl_seconds=self._ttl_seconds
            )
            logger.info("memory_session_cleared", session_id=session_id)
            return True
        except Exception as e:
            logger.error("memory_clear_session_failed", error=str(e))
            return False

    async def list_sessions(self, scope: str, limit: int = 20) -> list[str]:
        """List session ids for a scope, most recent first, skipping cleared ones."""
        if self._cache is None:
            return []
        try:
            raw = await self._cache.get(self._index_key(scope))
            ids: list[str] = json.loads(raw) if raw else []
            live = [i for i in reversed(ids) if await self._cache.get(self._make_key(scope, i))]
            return live[:limit]
        except Exception as e:
            logger.error("memory_list_sessions_failed", error=str(e))
            return []

    async def extend_ttl(self, scope: str, session_id: str) -> bool:
        """Extend session lifetime by re-writing it."""
        try:
            session = await self._load(scope, session_id)
            if session is None:
                return False
            await self._store(session)
            return True
        except Exception as e:
            logger.error("memory_extend_ttl_failed", error=str(e))
            return False
