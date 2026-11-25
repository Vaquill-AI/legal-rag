"""Shared base for actors: an injected language model, metrics and error wrapping.

An actor is one model call with a fixed role, a prompt built from constants and a typed result.
The model arrives through the constructor, so nothing here reads configuration or the environment.
"""

from __future__ import annotations

import inspect
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import ValidationError

from legal_rag.providers.protocols import LanguageModel, T
from legal_rag.providers.usage import estimate_tokens
from legal_rag.types import Message, RetrievedChunk
from legal_rag.verification._log import get_logger
from legal_rag.verification.grounding.actor_schemas import ActorMetrics

#: Metadata keys that may carry a human readable source name, in lookup order.
SOURCE_NAME_KEYS = ("source_name", "document_name", "filename", "title")


@dataclass(frozen=True)
class ActorConfig:
    """Per-actor model call limits.

    ``temperature`` is passed to ``generate_structured`` only when the injected model accepts a
    ``temperature`` argument, because the provider protocol does not require one.
    """

    max_tokens: int = 2000
    temperature: float = 0.3


class ActorError(Exception):
    """The model call failed, or the actor failed in an unexpected way."""


class ActorOutputError(ActorError):
    """The model replied, but the reply did not validate against the requested schema."""


def normalize_chunks(
    chunks: Sequence[RetrievedChunk | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return chunks as dicts that carry ``content`` and ``text``, ``metadata`` and ``score``.

    The actors read ``content`` and the hallucination detector reads ``text``.
    Both keys are filled from whichever one the caller supplied.
    """
    normalized: list[dict[str, Any]] = []
    for chunk in chunks or ():
        if isinstance(chunk, RetrievedChunk):
            entry: dict[str, Any] = {
                "id": chunk.id,
                "content": chunk.text,
                "text": chunk.text,
                "score": chunk.score,
                "metadata": dict(chunk.metadata),
            }
        else:
            entry = dict(chunk)
            text = entry.get("content") or entry.get("text") or ""
            entry["content"] = text
            entry["text"] = text
            entry["score"] = entry.get("score", 0.0)
            entry["metadata"] = dict(entry.get("metadata") or {})
        normalized.append(entry)
    return normalized


def source_name(chunk: Mapping[str, Any]) -> str | None:
    """The first non-empty source name found in a chunk's metadata, else None."""
    metadata = chunk.get("metadata") or {}
    for key in SOURCE_NAME_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _accepts_temperature(method: Any) -> bool:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "temperature" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class BaseActor(ABC):
    """Abstract base class for every actor.

    Subclasses set ``DEFAULT_CONFIG``, implement ``actor_name`` and ``execute``, and call
    ``call_structured`` for each model call so that metrics stay accurate.
    """

    DEFAULT_CONFIG: ClassVar[ActorConfig] = ActorConfig()

    def __init__(self, model: LanguageModel, config: ActorConfig | None = None) -> None:
        self.model = model
        self.config = config or self.DEFAULT_CONFIG
        self.logger = get_logger(f"legal_rag.agents.actors.{self.actor_name}")
        self._tokens_used = 0
        self._llm_calls = 0

    @property
    @abstractmethod
    def actor_name(self) -> str:
        """Unique identifier for this actor, such as ``planning``."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Core actor logic."""

    async def run_with_metrics(self, **kwargs: Any) -> tuple[Any, ActorMetrics]:
        """Run ``execute`` and return ``(result, ActorMetrics)``.

        A ``ValueError`` (a missing argument) and an ``ActorError`` propagate unchanged.
        Any other exception is wrapped in ``ActorError``.
        """
        start = time.perf_counter()
        error_message: str | None = None
        result: Any = None
        self._tokens_used = 0
        self._llm_calls = 0

        try:
            self.logger.info("actor_execution_started", **_safe_log_fields(kwargs))
            result = await self.execute(**kwargs)
            self.logger.info("actor_execution_completed")
        except ActorError as exc:
            error_message = str(exc)
            self.logger.error("actor_model_error", error_message=error_message)
            raise
        except ValueError as exc:
            error_message = str(exc)
            self.logger.error("actor_validation_error", error_message=error_message)
            raise
        except Exception as exc:
            error_message = str(exc)
            self.logger.error(
                "actor_unexpected_error",
                error_type=type(exc).__name__,
                error_message=error_message,
            )
            raise ActorError(f"Actor execution failed: {error_message}") from exc
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics = ActorMetrics(
                actor_name=self.actor_name,
                execution_time_ms=elapsed_ms,
                tokens_used=self._tokens_used,
                llm_calls=self._llm_calls,
                cache_hit=False,
                error_occurred=error_message is not None,
                error_message=error_message,
            )
            self.logger.info(
                "actor_metrics",
                execution_time_ms=elapsed_ms,
                tokens_used=self._tokens_used,
                llm_calls=self._llm_calls,
                error_occurred=error_message is not None,
            )

        return result, metrics

    async def call_structured(self, messages: Sequence[Message], schema: type[T]) -> T:
        """Ask the model for an instance of ``schema`` and account for the call.

        Raises ``ActorOutputError`` when the reply does not validate and ``ActorError`` for any
        other model failure.
        """
        kwargs: dict[str, Any] = {"max_tokens": self.config.max_tokens}
        if _accepts_temperature(self.model.generate_structured):
            kwargs["temperature"] = self.config.temperature

        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        self.logger.debug(
            "llm_call_started",
            schema=schema.__name__,
            max_tokens=self.config.max_tokens,
            message_count=len(messages),
        )
        try:
            result = await self.model.generate_structured(messages, schema, **kwargs)
        except (ValidationError, json.JSONDecodeError) as exc:
            self._llm_calls += 1
            self._tokens_used += prompt_tokens
            raise ActorOutputError(f"Model output did not match {schema.__name__}: {exc}") from exc
        except Exception as exc:
            self.logger.error(
                "llm_call_failed", schema=schema.__name__, error_type=type(exc).__name__
            )
            raise ActorError(f"Model call failed: {exc!s}") from exc

        self._llm_calls += 1
        self._tokens_used += prompt_tokens + estimate_tokens(result.model_dump_json())
        return result


def _safe_log_fields(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Log sizes and short scalars, never response text or chunk content."""
    fields: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in ("chunks", "available_chunks"):
            fields["chunks_count"] = len(value) if value else 0
        elif key == "query":
            fields["query_length"] = len(value) if value else 0
        elif isinstance(value, str):
            if len(value) > 100:
                fields[f"{key}_length"] = len(value)
            else:
                fields[key] = value
        elif isinstance(value, (int, float, bool, type(None))):
            fields[key] = value
        elif isinstance(value, (list, dict)):
            fields[f"{key}_count"] = len(value)
        else:
            fields[f"has_{key}"] = value is not None
    return fields
