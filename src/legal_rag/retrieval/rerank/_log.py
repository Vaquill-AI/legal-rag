"""A tiny logging adapter that accepts structured keyword fields on top of stdlib logging."""

from __future__ import annotations

import logging
from typing import Any


class KeywordLogger:
    """Wrap a stdlib logger so ``log.info("event", key=value)`` works without extra setup."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, fields: dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        exc_info = fields.pop("exc_info", None)
        suffix = " ".join(f"{k}={v!r}" for k, v in fields.items())
        self._logger.log(
            level,
            "%s %s" if suffix else "%s",
            event,
            *([suffix] if suffix else []),
            exc_info=exc_info,
        )

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)


def get_logger(name: str) -> KeywordLogger:
    """Logger under the ``legal_rag`` namespace."""
    return KeywordLogger(f"legal_rag.{name}")
