"""Structured-style logging on top of the standard library.

Chunking code logs events as ``logger.info("event_name", key=value)``.
This adapter keeps that call style while using plain ``logging`` underneath,
so an application can route the records with its own handlers and formatters.
Fields are attached as ``extra={"fields": {...}}`` and appended to the message.
"""

from __future__ import annotations

import logging
from typing import Any

_ROOT = "legal_rag.chunking"


class EventLogger:
    """A thin wrapper that accepts keyword fields on every logging call."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        if not self._logger.isEnabledFor(level):
            return
        suffix = " ".join(f"{k}={v!r}" for k, v in fields.items())
        message = f"{event} {suffix}".rstrip()
        self._logger.log(level, message, extra={"fields": fields}, exc_info=exc_info)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        exc_info = bool(fields.pop("exc_info", False))
        self._log(logging.ERROR, event, fields, exc_info=exc_info)

    def exception(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields, exc_info=True)


def get_logger(name: str | None = None) -> EventLogger:
    """Return an event logger under the ``legal_rag.chunking`` namespace."""
    suffix = name.rsplit(".", 1)[-1] if name else ""
    full = f"{_ROOT}.{suffix}" if suffix else _ROOT
    return EventLogger(logging.getLogger(full))
