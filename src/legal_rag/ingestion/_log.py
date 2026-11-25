"""Minimal key-value logger over the standard library.

Ingestion modules log an event name plus keyword fields.
This wrapper keeps that call style while using only ``logging``, so a host
application can route the records through any handler it likes.
"""

from __future__ import annotations

import logging
from typing import Any

_RESERVED = {"exc_info", "stack_info", "stacklevel"}


class KeyValueLogger:
    """Wrap a stdlib logger so ``logger.info("event", key=value)`` works."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, **fields: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        options = {key: fields.pop(key) for key in list(fields) if key in _RESERVED}
        message = event
        if fields:
            message += " " + " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.log(level, message, **options)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    warn = warning

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        fields.setdefault("exc_info", True)
        self._log(logging.ERROR, event, **fields)


def get_logger(name: str) -> KeyValueLogger:
    """Return a key-value logger for ``name``."""
    return KeyValueLogger(name)
