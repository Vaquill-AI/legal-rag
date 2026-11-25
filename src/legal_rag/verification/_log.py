"""Minimal logger that accepts structured keyword fields on top of the standard library.

``log.info("event", count=3)`` renders as ``event count=3`` through ``logging``.
Components use it so that no logging framework is required.
"""

from __future__ import annotations

import logging
from typing import Any


class _Logger:
    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, *args: Any, **fields: Any) -> None:
        if not self._log.isEnabledFor(level):
            return
        exc_info = fields.pop("exc_info", None)
        try:
            message = event % args if args else event
        except (TypeError, ValueError):
            message = event
        if fields:
            message += " " + " ".join(f"{k}={v!r}" for k, v in fields.items())
        self._log.log(level, message, exc_info=exc_info)

    def debug(self, event: str, *args: Any, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, *args, **fields)

    def info(self, event: str, *args: Any, **fields: Any) -> None:
        self._emit(logging.INFO, event, *args, **fields)

    def warning(self, event: str, *args: Any, **fields: Any) -> None:
        self._emit(logging.WARNING, event, *args, **fields)

    def error(self, event: str, *args: Any, **fields: Any) -> None:
        self._emit(logging.ERROR, event, *args, **fields)

    def exception(self, event: str, *args: Any, **fields: Any) -> None:
        fields.setdefault("exc_info", True)
        self._emit(logging.ERROR, event, *args, **fields)


def get_logger(name: str) -> _Logger:
    return _Logger(name)
