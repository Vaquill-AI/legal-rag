"""A thin adapter that lets stdlib loggers accept structured keyword fields.

``get_logger(name).info("event", key=value)`` logs ``event key=value`` through the standard
``logging`` module, so library code can log structured events without a logging dependency.
"""

from __future__ import annotations

import logging
from typing import Any


class KVLogger:
    """Wrap a stdlib logger so calls may carry ``key=value`` fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, fields: dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        exc_info = fields.pop("exc_info", None)
        fields.pop("extra", None)
        suffix = " ".join(f"{k}={v}" for k, v in fields.items())
        self._logger.log(level, f"{event} {suffix}".strip(), exc_info=exc_info)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        fields.setdefault("exc_info", True)
        self._log(logging.ERROR, event, fields)


def get_logger(name: str) -> KVLogger:
    return KVLogger(logging.getLogger(f"legal_rag.retrieval.{name}"))
