"""Jurisdiction plugins: opt-in retrieval behavior for one body of law.

A plugin implements ``JurisdictionPlugin`` (see ``legal_rag.pipeline.protocols``) and is looked up
by name here. The US plugin is registered as ``"us"``. Nothing is selected for you: a caller
asks for a plugin by name, builds it with its own retrievers and lookups, and hands it to the
pipeline.

    from legal_rag.pipeline import jurisdictions

    plugin = jurisdictions.get("us", case_retriever=cases, statute_retriever=statutes)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from legal_rag.pipeline.protocols import JurisdictionPlugin

PluginFactory = Callable[..., "JurisdictionPlugin"]
"""Builds a plugin. Keyword arguments given to ``get`` are passed straight through."""


class UnknownJurisdictionError(KeyError):
    """Raised by ``get`` for a name nobody registered."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown jurisdiction"


_REGISTRY: dict[str, PluginFactory] = {}


def _normalize(name: str) -> str:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("a jurisdiction name must be a non-empty string")
    return key


def register(name: str, plugin_factory: PluginFactory, *, replace: bool = False) -> None:
    """Register a plugin factory under ``name`` (case-insensitive).

    A name that is already taken is rejected unless ``replace=True``, so two packages cannot
    silently shadow each other.
    """
    key = _normalize(name)
    if not callable(plugin_factory):
        raise TypeError("plugin_factory must be callable")
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"jurisdiction {key!r} is already registered; pass replace=True to override"
        )
    _REGISTRY[key] = plugin_factory


def unregister(name: str) -> None:
    """Remove a registered name. Unknown names raise ``UnknownJurisdictionError``."""
    key = _normalize(name)
    if key not in _REGISTRY:
        raise UnknownJurisdictionError(_unknown_message(key))
    del _REGISTRY[key]


def get(name: str, **kwargs: Any) -> JurisdictionPlugin:
    """Build the plugin registered as ``name``, passing ``kwargs`` to its factory."""
    from legal_rag.pipeline.protocols import JurisdictionPlugin

    key = _normalize(name)
    factory = _REGISTRY.get(key)
    if factory is None:
        raise UnknownJurisdictionError(_unknown_message(key))
    plugin = factory(**kwargs)
    if not isinstance(plugin, JurisdictionPlugin):
        raise TypeError(f"factory for jurisdiction {key!r} did not return a JurisdictionPlugin")
    return plugin


def available() -> tuple[str, ...]:
    """Registered names, sorted."""
    return tuple(sorted(_REGISTRY))


def _unknown_message(key: str) -> str:
    return f"unknown jurisdiction {key!r}; available: {', '.join(available()) or 'none'}"


def _us_factory(**kwargs: Any) -> JurisdictionPlugin:
    from legal_rag.pipeline.jurisdictions.us import USJurisdiction

    return USJurisdiction(**kwargs)


register("us", _us_factory)

__all__ = [
    "PluginFactory",
    "UnknownJurisdictionError",
    "available",
    "get",
    "register",
    "unregister",
]
