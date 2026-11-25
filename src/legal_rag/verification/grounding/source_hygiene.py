"""Classify and filter the web and document sources shown alongside an answer.

Retrieved source lists often contain scraper telemetry, social media, dictionary pages, marketing blogs and paywalled pages that a reader cannot open.
This module buckets each source URL into a ``SourceTier`` so a caller can drop the unusable ones and rank the rest.

``JUNK`` and ``BLOCKED`` sources are dropped outright: neither can be legal authority.
Aggregators and paywalled hosts are ranked down rather than dropped, because for some opinions an aggregator copy is the only reachable full text.

The host lists are caller supplied.
The defaults only know generic non-authority hosts (social media, dictionaries) and official domains (``.gov``, ``.mil``, legislature hosts).
Pass your own ``free_public``, ``aggregator``, ``paywalled`` and ``blocked`` host sets to ``SourceHygiene`` to tune the ranking.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import IntEnum
from typing import Any
from urllib.parse import urlparse


class SourceTier(IntEnum):
    """Lower is better. The ordering is the ranking, so keep it meaningful."""

    OFFICIAL = 0  # .gov, courts, legislatures, official code publishers
    FREE_PUBLIC = 1  # free, stable, citable public sources
    AGGREGATOR = 2  # third-party mirrors: usable, not primary
    MARKETING = 3  # law-firm blogs, advocacy pages: not authority
    PAYWALLED = 4  # a reader cannot open it
    JUNK = 5  # telemetry, social media, dictionaries: never authority
    BLOCKED = 6  # hosts the caller never wants shown


# Dropped outright. Nothing in these buckets can be legal authority.
DROP_TIERS = frozenset({SourceTier.JUNK, SourceTier.BLOCKED})

_OFFICIAL_RE = re.compile(
    r"(?:^|\.)(?:gov|mil)$|(?:^|\.)legislature\.[a-z]{2}\.us$",
    re.IGNORECASE,
)

DEFAULT_JUNK_HOSTS: frozenset[str] = frozenset(
    {
        "instagram.com",
        "facebook.com",
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "x.com",
        "twitter.com",
        "pinterest.com",
        "quizlet.com",
        "thefreedictionary.com",
        "acronymfinder.com",
        "allacronyms.com",
        "encyclopedia.com",
    }
)

# Scraper telemetry that leaked into a source array; never a document.
_JUNK_PATH_RE = re.compile(r"crawl-ping|/ping\?|value=.*undefined", re.IGNORECASE)
_MARKETING_RE = re.compile(
    r"lawfirm|law-?firm|attorneys?\.com|lawyers?\.com|legalblog|lawblog",
    re.IGNORECASE,
)

# A URL the model half-invented: an unfilled template, an ellipsis, or a bare trailing parameter.
_PLACEHOLDER_URL_RE = re.compile(
    r"\.\.\.|[<>{}\[\]]|\bexample\.(?:com|org)\b|=(?:$|&)|/(?:URL|LINK|TODO)\b",
    re.IGNORECASE,
)

# Markdown inline link: [label](url).
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*(<?)([^)\s]+)\2\s*\)")


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _in(host: str, hosts: Iterable[str]) -> bool:
    return any(host == h or host.endswith("." + h) for h in hosts)


def is_malformed(url: str | None) -> bool:
    """True for entries that are not usable URLs at all (for example ``https://goto``)."""
    if not url or not url.strip():
        return True
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return True
    host = parsed.hostname or ""
    # A hostname with no dot is not routable on the public internet.
    return "." not in host


class SourceHygiene:
    """Bucket source URLs with caller-supplied host lists."""

    def __init__(
        self,
        *,
        free_public: Iterable[str] = (),
        aggregator: Iterable[str] = (),
        paywalled: Iterable[str] = (),
        blocked: Iterable[str] = (),
        junk: Iterable[str] = DEFAULT_JUNK_HOSTS,
    ) -> None:
        self.free_public = frozenset(h.lower() for h in free_public)
        self.aggregator = frozenset(h.lower() for h in aggregator)
        self.paywalled = frozenset(h.lower() for h in paywalled)
        self.blocked = frozenset(h.lower() for h in blocked)
        self.junk = frozenset(h.lower() for h in junk)

    def classify(self, url: str | None) -> SourceTier:
        """Bucket a URL. Unknown hosts are ``AGGREGATOR``: usable, not primary."""
        if is_malformed(url) or (url and _JUNK_PATH_RE.search(url)):
            return SourceTier.JUNK
        host = _host(url or "")
        if not host:
            return SourceTier.JUNK
        if _in(host, self.blocked):
            return SourceTier.BLOCKED
        if _in(host, self.junk):
            return SourceTier.JUNK
        if _in(host, self.paywalled):
            return SourceTier.PAYWALLED
        if _OFFICIAL_RE.search(host):
            return SourceTier.OFFICIAL
        if _in(host, self.free_public):
            return SourceTier.FREE_PUBLIC
        if _in(host, self.aggregator):
            return SourceTier.AGGREGATOR
        if _MARKETING_RE.search(host):
            return SourceTier.MARKETING
        return SourceTier.AGGREGATOR

    def should_drop(self, url: str | None) -> bool:
        """True for sources that must never be shown."""
        return self.classify(url) in DROP_TIERS

    def is_unusable_link(self, url: str | None) -> bool:
        """True when a URL must not be presented as a clickable link."""
        return (
            is_malformed(url)
            or bool(url and _PLACEHOLDER_URL_RE.search(url))
            or self.should_drop(url)
        )

    def filter_sources(
        self, sources: list[dict[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Drop junk and blocked sources. Returns ``(kept, dropped_count)``.

        Sources with no URL at all are KEPT: those are user documents and corpus chunks.
        """
        kept: list[dict[str, Any]] = []
        dropped = 0
        for source in sources or []:
            url = source.get("url") or (source.get("metadata") or {}).get("url")
            if url and self.should_drop(url):
                dropped += 1
                continue
            kept.append(source)
        return kept, dropped

    def sanitize_answer_links(self, text: str | None) -> tuple[str, int]:
        """Unlink citations whose URL is unusable, keeping the citation text intact.

        A citation is still a citation without a hyperlink.
        A citation wearing a DEAD link is worse than one wearing none, because the link asserts a check that did not happen.
        Returns ``(text, unlinked_count)``.
        """
        if not text:
            return text or "", 0
        removed = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal removed
            label, url = match.group(1), match.group(3)
            if self.is_unusable_link(url):
                removed += 1
                return label
            return match.group(0)

        return _MD_LINK_RE.sub(_replace, text), removed

    def prefer_official_url(self, candidates: list[str] | None) -> str | None:
        """Pick the best-tier URL among several for the same authority."""
        usable = [c for c in candidates or [] if c and not is_malformed(c)]
        if not usable:
            return None
        return min(usable, key=lambda u: (self.classify(u), len(u)))


_DEFAULT = SourceHygiene()


def classify_source(url: str | None) -> SourceTier:
    """``SourceHygiene().classify`` with the default host lists."""
    return _DEFAULT.classify(url)


def should_drop(url: str | None) -> bool:
    return _DEFAULT.should_drop(url)


def filter_sources(sources: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], int]:
    return _DEFAULT.filter_sources(sources)


def is_unusable_link(url: str | None) -> bool:
    return _DEFAULT.is_unusable_link(url)


def sanitize_answer_links(text: str | None) -> tuple[str, int]:
    return _DEFAULT.sanitize_answer_links(text)


def prefer_official_url(candidates: list[str] | None) -> str | None:
    return _DEFAULT.prefer_official_url(candidates)
