"""Source-list deduplication and stable ordering.

Retrieval often returns several chunks of the same case, statute or document, and the same
authority can arrive twice under different ids (a synthetic canonical entry and the real
record). These helpers collapse such duplicates into one source for display and citation.

Dedup key precedence:

1. ``group_id`` (an authoritative upstream id, for example a court-record cluster id).
2. Normalised case name plus normalised citation.
3. ``document_id``.
4. Normalised name-only or citation-only fallback.
5. Identity (no merge).

First occurrence wins on metadata, the longest excerpt is kept as content, and URL fields are
merged. All functions are idempotent. Sources may be dicts or objects with attributes.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

URL_FIELDS: tuple[str, ...] = (
    "pdf_url",
    "external_url",
    "url",
    "html_url",
    "xml_url",
    "file_url",
)
"""Fields merged from a duplicate into the survivor when the survivor lacks them."""


def _get(src: Any, key: str, default: Any = None) -> Any:
    if isinstance(src, dict):
        return src.get(key, (src.get("metadata") or {}).get(key, default))
    return getattr(src, key, (getattr(src, "metadata", None) or {}).get(key, default))


def _set(src: Any, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(src, dict):
        src[key] = value
        src.setdefault("metadata", {})[key] = value
    else:
        with contextlib.suppress(Exception):  # frozen or slotted objects are left unchanged
            setattr(src, key, value)


def _content(src: Any) -> str:
    if isinstance(src, dict):
        return src.get("content") or src.get("text") or ""
    return getattr(src, "text", "") or getattr(src, "content", "") or ""


def _set_content(src: Any, value: str) -> None:
    if isinstance(src, dict):
        src["content"] = value
        src["text"] = value
        return
    for attr in ("text", "content"):
        if hasattr(src, attr):
            with contextlib.suppress(Exception):  # leave frozen objects unchanged
                setattr(src, attr, value)
            return


def normalise_case_name(s: str) -> str:
    """Collapse case-name variants to one key ("Acme Inc. v. Doe" and "Acme v Doe" match)."""
    s = str(s or "").lower()
    s = re.sub(r"\bvs?\b\.?", "v", s)
    s = re.sub(r"\b(inc|llc|llp|corp|corporation|co|enterprises|enters|ltd)\.?\b", "", s)
    s = re.sub(r"[,.;:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:80]


def normalise_citation(s: str) -> str:
    """Collapse reporter-citation format variants ("123 U.S. 456 (2020)" and "123 US 456")."""
    s = str(s or "").lower()
    s = re.sub(r"\(\d{4}\)", "", s)
    s = re.sub(r"\bu\.\s*s\b\.?", "us", s)
    s = re.sub(r"\bf\.\s*(\d)d\b", r"f\1d", s)
    s = re.sub(r"\bs\.\s*ct\b\.?", "sct", s)
    s = re.sub(r"\bl\.\s*ed\.?(\s*2d)?\b", lambda m: "led2d" if m.group(1) else "led", s)
    s = re.sub(r"[,.;:§]", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:60]


def _name_of(src: Any) -> str:
    return _get(src, "case_name") or _get(src, "title") or _get(src, "filename") or ""


def _cite_of(src: Any) -> str:
    return _get(src, "primary_citation") or _get(src, "citation") or ""


def dedup_key(src: Any) -> str:
    """The key under which ``src`` is considered a duplicate."""
    group = _get(src, "group_id")
    if group:
        return f"group:{group}"
    norm_name = normalise_case_name(_name_of(src))
    norm_cite = normalise_citation(_cite_of(src))
    if norm_name and norm_cite:
        return f"name+cite:{norm_name}|{norm_cite}"
    doc_id = _get(src, "document_id")
    if doc_id:
        return f"doc:{doc_id}"
    if norm_name or norm_cite:
        return f"name:{norm_name}|cite:{norm_cite}"
    return f"id:{id(src)}"


def _merge_into(survivor: Any, src: Any) -> None:
    if len(_content(src)) > len(_content(survivor)):
        _set_content(survivor, _content(src))
    for fld in URL_FIELDS:
        if not _get(survivor, fld) and _get(src, fld):
            _set(survivor, fld, _get(src, fld))


def dedupe_sources(sources: list) -> list:
    """Collapse multiple chunks of the same case or statute into one source.

    A second pass merges an uncited entry into a cited one with the same normalised name,
    so "Acme v. Doe (123 F.3d 45)" and "Acme v. Doe" (no citation) become one.
    """
    seen: dict[str, Any] = {}
    order: list[str] = []
    for src in sources:
        key = dedup_key(src)
        if key not in seen:
            seen[key] = src
            order.append(key)
        else:
            _merge_into(seen[key], src)

    cited_by_name: dict[str, str] = {}
    uncited: list[str] = []
    for key in order:
        norm = normalise_case_name(_name_of(seen[key]))
        if not norm:
            continue
        if normalise_citation(_cite_of(seen[key])):
            cited_by_name.setdefault(norm, key)
        else:
            uncited.append(key)

    drops: set[str] = set()
    for key in uncited:
        norm = normalise_case_name(_name_of(seen[key]))
        if norm in cited_by_name and cited_by_name[norm] != key:
            _merge_into(seen[cited_by_name[norm]], seen[key])
            drops.add(key)
    return [seen[k] for k in order if k not in drops]


def stabilize_pinned_source_order(sources: list, scoped_document_ids: list[str] | None) -> list:
    """Put scoped documents first, in the caller's pin order, and keep the rest in place.

    Without this, source numbers follow per-turn rerank score, so the same pinned document is
    renumbered between turns and citations drift. Order only: nothing is dropped or mutated.
    """
    if not scoped_document_ids or not sources:
        return sources
    rank = {doc_id: i for i, doc_id in enumerate(scoped_document_ids)}
    pinned: list[tuple[int, int, Any]] = []
    rest: list = []
    for pos, src in enumerate(sources):
        doc_id = _get(src, "document_id")
        if doc_id in rank:
            pinned.append((rank[doc_id], pos, src))
        else:
            rest.append(src)
    if not pinned:
        return sources
    pinned.sort(key=lambda t: (t[0], t[1]))
    return [src for _, _, src in pinned] + rest


def build_chunk_to_post_dedup_remap(
    sources_pre_dedup: list, sources_post_dedup: list
) -> list[int | None]:
    """Map each pre-dedup source to its 1-indexed position after dedup, or ``None`` if dropped.

    Lets a context numbered before dedup be relabelled so every citation resolves to a
    source that is still shown.
    """
    key_to_post: dict[str, int] = {}
    name_to_post: dict[str, int] = {}
    for post_idx, src in enumerate(sources_post_dedup, start=1):
        key_to_post.setdefault(dedup_key(src), post_idx)
        norm = normalise_case_name(_name_of(src))
        if norm:
            name_to_post.setdefault(norm, post_idx)
    remap: list[int | None] = []
    for src in sources_pre_dedup:
        key = dedup_key(src)
        if key in key_to_post:
            remap.append(key_to_post[key])
            continue
        norm = normalise_case_name(_name_of(src))
        remap.append(name_to_post.get(norm) if norm else None)
    return remap


__all__ = [
    "build_chunk_to_post_dedup_remap",
    "dedup_key",
    "dedupe_sources",
    "normalise_case_name",
    "normalise_citation",
    "stabilize_pinned_source_order",
]
