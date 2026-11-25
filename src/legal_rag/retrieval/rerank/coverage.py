"""Retrieval coverage honesty gate.

When a user asks about their own documents and retrieval returns none of them, answering
anyway from general reference material, without saying so, is worse than an admission.
This module decides when to say so and builds the notice to prepend to the model context.

Which source types count as the user's own material and which as shared reference material is
configuration (``CoverageConfig``), not code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_OWN_MATERIAL_RE = re.compile(
    r"\b(?:my|our|the|this|that|these|those|attached|uploaded|provided|enclosed)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:document|documents|doc|file|files|contract|agreement|lease|complaint|answer|"
    r"motion|brief|petition|filing|filings|record|records|paperwork|exhibit|exhibits|"
    r"attachment|upload|pdf|letter|notice|memo|deposition|transcript|pleading|"
    r"affidavit|declaration|report|policy|invoice|statement)\b"
    r"|\b(?:in|from|per|according to|based on) (?:the|my|our|this|these|that|those|"
    r"attached|uploaded) (?:\w+\s+){0,2}?"
    r"(?:document|file|contract|record|filing|paperwork|upload|attachment)\b",
    re.IGNORECASE,
)

OWN_DOCUMENTS_EMPTY_NOTICE = (
    "[RETRIEVAL NOTICE - READ FIRST]\n"
    "No documents supplied by this user were retrieved for this question. "
    "Everything in the sources below is general reference material, NOT the "
    "user's own filings, contracts or records.\n"
    "You MUST:\n"
    "1. Tell the user plainly, at the start of your answer, that nothing was found "
    "in their documents for this question.\n"
    "2. NOT describe the material below as coming from their documents.\n"
    "3. NOT present a statute or rule below as governing their situation unless it is "
    "from the correct jurisdiction for their question.\n"
    "4. Suggest what would help (re-uploading the document, checking that it processed, "
    "or naming the specific filing) instead of answering as if you had read it.\n\n"
)


@dataclass(frozen=True)
class CoverageConfig:
    """Which source types are the user's own and which are shared reference material."""

    own_source_types: frozenset[str] = frozenset({"upload", "user_document"})
    reference_source_types: frozenset[str] = frozenset({"statute", "case", "reference", "web"})
    notice: str = OWN_DOCUMENTS_EMPTY_NOTICE
    own_material_pattern: re.Pattern[str] = field(default=_OWN_MATERIAL_RE)


DEFAULT_COVERAGE_CONFIG = CoverageConfig()


def source_type_of(item: Mapping[str, Any] | None) -> str:
    """A source's type, read from the top level or from nested ``metadata``."""
    if not item:
        return ""
    return str(item.get("source_type") or (item.get("metadata") or {}).get("source_type") or "")


def has_own_documents(
    sources: Sequence[Mapping[str, Any]] | None, config: CoverageConfig = DEFAULT_COVERAGE_CONFIG
) -> bool:
    """True when at least one retrieved source is the user's own document."""
    return any(source_type_of(s) in config.own_source_types for s in sources or [])


def has_reference_sources(
    sources: Sequence[Mapping[str, Any]] | None, config: CoverageConfig = DEFAULT_COVERAGE_CONFIG
) -> bool:
    """True when at least one retrieved source is shared reference material."""
    return any(source_type_of(s) in config.reference_source_types for s in sources or [])


def own_docs_missing(
    sources: Sequence[Mapping[str, Any]] | None,
    *,
    own_docs_requested: bool,
    config: CoverageConfig = DEFAULT_COVERAGE_CONFIG,
) -> bool:
    """True when the user's documents were searched, none came back, and reference material did.

    With no sources at all the pipeline already degrades honestly, so no notice is needed.
    """
    if not own_docs_requested or not sources:
        return False
    return not has_own_documents(sources, config) and has_reference_sources(sources, config)


def references_own_material(
    query: str | None, config: CoverageConfig = DEFAULT_COVERAGE_CONFIG
) -> bool:
    """True when the question is about the user's own documents, so a "not found" notice is on point."""
    return bool(query and config.own_material_pattern.search(query))


def apply_coverage_notice(
    context: str,
    sources: Sequence[Mapping[str, Any]] | None,
    *,
    own_docs_requested: bool,
    query: str | None = None,
    docs_pinned: bool = False,
    config: CoverageConfig = DEFAULT_COVERAGE_CONFIG,
) -> tuple[str, bool]:
    """Prepend the notice when own documents were requested but missing.

    Returns ``(context, notice_applied)``. The notice fires only when the turn actually involves
    the user's own material (documents pinned, or the question refers to them); a general legal
    question must not be answered with "nothing found in your documents".
    """
    if not own_docs_missing(sources, own_docs_requested=own_docs_requested, config=config):
        return context, False
    if not (docs_pinned or references_own_material(query, config)):
        return context, False
    return f"{config.notice}{context}", True
