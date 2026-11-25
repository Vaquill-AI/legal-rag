"""Combined US query analysis: one model call for jurisdiction, doctrine and area.

Each query can need up to three separate classification calls (jurisdiction, doctrine,
substantive area). When two or more of those dimensions miss their keyword path and a language
model is injected, this module replaces the separate calls with one combined call whose single
view of the whole query keeps the three signals mutually consistent.

Design:
- The keyword fast paths always run first (cheap, deterministic).
- The combined path needs an injected model and at least two keyword misses. Otherwise the
  three independent classifiers run.
- Everything fails open to the keyword verdict.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Final

from legal_rag.providers.protocols import Cache, LanguageModel
from legal_rag.retrieval.routing.doctrine_detector import (
    _LLM_DOCTRINE_MIN_CONFIDENCE,
    DoctrineMatch,
    _doctrine_catalog_for_prompt,
    detect_doctrine,
    detect_doctrine_fast,
    load_doctrines,
)
from legal_rag.retrieval.routing.jurisdiction_classifier import (
    _LLM_FALLBACK_THRESHOLD,
    Classification,
    _classify_keyword,
    classify_jurisdiction,
)
from legal_rag.retrieval.routing.topic_filters import (
    AreaResult,
    SubstantiveArea,
    _result_for,
    classify_substantive_area,
    classify_substantive_area_fast,
)
from legal_rag.types import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CombinedUSAnalysis:
    """The three US query signals, resolved together."""

    jurisdiction: Classification
    doctrine: DoctrineMatch | None
    area: AreaResult


def _norm_key(q: str) -> str:
    norm = re.sub(r"\s+", " ", (q or "").strip().lower())[:512]
    return hashlib.sha256(norm.encode()).hexdigest()[:32]


async def analyze_us_query(
    query: str, *, llm: LanguageModel | None = None, cache: Cache | None = None
) -> CombinedUSAnalysis:
    """Resolve jurisdiction, doctrine and substantive area for a US query.

    Runs the keyword fast paths, then (when a model is injected and two or more dimensions
    missed) one combined model call for the missing dimensions; otherwise the three
    independent classifiers. Fails open to the keyword verdict.
    """
    jur_kw = _classify_keyword(query)
    doc_kw = detect_doctrine_fast(query)
    area_kw = classify_substantive_area_fast(query)

    if llm is None:
        return CombinedUSAnalysis(jur_kw, doc_kw, area_kw)

    want_jur = jur_kw.confidence < _LLM_FALLBACK_THRESHOLD
    want_doc = doc_kw is None
    want_area = area_kw.area is SubstantiveArea.GENERAL

    if sum((want_jur, want_doc, want_area)) >= 2:
        return await _combined_path(
            query, jur_kw, doc_kw, area_kw, want_jur, want_doc, want_area, llm, cache
        )
    return await _individual_path(query, llm, cache)


async def _individual_path(
    query: str, llm: LanguageModel, cache: Cache | None
) -> CombinedUSAnalysis:
    """Three independent classifiers, run concurrently."""
    jur, doc, area = await asyncio.gather(
        classify_jurisdiction(query, llm=llm, cache=cache),
        detect_doctrine(query, llm=llm, cache=cache),
        classify_substantive_area(query, llm=llm, cache=cache),
    )
    return CombinedUSAnalysis(jur, doc, area)


async def _combined_path(
    query: str,
    jur_kw: Classification,
    doc_kw: DoctrineMatch | None,
    area_kw: AreaResult,
    want_jur: bool,
    want_doc: bool,
    want_area: bool,
    llm: LanguageModel,
    cache: Cache | None,
) -> CombinedUSAnalysis:
    """One combined LLM call for the missing dimensions; keyword hits preserved.

    A confident doctrine is the single source of truth for jurisdiction + area
    (consistency by construction, mirroring the keyword canon path): its curated
    canon tags override the LLM's separately-stated jurisdiction/area for the
    missing dimensions, so the combined verdict can never contradict itself the
    way three independent calls can. Dimensions the doctrine does not cover fall
    back to the LLM's direct fields. Fails open to the keyword verdict.
    """
    data = await _combined_llm(query, llm, cache)
    jur, doc, area = jur_kw, doc_kw, area_kw
    if data is None:
        return CombinedUSAnalysis(jur, doc, area)

    if want_doc:
        resolved_doc = _doctrine_from_llm(data)
        if resolved_doc is not None:
            doc = resolved_doc

    jur_set = area_set = False
    if doc is not None:
        if want_jur and doc.doctrine.jurisdiction in ("federal", "state"):
            jur = Classification(
                doc.doctrine.jurisdiction,
                doc.confidence,
                f"canonical doctrine: {doc.doctrine.id}",
                "llm",
            )
            jur_set = True
        if want_area:
            derived = _area_result(doc.doctrine.substantive_area, doc.confidence)
            if derived is not None:
                area = derived
                area_set = True

    if want_jur and not jur_set:
        resolved = _jurisdiction_from_llm(data)
        if resolved is not None:
            jur = resolved
    if want_area and not area_set:
        resolved_area = _area_from_llm(data)
        if resolved_area is not None:
            area = resolved_area

    return CombinedUSAnalysis(jur, doc, area)


def _jurisdiction_from_llm(data: dict[str, Any]) -> Classification | None:
    label = str(data.get("jurisdiction", "")).strip().lower()
    if label not in ("federal", "state", "ambiguous"):
        return None
    conf = _clamp(data.get("jurisdiction_confidence", data.get("confidence", 0.7)))
    return Classification(label, conf, "combined US analysis", "llm")


def _doctrine_from_llm(data: dict[str, Any]) -> DoctrineMatch | None:
    doctrine_id = data.get("doctrine_id")
    if not doctrine_id or str(doctrine_id).lower() in ("", "none", "null"):
        return None
    conf = _clamp(data.get("doctrine_confidence", data.get("confidence", 0.5)))
    # Same 0.6 floor the standalone doctrine fallback uses: a low-confidence
    # guess must not force-include canonical authority into retrieval.
    if conf < _LLM_DOCTRINE_MIN_CONFIDENCE:
        return None
    for d in load_doctrines():
        if d.id == doctrine_id:
            return DoctrineMatch(doctrine=d, confidence=conf, source="llm", matched_alias=None)
    logger.warning(
        "us_query_analysis.combined_unknown_doctrine", extra={"id": str(doctrine_id)[:60]}
    )
    return None


def _area_result(raw_area: Any, conf: float) -> AreaResult | None:
    """Build an AreaResult from a substantive-area string; None if unknown/general."""
    raw = str(raw_area).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        area = SubstantiveArea(raw)
    except ValueError:
        return None
    if area is SubstantiveArea.GENERAL:
        return None
    return _result_for(area, "llm", conf)


def _area_from_llm(data: dict[str, Any]) -> AreaResult | None:
    conf = _clamp(data.get("area_confidence", data.get("confidence", 0.7)))
    return _area_result(data.get("area", ""), conf)


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


_AREA_VALUES: Final[str] = ", ".join(a.value for a in SubstantiveArea)

_COMBINED_SYSTEM: Final[str] = (
    "You analyze a US legal question and return three fields in one JSON object.\n"
    "\n"
    "1. jurisdiction: 'federal', 'state', or 'ambiguous'. Most everyday legal "
    "questions are STATE law; choose 'federal' only when the question genuinely "
    "turns on federal law.\n"
    "   - state (the common case) = landlord-tenant/leases/security deposits, "
    "most contracts, torts and personal injury, real property and HOAs, family "
    "law and divorce, wills/trusts/probate, business formation and director "
    "duties, non-competes, at-will employment, and most crimes.\n"
    "   - federal = turns on the US Constitution (1st/2nd/4th/5th/14th-Amendment, "
    "equal protection, commerce clause, due process), a federal statute or "
    "agency (patent, copyright, federal trademark, securities, bankruptcy, "
    "immigration, Title VII, ERISA, antitrust, RICO), federal court rules "
    "(FRCP/FRE), or a named federal doctrine (qualified immunity, Brady, "
    "Twombly, Erie, Chevron/Loper Bright).\n"
    "   - ambiguous = genuinely unclear with no clear federal hook. When there "
    "is no federal hook, prefer 'state' for a private-law matter; do NOT default "
    "to 'federal'.\n"
    "2. doctrine_id: the single best-matching id from the doctrine catalog "
    "below, or null if none clearly applies. Only choose an id that squarely "
    "fits; prefer null over a weak guess.\n"
    f"3. area: one of [{_AREA_VALUES}], or 'general' if none clearly fits.\n"
    "\n"
    'Return STRICT JSON: {"jurisdiction": "...", "jurisdiction_confidence": 0-1, '
    '"doctrine_id": "..."|null, "doctrine_confidence": 0-1, "area": "...", '
    '"area_confidence": 0-1}. Keep the three fields mutually consistent.'
)


async def _combined_llm(
    query: str, llm: LanguageModel, cache: Cache | None
) -> dict[str, Any] | None:
    """One model call returning jurisdiction, doctrine_id and area, or None."""
    key = "us_analysis:" + _norm_key(query)
    if cache is not None:
        try:
            hit = await cache.get(key)
            if hit:
                cached = json.loads(hit)
                return cached if isinstance(cached, dict) else None
        except Exception:
            logger.warning("us_query_analysis.cache_read_failed")
    result = await _combined_llm_uncached(query, llm)
    if cache is not None and result is not None:
        try:
            await cache.set(key, json.dumps(result).encode(), ttl_seconds=3600)
        except Exception:
            logger.warning("us_query_analysis.cache_write_failed")
    return result


async def _combined_llm_uncached(query: str, llm: LanguageModel) -> dict[str, Any] | None:
    try:
        catalog = _doctrine_catalog_for_prompt()
        raw = await asyncio.wait_for(
            llm.generate(
                [
                    Message("system", _COMBINED_SYSTEM),
                    Message("user", f"{catalog}\n\n## Question\n{query[:1000]}"),
                ],
                max_tokens=1024,
                temperature=0.0,
            ),
            timeout=12.0,
        )
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("us_query_analysis.combined_failed", extra={"err": str(exc)[:120]})
        return None
