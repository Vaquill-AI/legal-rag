"""
Topic-aware filters for the US statutes corpus.

The US statute collection mixes USC (federal code) and CFR (federal
regulations). For some question topics, CFR results are pure noise: a
question about FRCP 56 summary judgment doesn't want a hit on
``43 C.F.R. § 4.111`` (Bureau of Land Management hearings procedures) just
because that section happens to be titled "Summary judgment". This module
detects those topics and tells the retriever to suppress one corpus.

Round 2 extends the simple civil-procedure suppression with a
**substantive-area classifier** that maps the question to the right USC
and CFR title set. For example, a "What is the Alice/Mayo two-step?"
question should retrieve only Title 35 USC + 37 CFR (patent), not 26 CFR
(tax) just because the IRS transfer-pricing reg shares the word
"transfer".

Two-stage classification, mirroring jurisdiction_classifier:

1. **Fast keyword path**: substring/regex match against known doctrine
   names, USC/CFR citations, and topical phrases. ~0ms, deterministic.

2. **LLM fallback** (injected language model): only fires when the keyword path
   returns ``GENERAL``. The LLM picks one of the known areas (or
   GENERAL) and returns the allowed USC/CFR title set.

Each area declares the canonical USC titles and CFR titles that contain
its substantive law. The retriever uses these to whitelist hits.

Source: Cornell LII USC / CFR title indexes, Wikipedia "List of titles of
the United States Code", and the eCFR title catalog.
"""
# ruff: noqa: BLE001

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib import resources
from typing import Any, Final

from legal_rag.providers.protocols import Cache, LanguageModel
from legal_rag.types import Message

logger = logging.getLogger(__name__)


def _load_data() -> dict[str, Any]:
    with (
        resources.files("legal_rag.retrieval.routing")
        .joinpath("data/topic_filters.json")
        .open("r", encoding="utf-8") as fh
    ):
        return json.load(fh)


_DATA = _load_data()


# ---------------------------------------------------------------------------
# Civil-procedure path (round 1, kept verbatim, still relied on by retriever)
# ---------------------------------------------------------------------------

_CIVIL_PROCEDURE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bFRCP\b", re.IGNORECASE),
    re.compile(r"\bF\.?\s*R\.?\s*Civ\.?\s*P\.?\b", re.IGNORECASE),
    re.compile(r"\bFRE\b|\bF\.?\s*R\.?\s*Evid\.?\b", re.IGNORECASE),
    re.compile(r"\bFRAP\b|\bF\.?\s*R\.?\s*App\.?\s*P\.?\b", re.IGNORECASE),
    re.compile(r"\bFed(?:eral)?\s+Rule\s+(?:of\s+)?(?:Civ|Crim|Evid|App)", re.IGNORECASE),
    re.compile(r"\brule\s*(?:11|12|15|23|26|37|41|45|50|54|56|59|60)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:summary\s+judgment|motion\s+to\s+dismiss|class\s+action|"
        r"diversity\s+jurisdiction|personal\s+jurisdiction|removal|"
        r"transfer\s+of\s+venue|venue\s+transfer|forum\s+non\s+conveniens|"
        r"joinder|interpleader|impleader|interlocutory|"
        r"discovery\s+sanctions|preclusion|res\s+judicata|collateral\s+estoppel)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:twombly|iqbal|erie|hanna|celotex|matsushita|anderson\s+v\.?\s+liberty)\b",
        re.IGNORECASE,
    ),
)


_ADMIN_PROC_CFR_TITLES: Final[frozenset[int]] = frozenset(_DATA["admin_proc_cfr_titles"])


def is_civil_procedure_question(query: str) -> bool:
    """Return True when the query is clearly about federal civil litigation procedure."""
    q = (query or "").strip()
    if not q:
        return False
    return any(p.search(q) for p in _CIVIL_PROCEDURE_PATTERNS)


def is_admin_proc_cfr_title(cfr_title_number: int | None) -> bool:
    """Return True for CFR titles that are administrative-procedure heavy."""
    return cfr_title_number in _ADMIN_PROC_CFR_TITLES


# ---------------------------------------------------------------------------
# Substantive-area classification (round 2)
# ---------------------------------------------------------------------------


class SubstantiveArea(str, Enum):
    """Coarse legal-area buckets that drive USC/CFR title filtering.

    Names are stable; downstream code matches on the string value.
    """

    PATENT = "patent"  # 35 USC + 37 CFR
    COPYRIGHT = "copyright"  # 17 USC + 37 CFR (parts 201-205)
    TRADEMARK = "trademark"  # 15 USC ch 22 + 37 CFR part 2
    SECURITIES = "securities"  # 15 USC 1933/1934 acts + 17 CFR
    TAX = "tax"  # 26 USC + 26 CFR
    LABOR = "labor"  # 29 USC + 29 CFR
    IMMIGRATION = "immigration"  # 8 USC + 8 CFR
    BANKRUPTCY = "bankruptcy"  # 11 USC
    ANTITRUST = "antitrust"  # 15 USC (Sherman/Clayton) + 16 CFR
    ENVIRONMENTAL = "environmental"  # 42 USC ch85/103/116 + 40 CFR
    CRIMINAL = "criminal"  # 18 USC + (some) 28 CFR
    CIVIL_RIGHTS = "civil_rights"  # 42 USC §1983 et seq + multiple CFR
    FOOD_DRUG = "food_drug"  # 21 USC + 21 CFR
    COMMUNICATIONS = "communications"  # 47 USC + 47 CFR
    TRANSPORTATION = "transportation"  # 49 USC + 49 CFR (also 14 USC/CFR)
    BANKING = "banking"  # 12 USC + 12 CFR
    ENERGY = "energy"  # 42 USC + 10 CFR + 18 CFR (FERC)
    HEALTHCARE = "healthcare"  # 42 USC + 42 CFR (CMS/HHS)
    EDUCATION = "education"  # 20 USC + 34 CFR
    CIVIL_PROCEDURE = "civil_procedure"  # FRCP / 28 USC (jurisdiction & venue)
    CONSTITUTIONAL = "constitutional"  # mostly case law, but 42 USC §1983 ties in
    EVIDENCE = "evidence"  # FRE / 28 USC
    GENERAL = "general"  # unclassified


_USC_TITLE_ALLOWLIST: Final[dict[SubstantiveArea, frozenset[int] | None]] = {
    SubstantiveArea(k): (None if v is None else frozenset(v))
    for k, v in _DATA["usc_title_allowlist"].items()
}


_CFR_TITLE_ALLOWLIST: Final[dict[SubstantiveArea, frozenset[int] | None]] = {
    SubstantiveArea(k): (None if v is None else frozenset(v))
    for k, v in _DATA["cfr_title_allowlist"].items()
}


# Keyword/regex cues per area for the fast path. Order matters: first match
# wins, so put narrow areas before broad ones (e.g. PATENT before GENERAL).
_AREA_PATTERNS: Final[tuple[tuple[SubstantiveArea, tuple[re.Pattern[str], ...]], ...]] = (
    (
        SubstantiveArea.PATENT,
        (
            re.compile(
                r"\b(?:patent(?:s|ability|eligibility|able)?|"
                r"35\s*U\.?S\.?C\.?\s*(?:§\s*)?(?:101|102|103|112|271|283|284|285)|"
                r"alice/?mayo|alice\s+corp|alice\s+v|mayo\s+collaborative|bilski|"
                r"§\s*101\b|"
                r"obviousness|anticipation|prior\s+art|claim\s+construction|markman|"
                r"inter\s+partes\s+review|\bIPR\b|PTAB|USPTO|patent\s+infringement)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.COPYRIGHT,
        (
            re.compile(
                r"\b(?:copyright(?:s|able|ed|ing)?|"
                r"17\s*U\.?S\.?C\.?\s*(?:§\s*)?(?:101|102|106|107|108|109|110|114|115|512)|"
                r"fair\s+use|DMCA|berne|first\s+sale|"
                # 🔴 "safe harbor" was here BARE until 2026-09-07, and it is the
                # most cross-cutting term this table has ever contained: tax
                # elections, ERISA plan designs, the PSLRA, anti-kickback
                # exceptions and worker classification all use it, and the DMCA
                # 512 sense is one of many. Every one of those questions came
                # back `copyright`.
                #
                # That deleted the answer rather than mislabelling it. The area
                # verdict is a HARD ALLOW-LIST -- copyright means 17 USC / 37
                # CFR only -- so `_search_statutes` dropped the question's own
                # corpus before ranking. Measured: `26 C.F.R. 1.263(a)-1` was at
                # dedupe rank 3 on a de-minimis safe-harbor question and was
                # thrown away.
                #
                # ⚠️ A wrong area is FINAL. `classify_substantive_area` is
                # keyword-first and calls the LLM only on a GENERAL miss, so
                # nothing downstream can correct a bad keyword hit. A missed
                # area costs one LLM call; a wrong one costs the answer. Keep
                # every cue here narrow enough that only a copyright question
                # can trip it.
                #
                # Qualified forms only. `DMCA`, `copyright` and the 17 U.S.C.
                # citations already match on their own, so these exist for the
                # DMCA question that never says "DMCA".
                r"(?:DMCA|copyright|service\s+provider|"
                r"(?:§\s*|section\s+)?512)\s+safe\s+harbors?|"
                r"notice[-\s]and[-\s]takedown|takedown\s+notice|"
                r"reproduction\s+right|public\s+performance|derivative\s+work|"
                r"works?\s+made\s+for\s+hire)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.TRADEMARK,
        (
            re.compile(
                r"\b(?:trademark|service\s+mark|trade\s+dress|lanham\s+act|"
                r"15\s*U\.?S\.?C\.?\s*§\s*11(?:14|25)|"
                r"likelihood\s+of\s+confusion|dilution|tarnishment|secondary\s+meaning|"
                r"distinctiveness|generic\s+mark|abercrombie\s+spectrum)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.SECURITIES,
        (
            re.compile(
                r"\b(?:securit(?:y|ies)\s+fraud|rule\s+10b-?5|10b-5|"
                r"insider\s+trading|PSLRA|scienter|"
                r"1933\s+act|1934\s+act|securities\s+act|exchange\s+act|"
                # 🔴 `section 11` and `section 12` were BARE here until
                # 2026-09-07. Those are the two most common section numbers in
                # any drafted document: a collective bargaining agreement's
                # section 11, a lease's section 12, a partnership agreement's
                # section 11. Each claimed 15 USC / 17 CFR and deleted the
                # title that actually answered the question.
                #
                # The Securities Act senses are real, but they never arrive
                # bare -- they come with "Securities Act", "1933 Act" or a
                # liability/claim noun, and the first two already match on
                # their own above.
                r"(?:securities\s+act|1933\s+act|'?33\s+act)\s+section\s+1[12]\b|"
                r"section\s+1[12](?:\([a-z0-9)(]+\))?\s+"
                r"(?:claim|liabilit(?:y|ies)|action|defense|defendant)|"
                r"materiality\s+securities|disclosure\s+securities|"
                r"15\s*U\.?S\.?C\.?\s*§?\s*7[78]|SEC\s+v\.)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.TAX,
        (
            re.compile(
                r"\b(?:tax(?:es|ation|able|payer|payers)?|"
                r"26\s*U\.?S\.?C\.?|internal\s+revenue\s+code|\bIRC\b|"
                r"\bIRS\b|tax\s+court|deduction|deductible|capital\s+gains|"
                r"transfer\s+pricing|tax\s+shelter)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.LABOR,
        (
            re.compile(
                r"\b(?:FMLA|FLSA|title\s+VII|ADEA|ADA\s+employment|"
                r"NLRA|NLRB|labor\s+relations|"
                r"29\s*U\.?S\.?C\.?|29\s*CFR|"
                r"wrongful\s+termination|workplace\s+harassment|"
                r"mcdonnell\s+douglas|sexual\s+harassment|disparate\s+(?:treatment|impact)|"
                r"employment\s+discrimination|hostile\s+work\s+environment)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.IMMIGRATION,
        (
            re.compile(
                r"\b(?:immigration|deport(?:ation|able|ed)?|asylum|refugee|"
                r"INA\b|8\s*U\.?S\.?C\.?|naturalization|visa|green\s+card|"
                r"removal\s+proceedings|cancellation\s+of\s+removal|"
                r"adjustment\s+of\s+status|\bDACA\b|\bTPS\b)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.BANKRUPTCY,
        (
            re.compile(
                r"\b(?:bankruptcy|chapter\s+(?:7|11|12|13)|11\s*U\.?S\.?C\.?|"
                r"automatic\s+stay|dischargeable|nondischargeable|"
                # 🔴 `trustee` was BARE here until 2026-09-07 and claimed
                # 11 + 28 USC for every kind of trustee in American law: ERISA
                # plan trustees (29 USC), indenture trustees (15 USC), trustees
                # of a charitable remainder trust (26 USC), plain state-law
                # fiduciaries. Same hard-allow-list consequence as the
                # `safe harbor` case above: the question's own corpus is
                # dropped before ranking, and the LLM fallback never runs
                # because the keyword already "succeeded".
                #
                # `bankruptcy` and `chapter 7|11|12|13` already match the
                # bankruptcy sense, so the bare word carried only false
                # positives. These qualified forms exist for the two that it
                # was genuinely carrying: the United States Trustee and the
                # panel/standing trustees, who are named without either.
                r"(?:united\s+states|U\.?S\.?|panel|standing|bankruptcy)\s+trustee|"
                r"trustee\s+in\s+bankruptcy|"
                r"adversary\s+proceeding|cram\s*down)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.ANTITRUST,
        (
            re.compile(
                r"\b(?:antitrust|sherman\s+act|clayton\s+act|FTC\s+act|"
                r"price\s+fixing|monopol(?:y|ize|ization|istic)|"
                r"hart-?scott-?rodino|\bHSR\b|tying\s+arrangement|"
                r"vertical\s+restraint|horizontal\s+restraint|"
                r"rule\s+of\s+reason|per\s+se\s+illegal|"
                r"15\s*U\.?S\.?C\.?\s*§?\s*[127])\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.ENVIRONMENTAL,
        (
            re.compile(
                r"\b(?:CERCLA|RCRA|NEPA|CWA|CAA|TSCA|"
                r"clean\s+(?:air|water)\s+act|"
                r"endangered\s+species|environmental\s+impact|"
                r"superfund|hazardous\s+waste|EPA\s+regulation|"
                r"40\s*CFR|chevron\s+deference\s+EPA)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.CIVIL_RIGHTS,
        (
            re.compile(
                r"\b(?:42\s*U\.?S\.?C\.?\s*§?\s*1983|section\s+1983|"
                r"qualified\s+immunity|monell|absolute\s+immunity|"
                r"civil\s+rights\s+act|equal\s+protection|"
                r"due\s+process\s+clause|fourteenth\s+amendment|"
                r"§\s*1981|§\s*1985|§\s*1988|"
                r"voting\s+rights\s+act|fair\s+housing\s+act|"
                r"title\s+IX|disparate\s+impact\s+race)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.CRIMINAL,
        (
            re.compile(
                r"\b(?:18\s*U\.?S\.?C\.?|federal\s+criminal|"
                r"brady\s+(?:rule|disclosure)|strickland|miranda|"
                r"fourth\s+amendment|search\s+(?:and\s+)?seizure|"
                r"exclusionary\s+rule|fruit\s+of\s+the\s+poisonous|"
                r"sixth\s+amendment|right\s+to\s+counsel|"
                r"ineffective\s+assistance|"
                r"RICO|18\s*U\.?S\.?C\.?\s*§?\s*1962|"
                r"federal\s+drug|controlled\s+substances\s+act|\bCSA\b|"
                r"actual\s+possession|constructive\s+possession)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.FOOD_DRUG,
        (
            re.compile(
                r"\b(?:FDA|food.{0,10}drug\s+act|FDCA|"
                r"21\s*U\.?S\.?C\.?|21\s*CFR|"
                r"hatch-?waxman|orange\s+book|"
                r"new\s+drug\s+application|\bNDA\b|\bANDA\b|"
                r"medical\s+device|drug\s+labeling)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.COMMUNICATIONS,
        (
            re.compile(
                r"\b(?:FCC|telecommunications\s+act|"
                r"47\s*U\.?S\.?C\.?|47\s*CFR|"
                r"net\s+neutrality|section\s+230|"
                r"communications\s+decency|CDA\b)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.TRANSPORTATION,
        (
            re.compile(
                r"\b(?:FAA\s+regulation|aviation\s+law|airline\s+deregulation|"
                r"NHTSA|department\s+of\s+transportation\s+regulation|"
                r"49\s*U\.?S\.?C\.?|49\s*CFR|14\s*CFR)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.BANKING,
        (
            re.compile(
                r"\b(?:bank\s+holding|federal\s+reserve\s+act|"
                r"12\s*U\.?S\.?C\.?|12\s*CFR|"
                r"truth\s+in\s+lending|TILA|dodd-?frank|"
                r"national\s+bank\s+act|OCC\b|FDIC)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.ENERGY,
        (
            re.compile(
                r"\b(?:FERC|federal\s+energy|natural\s+gas\s+act|"
                r"federal\s+power\s+act|nuclear\s+regulatory|NRC\b|"
                r"10\s*CFR|18\s*CFR)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.HEALTHCARE,
        (
            re.compile(
                r"\b(?:HIPAA|medicare|medicaid|ACA\s+aca|affordable\s+care|"
                r"EMTALA|ERISA\s+health|stark\s+law|anti-?kickback|"
                r"42\s*U\.?S\.?C\.?\s*§?\s*1320|42\s*CFR|"
                r"\bCMS\b|HHS\b)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.EDUCATION,
        (
            re.compile(
                r"\b(?:IDEA\s+education|FERPA|title\s+IX\s+education|"
                r"individuals\s+with\s+disabilities\s+education|"
                r"20\s*U\.?S\.?C\.?|34\s*CFR|"
                r"department\s+of\s+education)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    # EVIDENCE comes BEFORE CIVIL_PROCEDURE so "FRE 802" routes here, not to
    # CIVIL_PROCEDURE (whose pattern set also recognises \bFRE\b).
    (
        SubstantiveArea.EVIDENCE,
        (
            re.compile(
                r"\b(?:hearsay|FRE\s+\d|federal\s+rules?\s+of\s+evidence|"
                r"rule\s+(?:401|403|404|407|801|802|803|804|901|902))\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        SubstantiveArea.CIVIL_PROCEDURE,
        _CIVIL_PROCEDURE_PATTERNS,
    ),
    (
        SubstantiveArea.CONSTITUTIONAL,
        (
            re.compile(
                r"\b(?:first\s+amendment|free\s+speech|content[-\s]based|"
                r"content[-\s]neutral|time\s+place\s+manner|"
                r"second\s+amendment|bruen|heller|"
                r"fifth\s+amendment\s+takings|takings\s+clause|"
                r"establishment\s+clause|free\s+exercise|"
                r"commerce\s+clause|wickard|lopez|raich)\b",
                re.IGNORECASE,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class AreaResult:
    """The classifier's verdict for a question."""

    area: SubstantiveArea
    allowed_usc_titles: frozenset[int] | None  # None = no filter
    allowed_cfr_titles: frozenset[int] | None  # None = no filter
    source: str  # "keyword" | "llm" | "default"
    confidence: float


def _result_for(area: SubstantiveArea, source: str, confidence: float) -> AreaResult:
    return AreaResult(
        area=area,
        allowed_usc_titles=_USC_TITLE_ALLOWLIST[area],
        allowed_cfr_titles=_CFR_TITLE_ALLOWLIST[area],
        source=source,
        confidence=confidence,
    )


def _doctrine_area(query: str) -> SubstantiveArea | None:
    """The canon doctrine's substantive_area if the query names one and that
    area is specific (not GENERAL), else None.

    Single source of truth: the area is read off `canonical_doctrines.json` via
    the shared fast alias matcher, so a doctrine-named query gets the same area
    the doctrine detector and jurisdiction classifier see. Pure: only the
    one-time cached canon load.
    """
    from legal_rag.retrieval.routing.doctrine_detector import detect_doctrine_fast

    match = detect_doctrine_fast(query)
    if match is None:
        return None
    try:
        area = SubstantiveArea(match.doctrine.substantive_area)
    except ValueError:
        return None
    return area if area is not SubstantiveArea.GENERAL else None


def classify_substantive_area_fast(query: str) -> AreaResult:
    """Keyword-only classifier. Returns GENERAL on no match."""
    q = (query or "").strip()
    if not q:
        return _result_for(SubstantiveArea.GENERAL, "default", 0.0)
    for area, patterns in _AREA_PATTERNS:
        if any(p.search(q) for p in patterns):
            return _result_for(area, "keyword", 0.9)
    # Canonical-doctrine consultation: a doctrine-named query whose curated area
    # is specific supplies the area when the surface-word patterns miss, keeping
    # area consistent with jurisdiction and doctrine detection.
    canon_area = _doctrine_area(q)
    if canon_area is not None:
        return _result_for(canon_area, "canon", 0.85)
    return _result_for(SubstantiveArea.GENERAL, "default", 0.3)


def _cache_key(q: str) -> str:
    norm = re.sub(r"\s+", " ", (q or "").strip().lower())[:512]
    return hashlib.sha256(norm.encode()).hexdigest()[:32]


_LLM_SYSTEM: Final[str] = (
    "You classify US legal questions into one of these substantive areas:\n"
    + ", ".join(a.value for a in SubstantiveArea if a is not SubstantiveArea.GENERAL)
    + ", or 'general' if none clearly fits.\n"
    + 'Return STRICT JSON: {"area":"<one of the above>","confidence":0-1,'
    + '"reason":"≤20 words"}. Be conservative; prefer \'general\' over guessing.'
)


async def _llm_classify_area(question: str, llm: LanguageModel) -> AreaResult | None:
    """LLM fallback when keyword path returns GENERAL.

    Returns None on failure / unknown area; caller keeps the keyword default.
    """
    try:
        raw = await asyncio.wait_for(
            llm.generate(
                [Message("system", _LLM_SYSTEM), Message("user", f"Question:\n{question[:1000]}")],
                max_tokens=1024,
                temperature=0.0,
            ),
            timeout=10.0,
        )
        data = json.loads(raw or "{}")
        label = (data.get("area") or "").strip().lower()
        # Tolerate small variations in the LLM's label spelling: dashes/spaces
        # vs underscores, plural vs singular, "patent_law" vs "patent".
        label = label.replace("-", "_").replace(" ", "_")
        if label.endswith("s"):
            singular = label[:-1]
        else:
            singular = label
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        try:
            area = SubstantiveArea(label)
        except ValueError:
            try:
                area = SubstantiveArea(singular)
            except ValueError:
                # Best-effort substring match against known area names.
                area = None
                for a in SubstantiveArea:
                    if label and a.value in label:
                        area = a
                        break
                if area is None:
                    logger.warning(
                        "topic_filter.llm_unknown_area",
                        extra={"label": label[:40]},
                    )
                    return None
        return _result_for(area, "llm", conf)
    except Exception as exc:
        logger.warning("topic_filter.llm_failed", extra={"err": str(exc)[:120]})
        return None


async def classify_substantive_area(
    query: str,
    *,
    llm: LanguageModel | None = None,
    cache: Cache | None = None,
    cache_ttl_seconds: int = 3600,
) -> AreaResult:
    """Async entry point. Keyword first; the injected model only on a GENERAL miss."""
    fast = classify_substantive_area_fast(query)
    if fast.area is not SubstantiveArea.GENERAL or llm is None:
        return fast
    key = "area:" + _cache_key(query)
    if cache is not None:
        try:
            hit = await cache.get(key)
            if hit:
                cached = json.loads(hit)
                return _result_for(
                    SubstantiveArea(cached["area"]), cached["source"], cached["confidence"]
                )
        except Exception:
            logger.warning("topic_filter.cache_read_failed")
    verdict = await _llm_classify_area(query, llm) or fast
    if cache is not None:
        try:
            payload = {
                "area": verdict.area.value,
                "source": verdict.source,
                "confidence": verdict.confidence,
            }
            await cache.set(key, json.dumps(payload).encode(), ttl_seconds=cache_ttl_seconds)
        except Exception:
            logger.warning("topic_filter.cache_write_failed")
    return verdict


@lru_cache(maxsize=128)
def classify_substantive_area_sync(query: str) -> AreaResult:
    """Sync entry. Keyword only, for tests and contexts without an event loop."""
    return classify_substantive_area_fast(query)


def is_usc_title_allowed(usc_title: int | None, result: AreaResult) -> bool:
    """True iff the USC title is allowed under the classified area.

    None means: keep (we don't know what title the chunk belongs to).
    """
    if usc_title is None:
        return True
    if result.allowed_usc_titles is None:
        return True
    return usc_title in result.allowed_usc_titles


def is_cfr_title_allowed(cfr_title: int | None, result: AreaResult) -> bool:
    """True iff the CFR title is allowed under the classified area.

    None means: keep (we don't know).
    An empty frozenset means: suppress all CFR.
    """
    if cfr_title is None:
        return True
    if result.allowed_cfr_titles is None:
        return True
    return cfr_title in result.allowed_cfr_titles


__all__ = [
    "AreaResult",
    "SubstantiveArea",
    "classify_substantive_area",
    "classify_substantive_area_fast",
    "classify_substantive_area_sync",
    "is_admin_proc_cfr_title",
    "is_cfr_title_allowed",
    "is_civil_procedure_question",
    "is_usc_title_allowed",
]
