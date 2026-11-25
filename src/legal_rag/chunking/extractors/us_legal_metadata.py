"""Structured metadata extraction for US legal documents.

A simple contract-oriented regex extractor yields only a title, parties, dates and a
jurisdiction. It has no notion of a court docket number, a court, or a litigation
document type, so those fields stay empty for court filings (complaints, motions,
orders, deeds, dockets). An empty case number is also what lets documents from
different cases mix together in retrieval.

This module fills that gap with a hybrid extractor:

- a language model (injected through the ``LanguageModel`` protocol, structured
  output) reads the caption region and pulls the structured fields (document type,
  primary case number, court, court level, jurisdiction, parties, filing date,
  judge, citations);
- a deterministic regex (``case_numbers.extract_case_numbers``) scans the whole
  document for every docket number, for high-recall multi-case detection.

It is defensive: any model failure degrades to the regex-only result rather than
raising, so ingestion never breaks because extraction was unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from legal_rag.chunking._logging import get_logger
from legal_rag.chunking.extractors.case_numbers import extract_case_numbers, normalize_case_number
from legal_rag.providers.protocols import LanguageModel
from legal_rag.types import Message

logger = get_logger("us_legal_metadata")

_VOCAB_FILE = Path(__file__).resolve().parent.parent / "data" / "us_document_vocabulary.json"


def _load_vocabulary() -> dict[str, Any]:
    with _VOCAB_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


_VOCAB = _load_vocabulary()

# Document types the model classifies into: litigation first, then transactional.
DOCUMENT_TYPES: tuple[str, ...] = tuple(_VOCAB["document_types"])

COURT_LEVELS: tuple[str, ...] = tuple(_VOCAB["court_levels"])

SYSTEM_PROMPT = (
    "You extract structured metadata from a US legal document. You are given the "
    "top portion of the document (where the caption appears). Return JSON only.\n\n"
    "Rules:\n"
    "- document_type: classify the document. For a court filing pick the litigation "
    "type (complaint, answer, motion, order, judgment, notice, petition, brief, "
    "affidavit, declaration, subpoena, discovery, pleading). Use deed/contract/"
    "agreement/lease/will_or_trust for transactional docs, correspondence for "
    "letters/emails, exhibit/transcript/financial/scan_or_image where they fit, "
    "else other.\n"
    "- primary_case_number: the docket/case number in the CAPTION of THIS document, "
    "verbatim as written (e.g. '2025-CA-2720', 'JU-2023-000406.01', "
    "'1:23-cv-00456'). Null if the document has no caption case number.\n"
    "- court: the full court name from the caption (e.g. 'Circuit Court of the Fifth "
    "Judicial Circuit, Marion County, Florida'). Null if absent.\n"
    "- court_level: classify the court.\n"
    "- jurisdiction: the US state name (e.g. 'Florida') or 'federal'. Null if unclear.\n"
    "- plaintiffs / defendants: the named parties (use petitioners as plaintiffs, "
    "respondents as defendants). Empty arrays if none.\n"
    "- filing_date / judge: from the caption or signature block, else null.\n"
    "- citations: case or statute citations referenced in the visible text.\n\n"
    "Extract only what is actually present. Do not guess or invent. Use null / empty "
    "for anything not clearly stated."
)


class CaptionMetadata(BaseModel):
    """The structured fields the model is asked to return."""

    document_type: str = "other"
    primary_case_number: str | None = None
    court: str | None = None
    court_level: str = "unknown"
    jurisdiction: str | None = None
    plaintiffs: list[str] = Field(default_factory=list)
    defendants: list[str] = Field(default_factory=list)
    filing_date: str | None = None
    judge: str | None = None
    citations: list[str] = Field(default_factory=list)


# Caption and signature usually live in the first few thousand characters.
# The case-number regex still scans the full text.
_LLM_HEAD_CHARS = 7000


def _clean(value: str | None) -> str | None:
    return (value or "").strip() or None


async def extract_us_legal_metadata(
    text: str | None,
    *,
    filename: str | None = None,
    llm: LanguageModel | None = None,
    head_chars: int = _LLM_HEAD_CHARS,
) -> dict[str, Any]:
    """Extract structured US legal metadata from document text.

    Args:
        text: Full document text.
        filename: Optional file name, also scanned for case numbers.
        llm: Language model used to read the caption. When omitted, only the
            regex fields are populated.
        head_chars: How much of the document's start is shown to the model.

    Returns:
        A dict with: document_type, case_number (primary, verbatim), case_number_key,
        case_numbers (normalized keys for every referenced case, for filtering and
        multi-case detection), court, court_level, jurisdiction, plaintiffs,
        defendants, filing_date, judge, citations. Never raises.
    """
    full = text or ""
    # High-recall and deterministic: every case number anywhere in the document and filename.
    regex_cases = extract_case_numbers(full) | extract_case_numbers(filename or "")

    base: dict[str, Any] = {
        "document_type": None,
        "case_number": None,
        "case_number_key": next(iter(regex_cases)) if len(regex_cases) == 1 else None,
        "case_numbers": sorted(regex_cases),
        "court": None,
        "court_level": None,
        "jurisdiction": None,
        "plaintiffs": [],
        "defendants": [],
        "filing_date": None,
        "judge": None,
        "citations": [],
    }

    if not full.strip() or llm is None:
        return base

    try:
        head = full[:head_chars]
        user_content = (
            f"FILENAME: {filename}\n\n" if filename else ""
        ) + f"DOCUMENT (top portion):\n{head}"

        data = await llm.generate_structured(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ],
            CaptionMetadata,
            max_tokens=1024,
        )

        doc_type = data.document_type if data.document_type in DOCUMENT_TYPES else "other"
        court_level = data.court_level if data.court_level in COURT_LEVELS else "unknown"
        base["document_type"] = doc_type
        base["court"] = _clean(data.court)
        base["court_level"] = court_level
        base["jurisdiction"] = _clean(data.jurisdiction)
        base["plaintiffs"] = [p for p in data.plaintiffs if p]
        base["defendants"] = [p for p in data.defendants if p]
        base["filing_date"] = _clean(data.filing_date)
        base["judge"] = _clean(data.judge)
        base["citations"] = [c for c in data.citations if c]

        primary_raw = _clean(data.primary_case_number)
        base["case_number"] = primary_raw
        # Fold the caption number into the normalized set so it is matchable.
        # normalize_case_number() handles compact, county-prefixed and uniform
        # forms that the body-text regex will not. If the model gave no primary
        # but the text had exactly one case, use that as the primary.
        if primary_raw:
            primary_key = normalize_case_number(primary_raw)
            base["case_numbers"] = sorted(regex_cases | ({primary_key} if primary_key else set()))
            # Prefer the normalized canonical form so the same case written
            # differently across documents matches.
            base["case_number_key"] = primary_key
        elif len(regex_cases) == 1:
            only = next(iter(regex_cases))
            base["case_number"] = only
            base["case_number_key"] = only

        logger.info(
            "us_legal_metadata_extracted",
            document_type=base["document_type"],
            has_case_number=bool(base["case_number"]),
            n_cases=len(base["case_numbers"]),
            court_level=base["court_level"],
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully by design
        logger.warning("us_legal_metadata_extract_failed", error=str(exc))
        # Degrade to the regex-only result; never break ingestion.

    return base
