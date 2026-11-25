"""Data-file driven components: precedent, language, case numbers, hygiene."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

import legal_rag.chunking as chunking_pkg
from legal_rag.chunking.extractors import extract_case_numbers, normalize_case_number
from legal_rag.chunking.multilingual import LanguageDetector
from legal_rag.chunking.multilingual.language_detector import ScriptType
from legal_rag.chunking.temporal import (
    PrecedentStatus,
    PrecedentTracker,
    extract_year_from_citation,
)
from legal_rag.chunking.temporal.precedent import available_citation_groups

PACKAGE_DIR = Path(chunking_pkg.__file__).parent


def test_year_extraction_defaults_to_us_then_commonwealth():
    assert extract_year_from_citation("Smith v. Jones, 123 F.3d 456 (9th Cir. 1998)") == 1998
    assert extract_year_from_citation("2020 WL 123456") == 2020
    assert extract_year_from_citation("[2019] UKSC 12") == 2019
    assert extract_year_from_citation("no year here") is None


def test_regional_reporters_are_opt_in():
    assert "india" in available_citation_groups()
    assert extract_year_from_citation("AIR 1976 SC 1207", ["us"]) is None
    assert extract_year_from_citation("AIR 1976 SC 1207", ["us", "india"]) == 1976


async def test_tracker_seeded_with_us_overrulings():
    tracker = PrecedentTracker()
    validity = await tracker.check_validity("163 U.S. 537")
    assert validity.status == PrecedentStatus.OVERRULED
    assert validity.is_valid is False
    assert "347 U.S. 483" in "".join(validity.related_citations)


def test_registered_overrulings_stay_on_the_instance():
    first, second = PrecedentTracker(), PrecedentTracker()
    first.register_overruling("1 U.S. 1", "2 U.S. 2", date(2000, 1, 1), "Court", "test")
    assert "1 US 1" in first.known_overruled_cases
    assert "1 US 1" not in second.known_overruled_cases


def test_temporal_validation_rejects_impossible_overruling():
    result = PrecedentTracker().validate_temporal_relationship(
        "500 U.S. 1 (2000)", "400 U.S. 1 (1990)", date(1990, 1, 1)
    )
    assert result.is_valid is False


async def test_extract_citations_us_default():
    found = await PrecedentTracker().extract_citations("See 410 U.S. 113 and 2020 WL 99999.")
    assert "410 U.S. 113" in found
    assert "2020 WL 99999" in found


def test_language_detector_english_legal_text_is_not_romanized():
    info = LanguageDetector().detect(
        "The petition for bail was denied and the appeal is pending before the court."
    )
    assert info.primary_language == "en"
    assert info.is_transliterated is False


def test_language_detector_scripts():
    detector = LanguageDetector()
    assert detector.detect("هذا اختبار").script == ScriptType.ARABIC
    assert detector.detect("यह एक परीक्षण है").primary_language == "hi"
    assert detector.detect("").primary_language == "unknown"


def test_case_number_normalization():
    keys = {normalize_case_number(v) for v in ("2025-CA-2720", "25CA002720AX", "42-2025-CA-002720")}
    assert keys == {"CA-2025-2720"}
    assert extract_case_numbers("Case No. 2:25-cv-01234 was filed") == {"CV-2025-1234"}
    assert extract_case_numbers(None) == set()


def test_data_files_are_valid_json():
    files = list((PACKAGE_DIR / "data").glob("*.json"))
    assert len(files) >= 5
    for path in files:
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


FORBIDDEN = [
    r"\bfrom app\b",
    r"\bimport app\b",
    r"\bfrom scripts\b",
    r"os\.environ",
    r"getenv",
    r"user_id",
    r"organization_id",
    r"matter_id",
    r"vaquill",
    r"qdrant",
    r"voyage",
    "—",
    "–",
]


def test_sources_are_free_of_product_and_tenancy_strings():
    pattern = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    offenders = []
    for path in PACKAGE_DIR.rglob("*"):
        if path.suffix in {".py", ".json"}:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{number}")
    assert offenders == []


def test_unknown_pack_error_is_clear():
    from legal_rag.chunking.document_types import DocumentTypeClassifier

    with pytest.raises(KeyError):
        DocumentTypeClassifier(["missing"])
