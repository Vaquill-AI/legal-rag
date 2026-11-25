"""Case-name extraction ignores leading signal words; default headings are US-first."""

import pytest

from legal_rag.chunking import LegalSectionDetector
from legal_rag.chunking.document_types import DocumentType, UniversalSectionDetector
from legal_rag.chunking.extractors.legal_ner import (
    HybridLegalNERExtractor,
    strip_leading_signal_words,
)


async def test_case_name_drops_leading_preposition():
    text = "In Smith v. Jones, 123 F.3d 456 (9th Cir. 1999), the court held that the claim failed."
    assert await HybridLegalNERExtractor().extract_case_names(text) == ["Smith v. Jones"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See Smith v. Jones, 5 U.S. 1 (1803).", "Smith v. Jones"),
        ("Cf. Smith v. Jones, 5 U.S. 1 (1803).", "Smith v. Jones"),
        ("Under Brown v. Board of Education (1954), segregation ended.", "Brown v. Board"),
        ("Per Marbury v. Madison (1803), review is settled.", "Marbury v. Madison"),
        ("As held in Smith v. Jones, 410 U.S. 113, the court agreed.", "Smith v. Jones"),
    ],
)
async def test_case_name_drops_signal_words(text, expected):
    assert await HybridLegalNERExtractor().extract_case_names(text) == [expected]


async def test_signal_prefixed_and_bare_mentions_deduplicate():
    text = (
        "In Smith v. Jones, 5 U.S. 1, the rule applies. Later, Smith v. Jones, 5 U.S. 1, was cited."
    )
    assert await HybridLegalNERExtractor().extract_case_names(text) == ["Smith v. Jones"]


async def test_transformer_path_also_strips_signal_words():
    class Classifier:
        async def token_classification(self, text):
            return [
                {"entity_group": "ORG", "word": "In Acme", "end": 7},
                {"entity_group": "ORG", "word": "Beta", "end": 17},
            ]

    extractor = HybridLegalNERExtractor(token_classifier=Classifier())
    assert await extractor.extract_case_names("In Acme v. Beta was decided.") == ["Acme v. Beta"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("In Smith", "Smith"),
        ("See Smith", "Smith"),
        ("See Also Smith", "Smith"),
        ("Cf. Smith", "Smith"),
        ("Under Smith", "Smith"),
        ("Per Smith", "Smith"),
        ("As held in Smith", "Smith"),
        ("See In re Gault", "In re Gault"),
        ("Smith", "Smith"),
    ],
)
def test_strip_leading_signal_words(raw, expected):
    assert strip_leading_signal_words(raw) == expected


@pytest.mark.parametrize("name", ["In re Gault", "Ex parte Young", "Matter of Smith", "See"])
def test_legitimate_names_are_kept(name):
    assert strip_leading_signal_words(name) == name


_INDIA_AND_COMMONWEALTH_HEADINGS = [
    "CORAM: Justice A, Justice B",
    "WRIT PETITION (C) NO. 12 OF 2020",
    "SPECIAL LEAVE PETITION (CIVIL) NO. 5 OF 2019",
    "HON'BLE MR. JUSTICE A. KUMAR",
]


def _default_legal_section_patterns():
    return [p for patterns in LegalSectionDetector.SECTION_PATTERNS.values() for p in patterns]


def _default_universal_judgment_patterns():
    judgment = UniversalSectionDetector.SECTION_PATTERNS[DocumentType.JUDGMENT]
    return [pattern for rows in judgment.values() for pattern, _level in rows]


@pytest.mark.parametrize("heading", _INDIA_AND_COMMONWEALTH_HEADINGS)
def test_default_section_headings_have_no_india_or_commonwealth_forms(heading):
    for pattern in _default_legal_section_patterns():
        assert not pattern.search(heading), pattern.pattern
    for pattern in _default_universal_judgment_patterns():
        assert not pattern.search(heading), pattern.pattern


def test_default_detectors_do_not_split_on_removed_headings():
    text = "CORAM: Justice A, Justice B\n\nThe parties dispute the lease.\n"
    boundaries = LegalSectionDetector().detect(text)
    assert [b.content for b in boundaries] == [text]
    assert all(
        "CORAM" not in b.matched_text.upper()
        for b in UniversalSectionDetector().detect(text, DocumentType.JUDGMENT)
        if getattr(b, "matched_text", None)
    )


def test_us_first_headings_still_detected_by_default():
    text = "HEADNOTE: Lease dispute.\n\nBEFORE: Judge Rivera\n\nJUDGMENT: The appeal fails.\n"
    sections = {b.section.value for b in LegalSectionDetector().detect(text)}
    assert "headnote" in sections
    assert "judgment" in sections
