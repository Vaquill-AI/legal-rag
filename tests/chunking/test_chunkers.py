"""Chunkers, document type detection and section detection."""

import pytest

from legal_rag.chunking import (
    DocumentType,
    EmailChunker,
    LegalSectionDetector,
    RecursiveLegalChunker,
    SemchunkLegalChunker,
    get_document_type_classifier,
    get_universal_section_detector,
)
from legal_rag.chunking.document_types import available_pattern_packs, load_pattern_packs
from legal_rag.chunking.tokens import ApproximateEncoding, count_tokens

JUDGMENT = (
    """SUPREME COURT OF THE UNITED STATES

Smith v. Jones, No. 21-1234

OPINION OF THE COURT

JUSTICE ROBERTS delivered the opinion of the Court.

1. This case concerns whether the statute applies to the petitioner. """
    + ("The court reviewed the record and considered the arguments of both parties. " * 80)
    + """

ORDER:

The judgment of the court of appeals is affirmed.
"""
)

CONTRACT = """MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is made and entered into on January 1, 2024, between Acme Corp and Beta LLC.

ARTICLE I DEFINITIONS
"Confidential Information" means all non-public information.

ARTICLE II TERM
The Receiving Party shall protect the Disclosing Party's information for two years.

ARTICLE III GOVERNING LAW
This Agreement is governed by the laws of Delaware. IN WITNESS WHEREOF the parties have signed.
"""

EMAIL = """From: alice@example.com
To: bob@example.com
Subject: Contract review
Date: Mon, 1 Jan 2024 10:00:00 +0000

Hi Bob,

Please review the attached agreement and send comments by Friday.

Thanks,
Alice

On Sun, 31 Dec 2023, Bob wrote:
> Sounds good, send it over.
"""


def test_token_counting_and_fallback():
    assert count_tokens("") == 0
    assert count_tokens("The court affirmed the judgment.") > 3
    approx = ApproximateEncoding()
    pieces = approx.encode("two words")
    assert approx.decode(pieces) == "two words"


def test_classifier_recognises_us_documents():
    classifier = get_document_type_classifier()
    assert classifier.classify(JUDGMENT)[0] == DocumentType.JUDGMENT
    assert classifier.classify(CONTRACT)[0] == DocumentType.CONTRACT
    assert classifier.classify("")[0] == DocumentType.UNKNOWN


def test_us_pack_is_default_and_regional_pack_is_opt_in():
    assert "us" in available_pattern_packs()
    notice = "FEDERAL REGISTER\nAGENCY: Office of Example\nACTION: Notice\nDocket No. 2024-1"
    assert get_document_type_classifier().classify(notice)[0] == DocumentType.CIRCULAR
    default = load_pattern_packs()
    assert DocumentType.JUDGMENT not in default
    extra = load_pattern_packs(["us", "india"])
    assert len(extra[DocumentType.JUDGMENT]) >= 4
    with pytest.raises(KeyError):
        load_pattern_packs(["nowhere"])


def test_pack_changes_classification_only_when_selected():
    text = "RESERVE BANK OF INDIA\nMASTER DIRECTION\nRBI/2020/12"
    plain = get_document_type_classifier()
    packed = get_document_type_classifier(["us", "india"])
    assert len(packed._patterns[DocumentType.CIRCULAR]) > len(
        plain._patterns[DocumentType.CIRCULAR]
    )
    assert packed.classify(text)[0] == DocumentType.CIRCULAR


def test_universal_detector_finds_sections():
    boundaries = get_universal_section_detector().detect(CONTRACT)
    assert boundaries
    assert all(0.0 <= b.confidence <= 1.0 for b in boundaries)


def test_legacy_detector_finds_judgment_sections():
    sections = LegalSectionDetector().detect(JUDGMENT)
    assert isinstance(sections, list)


async def test_recursive_chunker_covers_text_with_overlap():
    chunker = RecursiveLegalChunker(chunk_size=128, chunk_overlap=20, min_chunk_size=10)
    result = await chunker.chunk(JUDGMENT, "doc-1")
    assert len(result.chunks) > 2
    assert result.total_chunks == len(result.chunks)
    assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))
    assert all(c.document_id == "doc-1" for c in result.chunks)
    assert all(c.content.strip() for c in result.chunks)


async def test_recursive_chunker_is_deterministic_across_concurrency():
    sequential = RecursiveLegalChunker(chunk_size=128, chunk_overlap=20, max_concurrency=1)
    parallel = RecursiveLegalChunker(chunk_size=128, chunk_overlap=20, max_concurrency=4)
    a = await sequential.chunk(CONTRACT * 20, "d")
    b = await parallel.chunk(CONTRACT * 20, "d")
    assert [c.content for c in a.chunks] == [c.content for c in b.chunks]


async def test_recursive_chunker_empty_text():
    result = await RecursiveLegalChunker().chunk("   ", "doc-empty")
    assert list(result.chunks) == []


async def test_semchunk_chunker_links_neighbours():
    chunker = SemchunkLegalChunker(chunk_size=128, chunk_overlap=16, min_chunk_size=10)
    result = await chunker.chunk(JUDGMENT, "doc-2")
    assert len(result.chunks) >= 2


def test_semchunk_citations_use_injected_extractor():
    class Cite:
        full_citation = "410 U.S. 113"
        position = 3
        year = 1973

    class Extractor:
        def extract(self, text):
            return [Cite()]

    without = SemchunkLegalChunker()
    assert without.extract_citations_from_chunk("See 410 U.S. 113.", 0) == []
    with_extractor = SemchunkLegalChunker(citation_extractor=Extractor())
    found = with_extractor.extract_citations_from_chunk("See 410 U.S. 113.", 0)
    assert found[0]["citation_text"] == "410 U.S. 113"
    assert found[0]["year"] == 1973
    assert found[0]["court"] is None


async def test_email_chunker_keeps_headers_and_quotes_apart():
    result = await EmailChunker().chunk(EMAIL, "mail-1")
    assert result.chunks
    joined = "\n".join(c.content for c in result.chunks)
    assert "Contract review" in joined
