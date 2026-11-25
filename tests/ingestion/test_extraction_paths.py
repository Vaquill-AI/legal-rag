"""Real extraction paths run against fixtures generated in ``tmp_path``.

No binary file is committed. Each test builds a small document with the format's own
library, runs the extractor or ingestor on it, and asserts on the extracted text.
A format whose library is not installed is skipped, not failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from legal_rag.ingestion.email_file import EmailFileIngestor
from legal_rag.ingestion.extractors.office import OfficeExtractor
from legal_rag.ingestion.extractors.text import TextExtractor

# ---------------------------------------------------------------------------
# Office formats
# ---------------------------------------------------------------------------


async def test_docx_extracts_headings_paragraphs_and_tables(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Master Services Agreement", level=1)
    document.add_paragraph("The Supplier shall deliver the goods within thirty days.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Party"
    table.cell(0, 1).text = "Role"
    table.cell(1, 0).text = "Acme Corp"
    table.cell(1, 1).text = "Supplier"
    path = tmp_path / "agreement.docx"
    document.save(str(path))

    result = await OfficeExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert "Master Services Agreement" in result.text
    assert "within thirty days" in result.text
    assert "Acme Corp | Supplier" in result.text
    assert result.tables
    assert result.tables[0]["data"] == [["Party", "Role"], ["Acme Corp", "Supplier"]]


async def test_docx_with_no_text_is_not_an_error(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "blank.docx"
    docx.Document().save(str(path))

    result = await OfficeExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert result.text.strip() == ""


async def test_xlsx_extracts_every_sheet_with_cell_values(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    fees = workbook.active
    fees.title = "Fees"
    fees.append(["Item", "Amount"])
    fees.append(["Retainer", 5000])
    workbook.create_sheet("Notes").append(["Indemnity cap applies"])
    path = tmp_path / "schedule.xlsx"
    workbook.save(str(path))

    result = await OfficeExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert result.sheets == ["Fees", "Notes"]
    assert "Retainer | 5000" in result.text
    assert "Indemnity cap applies" in result.text
    assert result.tables
    assert result.tables[0]["data"] == [["Item", "Amount"], ["Retainer", "5000"]]


async def test_pptx_extracts_slide_title_and_body(tmp_path: Path) -> None:
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Deal Overview"
    slide.placeholders[1].text = "Closing scheduled for March"
    path = tmp_path / "deck.pptx"
    presentation.save(str(path))

    result = await OfficeExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert "Deal Overview" in result.text
    assert "Closing scheduled for March" in result.text
    assert result.slides
    assert len(result.slides) == 1


async def test_pptx_speaker_notes_are_extracted_under_a_label(tmp_path: Path) -> None:
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    first = presentation.slides.add_slide(presentation.slide_layouts[1])
    first.shapes.title.text = "Deal Overview"
    first.placeholders[1].text = "Closing scheduled for March"
    first.notes_slide.notes_text_frame.text = "Mention the escrow release before the vote"
    second = presentation.slides.add_slide(presentation.slide_layouts[1])
    second.shapes.title.text = "Risks"
    path = tmp_path / "deck.pptx"
    presentation.save(str(path))

    result = await OfficeExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert "Mention the escrow release before the vote" in result.text
    assert "Speaker notes:\nMention the escrow release" in result.text
    assert (
        result.slides and result.slides[0]["notes"] == "Mention the escrow release before the vote"
    )
    assert "escrow" not in result.slides[0]["text"]
    assert "notes" not in result.slides[1]
    assert result.pages and "Speaker notes:" in result.pages[0]["text"]
    assert all("Speaker notes:" not in page["text"] for page in result.pages[1:])


async def test_corrupt_office_file_reports_failure_instead_of_raising() -> None:
    pytest.importorskip("docx")

    result = await OfficeExtractor().extract(b"this is not a zip archive", "broken.docx")

    assert not result.success
    assert result.error


# ---------------------------------------------------------------------------
# HTML and RTF
# ---------------------------------------------------------------------------


async def test_html_keeps_visible_text_and_drops_scripts_and_styles(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    path = tmp_path / "opinion.html"
    path.write_text(
        "<html><head><title>Opinion</title>"
        "<style>.hidden { display: none; }</style>"
        "<script>var internalToken = 'SCRIPT-ONLY';</script></head>"
        "<body><h1>Smith v. Jones</h1>"
        "<p>The court held the clause void.</p>"
        "<table><tr><td>Count</td><td>Two</td></tr></table></body></html>",
        encoding="utf-8",
    )

    result = await TextExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert "Smith v. Jones" in result.text
    assert "The court held the clause void." in result.text
    assert "Count" in result.text
    assert "SCRIPT-ONLY" not in result.text
    assert "display: none" not in result.text
    assert "<p>" not in result.text


async def test_html_title_is_recovered_into_metadata_and_not_the_body() -> None:
    pytest.importorskip("bs4")
    html = (
        b"<html><head><title>  Smith v. Jones, Opinion </title></head>"
        b"<body><p>The court held the clause void.</p></body></html>"
    )

    result = await TextExtractor().extract(html, "opinion.html")

    assert result.success, result.error
    assert result.metadata["title"] == "Smith v. Jones, Opinion"
    assert "Smith v. Jones" not in result.text
    assert "The court held the clause void." in result.text


async def test_rtf_strips_control_words_and_keeps_text(tmp_path: Path) -> None:
    pytest.importorskip("striprtf")
    path = tmp_path / "lease.rtf"
    path.write_bytes(
        rb"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times New Roman;}}\f0\fs24 "
        rb"\b Lease Agreement\b0\par The tenant shall pay rent monthly.\par}"
    )

    result = await TextExtractor().extract(path.read_bytes(), path.name)

    assert result.success, result.error
    assert "Lease Agreement" in result.text
    assert "The tenant shall pay rent monthly." in result.text
    assert "\\rtf1" not in result.text
    assert "\\par" not in result.text
    assert "Times New Roman" not in result.text


# ---------------------------------------------------------------------------
# MSG
# ---------------------------------------------------------------------------


async def test_msg_that_is_not_ole2_fails_cleanly() -> None:
    pytest.importorskip("extract_msg")

    result = await EmailFileIngestor().ingest(file_content=b"not an ole file", filename="x.msg")

    assert not result.success
    assert result.error


def test_msg_roundtrip_needs_a_fixture_writer() -> None:
    extract_msg = pytest.importorskip("extract_msg")
    # extract-msg only reads .msg files. Without a writer there is no way to generate a
    # valid fixture in tmp_path, and committing a binary one is against the rules here.
    if not hasattr(extract_msg, "MSGBuilder"):
        pytest.skip("extract-msg cannot build a .msg fixture")


# ---------------------------------------------------------------------------
# PDF layout path (AGPL extra, optional)
# ---------------------------------------------------------------------------


@pytest.fixture
def pymupdf_modules() -> Any:
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    return pymupdf


def _one_page_pdf(pymupdf: Any, path: Path, *, extra_drawings: int = 0) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    body = "The Seller shall convey the property to the Buyer free of all liens. " * 6
    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 700),
        f"PURCHASE AGREEMENT\n\n{body}",
        fontsize=11,
    )
    for i in range(extra_drawings):
        y = 720 + (i % 40)
        page.draw_line((50 + i, y), (60 + i, y + 1))
    document.save(str(path))
    document.close()
    return path


async def test_pdf_layout_path_extracts_text_from_a_generated_page(
    tmp_path: Path, pymupdf_modules: Any
) -> None:
    from legal_rag.ingestion.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor, PyMuPDFConfig

    path = _one_page_pdf(pymupdf_modules, tmp_path / "agreement.pdf")

    result = await PyMuPDF4LLMIngestor(PyMuPDFConfig(force_ocr=False)).ingest(
        file_content=path.read_bytes(), filename=path.name, document_id="doc-1"
    )

    assert result.success, result.error
    assert result.page_count == 1
    assert "PURCHASE AGREEMENT" in result.content
    assert "free of all liens" in result.content


async def test_pdf_with_tight_graphics_limits_still_extracts_text(
    tmp_path: Path, pymupdf_modules: Any
) -> None:
    from legal_rag.ingestion.pymupdf4llm_ingestor import PyMuPDF4LLMIngestor, PyMuPDFConfig

    path = _one_page_pdf(pymupdf_modules, tmp_path / "busy.pdf", extra_drawings=60)
    config = PyMuPDFConfig(
        force_ocr=False, graphics_limit=10, max_drawings_per_page=5, image_size_limit=0.5
    )

    result = await PyMuPDF4LLMIngestor(config).ingest(
        file_content=path.read_bytes(), filename=path.name, document_id="doc-2"
    )

    assert result.success, result.error
    assert "PURCHASE AGREEMENT" in result.content


# ---------------------------------------------------------------------------
# PyMuPDFConfig limits
# ---------------------------------------------------------------------------


def test_pymupdf_config_limits_default_to_module_constants_and_can_be_overridden() -> None:
    from legal_rag.ingestion import pymupdf4llm_ingestor as module

    defaults = module.PyMuPDFConfig()
    assert defaults.graphics_limit == module.DEFAULT_GRAPHICS_LIMIT
    assert defaults.image_size_limit == module.DEFAULT_IMAGE_SIZE_LIMIT
    assert defaults.max_drawings_per_page == module.DEFAULT_MAX_DRAWINGS_PER_PAGE
    assert defaults.graphics_limit > 0
    assert 0 < defaults.image_size_limit < 1
    assert defaults.max_drawings_per_page >= defaults.graphics_limit

    tuned = module.PyMuPDFConfig(graphics_limit=7, image_size_limit=0.2, max_drawings_per_page=9)
    assert (tuned.graphics_limit, tuned.image_size_limit, tuned.max_drawings_per_page) == (
        7,
        0.2,
        9,
    )
