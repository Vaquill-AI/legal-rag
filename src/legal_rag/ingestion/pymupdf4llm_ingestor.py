# ruff: noqa: BLE001, SIM115
# Ingestion degrades per document instead of failing a batch, hence the broad excepts.
"""
PDF layout ingestor built on PyMuPDF4LLM (optional module).

LICENCE NOTICE: this module drives PyMuPDF, PyMuPDF4LLM and, optionally, PyMuPDF Pro and
PyMuPDF Layout.
PyMuPDF and PyMuPDF4LLM are distributed under the GNU AGPL or a commercial licence from the
vendor, and the Pro and Layout add-ons need a commercial licence key.
legal-rag does not bundle or relicense any of them.
Before you import this module in a product, check that the licence you hold covers your use.
Nothing in the legal-rag package imports this module at import time; it is loaded only when
you import it yourself or the router asks for the layout parser.

Features:
- Optional layout analysis (box classes such as title, section-header and text)
- Pre-flight OCR check: skips the OCR pass entirely for native (text-layer) PDFs
- Single-parse optimization: parses the document once instead of re-parsing
- Parallel page-range extraction, with a subprocess path for daemonic callers
- Optional host-wide gate (``ExtractionGate``) that limits concurrent heavy extractions

Tuning lives in ``PyMuPDFConfig``, passed to the constructor.
No settings, environment variables or external services are read.
"""

import contextlib
import json
import multiprocessing
import os
import subprocess  # nosec B404 - invoke our own `-m` helper with internally-generated paths; no shell, no user input
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from legal_rag.ingestion._log import get_logger
from legal_rag.ingestion.base import (
    BaseIngestor,
    DocumentSection,
    DocumentType,
    IngestorResult,
)
from legal_rag.ingestion.protocols import ExtractionGate
from legal_rag.ingestion.text_cleaner import clean_ocr_artifacts

logger = get_logger("legal_rag.ingestion.pymupdf4llm")

# ============================================
# PyMuPDF Pro Initialization
# ============================================
PYMUPDF_PRO_AVAILABLE = False
PYMUPDF_AVAILABLE = False
PYMUPDF4LLM_AVAILABLE = False
OCR_AVAILABLE = False

# Detect PyMuPDF and PyMuPDF Pro. Pro is unlocked explicitly, with a key you supply,
# by ``unlock_pymupdf_pro`` (the ingestor constructor calls it when ``pro_api_key`` is set).
try:
    import pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:
    pymupdf = None  # type: ignore
    logger.warning("pymupdf_not_installed", message="Install with: pip install pymupdf")


def unlock_pymupdf_pro(api_key: str) -> bool:
    """Unlock PyMuPDF Pro with ``api_key``. Returns True when Pro is available afterwards."""
    global PYMUPDF_PRO_AVAILABLE
    if not (PYMUPDF_AVAILABLE and api_key):
        return PYMUPDF_PRO_AVAILABLE
    try:
        import pymupdf.pro

        pymupdf.pro.unlock(api_key)
        PYMUPDF_PRO_AVAILABLE = True
        logger.debug("pymupdf_pro_unlocked", message="Layout analysis enabled")
    except ImportError:
        logger.info("pymupdf_pro_not_installed", message="Install with: pip install pymupdfpro")
    except Exception as e:
        logger.warning("pymupdf_pro_unlock_failed", error=str(e))
    return PYMUPDF_PRO_AVAILABLE


# ============================================
# PyMuPDF Layout Activation (MUST be before pymupdf4llm import)
# ============================================
# PyMuPDF Layout is a GNN-based layout analysis addon that provides:
# - Semantic element detection (title, section-header, text, table, caption, etc.)
# - Better structure understanding for complex documents
# - Headers/footers, list items, and caption detection
#
# When imported, it changes pymupdf4llm's output format:
# - With layout: chunks have 'page_boxes' key (semantic element boxes with class info)
# - Without layout: chunks have 'words' key (unused - we only need semantic classes)
PYMUPDF_LAYOUT_AVAILABLE = False
try:
    import pymupdf.layout

    PYMUPDF_LAYOUT_AVAILABLE = True
    logger.info(
        "pymupdf_layout_activated", message="GNN layout analysis enabled for semantic detection"
    )
except ImportError:
    logger.warning(
        "pymupdf_layout_import_failed",
        message="pymupdf-layout not available - falling back to word-level extraction",
    )

# Import pymupdf4llm high-level API (AFTER layout import)
try:
    import pymupdf4llm

    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    pymupdf4llm = None  # type: ignore
    logger.warning("pymupdf4llm_not_installed", message="Install with: pip install pymupdf4llm")

# Check for OCR dependencies
try:
    import cv2  # noqa: F401

    OCR_AVAILABLE = True
except ImportError:
    logger.info(
        "opencv_not_installed",
        message="OCR disabled - install opencv-python for scanned PDF support",
    )

# Legacy non-Unicode fonts that extract as printable-but-wrong gibberish
# (glyphs map to ASCII/Latin code points that render as an Indic script). The
# printable-ratio check cannot catch these because the bytes ARE printable, so
# we match on font name and route to OCR. Names are matched case-insensitively
# as substrings, so "Kruti" covers "Kruti Dev 010", "DevLys" covers "DevLys 010",
# etc. Kept conservative to avoid colliding with legitimate Unicode font names.
LEGACY_FONT_INDICATORS = [
    # Punjabi (Gurmukhi)
    "Asees",
    "Akhar",
    "AnmolLipi",
    "Satluj",
    "Joy",
    # Hindi / Marathi / Sanskrit (Devanagari)
    "Kruti",  # Krutidev / Kruti Dev
    "DevLys",
    "Shusha",
    "Chanakya",
    "Walkman",
    "Shivaji",
    "Yogesh",
    "Agra",
    # Gujarati
    "Krishna",
    "Saral",
    # Tamil
    "TSCII",
    "Bamini",
    "TAB-",
    "TAM-",
    # Generic legacy Indic vendors
    "Shree",  # Shree-Dev / Shree-Lipi families
    "APS",
    "ISFOC",
]


def _is_page_text_readable(text: str) -> bool:
    """True if a page's extracted text is usable (not glyph-index noise)."""
    from legal_rag.ingestion.text_quality import is_text_readable

    return is_text_readable(text)


# A "junk" token in form noise: a single stray char, or a run made only of
# form-mark punctuation (underscores, dashes, pipes, dots used for fill-in rules
# and checkbox glyphs). Multi-digit/decimal numbers ("1,234.56") are NOT junk,
# which keeps legitimate numeric tables from tripping the form-noise gate.
_FORM_MARK_CHARS = frozenset("_-.|=*~")


def _looks_like_form_noise(text: str) -> bool:
    """True if a page extracted *some* text, but it is scattered fill-in-form
    marks (underscores, checkbox glyphs, isolated letters/digits) rather than
    real prose. A scanned form whose text layer is this kind of junk reads as
    "readable" to every text_quality check (printable ASCII, no replacement
    chars, has whitespace), so it slips past the readability gate and the junk
    gets embedded instead of routing to OCR.

    Deliberately specific to avoid false-positives on legitimately sparse pages
    (cover/signature pages: real words, high alpha density) and numeric tables
    (multi-char number tokens, so junk-token density stays low). Both signals
    must fire together.
    """
    stripped = text.strip()
    if len(stripped) < OCR_MIN_CHARS_PER_PAGE:
        return False  # too little to judge; the char_count gate already handles it

    tokens = stripped.split()
    if not tokens:
        return False

    junk_tokens = sum(1 for t in tokens if len(t) <= 1 or all(c in _FORM_MARK_CHARS for c in t))
    junk_ratio = junk_tokens / len(tokens)

    non_ws = [c for c in stripped if not c.isspace()]
    alpha_ratio = (sum(1 for c in non_ws if c.isalpha()) / len(non_ws)) if non_ws else 0.0

    # Real prose sits near alpha_ratio ~0.75+ with few single-char tokens.
    # Form noise inverts both: mostly stray marks/digits, little alphabetic text.
    return junk_ratio >= 0.55 and alpha_ratio < 0.55


# If this ratio (or more) of sampled pages look like fill-in-form junk, the
# text layer is unusable scanned-form noise: route to OCR. Set as high as the
# scanned-page bar so a single sparse page never triggers needless OCR spend.
OCR_FORM_NOISE_THRESHOLD = 0.70


# Unicode replacement character - indicates font encoding issues
REPLACEMENT_CHARACTER = chr(0xFFFD)

# Box classes to skip when building clean text
SKIP_BOX_CLASSES = {"page-header", "page-footer"}

# Box classes that get semantic prefixes for embedding boost
BOOST_BOX_CLASSES = {
    "title": "[TITLE] # ",
    "section-header": "[SECTION] ## ",
}

# ============================================
# OCR Analysis Data Classes
# ============================================

# Smart sampling: how many pages to sample for the pre-flight OCR check
OCR_SAMPLE_PAGE_COUNT = 10
# A page needs at least this many chars to be considered "text-rich"
OCR_MIN_CHARS_PER_PAGE = 50
# If this ratio (or more) of sampled pages are scanned, route to Mistral OCR
OCR_SCANNED_THRESHOLD = 0.70
# Document-level density floor: the mean chars/page below which a text layer is
# captions and page numbers rather than content.
#
# The per-page bar above cannot catch a header-only scan, because a printed case
# caption is real, readable text: it clears 50 chars, is not garbled and is not
# form noise, so every per-page signal passes it. A multi-page filing whose text
# layer holds only stamps would then be treated as born-digital and never reach OCR.
#
# 250 sits well above header-stamp density (about 120) and far below a dense page of
# a legal filing (1,500 to 3,000). Applied only to multi-page documents so a genuinely
# short one-pager (receipt, cover note) is not dragged into needless OCR.
OCR_MIN_MEAN_CHARS_PER_PAGE = 250
OCR_DENSITY_MIN_PAGES = 3

# PyMuPDF4LLM extraction limits. Each is the default of the matching ``PyMuPDFConfig``
# field, so callers override them per ingestor through the config, not by patching these.

# Above this many vector drawings on a page, graphics analysis is skipped for it (text is still extracted).
DEFAULT_GRAPHICS_LIMIT = 5000
# Images covering less than this fraction of the page area (logos, rules, bullets) are ignored.
DEFAULT_IMAGE_SIZE_LIMIT = 0.05
# A sampled page with more drawings than this (CAD plans, vector maps) is treated as too heavy to lay out.
DEFAULT_MAX_DRAWINGS_PER_PAGE = 20000
# Fallback render resolution (dpi) for page images when a worker config omits one.
DEFAULT_RENDER_DPI = 200


@dataclass
class PageOCRClassification:
    """Classification of a single sampled page for OCR routing."""

    page_index: int
    char_count: int
    has_replacement_chars: bool
    is_text_rich: bool  # char_count >= threshold AND no replacement chars
    looks_like_form_noise: bool = False  # scattered fill-in-form marks, not prose


@dataclass
class OCRAnalysis:
    """Result of the smart pre-flight OCR analysis.

    Used by the router to decide between PyMuPDF (free), Mistral OCR ($2/1K pages),
    or hybrid routing (split PDF by page type).
    """

    needs_ocr: bool  # True = route entire doc to Mistral OCR
    scanned_ratio: float  # 0.0 to 1.0
    sampled_pages: list[PageOCRClassification] = field(default_factory=list)
    total_pages: int = 0
    strategy: str = "all_text"  # "all_scanned" | "all_text" | "mixed"


def _compute_sample_indices(
    total_pages: int, max_samples: int = OCR_SAMPLE_PAGE_COUNT
) -> list[int]:
    """Compute page indices to sample, spread across the document.

    Strategy:
    - If total_pages <= max_samples: sample ALL pages
    - Otherwise: first 5 pages + 5 evenly spaced from the remainder

    This avoids being fooled by blank cover pages while still catching
    scanned appendices near the end.
    """
    if total_pages <= max_samples:
        return list(range(total_pages))

    # First 5 pages (catches cover + early content)
    front = list(range(min(5, total_pages)))

    # 5 evenly spaced pages from the remainder
    remaining_count = max_samples - len(front)
    remainder_start = len(front)
    remainder_size = total_pages - remainder_start
    spread = [
        remainder_start + i * remainder_size // remaining_count for i in range(remaining_count)
    ]

    # Deduplicate and sort
    return sorted(set(front + spread))


def group_page_ranges(analysis: OCRAnalysis) -> list[tuple[int, int, str]]:
    """Group pages into contiguous ranges of the same type for hybrid routing.

    Uses sampled page classifications and forward-fills gaps between samples.
    Merges tiny ranges (< 3 pages) into the surrounding range to avoid
    excessive splitting overhead.

    Returns:
        List of (start_page, end_page_exclusive, "text"|"ocr") tuples.
    """
    if not analysis.sampled_pages or analysis.total_pages == 0:
        return [(0, analysis.total_pages, "text")]

    # Build full classification array, forward-filling from sampled pages
    classifications: list[bool] = [True] * analysis.total_pages  # True = text_rich
    sorted_samples = sorted(analysis.sampled_pages, key=lambda p: p.page_index)

    # Set sampled pages
    for sample in sorted_samples:
        classifications[sample.page_index] = sample.is_text_rich

    # Forward-fill: each unsampled page inherits from the nearest prior sample
    sample_indices = {s.page_index for s in sorted_samples}
    last_known = sorted_samples[0].is_text_rich if sorted_samples else True
    for i in range(analysis.total_pages):
        if i in sample_indices:
            last_known = classifications[i]
        else:
            classifications[i] = last_known

    # Group contiguous runs
    ranges: list[tuple[int, int, str]] = []
    current_start = 0
    current_type = "text" if classifications[0] else "ocr"

    for i in range(1, analysis.total_pages):
        page_type = "text" if classifications[i] else "ocr"
        if page_type != current_type:
            ranges.append((current_start, i, current_type))
            current_start = i
            current_type = page_type
    ranges.append((current_start, analysis.total_pages, current_type))

    # Merge tiny ranges (< 3 pages) into the surrounding range
    if len(ranges) <= 1:
        return ranges

    merged: list[tuple[int, int, str]] = [ranges[0]]
    for start, end, rtype in ranges[1:]:
        page_count = end - start
        prev_start, _prev_end, prev_type = merged[-1]
        if page_count < 3:
            # Absorb into previous range
            merged[-1] = (prev_start, end, prev_type)
        elif rtype == prev_type:
            # Consolidate adjacent same-type ranges
            merged[-1] = (prev_start, end, prev_type)
        else:
            merged.append((start, end, rtype))

    return merged


# ============================================
# Parallel Processing Configuration
# ============================================
# Only parallelize for large documents - overhead isn't worth it for small ones.
PARALLEL_PAGE_THRESHOLD = 50  # Min pages before parallel kicks in
MIN_PAGES_PER_WORKER = 32  # At least 32 pages per worker to amortize spawn cost
# Target Hetzner VM has 16 vCPU. 14 leaves 2 cores for the uvicorn API +
# ambient task-queue beat tasks (health checks, DLQ monitor, announcement scheduler)
# so a single big PDF parse does not starve the rest of the system.
MAX_PARALLEL_WORKERS = 14  # Cap workers to leave CPU for other tasks

# Subprocess fallback for daemonic callers (e.g. task-queue prefork workers).
# Python forbids daemonic processes from spawning ProcessPoolExecutor children,
# so we delegate extraction to a non-daemonic helper script. See
# legal_rag/ingestion/pymupdf_extract_subprocess.py for the protocol.
SUBPROCESS_HELPER_MODULE = "legal_rag.ingestion.pymupdf_extract_subprocess"
# Maximum wall-clock budget for the subprocess call. Must fit inside the
# parse task's hard time_limit (900s in document_pipeline.py); we cap at
# 850s to leave 50s headroom for metadata extraction and result handling.
# Used directly when the page count is unknown; otherwise see
# `compute_subprocess_timeout()` for the per-page scaling.
SUBPROCESS_TIMEOUT_SECONDS = 850
# Floor for the per-page timeout. Even a 10-page PDF gets at least this
# much budget so a slow-OCR or table-heavy page does not artificially
# exceed the timeout. Lower values cut the bad-doc detection latency
# but risk false-positive timeouts on legitimate small docs with
# scanned pages.
SUBPROCESS_TIMEOUT_FLOOR_SECONDS = 300
# Per-page wall-clock budget. Empirical: a 1343-page calculus textbook
# parses in ~111s with 14 workers, so worst-case ~0.1s/page on the M4
# Max baseline. Production VMs are slower per-core; 0.5s/page gives a
# 5x safety margin over the measured baseline and still keeps the
# computed budget well under the 850s ceiling for documents up to
# ~1700 pages. Anything bigger hits the ceiling and falls back to the
# host-level OOM/CPU defences (RLIMIT_AS, RLIMIT_CPU).
SUBPROCESS_TIMEOUT_PER_PAGE_SECONDS = 0.5


def compute_subprocess_timeout(total_pages: int) -> int:
    """Return a per-document subprocess timeout that scales with page
    count, clamped between SUBPROCESS_TIMEOUT_FLOOR_SECONDS and
    SUBPROCESS_TIMEOUT_SECONDS.

    Why scale instead of using a flat 850s for everything:

      * A 30-page contract that spends >5 minutes in pymupdf is almost
        certainly stuck on a graphics-heavy page or DRM glitch, not
        legitimately busy. Returning to the single-threaded fallback
        sooner lets the user see "FAILED" inside the parse task's hard
        time_limit instead of after a near-15-minute hang.
      * Large textbooks still get the full 850s ceiling because the
        per-page allowance crosses the cap by ~1700 pages.

    The function is deliberately pure and module-level so callers can
    unit-test the policy without instantiating the ingestor.
    """
    if total_pages <= 0:
        # Unknown page count (corrupt PDF, weird format). Use the
        # ceiling so the subprocess gets the full budget rather than
        # being prematurely killed by the floor.
        return SUBPROCESS_TIMEOUT_SECONDS
    scaled = int(total_pages * SUBPROCESS_TIMEOUT_PER_PAGE_SECONDS)
    return max(SUBPROCESS_TIMEOUT_FLOOR_SECONDS, min(scaled, SUBPROCESS_TIMEOUT_SECONDS))


# Defensive resource limits for the subprocess tree (parent + N workers).
# A hostile or pathological PDF can otherwise pin up to MAX_PARALLEL_WORKERS
# cores for SUBPROCESS_TIMEOUT_SECONDS and allocate unbounded memory.
# Applied via preexec_fn on POSIX; silently skipped on Windows.
SUBPROCESS_MAX_ADDRESS_SPACE_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB across tree
SUBPROCESS_MAX_CPU_SECONDS = 1800  # aggregate CPU time across children

# Host-wide gate for concurrent subprocess extractions. Documents below
# ``PyMuPDFConfig.heavy_page_threshold`` bypass the gate (small documents spawn only a few
# children). Larger ones acquire a slot from the injected ``ExtractionGate`` so at most
# ``PyMuPDFConfig.gate_limit`` process trees run at once on the host.


def _page_bucket(total_pages: int) -> str:
    """Bucket page counts so dashboards can group extraction outcomes by
    document size class without us needing high-cardinality labels."""
    if total_pages < 100:
        return "<100"
    if total_pages < 300:
        return "100-299"
    if total_pages < 1000:
        return "300-999"
    return "1000+"


def _emit_subprocess_outcome(
    *,
    outcome: str,
    total_pages: int,
    duration_s: float,
    gate_active: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    """Single structured-log emit for every terminal path of the subprocess
    extractor. Ops dashboards count occurrences per `outcome` and chart the
    `duration_s` histogram per `page_bucket`. Outcomes:

      * success                      - subprocess returned a valid payload
      * timeout                      - subprocess.run exceeded SUBPROCESS_TIMEOUT_SECONDS
      * subprocess_crash             - non-zero return code or process killed
      * bad_payload                  - JSON deserialised to a non-list / wrong shape
      * subprocess_reported_error    - child wrote the error sentinel
      * exception                    - unexpected exception in the parent path
      * gate_timeout                 - extraction gate could not be acquired in time
      * output_owner_mismatch        - shared-VM defence: out file not owned by us
    """
    payload: dict[str, Any] = {
        "outcome": outcome,
        "total_pages": total_pages,
        "page_bucket": _page_bucket(total_pages),
        "duration_seconds": round(duration_s, 3),
        "gate_active": gate_active,
    }
    if extra:
        payload.update(extra)
    logger.info("pymupdf_subprocess_outcome", **payload)


def _build_subprocess_rlimit_preexec():
    """Construct a preexec_fn that caps the subprocess tree's resource use.

    Imported only at call time because `resource` is POSIX-only. On macOS/Linux
    this installs `RLIMIT_AS` (virtual memory) and `RLIMIT_CPU` on the parent
    subprocess; child processes inherit these limits via fork/spawn.

    Kept as a factory so unit tests can mock it without patching the os layer.
    """
    import resource  # POSIX-only - caller guards with os.name == "posix"

    def _preexec() -> None:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_AS,
                (SUBPROCESS_MAX_ADDRESS_SPACE_BYTES, SUBPROCESS_MAX_ADDRESS_SPACE_BYTES),
            )
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (SUBPROCESS_MAX_CPU_SECONDS, SUBPROCESS_MAX_CPU_SECONDS),
            )

    return _preexec


def sweep_stale_subprocess_tempdirs(max_age_seconds: int = 3600) -> int:
    """Delete orphaned subprocess tempdirs from prior SIGKILL events.

    Call this from a worker-ready signal or app startup so repeated crashes
    don't fill /tmp. Returns the number of tempdirs removed.
    """
    root = tempfile.gettempdir()
    now = time.time()
    cleaned = 0
    try:
        with os.scandir(root) as it:
            for entry in it:
                if not entry.name.startswith("pymupdf_subproc_"):
                    continue
                try:
                    age = now - entry.stat().st_mtime
                except OSError:
                    continue
                if age < max_age_seconds:
                    continue
                with contextlib.suppress(OSError):
                    # shutil.rmtree would follow symlinks; we prefer not to
                    # traverse, so walk and unlink.
                    import shutil

                    shutil.rmtree(entry.path, ignore_errors=True)
                    cleaned += 1
    except OSError:
        pass
    return cleaned


def _parse_page_range(args: tuple[str, list[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Worker function for parallel page processing (runs in child process).

    Must be module-level for ProcessPoolExecutor pickling.
    Initializes PyMuPDF Pro + Layout in each child process.

    Args:
        args: Tuple of (pdf_path, page_list, config_dict)

    Returns:
        List of page chunk dicts from pymupdf4llm.to_markdown
    """
    pdf_path, page_list, config_dict = args

    # Initialize PyMuPDF Pro in child process
    import pymupdf

    try:
        import pymupdf.pro

        pro_key = config_dict.get("pro_api_key")
        if pro_key:
            pymupdf.pro.unlock(pro_key)
    except ImportError:
        pass

    # Activate layout engine in child process
    try:
        import pymupdf.layout
    except ImportError:
        pass

    import pymupdf4llm

    to_md_kwargs: dict[str, Any] = {
        "pages": page_list,
        "page_chunks": True,
        "extract_words": False,
        "force_text": config_dict.get("force_text", True),
        "table_strategy": config_dict.get("table_strategy", "lines_strict"),
        "graphics_limit": config_dict.get("graphics_limit", DEFAULT_GRAPHICS_LIMIT),
        "image_size_limit": config_dict.get("image_size_limit", DEFAULT_IMAGE_SIZE_LIMIT),
        "dpi": config_dict.get("dpi", DEFAULT_RENDER_DPI),
    }
    # Only pass ignore_graphics when True - older pymupdf4llm versions don't accept it.
    if config_dict.get("ignore_graphics"):
        to_md_kwargs["ignore_graphics"] = True

    try:
        raw_chunks = pymupdf4llm.to_markdown(pdf_path, **to_md_kwargs)
    except AssertionError:
        # pymupdf4llm 0.2.9 raises AssertionError inside
        # `pymupdf._as_pdf_document(doc)` for some Office-converted PDFs
        # (form-field extraction asserts when the underlying document
        # isn't a real PDF). Falling back to plain `page.get_text("text")`
        # gets us the same readable text without the GNN layout
        # machinery that triggered the assert.
        #
        # Safe to do here because we ARE inside a fresh non-daemonic
        # process (either ProcessPoolExecutor child or subprocess.run
        # child). pymupdf.open is not the fork-hazard concern in this
        # context.
        doc = pymupdf.open(pdf_path)
        try:
            sanitized: list[dict[str, Any]] = []
            for page_idx in page_list:
                if 0 <= page_idx < len(doc):
                    page = doc[page_idx]
                    sanitized.append(
                        {
                            "text": page.get_text("text"),
                            "metadata": {"page": page_idx},
                            "tables": [],
                            "images": [],
                            "page_boxes": [],
                        }
                    )
        finally:
            doc.close()
        return sanitized

    # Sanitize results for pickling - pymupdf4llm may embed lambdas in chunk dicts.
    # Only keep the keys we actually use downstream.
    sanitized = []
    for chunk in raw_chunks:
        sanitized.append(
            {
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "tables": chunk.get("tables", []),
                "images": chunk.get("images", []),
                "page_boxes": chunk.get("page_boxes", []),
            }
        )
    return sanitized


@dataclass
class PyMuPDFConfig:
    """Tuning for ``PyMuPDF4LLMIngestor``.

    Defaults are conservative starting points; measure against your own documents.

    Args:
        force_ocr: True always OCR, False never, None auto-detect.
        ocr_dpi: Render resolution for OCR.
        min_chars_threshold: Minimum characters per page below which a page counts as scanned.
        table_strategy: Table detection strategy passed to PyMuPDF4LLM.
        force_text: Keep the text layer when OCR is used.
        graphics_limit: Skip vector graphics analysis on a page above this many drawings
            (default ``DEFAULT_GRAPHICS_LIMIT``).
        image_size_limit: Ignore images covering less than this fraction of the page
            (default ``DEFAULT_IMAGE_SIZE_LIMIT``).
        max_drawings_per_page: Pages with more drawings than this are treated as too heavy
            (default ``DEFAULT_MAX_DRAWINGS_PER_PAGE``).
        pro_api_key: PyMuPDF Pro licence key, if you hold one.
        heavy_page_threshold: Page count at or above which the extraction gate applies.
        gate_limit: Maximum concurrent heavy extractions (0 disables the gate).
        gate_wait_seconds: How long to wait for a gate slot.
    """

    force_ocr: bool | None = None
    ocr_dpi: int = 300
    min_chars_threshold: int = 200
    table_strategy: str = "lines_strict"
    force_text: bool = True
    graphics_limit: int = DEFAULT_GRAPHICS_LIMIT
    image_size_limit: float = DEFAULT_IMAGE_SIZE_LIMIT
    max_drawings_per_page: int = DEFAULT_MAX_DRAWINGS_PER_PAGE
    pro_api_key: str = ""
    heavy_page_threshold: int = 300
    gate_limit: int = 3
    gate_wait_seconds: float = 120.0


class PyMuPDF4LLMIngestor(BaseIngestor):
    """
    Primary document ingestor using PyMuPDF4LLM with GNN layout analysis.

    Supports:
    - PDF (always)
    - XPS, EPUB (native PyMuPDF formats)
    - Office formats when PyMuPDF Pro unlocked: DOC/DOCX, XLS/XLSX, PPT/PPTX, HWP/HWPX

    Provides:
    - Superior text extraction for born-digital documents
    - Box-class detection for semantic chunking
    - Automatic OCR for scanned documents
    - Legacy font detection for quality warnings

    Note: Highlighting uses search-based char positions (no bbox storage).
    """

    # Base formats (always supported)
    _BASE_FORMATS: ClassVar[set[str]] = {"pdf", "xps", "epub"}

    # Office formats (supported when PyMuPDF Pro is unlocked)
    _PRO_FORMATS: ClassVar[set[str]] = {"doc", "docx", "xls", "xlsx", "ppt", "pptx", "hwp", "hwpx"}

    @property
    def SUPPORTED_FORMATS(self) -> set[str]:
        """Get supported formats based on PyMuPDF Pro availability."""
        if PYMUPDF_PRO_AVAILABLE:
            return self._BASE_FORMATS | self._PRO_FORMATS
        return self._BASE_FORMATS

    def __init__(
        self,
        config: PyMuPDFConfig | None = None,
        *,
        extraction_gate: ExtractionGate | None = None,
    ):
        """Initialize the PyMuPDF4LLM ingestor.

        Args:
            config: Tuning values. Defaults are used when omitted.
            extraction_gate: Optional host-wide gate for heavy parallel extractions.
        """
        self.config = config or PyMuPDFConfig()
        self.extraction_gate = extraction_gate
        if self.config.pro_api_key:
            unlock_pymupdf_pro(self.config.pro_api_key)

    async def supports_format(self, filename: str) -> bool:
        """
        Check if this ingestor supports the given file format.

        Returns True for:
        - PDF, XPS, EPUB (always)
        - Office formats (DOC/DOCX, XLS/XLSX, PPT/PPTX, HWP/HWPX) when Pro unlocked
        """
        if not PYMUPDF4LLM_AVAILABLE:
            return False
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in self.SUPPORTED_FORMATS

    async def ingest(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        filename: str = "document.pdf",
        document_id: str | None = None,
        **kwargs,
    ) -> IngestorResult:
        """
        Ingest a document using PyMuPDF4LLM.

        Supports:
        - PDF (always)
        - XPS, EPUB (native formats)
        - Office formats when Pro unlocked: DOC/DOCX, XLS/XLSX, PPT/PPTX, HWP/HWPX

        Args:
            file_path: Path to the document file
            file_content: Raw document bytes (alternative to file_path)
            filename: Original filename
            document_id: Unique document identifier
            **kwargs: Additional options:
                - force_ocr: Force OCR (True/False/None=auto)
                - ocr_dpi: OCR DPI (default: config value)

        Returns:
            IngestorResult with extracted content and metadata
        """
        start_time = time.time()
        doc_id = document_id or "unknown"

        # Check if pymupdf4llm is available
        if not PYMUPDF4LLM_AVAILABLE:
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error="pymupdf4llm not installed - run: pip install pymupdf4llm",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Get options from kwargs or config
        force_ocr = kwargs.get("force_ocr", self.config.force_ocr)
        ocr_dpi = kwargs.get("ocr_dpi", self.config.ocr_dpi)
        min_chars = kwargs.get("min_chars", self.config.min_chars_threshold)

        # Handle file content vs file path
        temp_file = None
        working_path = file_path

        try:
            if file_content is not None and file_path is None:
                # Write content to temp file (pymupdf4llm needs file path)
                # Preserve original extension for Office format detection
                ext = Path(filename).suffix or ".pdf"
                temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb")
                temp_file.write(file_content)
                temp_file.close()
                working_path = temp_file.name

            if working_path is None:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error="Either file_path or file_content must be provided",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Extract using PyMuPDF4LLM
            result = self._extract_with_pymupdf4llm(
                pdf_path=working_path,
                force_ocr=force_ocr,
                ocr_dpi=ocr_dpi,
                min_chars_threshold=min_chars,
            )

            if "error" in result:
                return IngestorResult(
                    document_id=doc_id,
                    filename=filename,
                    content="",
                    success=False,
                    error=result["error"],
                    parser_name=result.get("parser_name", "pymupdf4llm"),
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Extract sections from boxes
            sections = self._extract_sections_from_boxes(result.get("boxes", []))

            # Build IngestorResult
            processing_time = (time.time() - start_time) * 1000
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content=result.get("text_clean", result.get("text", "")),  # ✅ Use cleaned text
                text_clean=result.get("text_clean"),
                sections=sections,
                document_type=DocumentType.PDF,
                page_count=result.get("page_count", 0),
                metadata=result.get("metadata", {}),
                # PyMuPDF4LLM specific
                boxes=result.get("boxes"),
                pages=result.get("pages"),
                box_class_counts=result.get("box_class_counts"),
                ocr_used=result.get("ocr_used", False),
                supports_positions=result.get("supports_positions", True),
                total_chars=result.get("total_chars", 0),
                parser_name=result.get("parser_name", "pymupdf4llm-pro-layout"),
                has_legacy_fonts=result.get("has_legacy_fonts", False),
                legacy_fonts_found=result.get("legacy_fonts_found", []),
                # ✅ OCR artifact cleaning metadata
                artifacts_removed=result.get("artifacts_removed", 0),
                cleaning_applied=result.get("cleaning_applied", False),
                # Processing info
                success=True,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.exception(
                "pymupdf4llm_extraction_failed",
                document_id=doc_id,
                filename=filename,
                error=str(e),
            )
            return IngestorResult(
                document_id=doc_id,
                filename=filename,
                content="",
                success=False,
                error=f"PyMuPDF4LLM extraction failed: {e!s}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        finally:
            # Clean up temp file
            if temp_file is not None:
                with contextlib.suppress(Exception):
                    Path(temp_file.name).unlink(missing_ok=True)

    def _analyze_ocr_needs_from_bytes(self, file_content: bytes) -> OCRAnalysis:
        """
        Smart pre-flight OCR analysis from bytes.

        Samples pages spread across the document (not just the first few) to
        avoid being fooled by blank cover pages. Classifies each page individually
        and uses a ratio threshold to decide routing.

        Used by the router to choose between PyMuPDF, Mistral OCR, or hybrid routing.
        """
        if pymupdf is None:
            return OCRAnalysis(
                needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
            )

        try:
            doc = pymupdf.open(stream=file_content, filetype="pdf")
            return self._analyze_ocr_needs_doc(doc)
        except Exception:
            return OCRAnalysis(
                needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
            )

    def _analyze_ocr_needs(self, pdf_path: str) -> OCRAnalysis:
        """
        Smart pre-flight OCR analysis from file path.

        Same logic as _analyze_ocr_needs_from_bytes but opens from disk.
        Used internally by the PyMuPDF ingestor during its own ingest() flow.
        """
        if pymupdf is None:
            return OCRAnalysis(
                needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
            )

        try:
            doc = pymupdf.open(pdf_path)
            return self._analyze_ocr_needs_doc(doc)
        except Exception:
            return OCRAnalysis(
                needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
            )

    def _analyze_ocr_needs_doc(self, doc: Any) -> OCRAnalysis:
        """
        Core OCR analysis logic on an opened PyMuPDF document.

        Sampling strategy:
        - Documents <= 10 pages: sample ALL pages
        - Documents > 10 pages: first 5 + 5 evenly spaced from the remainder

        Per-page classification:
        - Text-rich: char_count >= OCR_MIN_CHARS_PER_PAGE AND no replacement chars
        - Scanned: everything else

        Decision:
        - scanned_ratio >= 0.70: needs_ocr=True, strategy="all_scanned"
        - scanned_ratio == 0.0: needs_ocr=False, strategy="all_text"
        - 0.0 < scanned_ratio < 0.70: needs_ocr=False, strategy="mixed"
        """
        try:
            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                return OCRAnalysis(
                    needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
                )

            sample_indices = _compute_sample_indices(total_pages)
            sampled: list[PageOCRClassification] = []

            for page_idx in sample_indices:
                page = doc[page_idx]
                text = page.get_text("text")
                char_count = len(text.strip())
                has_replacement = REPLACEMENT_CHARACTER in text
                # A page is text-rich only if it yields enough chars AND those
                # chars are actually readable. A Type3 / CID font with no
                # ToUnicode map extracts as control-byte noise: high char_count,
                # no U+FFFD, but unusable. Treat those pages as scanned so the
                # whole doc routes to OCR instead of wasting a pymupdf parse.
                is_text_rich = (
                    char_count >= OCR_MIN_CHARS_PER_PAGE
                    and not has_replacement
                    and _is_page_text_readable(text)
                )

                sampled.append(
                    PageOCRClassification(
                        page_index=page_idx,
                        char_count=char_count,
                        has_replacement_chars=has_replacement,
                        is_text_rich=is_text_rich,
                        # Only meaningful for pages that LOOK text-rich; a truly
                        # scanned page is already routed to OCR by is_text_rich.
                        looks_like_form_noise=(is_text_rich and _looks_like_form_noise(text)),
                    )
                )

            doc.close()

            scanned_count = sum(1 for p in sampled if not p.is_text_rich)
            scanned_ratio = scanned_count / len(sampled) if sampled else 1.0

            # Garbled text-layer detection: a scanned form can carry a junk text
            # layer (Unicode replacement chars from a bad prior OCR) that is
            # "text-rich" by char count, so it passes the scanned_ratio gate and
            # the junk gets used instead of running OCR. If a meaningful share of
            # pages show replacement chars, force OCR to recover real text.
            garbled_ratio = (
                sum(1 for p in sampled if p.has_replacement_chars) / len(sampled)
                if sampled
                else 0.0
            )

            # Form-noise detection: a scanned fill-in form (court answer forms,
            # checkbox pleadings) can carry a text layer that is only scattered
            # marks ("_I_", "- 21", underscores) yet reads as text-rich. Embedding
            # that junk is what caused cross-document confusion. If most pages look
            # like form junk, route to OCR to recover the real printed text.
            form_noise_ratio = (
                sum(1 for p in sampled if p.looks_like_form_noise) / len(sampled)
                if sampled
                else 0.0
            )

            # Header-only detection: a scanned filing often carries a text layer
            # holding just the printed case caption and page number. That is real,
            # readable, non-junk text, so every per-page signal above passes it and
            # the document scores scanned_ratio 0.0 ("all_text") -- the hybrid OCR
            # path is never given anything to do. Mean density is the only signal
            # that separates "a page of content" from "a page of letterhead".
            mean_chars_per_page = (
                sum(p.char_count for p in sampled) / len(sampled) if sampled else 0.0
            )
            text_layer_too_sparse = (
                total_pages >= OCR_DENSITY_MIN_PAGES
                and mean_chars_per_page < OCR_MIN_MEAN_CHARS_PER_PAGE
            )

            if (
                scanned_ratio >= OCR_SCANNED_THRESHOLD
                or garbled_ratio >= 0.30
                or form_noise_ratio >= OCR_FORM_NOISE_THRESHOLD
                or text_layer_too_sparse
            ):
                strategy = "all_scanned"
                needs_ocr = True
            elif scanned_ratio == 0.0:
                strategy = "all_text"
                needs_ocr = False
            else:
                strategy = "mixed"
                needs_ocr = False

            logger.debug(
                "ocr_analysis_complete",
                total_pages=total_pages,
                sampled_count=len(sampled),
                scanned_ratio=round(scanned_ratio, 2),
                garbled_ratio=round(garbled_ratio, 2),
                form_noise_ratio=round(form_noise_ratio, 2),
                mean_chars_per_page=round(mean_chars_per_page, 1),
                text_layer_too_sparse=text_layer_too_sparse,
                strategy=strategy,
                needs_ocr=needs_ocr,
                sample_indices=[p.page_index for p in sampled],
                text_rich_pages=[p.page_index for p in sampled if p.is_text_rich],
            )

            return OCRAnalysis(
                needs_ocr=needs_ocr,
                scanned_ratio=scanned_ratio,
                sampled_pages=sampled,
                total_pages=total_pages,
                strategy=strategy,
            )

        except Exception:
            doc.close()
            return OCRAnalysis(
                needs_ocr=True, scanned_ratio=1.0, total_pages=0, strategy="all_scanned"
            )

    def _check_needs_ocr_from_bytes(
        self, file_content: bytes, sample_pages: int = 10, min_chars: int = 200
    ) -> bool:
        """Backward-compatible wrapper. Returns True if OCR is needed."""
        return self._analyze_ocr_needs_from_bytes(file_content).needs_ocr

    def _check_needs_ocr(self, pdf_path: str, sample_pages: int = 10, min_chars: int = 200) -> bool:
        """Backward-compatible wrapper. Returns True if OCR is needed."""
        return self._analyze_ocr_needs(pdf_path).needs_ocr

    @staticmethod
    def _count_max_drawings_per_page(pdf_path: str, sample_pages: int = 5) -> int:
        """
        Count the maximum number of vector drawing paths on any sampled page.

        Uses Page.get_cdrawings() (the fast C-level variant) to check the first
        N pages without loading full text or running layout analysis.

        Award certificates, decorative letterheads, and graphical court documents
        can have 1,000 to 10,000+ vector paths per page, causing the GNN layout model
        in pymupdf.layout to hang for many minutes.  Anything above
        config.PYMUPDF_MAX_DRAWINGS_PER_PAGE (default 500) triggers
        ignore_graphics=True so GNN skips graphic paths entirely.

        Args:
            pdf_path: Path to the PDF file.
            sample_pages: How many pages to sample (first N pages only).

        Returns:
            Maximum drawing-path count across sampled pages (0 on any error).
        """
        if pymupdf is None:
            return 0
        try:
            doc = pymupdf.open(pdf_path)
            pages_to_check = min(sample_pages, len(doc))
            max_drawings = 0
            for i in range(pages_to_check):
                # get_cdrawings is significantly faster than get_drawings
                count = len(doc[i].get_cdrawings())
                max_drawings = max(max_drawings, count)
            doc.close()
            return max_drawings
        except Exception:
            return 0

    @staticmethod
    def _fallback_extract_text_only(pdf_path: str) -> list[dict[str, Any]]:
        """Last-ditch extraction that never invokes pymupdf4llm's GNN path.

        Used in two situations:
          1. Office files in a daemonic worker, where calling
             pymupdf4llm.to_markdown would deadlock on inherited
             multi-threaded PyMuPDF Pro state.
          2. Any pdf-style file where pymupdf4llm.to_markdown raised an
             AssertionError inside form-field extraction (a known
             upstream bug on some DOCX-converted PDFs).

        Produces page chunks with the same shape as
        pymupdf4llm.to_markdown(page_chunks=True): each entry has `text`
        and `metadata.page`. No `tables`, `images`, or `page_boxes`,
        because those require the GNN layout path we are explicitly
        avoiding here.
        """
        if pymupdf is None:  # type: ignore[truthy-bool]
            return []
        doc = pymupdf.open(pdf_path)
        page_chunks: list[dict[str, Any]] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_chunks.append(
                    {
                        "text": page.get_text("text"),
                        "metadata": {"page": page_num},
                    }
                )
        finally:
            doc.close()
        return page_chunks

    def _detect_legacy_fonts(self, doc: Any) -> dict[str, Any]:
        """
        Detect legacy non-Unicode fonts that extract as garbage.

        Args:
            doc: Open pymupdf.Document object

        Returns:
            Dictionary with legacy font detection results
        """
        all_fonts: set[str] = set()
        legacy_fonts: set[str] = set()

        try:
            for page in doc:
                fonts = page.get_fonts()
                for font in fonts:
                    font_name = font[3] if len(font) > 3 else ""
                    all_fonts.add(font_name)

                    font_name_lower = font_name.lower()
                    for legacy in LEGACY_FONT_INDICATORS:
                        if legacy.lower() in font_name_lower:
                            legacy_fonts.add(font_name)

            return {
                "has_legacy_fonts": len(legacy_fonts) > 0,
                "legacy_fonts_found": list(legacy_fonts),
                "all_fonts": list(all_fonts),
            }
        except Exception as e:
            return {
                "has_legacy_fonts": False,
                "legacy_fonts_found": [],
                "all_fonts": [],
                "font_detection_error": str(e),
            }

    def _extract_with_pymupdf4llm(
        self,
        pdf_path: str,
        force_ocr: bool | None = None,
        ocr_dpi: int = 200,
        min_chars_threshold: int = 100,
    ) -> dict[str, Any]:
        """
        Extract text from PDF using pymupdf4llm's to_markdown API.

        Uses the correct pymupdf4llm 0.2.x API (to_markdown with page_chunks=True).

        Optimizations applied (mechanisms; multipliers unmeasured):
        1. Pre-flight OCR check (skips the OCR pass for native PDFs)
        2. Pre-flight graphics check - sets ignore_graphics=True for award/certificate
           PDFs with complex vector art that would otherwise hang GNN layout analysis
        3. Optimized string building (single join instead of repeated concatenation)
        4. Single PDF open for font detection

        Note: No bbox coordinates stored. Highlighting uses char positions +
        PyMuPDF search_for() at render time (via bbox_extractor.py).

        Args:
            pdf_path: Path to PDF file
            force_ocr: True=always, False=never, None=auto-detect
            ocr_dpi: OCR DPI (200=fast, 400=quality)
            min_chars_threshold: Min chars for successful extraction

        Returns:
            Dictionary with extraction results
        """
        try:
            # Determine fork-hazard situation EARLY so the password pre-flight
            # can be skipped for office files in daemonic workers. We learned
            # the hard way that `pymupdf.open()` on .xlsx and .docx is NOT a
            # fast header read - PyMuPDF Pro fully converts the Office file
            # to PDF inside that one call, and that conversion is exactly
            # the path that deadlocks in a forked child. So even the
            # password pre-flight has to be deferred to the subprocess.
            ext_for_office_check = Path(pdf_path).suffix.lower().lstrip(".")
            is_office_for_preflight = ext_for_office_check in self._PRO_FORMATS
            is_daemonic_for_preflight = multiprocessing.current_process().daemon
            skip_password_preflight = is_office_for_preflight and is_daemonic_for_preflight

            # Pre-flight password check: detect encrypted PDFs before attempting extraction.
            # pymupdf4llm crashes with cryptic errors on password-protected files.
            # SKIPPED for office files in daemonic workers (see comment above).
            # The subprocess will surface password-protected files via the
            # standard pymupdf4llm error path, which we already wrap.
            if PYMUPDF_AVAILABLE and not skip_password_preflight:
                import pymupdf

                _preflight_doc = pymupdf.open(pdf_path)
                _needs_pass = _preflight_doc.needs_pass
                _preflight_doc.close()
                if _needs_pass:
                    return {
                        "error": (
                            "This PDF is password-protected. Please remove the password "
                            "and re-upload, or provide an unprotected version of the document."
                        ),
                        "parser_name": "pymupdf4llm",
                        "is_password_protected": True,
                    }

            # =================================================================
            # Pre-flight strategy (FORK HAZARD AWARE)
            # =================================================================
            # The next two pre-flights - OCR detection and drawings count - both
            # call pymupdf.open() and iterate pages, which kicks PyMuPDF Pro
            # into its rendering / native-thread codepaths. In a task-queue prefork
            # worker child whose master has been alive for hours, those code
            # paths can block forever on locks inherited from helper threads
            # that died at fork time (we caught a real instance of this on a
            # 14 KB XLSX: stack stuck inside _ssl__SSLSocket_read → take_gil
            # → _pthread_cond_wait, no progress for 15 min, SIGKILL'd by the
            # task time_limit).
            #
            # PDFs avoid this two ways: (a) they don't trigger Pro's heavier
            # Excel→PDF conversion path, and (b) >= PARALLEL_PAGE_THRESHOLD
            # PDFs already route through `_extract_via_subprocess` which is
            # fork-immune. Office formats have no such escape hatch - the
            # pre-flights happen BEFORE the dispatcher can route to subprocess.
            #
            # Mitigation: for office files in a daemonic worker, skip the
            # OCR + drawings pre-flights entirely and use sensible defaults.
            # Office files in our domain are essentially never scanned (those
            # come as PDF) and essentially never graphics-heavy (Excel cells,
            # not vector art). The password check stays inline because it's
            # one boolean header read - fast, never enters the heavy paths.
            ext = Path(pdf_path).suffix.lower().lstrip(".")
            is_office = ext in self._PRO_FORMATS
            is_daemonic = multiprocessing.current_process().daemon
            skip_dangerous_preflights = is_office and is_daemonic

            if skip_dangerous_preflights:
                use_ocr = bool(force_ocr)  # honour explicit override; otherwise False
                ignore_graphics = False
                logger.info(
                    "pymupdf4llm_office_preflight_skipped",
                    pdf_path=pdf_path,
                    ext=ext,
                    msg=(
                        "Skipping OCR + drawings pre-flight for office file in "
                        "daemonic worker (fork-hazard avoidance). Defaults: "
                        "use_ocr=False, ignore_graphics=False. Subprocess will "
                        "do its own page-count + extraction in a fresh process."
                    ),
                )
            else:
                # Pre-flight OCR check
                use_ocr = self._check_needs_ocr(pdf_path) if force_ocr is None else force_ocr

                # Pre-flight graphics check: award certificates and graphical PDFs can have
                # thousands of vector drawing paths per page.  pymupdf.layout's GNN model
                # tries to classify every path and can freeze for 15+ minutes on such files.
                # When any sampled page exceeds MAX_DRAWINGS_PER_PAGE we pass
                # ignore_graphics=True to to_markdown() so GNN skips graphic paths entirely
                # (text extraction and table detection are unaffected).
                ignore_graphics = False
                if PYMUPDF_LAYOUT_AVAILABLE:
                    max_drawings = self._count_max_drawings_per_page(pdf_path)
                    if max_drawings > self.config.max_drawings_per_page:
                        ignore_graphics = True
                        logger.warning(
                            "pymupdf4llm_layout_graphics_suppressed",
                            pdf_path=pdf_path,
                            max_drawings_per_page=max_drawings,
                            threshold=self.config.max_drawings_per_page,
                            message=(
                                "Graphics-heavy PDF detected - setting ignore_graphics=True "
                                "to prevent GNN layout analysis from hanging"
                            ),
                        )

            logger.debug(
                "pymupdf4llm_extraction_start",
                pdf_path=pdf_path,
                use_ocr=use_ocr,
                ocr_dpi=ocr_dpi,
                layout_enabled=PYMUPDF_LAYOUT_AVAILABLE,
                ignore_graphics=ignore_graphics,
                skipped_preflights=skip_dangerous_preflights,
            )

            # Use correct pymupdf4llm API: to_markdown with page_chunks=True
            # With layout engine: chunks have 'page_boxes' with semantic class info
            # (title, section-header, text, table, etc.) used by chunker for prefixes.
            # extract_words=False: skip word-level bbox extraction (2.7x overhead, unused).

            # Quick page count check for parallel processing decision.
            # For office files in daemonic workers we ALSO skip this (it calls
            # pymupdf.open which on .xlsx triggers the same Pro path that hangs).
            # Pass total_page_count=0 so the subprocess timeout falls back to the
            # ceiling and the parallelism heuristic just routes to subprocess
            # anyway (which doesn't need exact page count up-front).
            import pymupdf

            if skip_dangerous_preflights:
                total_page_count = 0
            else:
                doc_for_count = pymupdf.open(pdf_path)
                total_page_count = len(doc_for_count)
                doc_for_count.close()

            common_kwargs: dict[str, Any] = {
                "force_text": self.config.force_text if use_ocr else True,
                "table_strategy": self.config.table_strategy,
                "graphics_limit": self.config.graphics_limit,
                "image_size_limit": self.config.image_size_limit,
                "dpi": ocr_dpi,
            }
            # Only pass ignore_graphics when True - older pymupdf4llm versions don't accept it.
            if ignore_graphics:
                common_kwargs["ignore_graphics"] = True

            wants_parallel = total_page_count >= PARALLEL_PAGE_THRESHOLD
            # Office formats (DOC/DOCX/XLS/XLSX/PPT/PPTX/HWP/HWPX) MUST never
            # be processed inline by pymupdf4llm.to_markdown when we're inside
            # a daemonic worker. PyMuPDF Pro spawns native threads on import
            # (fontconfig + GNN model loaders); when task-queue prefork forks the
            # worker after `import pymupdf.layout`, only the calling thread
            # survives in the child but locks held by the now-dead helper
            # threads remain LOCKED forever. The first PyMuPDF call for an
            # office file in such a child blocks until the 15-min task-queue
            # time_limit kills it. PDFs avoid this because their parallel
            # path already routes through the non-daemonic subprocess helper
            # below; we extend the same protection to office files
            # regardless of page count.
            page_chunks: list[dict[str, Any]] | None = None

            if wants_parallel and not is_daemonic:
                # Direct parallel path: parent is non-daemonic, so we can spawn
                # ProcessPoolExecutor children in-process.
                page_chunks = self._extract_parallel(pdf_path, total_page_count, common_kwargs)
            elif wants_parallel and is_daemonic:
                # task-queue prefork workers are daemonic and cannot spawn pool
                # children directly. Delegate to a non-daemonic subprocess which
                # CAN fan out via ProcessPoolExecutor. Falls back to
                # single-threaded extraction below if the subprocess fails.
                page_chunks = self._extract_via_subprocess(
                    pdf_path, total_page_count, common_kwargs
                )
            elif is_office and is_daemonic:
                # Even small office files must use the subprocess helper in a
                # daemonic worker (see fork-deadlock comment above). The
                # helper plans a single-worker page range for tiny docs, which
                # gives us a fresh non-daemonic Python process - exactly what
                # we need to avoid inheriting locked PyMuPDF Pro thread state.
                page_chunks = self._extract_via_subprocess(
                    pdf_path, total_page_count, common_kwargs
                )

            if page_chunks is None:
                # Single-threaded path: small documents, or subprocess fallback.
                # IMPORTANT: for office files in a daemonic worker we MUST NOT
                # call ANY pymupdf.open() here - including via
                # _fallback_extract_text_only - because the .xlsx/.docx
                # path triggers PyMuPDF Pro's Excel→PDF conversion which
                # is the fork-hazard we are routing around.
                #
                # The subprocess (`_parse_page_range`) now has its own
                # AssertionError fallback that returns plain-text chunks
                # for office files even when pymupdf4llm.to_markdown
                # asserts on form-field extraction, so this branch should
                # almost never fire. If it does (genuine subprocess crash,
                # OOM, gate timeout), return empty chunks - the router
                # will see "low text content" and trigger MultiFormat
                # fallback which lives in another module entirely.
                if is_office and is_daemonic:
                    logger.warning(
                        "pymupdf4llm_office_subprocess_failed",
                        msg=(
                            "Subprocess extraction returned no chunks for an "
                            "office file in a daemonic worker. Returning empty "
                            "chunks - calling pymupdf.open inline here would "
                            "re-trigger the fork-deadlock we just routed around."
                        ),
                        ext=ext,
                    )
                    page_chunks = []
                else:
                    try:
                        page_chunks = pymupdf4llm.to_markdown(
                            pdf_path,
                            page_chunks=True,
                            extract_words=False,
                            **common_kwargs,
                        )
                    except AssertionError:
                        # pymupdf4llm crashes on some DOCX-converted PDFs when
                        # extracting form fields (pymupdf._as_pdf_document fails).
                        # Retry with form field extraction disabled.
                        logger.warning(
                            "pymupdf4llm_assertion_retry",
                            msg="Form field extraction failed, retrying without it",
                        )
                        page_chunks = self._fallback_extract_text_only(pdf_path)

            # Process pages
            full_text_parts: list[str] = []
            clean_text_parts: list[str] = []
            all_boxes: list[dict[str, Any]] = []
            processed_pages: list[dict[str, Any]] = []

            page_count = len(page_chunks)
            metadata: dict[str, Any] = {}

            for page_idx, chunk in enumerate(page_chunks):
                page_text = chunk.get("text", "")
                page_metadata = chunk.get("metadata", {})
                full_text_parts.append(page_text)

                # Get document metadata from first page
                if page_idx == 0 and page_metadata:
                    metadata = {
                        "title": page_metadata.get("title", ""),
                        "author": page_metadata.get("author", ""),
                        "subject": page_metadata.get("subject", ""),
                        "creator": page_metadata.get("creator", ""),
                        "producer": page_metadata.get("producer", ""),
                        "file_path": page_metadata.get("file_path", pdf_path),
                        "page_count": page_metadata.get("page_count", page_count),
                    }

                # Clean text (remove markdown formatting for search)
                page_clean_text = page_text

                # Extract semantic boxes from GNN layout engine (no bbox coords stored).
                # page_boxes provides class info (title, section-header, text, table, etc.)
                # used by chunker for [TITLE]/[SECTION] prefixes and header/footer filtering.
                page_boxes = chunk.get("page_boxes", [])

                for box in page_boxes:
                    box_class = box.get("class", "text")
                    pos = box.get("pos", [])
                    if len(pos) >= 2:
                        box_text = page_text[pos[0] : pos[1]].strip()
                    else:
                        box_text = ""

                    if box_text:
                        all_boxes.append(
                            {
                                "page_number": page_idx + 1,
                                "box_class": box_class,
                                "text": box_text,
                            }
                        )

                clean_text_parts.append(page_clean_text)

                # Build page data
                processed_page: dict[str, Any] = {
                    "page_number": page_idx + 1,
                    "text": page_text,
                    "text_clean": page_clean_text,
                    "char_count": len(page_text),
                    "word_count": len(page_text.split()),
                    "tables": len(chunk.get("tables", [])),
                    "images": len(chunk.get("images", [])),
                }

                processed_pages.append(processed_page)

            # Combine text
            full_text = "\n\n".join(full_text_parts)
            clean_text = "\n\n".join(clean_text_parts)

            # ✅ Apply OCR artifact cleaning to remove image placeholders
            cleaned_text, cleaning_stats = clean_ocr_artifacts(
                clean_text, preserve_structure=True, return_stats=True
            )

            # Box class stats
            box_class_counts: dict[str, int] = {}
            for box in all_boxes:
                cls = box["box_class"]
                box_class_counts[cls] = box_class_counts.get(cls, 0) + 1

            # Detect legacy fonts. SKIPPED for office files in daemonic
            # workers because the helper opens the file via pymupdf.open(),
            # which on .xlsx/.docx kicks the same Excel→PDF conversion path
            # that fork-deadlocks. Legacy fonts are a common
            # concern (Asees, Akhar, AnmolLipi); office files don't carry
            # those non-Unicode legacy fonts in practice.
            if skip_dangerous_preflights:
                font_info: dict[str, Any] = {
                    "has_legacy_fonts": False,
                    "legacy_fonts_found": [],
                    "all_fonts": [],
                }
            else:
                font_info = self._detect_legacy_fonts_from_parsed(None, pdf_path)

            total_chars = len(cleaned_text.strip())

            # Determine extraction mode for parser name
            extraction_mode = "layout" if PYMUPDF_LAYOUT_AVAILABLE else "words"
            parser_name = (
                f"pymupdf4llm-pro-{extraction_mode}"
                if PYMUPDF_PRO_AVAILABLE
                else f"pymupdf4llm-{extraction_mode}"
            )

            logger.debug(
                "pymupdf4llm_extraction_complete",
                pdf_path=pdf_path,
                page_count=page_count,
                total_chars=total_chars,
                box_count=len(all_boxes),
                ocr_used=use_ocr,
                layout_enabled=PYMUPDF_LAYOUT_AVAILABLE,
                extraction_mode=extraction_mode,
                # ✅ OCR artifact cleaning metrics
                artifacts_removed=cleaning_stats.artifacts_removed,
                char_reduction_pct=round(cleaning_stats.reduction_pct, 2),
            )

            return {
                "text": full_text,
                "text_clean": cleaned_text,  # ✅ Now uses cleaned text (artifacts removed)
                "pages": processed_pages,
                "boxes": all_boxes,
                "box_class_counts": box_class_counts,
                "page_count": page_count,
                "total_chars": total_chars,
                "metadata": metadata,
                "supports_positions": True,
                "parser_name": parser_name,
                "ocr_used": use_ocr,
                "layout_enabled": PYMUPDF_LAYOUT_AVAILABLE,
                "has_legacy_fonts": font_info.get("has_legacy_fonts", False),
                "legacy_fonts_found": font_info.get("legacy_fonts_found", []),
                # ✅ OCR artifact cleaning metadata
                "artifacts_removed": cleaning_stats.artifacts_removed,
                "cleaning_applied": True,
            }

        except Exception as e:
            logger.exception("pymupdf4llm_extraction_error", error=str(e))
            extraction_mode = "layout" if PYMUPDF_LAYOUT_AVAILABLE else "words"
            error_parser_name = (
                f"pymupdf4llm-pro-{extraction_mode}"
                if PYMUPDF_PRO_AVAILABLE
                else f"pymupdf4llm-{extraction_mode}"
            )
            return {
                "error": f"pymupdf4llm extraction failed: {e!s}",
                "parser_name": error_parser_name,
            }

    def _extract_parallel(
        self,
        pdf_path: str,
        total_pages: int,
        common_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract text from a large PDF in parallel using ProcessPoolExecutor.

        Splits pages into ranges, sends each to a child process running
        _parse_page_range(), then merges results in page order.

        Args:
            pdf_path: Path to the PDF file
            total_pages: Total page count
            common_kwargs: Shared kwargs for pymupdf4llm.to_markdown

        Returns:
            Merged list of page chunk dicts (same format as single-threaded path)
        """
        # Shared planner - same implementation used by the subprocess helper,
        # so output is byte-identical regardless of which code path ran.
        from legal_rag.ingestion.pymupdf_extract_subprocess import plan_page_ranges

        page_ranges = plan_page_ranges(total_pages, MAX_PARALLEL_WORKERS, MIN_PAGES_PER_WORKER)
        if not page_ranges:
            return []

        num_workers = len(page_ranges)

        logger.debug(
            "parallel_extraction_start",
            total_pages=total_pages,
            num_workers=num_workers,
            pages_per_worker=[len(r) for r in page_ranges],
        )

        # Build config dict for child processes (must be picklable by ProcessPoolExecutor)
        config_dict = {
            "pro_api_key": self.config.pro_api_key,
            **common_kwargs,
        }

        worker_args = [(pdf_path, pr, config_dict) for pr in page_ranges]

        # submit + as_completed instead of pool.map so we can attribute
        # worker crashes (BrokenProcessPool / OOM / pymupdf signal) to the
        # exact page range that failed.
        from concurrent.futures import as_completed as _as_completed

        results_by_first_page: dict[int, list[dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            future_to_range = {pool.submit(_parse_page_range, wa): wa[1] for wa in worker_args}
            for fut in _as_completed(future_to_range):
                pages = future_to_range[fut]
                first, last = pages[0], pages[-1]
                try:
                    results_by_first_page[first] = fut.result()
                except Exception as e:
                    raise RuntimeError(
                        f"pymupdf worker failed on pages {first}..{last}: {type(e).__name__}: {e}"
                    ) from e

        # Reassemble in ascending page order - as_completed yields in
        # completion order, so we must sort the map keys explicitly.
        merged: list[dict[str, Any]] = []
        for first in sorted(results_by_first_page):
            merged.extend(results_by_first_page[first])

        logger.debug(
            "parallel_extraction_complete",
            total_pages=total_pages,
            num_workers=num_workers,
            merged_chunks=len(merged),
        )

        return merged

    def _extract_via_subprocess(
        self,
        pdf_path: str,
        total_pages: int,
        common_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """
        Extract pages in parallel by delegating to a non-daemonic subprocess.

        This is the path taken when `multiprocessing.current_process().daemon`
        is True (task-queue prefork workers). The subprocess CAN spawn its own
        ProcessPoolExecutor children, so we get the same fan-out as the
        in-process parallel path.

        Hardening applied:
          - JSON (not pickle) for subprocess I/O. Payload is primitives only,
            removes the "attacker drops pickle at guessable path" RCE surface.
          - PyMuPDF Pro API key passed via env, NOT the on-disk JSON config.
            Prevents post-crash leakage via leftover tempfiles.
          - TemporaryDirectory context manager for atomic cleanup; even under
            a non-SIGKILL exception the entire directory goes away. For SIGKILL
            we rely on the worker-startup sweep (see sweep_stale_subprocess_tempdirs, called from
            a worker init hook).
          - POSIX RLIMIT_AS / RLIMIT_CPU applied to the subprocess tree via
            preexec_fn so an adversarial PDF can't pin 14 cores for 13 minutes
            or OOM the host.

        Returns:
            Merged list of page chunk dicts (identical shape to
            `_extract_parallel`), or None on failure. When None is returned the
            caller falls back to the single-threaded path.
        """
        t0 = time.monotonic()

        # Per-document subprocess timeout. Smaller PDFs get a shorter
        # budget so a stuck pymupdf bug surfaces sooner; large textbooks
        # get the full 850s ceiling. See compute_subprocess_timeout()
        # for the policy. The host-level RLIMIT_CPU/RLIMIT_AS guards
        # remain the absolute backstop regardless of this value.
        subprocess_timeout = compute_subprocess_timeout(total_pages)

        # Gate only big extractions so small/medium docs stay fast.
        # For pages below the heavy threshold the ProcessPoolExecutor fans
        # out to at most `total_pages // MIN_PAGES_PER_WORKER` children
        # (e.g. 3 for 100 pages), which the host absorbs easily even with
        # all 8 task-queue slots busy. The gate targets the catastrophe case:
        # multiple 1000+ page docs each spawning 14 children in parallel.
        heavy_threshold = self.config.heavy_page_threshold
        gate_limit = self.config.gate_limit
        gate_wait = self.config.gate_wait_seconds
        use_gate = (
            self.extraction_gate is not None and total_pages >= heavy_threshold and gate_limit > 0
        )

        try:
            guard = (
                self.extraction_gate.guard(limit=gate_limit, max_wait_seconds=gate_wait)
                if use_gate and self.extraction_gate is not None
                else contextlib.nullcontext("disabled")
            )
            with guard as gate_token:
                if use_gate and gate_token is None:
                    # The gate said no within the wait budget. Do NOT try to
                    # run the subprocess unguarded (that is exactly the
                    # thrash we are preventing). Return None so the caller
                    # falls back to the single-threaded extraction path.
                    logger.warning(
                        "subprocess_extraction_gate_timeout",
                        total_pages=total_pages,
                        limit=gate_limit,
                        waited_s=gate_wait,
                    )
                    _emit_subprocess_outcome(
                        outcome="gate_timeout",
                        total_pages=total_pages,
                        duration_s=time.monotonic() - t0,
                        gate_active=use_gate,
                        extra={"gate_limit": gate_limit, "wait_budget_s": gate_wait},
                    )
                    return None
                if use_gate:
                    logger.debug(
                        "subprocess_extraction_gate_acquired",
                        total_pages=total_pages,
                        limit=gate_limit,
                        token_kind=(
                            "disabled"
                            if gate_token == "disabled"
                            else "fail-open"
                            if gate_token == "fail-open"
                            else "acquired"
                        ),
                    )

                with tempfile.TemporaryDirectory(prefix="pymupdf_subproc_") as tmpdir:
                    config_path = os.path.join(tmpdir, "config.json")
                    out_path = os.path.join(tmpdir, "out.json")

                    cfg = {
                        "pdf_path": pdf_path,
                        "total_pages": total_pages,
                        "common_kwargs": common_kwargs,
                        # pro_api_key intentionally excluded here; passed on stdin.
                        "num_workers": MAX_PARALLEL_WORKERS,
                        "min_pages_per_worker": MIN_PAGES_PER_WORKER,
                    }
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f)

                    logger.info(
                        "subprocess_extraction_start",
                        pdf_path=pdf_path,
                        total_pages=total_pages,
                        timeout_s=subprocess_timeout,
                        gate_active=use_gate,
                    )

                    # The child inherits the parent environment. The Pro key goes to it on
                    # standard input, so it never touches disk (a SIGKILL of the child
                    # leaves no key in a leftover tempfile).
                    pro_key_input = (self.config.pro_api_key or "").encode("utf-8")

                    # Defensive resource limits for the subprocess tree.
                    # POSIX only. On Windows / non-POSIX we skip the preexec_fn.
                    preexec = _build_subprocess_rlimit_preexec() if os.name == "posix" else None

                    # argv array (shell=False), interpreter is sys.executable,
                    # module path is a module-level constant, both file paths come from the
                    # TemporaryDirectory we just created. No untrusted input reaches argv.
                    result = subprocess.run(  # nosec B603
                        [
                            sys.executable,
                            "-m",
                            SUBPROCESS_HELPER_MODULE,
                            "--config",
                            config_path,
                            "--out",
                            out_path,
                        ],
                        timeout=subprocess_timeout,
                        capture_output=True,
                        check=False,
                        input=pro_key_input,
                        preexec_fn=preexec,
                    )

                    if result.returncode != 0:
                        stderr_tail = (result.stderr or b"").decode("utf-8", errors="replace")[
                            -2000:
                        ]
                        logger.warning(
                            "subprocess_extraction_failed",
                            returncode=result.returncode,
                            stderr_tail=stderr_tail,
                        )
                        _emit_subprocess_outcome(
                            outcome="subprocess_crash",
                            total_pages=total_pages,
                            duration_s=time.monotonic() - t0,
                            gate_active=use_gate,
                            extra={"returncode": result.returncode},
                        )
                        return None

                    # Sanity-check the output file's ownership + permissions before
                    # deserializing. In a single-UID container this is always fine,
                    # but on a shared VM it prevents a co-tenant at the same UID
                    # from swapping in crafted JSON between our subprocess.run
                    # return and our open().
                    try:
                        st = os.stat(out_path)
                        if st.st_uid != os.getuid():
                            logger.warning(
                                "subprocess_output_unexpected_owner",
                                expected_uid=os.getuid(),
                                actual_uid=st.st_uid,
                            )
                            _emit_subprocess_outcome(
                                outcome="output_owner_mismatch",
                                total_pages=total_pages,
                                duration_s=time.monotonic() - t0,
                                gate_active=use_gate,
                            )
                            return None
                    except OSError as stat_err:
                        logger.warning(
                            "subprocess_output_stat_failed",
                            error=str(stat_err),
                        )
                        _emit_subprocess_outcome(
                            outcome="output_stat_failed",
                            total_pages=total_pages,
                            duration_s=time.monotonic() - t0,
                            gate_active=use_gate,
                        )
                        return None

                    with open(out_path, encoding="utf-8") as f:
                        payload = json.load(f)

                    if isinstance(payload, dict) and payload.get("__error__"):
                        logger.warning(
                            "subprocess_extraction_reported_error",
                            error=payload.get("__error__"),
                            error_type=payload.get("__error_type__"),
                            page_range=payload.get("__page_range__"),
                        )
                        _emit_subprocess_outcome(
                            outcome="subprocess_reported_error",
                            total_pages=total_pages,
                            duration_s=time.monotonic() - t0,
                            gate_active=use_gate,
                            extra={"child_error_type": payload.get("__error_type__")},
                        )
                        return None

                    if not isinstance(payload, list):
                        logger.warning(
                            "subprocess_extraction_bad_payload",
                            payload_type=type(payload).__name__,
                        )
                        _emit_subprocess_outcome(
                            outcome="bad_payload",
                            total_pages=total_pages,
                            duration_s=time.monotonic() - t0,
                            gate_active=use_gate,
                            extra={"payload_type": type(payload).__name__},
                        )
                        return None

                    elapsed = time.monotonic() - t0
                    logger.info(
                        "subprocess_extraction_complete",
                        total_pages=total_pages,
                        merged_chunks=len(payload),
                        elapsed_s=round(elapsed, 2),
                    )
                    _emit_subprocess_outcome(
                        outcome="success",
                        total_pages=total_pages,
                        duration_s=elapsed,
                        gate_active=use_gate,
                        extra={"merged_chunks": len(payload)},
                    )
                    return payload

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            logger.warning(
                "subprocess_extraction_timed_out",
                total_pages=total_pages,
                timeout_s=subprocess_timeout,
                elapsed_s=round(elapsed, 2),
            )
            _emit_subprocess_outcome(
                outcome="timeout",
                total_pages=total_pages,
                duration_s=elapsed,
                gate_active=use_gate,
                extra={"timeout_s": subprocess_timeout},
            )
            return None
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                "subprocess_extraction_exception",
                error=str(e),
                error_type=type(e).__name__,
            )
            _emit_subprocess_outcome(
                outcome="exception",
                total_pages=total_pages,
                duration_s=elapsed,
                gate_active=use_gate,
                extra={"error_type": type(e).__name__},
            )
            return None

    def _detect_legacy_fonts_from_parsed(self, parsed_doc: Any, pdf_path: str) -> dict[str, Any]:
        """Detect legacy fonts using parsed document or file."""
        try:
            # Try using internal document (with null safety - 2026-01-19 audit fix)
            if hasattr(parsed_doc, "_doc") and parsed_doc._doc is not None:
                return self._detect_legacy_fonts(parsed_doc._doc)
            # Fallback to opening file
            if pymupdf is not None:
                doc = pymupdf.open(pdf_path)
                result = self._detect_legacy_fonts(doc)
                doc.close()
                return result
            return {"has_legacy_fonts": False, "legacy_fonts_found": [], "all_fonts": []}
        except Exception:
            return {"has_legacy_fonts": False, "legacy_fonts_found": [], "all_fonts": []}

    def _extract_sections_from_boxes(self, boxes: list[dict[str, Any]]) -> list[DocumentSection]:
        """
        Extract document sections from box-class data.

        Uses box_class to identify section headers and titles.
        """
        sections: list[DocumentSection] = []
        current_section: DocumentSection | None = None
        section_content_parts: list[str] = []

        for box in boxes:
            box_class = box.get("box_class", "text")
            text = box.get("text", "")
            page_number = box.get("page_number", 1)

            if box_class in ("title", "section-header"):
                # Save previous section
                if current_section is not None:
                    current_section.content = " ".join(section_content_parts)
                    sections.append(current_section)
                    section_content_parts = []

                # Start new section
                level = 1 if box_class == "title" else 2
                current_section = DocumentSection(
                    title=text.strip(),
                    content="",
                    level=level,
                    page_number=page_number,
                    metadata={"box_class": box_class},
                )

            elif box_class not in SKIP_BOX_CLASSES:
                section_content_parts.append(text)

        # Save final section
        if current_section is not None:
            current_section.content = " ".join(section_content_parts)
            sections.append(current_section)

        return sections
