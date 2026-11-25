"""Hygiene scan of the tool schemas: no private names, no deployment coupling, no em dashes."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TOOLS_PATH = Path(__file__).resolve().parents[2] / "src" / "legal_rag" / "agents" / "tools.py"

FORBIDDEN = [
    "vaquill",
    "supabase",
    "qdrant",
    "organization_id",
    "user_id",
    "matter_id",
    "app.",
    "os.environ",
    "settings",
    "india",
    "courtlistener",
    "voyage",
    chr(0x2014),
    chr(0x2013),
]
FORBIDDEN_PATTERNS = {
    "ticket tag": r"\b(?:round|gap|edge|bug-fix|fix)\s*#?\d+",
    "internal doc path": r"\bdocs/",
    "corpus size figure": r"\b\d[\d,.]*\s*[KMB]\+?\s+(?:chunks|sections|opinions|points|documents)\b",
    "dated comment": r"#.*\b20\d\d-\d\d-\d\d\b",
}
ALLOWED_IMPORTS = {"__future__", "collections", "copy", "dataclasses", "re", "typing"}


@pytest.fixture(scope="module")
def source() -> str:
    return TOOLS_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_forbidden_text_is_absent(source: str, needle: str) -> None:
    assert needle not in source.lower(), f"{needle!r} must not appear in tools.py"


@pytest.mark.parametrize("label", sorted(FORBIDDEN_PATTERNS))
def test_forbidden_patterns_are_absent(source: str, label: str) -> None:
    match = re.search(FORBIDDEN_PATTERNS[label], source, flags=re.IGNORECASE)
    assert match is None, f"{label}: {match.group(0) if match else ''}"


def test_only_a_section_sign_is_non_ascii(source: str) -> None:
    extra = {ch for ch in source if ord(ch) > 127} - {"§"}
    assert not extra, f"unexpected non-ASCII characters: {sorted(extra)}"


def test_imports_are_standard_library_only(source: str) -> None:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not expected here"
            roots.add((node.module or "").split(".")[0])
    assert roots <= ALLOWED_IMPORTS, f"unexpected imports: {sorted(roots - ALLOWED_IMPORTS)}"
