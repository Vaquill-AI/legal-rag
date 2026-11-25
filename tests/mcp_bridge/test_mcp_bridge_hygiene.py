"""The bridge ships product-neutral and dependency-light: scan its source for leaks."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import legal_rag.mcp_bridge as bridge

BRIDGE_DIR = Path(bridge.__file__).parent

#: (label, pattern). Names and identifiers match case-insensitively; `app.` only as a
#: whole word so prose such as "wrapping." cannot trip it.
FORBIDDEN = [
    ("vaquill", re.compile(r"vaquill", re.IGNORECASE)),
    ("supabase", re.compile(r"supabase", re.IGNORECASE)),
    ("qdrant", re.compile(r"qdrant", re.IGNORECASE)),
    ("workspace", re.compile(r"workspace", re.IGNORECASE)),
    ("organization_id", re.compile(r"organization_id", re.IGNORECASE)),
    ("user_id", re.compile(r"user_id", re.IGNORECASE)),
    ("app.", re.compile(r"\bapp\.")),
    ("os.environ", re.compile(r"os\.environ")),
    ("an em dash", re.compile(chr(0x2014))),
]


def _sources() -> list[Path]:
    files = sorted(BRIDGE_DIR.glob("*.py"))
    assert {f.name for f in files} >= {
        "__init__.py",
        "catalogue.py",
        "dispatch.py",
        "naming.py",
        "openapi_enum.py",
        "result.py",
        "schema.py",
    }
    return files


def test_no_forbidden_string_appears_in_the_bridge_source():
    hits = [
        f"{path.name}:{number}: {label}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for label, pattern in FORBIDDEN
        if pattern.search(line)
    ]
    assert hits == []


def test_the_bridge_imports_nothing_outside_the_standard_library_and_itself():
    allowed_roots = {"__future__", "legal_rag", *sys.stdlib_module_names}
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            roots = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            offenders += [f"{path.name}: {root}" for root in roots if root not in allowed_roots]
    # `mcp` is imported lazily inside `ToolResult.to_mcp`, and only there.
    assert offenders == ["result.py: mcp"]


def test_importing_the_bridge_does_not_import_mcp_or_httpx():
    code = (
        "import sys, legal_rag.mcp_bridge;"
        "bad = sorted(m for m in ('mcp', 'httpx') if m in sys.modules);"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
