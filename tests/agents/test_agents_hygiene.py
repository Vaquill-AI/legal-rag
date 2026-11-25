"""The agents package ships product-neutral: scan its source for private names and leaks."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from legal_rag import agents

AGENTS_DIR = Path(agents.__file__).parent

#: Modules covered by their own tests. The tool-schema module is checked there.
EXCLUDED = {"tools.py"}

#: (label, pattern). Names match case-insensitively; `app.` only as a whole word so that prose
#: such as "wrapping." cannot trip it.
FORBIDDEN = [
    ("vaquill", re.compile(r"vaquill", re.IGNORECASE)),
    ("supabase", re.compile(r"supabase", re.IGNORECASE)),
    ("qdrant", re.compile(r"qdrant", re.IGNORECASE)),
    ("openai", re.compile(r"openai", re.IGNORECASE)),
    ("organization_id", re.compile(r"organization_id", re.IGNORECASE)),
    ("user_id", re.compile(r"user_id", re.IGNORECASE)),
    ("matter_id", re.compile(r"matter_id", re.IGNORECASE)),
    ("app.", re.compile(r"\bapp\.")),
    ("os.environ", re.compile(r"os\.environ")),
    ("settings", re.compile(r"settings", re.IGNORECASE)),
    ("gpt-", re.compile(r"gpt-", re.IGNORECASE)),
    ("an em dash", re.compile(chr(0x2014))),
]


def _sources() -> list[Path]:
    files = sorted(p for p in AGENTS_DIR.rglob("*.py") if p.name not in EXCLUDED)
    names = {p.relative_to(AGENTS_DIR).as_posix() for p in files}
    assert names >= {
        "__init__.py",
        "generation.py",
        "actors/__init__.py",
        "actors/base.py",
        "actors/planning.py",
        "actors/validation.py",
        "actors/refinement.py",
        "actors/deterministic.py",
    }
    return files


def test_no_forbidden_string_appears_in_the_agents_source():
    hits = [
        f"{path.relative_to(AGENTS_DIR)}:{number}: {label}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for label, pattern in FORBIDDEN
        if pattern.search(line)
    ]
    assert hits == []


def test_the_agents_import_only_the_standard_library_pydantic_and_the_library():
    allowed_roots = {"__future__", "legal_rag", "pydantic", *sys.stdlib_module_names}
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            roots = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            offenders += [f"{path.name}: {root}" for root in roots if root not in allowed_roots]
    assert offenders == []


def test_importing_the_agents_package_loads_no_actor_and_no_provider_sdk():
    code = (
        "import sys, legal_rag.agents;"
        "bad = sorted(m for m in sys.modules if m.startswith('legal_rag.agents.'));"
        "assert not bad, bad;"
        "sdk = sorted(m for m in ('openai', 'anthropic', 'httpx') if m in sys.modules);"
        "assert not sdk, sdk"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
