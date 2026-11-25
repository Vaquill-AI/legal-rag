"""Guards that keep the retrieval code free of product-specific content."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "legal_rag"
SCOPES = [
    ROOT / "retrieval" / "base.py",
    ROOT / "retrieval" / "fusion.py",
    ROOT / "retrieval" / "strategies",
    ROOT / "retrieval" / "rerank",
    ROOT / "providers" / "adapters",
]
FORBIDDEN = {
    "private package import": re.compile(r"^\s*(?:from|import)\s+(?:app|scripts)\b", re.MULTILINE),
    "environment read": re.compile(r"os\.environ|os\.getenv|get_settings|\bsettings\.\w"),
    "tenancy key": re.compile(r"\b(?:user_id|organization_id|matter_id)\b"),
    "product name": re.compile(r"vaquill", re.IGNORECASE),
    "em dash": re.compile("—"),
    "indian wording": re.compile(r"\b(?:Hon'ble|IPC|CrPC|Indian)\b"),
}


def python_files():
    for scope in SCOPES:
        yield from ([scope] if scope.is_file() else sorted(scope.rglob("*.py")))


def test_files_exist():
    assert len(list(python_files())) > 20


def test_no_forbidden_content():
    problems = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                problems.append(f"{path.relative_to(ROOT)}: {label}")
    assert not problems, problems
