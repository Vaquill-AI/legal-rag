"""The pipeline package must stay free of private product, tenancy, vendor and settings names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "src" / "legal_rag" / "pipeline"

FORBIDDEN = {
    "vaquill": re.compile(r"vaquill", re.IGNORECASE),
    "supabase": re.compile(r"supabase", re.IGNORECASE),
    "qdrant": re.compile(r"qdrant", re.IGNORECASE),
    "organization_id": re.compile(r"organization_id"),
    "user_id": re.compile(r"user_id"),
    "matter_id": re.compile(r"matter_id"),
    "private app import": re.compile(r"(?<![A-Za-z_])app\."),
    "os.environ": re.compile(r"os\.environ|getenv"),
    "settings": re.compile(r"settings", re.IGNORECASE),
    "gpt-": re.compile(r"gpt-", re.IGNORECASE),
    "voyage": re.compile(r"voyage", re.IGNORECASE),
    "courtlistener": re.compile(r"courtlistener", re.IGNORECASE),
    "em dash": re.compile("—|–"),
    "ticket tag": re.compile(r"Round \d|bug-?fix #\d", re.IGNORECASE),
}


def _files() -> list[Path]:
    files = sorted(
        p for p in PIPELINE_DIR.rglob("*") if p.is_file() and p.suffix in {".py", ".json"}
    )
    assert files, f"no pipeline files found under {PIPELINE_DIR}"
    return files


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_pipeline_package_has_no_forbidden_text(label: str):
    pattern = FORBIDDEN[label]
    hits = []
    for path in _files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(PIPELINE_DIR)}:{number}: {line.strip()}")
    assert not hits, f"{label} found:\n" + "\n".join(hits)
