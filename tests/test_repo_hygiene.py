"""Repo-wide guard: default code paths carry no jurisdiction-specific vocabulary from outside the US.

The opt-in data packs under ``chunking/data`` are the one place such vocabulary may live,
because a caller has to ask for a pack by name. Everything else must stay neutral, so a
query that mentions "FIR" or "SLP" is not treated as a legal concept by default.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "legal_rag"
OPT_IN_DATA = SRC / "chunking" / "data"

FORBIDDEN = {
    "commercial database names": re.compile(r"\b(?:SCC Online|Manupatra)\b", re.IGNORECASE),
    "honorific wording": re.compile(r"\bHon'ble\b", re.IGNORECASE),
    "indian statutes and bodies": re.compile(
        r"\b(?:IPC|CrPC|SARFAESI|ITAT|CESTAT|NCLT|Lok Adalat|Lokpal)\b", re.IGNORECASE
    ),
    "indian procedure terms": re.compile(
        r"\b(?:chargesheet|anticipatory bail|curative petition|special leave petition)\b",
        re.IGNORECASE,
    ),
    "indian keyword tokens": re.compile(r'"(?:fir|slp|pil|fera|poca|dpdp)"'),
}


def _default_sources() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if OPT_IN_DATA not in p.parents]


def test_default_code_paths_have_no_non_us_jurisdiction_vocabulary():
    hits: list[str] = []
    for path in _default_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in FORBIDDEN.items():
                if pattern.search(line):
                    hits.append(f"{path.relative_to(SRC)}:{number} ({label}): {line.strip()[:80]}")
    assert not hits, "\n".join(hits)


def test_the_scan_covers_the_package():
    assert len(_default_sources()) > 100
