"""Aggregate router comparison logs into a misroute report.

A deployment can log one structured record per turn that holds the new router's verdict and
the verdict of a baseline (an older rule cascade, say).
Those records mean little until aggregated.
This module reads JSON lines with ``event == "router_comparison"`` and prints where the router
and the baseline disagree, clustered so recurring patterns stand out.
It ignores every other line, so it is safe to point at a full application log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass

EVENT_NAME = "router_comparison"


@dataclass(frozen=True)
class _Row:
    intent: str
    baseline: str
    action: str
    confidence: float
    disagree: bool
    doc_count: int
    has_scope: bool
    has_pasted_body: bool
    request_id: str | None


def parse_line(line: str) -> _Row | None:
    """Parse one JSON log line into a row, or None if it is not a comparison record."""
    line = line.strip()
    if not line or EVENT_NAME not in line:
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("event") != EVENT_NAME:
        return None
    return _Row(
        intent=str(d.get("intent", "?")),
        baseline=str(d.get("baseline_intent", "?")),
        action=str(d.get("action", "?")),
        confidence=float(d.get("confidence") or 0.0),
        disagree=bool(d.get("disagree")),
        doc_count=int(d.get("doc_count") or 0),
        has_scope=bool(d.get("has_scope")),
        has_pasted_body=bool(d.get("has_pasted_body")),
        request_id=d.get("request_id"),
    )


def load(lines: list[str]) -> list[_Row]:
    return [r for line in lines if (r := parse_line(line)) is not None]


def build_report(
    rows: list[_Row], *, default_intent: str = "answer", min_cluster: int = 2, examples: int = 3
) -> str:
    if not rows:
        return f"No {EVENT_NAME} lines found in input.\n"
    n = len(rows)
    disagree = [r for r in rows if r.disagree]
    out = [
        f"ROUTER COMPARISON REPORT: {n} turns",
        f"  agreement with baseline: {100 * (n - len(disagree)) / n:.1f}%  ({len(disagree)} disagree)",
    ]
    would_flip = [r for r in disagree if r.action == "execute"]
    out.append(f"  disagreements the router would execute: {len(would_flip)}")
    tool_to_default = [
        r for r in disagree if r.intent == default_intent and r.baseline != default_intent
    ]
    out.append(
        f"  baseline fired a tool where the router would {default_intent}: {len(tool_to_default)}"
    )
    default_to_tool = [
        r for r in disagree if r.baseline == default_intent and r.intent != default_intent
    ]
    out.append(f"  router wants a tool where the baseline answered: {len(default_to_tool)}")
    clusters: Counter[tuple[str, str]] = Counter((r.baseline, r.intent) for r in disagree)
    shown = [(k, c) for k, c in clusters.most_common() if c >= min_cluster]
    if shown:
        out.append(f"DISAGREEMENT CLUSTERS (baseline -> router, count >= {min_cluster}):")
        for (baseline, intent), cnt in shown:
            sample = [r for r in disagree if r.baseline == baseline and r.intent == intent]
            avg = sum(r.confidence for r in sample) / len(sample)
            out.append(f"  {baseline:>14} -> {intent:<18} {cnt:>4}  (avg conf {avg:.2f})")
            ids = ", ".join(str(r.request_id) for r in sample[:examples] if r.request_id)
            if ids:
                out.append(f"      e.g. request_id: {ids}")
    else:
        out.append(f"No disagreement cluster reached the min-count threshold ({min_cluster}).")
    out.append("ROUTER ACTION MIX:")
    for action, cnt in Counter(r.action for r in rows).most_common():
        out.append(f"  {action:<18} {cnt:>5}  ({100 * cnt / n:.1f}%)")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="JSONL log file, or '-' for stdin")
    ap.add_argument("--min-cluster", type=int, default=2)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args(argv)
    if args.source == "-":
        lines = sys.stdin.read().splitlines()
    else:
        with open(args.source, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    sys.stdout.write(
        build_report(load(lines), min_cluster=args.min_cluster, examples=args.examples)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
