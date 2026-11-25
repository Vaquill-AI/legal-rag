"""A labeled-set accuracy and confusion harness for the router.

Run a list of ``Case`` objects through ``route`` and report accuracy per tag, the misses and
the confusion pairs.
Use it to tune a classifier prompt or to compare two taxonomies.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from legal_rag.providers.protocols import Embedder, LanguageModel
from legal_rag.retrieval.routing.config import Taxonomy
from legal_rag.retrieval.routing.decision import RouteDecision
from legal_rag.retrieval.routing.router import route
from legal_rag.retrieval.routing.state import RouterState, RoutingInput


@dataclass(frozen=True)
class Case:
    """One labeled turn. ``accept`` lists alternative intents that also count as correct."""

    query: str
    state: RouterState
    expected: str
    accept: tuple[str, ...] = field(default_factory=tuple)
    tag: str = "general"
    pasted_body: str | None = None

    def ok(self, intent: str) -> bool:
        return intent == self.expected or intent in self.accept


@dataclass
class EvalReport:
    total: int
    hits: int
    by_tag: dict[str, tuple[int, int]]
    misses: list[tuple[Case, str, float]]
    confusion: Counter[str]

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 0.0


async def evaluate(
    cases: list[Case],
    taxonomy: Taxonomy,
    *,
    llm: LanguageModel | None,
    embedder: Embedder | None = None,
    concurrency: int = 8,
) -> EvalReport:
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: Case) -> tuple[Case, RouteDecision]:
        if case.pasted_body:
            ri = RoutingInput(
                case.query, f"{case.query}\n\n{case.pasted_body}", has_pasted_body=True
            )
        else:
            ri = RoutingInput(case.query, case.query)
        async with sem:
            return case, await route(ri, case.state, taxonomy=taxonomy, llm=llm, embedder=embedder)

    results = await asyncio.gather(*[_one(c) for c in cases])
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: Counter[str] = Counter()
    misses: list[tuple[Case, str, float]] = []
    hits = 0
    for case, d in results:
        tally[case.tag][1] += 1
        if case.ok(d.intent):
            hits += 1
            tally[case.tag][0] += 1
        else:
            misses.append((case, d.intent, d.confidence))
            confusion[f"{case.expected} -> {d.intent}"] += 1
    return EvalReport(
        total=len(results),
        hits=hits,
        by_tag={t: (v[0], v[1]) for t, v in tally.items()},
        misses=misses,
        confusion=confusion,
    )


def format_report(report: EvalReport, *, misses_only: bool = False) -> str:
    out: list[str] = [f"ROUTER EVAL: {report.total} cases"]
    if not misses_only:
        for tag in sorted(report.by_tag):
            ok, n = report.by_tag[tag]
            out.append(f"  {tag:14} {ok:>3}/{n:<3}")
    out.append(f"OVERALL: {report.hits}/{report.total} = {100 * report.accuracy:.0f}%")
    for case, got, conf in report.misses:
        out.append(
            f"  MISS [{case.tag}] {case.expected} -> {got} ({conf:.2f}) :: {case.query[:70]}"
        )
    for pair, cnt in report.confusion.most_common():
        out.append(f"  CONFUSION {pair}: {cnt}")
    return "\n".join(out) + "\n"
