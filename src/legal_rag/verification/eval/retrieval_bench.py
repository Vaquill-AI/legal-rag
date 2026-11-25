"""Retrieval-recall benchmark with error attribution (offline, deterministic).

Separate from the end-to-end answer metrics (`ragas_eval`): this
measures whether the RIGHT passages are retrieved, and WHERE a gold passage is
lost (dense, sparse, fusion, rerank, or context-budget), so a bad answer can be
attributed to a retrieval miss vs a generation miss.

The core here is pure and dependency-free: metric functions plus a driver that
takes an injected ``retrieve_fn``. In a unit test the retriever is a stub (no
network or cost); in a live setting ``retrieve_fn`` wraps your real pipeline so
the same metrics run against it. Ground-truth construction (synthetic,
weak-label-mined, curated) is a separate step.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Ordered retrieval stages, coarse to fine. A gold id should survive each; the
# first stage that drops it localizes the failure.
STAGE_ORDER: tuple[str, ...] = ("dense", "sparse", "fused", "reranked", "context")


@dataclass(frozen=True)
class LabeledQuery:
    """One benchmark item: a query and the passage ids that truly answer it."""

    query: str
    gold_ids: frozenset[str]
    jurisdiction: str = "US"
    query_type: str = "semantic"  # "structured" | "semantic"
    corpus_snapshot: str = ""  # so a stale label is reportable, not silently scored
    source: str = "curated"  # "curated" | "synthetic" | "weak_label" | "self_retrieval"
    # How gold_ids should be scored:
    #   "all" - gold is a SET OF DISTINCT relevant documents (e.g. the several
    #           sources an answer cited); recall is the fraction of them found.
    #   "any" - gold is ONE document under several ALIASES (e.g. a statute indexed
    #           under 17 chapter paths); finding any one is complete success, so
    #           fraction-based recall would wrongly report 1/17 for a perfect hit.
    gold_mode: str = "all"  # "all" | "any"


@dataclass(frozen=True)
class StageIds:
    """The ranked ids a single retrieval run produced at each stage.

    A missing stage (None) is skipped in attribution (e.g. a tier without a
    separate sparse pass). ``context`` is the final set that reaches the LLM.
    """

    dense: Sequence[str] | None = None
    sparse: Sequence[str] | None = None
    fused: Sequence[str] | None = None
    reranked: Sequence[str] | None = None
    context: Sequence[str] = field(default_factory=tuple)
    # id -> human label (citation / title). Optional, for auditing a run by eye:
    # bare ids are unreadable when judging whether retrieval was actually good.
    labels: dict[str, str] = field(default_factory=dict)

    def by_stage(self) -> dict[str, Sequence[str] | None]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "fused": self.fused,
            "reranked": self.reranked,
            "context": self.context,
        }


# ── Pure metric functions ──────────────────────────────────────────────────


def recall_at_k(
    retrieved: Sequence[str], gold: Iterable[str], k: int, *, mode: str = "all"
) -> float:
    """Recall of gold in the top-k. 1.0 if there is no gold.

    ``mode="all"``: gold is a set of distinct relevant documents, so recall is the
    fraction found. ``mode="any"``: gold is one document under aliases, so finding
    any one is complete success (1.0) and fraction-based recall would understate a
    perfect hit as 1/len(aliases).
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    hits = set(retrieved[:k]) & gold_set
    if mode == "any":
        return 1.0 if hits else 0.0
    return len(hits) / len(gold_set)


def mrr_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Reciprocal rank of the FIRST gold id in the top-k (0.0 if none)."""
    gold_set = set(gold)
    for idx, rid in enumerate(retrieved[:k], start=1):
        if rid in gold_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int, *, mode: str = "all") -> float:
    """Binary-relevance nDCG at k (1.0 when no gold to find).

    In ``mode="any"`` gold is one document under aliases, so only the FIRST hit is
    credited (counting every alias would push nDCG above 1.0); the score is then
    just the rank discount, 1.0 for a rank-1 hit.
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    if mode == "any":
        for idx, rid in enumerate(retrieved[:k], start=1):
            if rid in gold_set:
                return 1.0 / math.log2(idx + 1)
        return 0.0
    dcg = sum(
        1.0 / math.log2(idx + 1)
        for idx, rid in enumerate(retrieved[:k], start=1)
        if rid in gold_set
    )
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg else 0.0


def stage_attribution(stages: StageIds, gold: Iterable[str], k: int) -> str:
    """Localize where the gold passages are lost.

    Returns the FIRST stage (in STAGE_ORDER) whose top-k contains no gold id,
    or "retrieved" when the gold survives all the way to the context. A stage
    that was not run (None) is skipped, not blamed.
    """
    gold_set = set(gold)
    if not gold_set:
        return "retrieved"
    by_stage = stages.by_stage()
    for stage in STAGE_ORDER:
        ids = by_stage.get(stage)
        if ids is None:
            continue
        if not (set(ids[:k]) & gold_set):
            return stage  # gold absent by this stage -> lost here
    return "retrieved"


# ── Aggregate driver ───────────────────────────────────────────────────────

# A retriever takes a query and returns the per-stage ranked ids for it.
RetrieveFn = Callable[[str], StageIds]
# Async variant: wraps the real (async) retrieval pipeline in the live env.
AsyncRetrieveFn = Callable[[str], Awaitable[StageIds]]


@dataclass
class BenchReport:
    """Aggregate benchmark results plus per-slice and attribution breakdowns."""

    n: int
    recall: dict[int, float]
    mrr: dict[int, float]
    ndcg: dict[int, float]
    attribution: dict[str, int]  # stage -> count of queries losing gold there
    by_query_type: dict[str, dict[int, float]]  # query_type -> {k: recall}
    by_jurisdiction: dict[str, dict[int, float]]
    # One record per query (query, gold, what was retrieved, hit/miss, where lost)
    # so a run is auditable by eye, not just an aggregate number.
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "recall": self.recall,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "attribution": self.attribution,
            "by_query_type": self.by_query_type,
            "by_jurisdiction": self.by_jurisdiction,
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(pairs: Sequence[tuple[LabeledQuery, StageIds]], ks: Sequence[int]) -> BenchReport:
    """Aggregate (item, stages) pairs into a report. Metrics use the ``context``
    stage (what reaches the LLM); attribution walks all stages to localize loss."""
    max_k = max(ks)
    recall_acc: dict[int, list[float]] = {k: [] for k in ks}
    mrr_acc: dict[int, list[float]] = {k: [] for k in ks}
    ndcg_acc: dict[int, list[float]] = {k: [] for k in ks}
    attribution: dict[str, int] = {}
    by_type: dict[str, dict[int, list[float]]] = {}
    by_juris: dict[str, dict[int, list[float]]] = {}

    details: list[dict] = []
    for item, stages in pairs:
        ctx = stages.context
        mode = item.gold_mode
        for k in ks:
            recall_acc[k].append(recall_at_k(ctx, item.gold_ids, k, mode=mode))
            mrr_acc[k].append(mrr_at_k(ctx, item.gold_ids, k))
            ndcg_acc[k].append(ndcg_at_k(ctx, item.gold_ids, k, mode=mode))
        where = stage_attribution(stages, item.gold_ids, max_k)
        attribution[where] = attribution.get(where, 0) + 1
        gold_set = set(item.gold_ids)
        details.append(
            {
                "query": item.query,
                "source": item.source,
                "jurisdiction": item.jurisdiction,
                "query_type": item.query_type,
                "gold_mode": mode,
                "gold_ids": sorted(gold_set),
                # what actually came back, with a readable label where we have one
                "retrieved": [
                    {"id": rid, "label": stages.labels.get(rid, ""), "is_gold": rid in gold_set}
                    for rid in list(ctx)[:max_k]
                ],
                "hit_rank": next(
                    (i for i, rid in enumerate(list(ctx)[:max_k], 1) if rid in gold_set), None
                ),
                "recall": {k: recall_at_k(ctx, item.gold_ids, k, mode=mode) for k in ks},
                "attribution": where,
            }
        )
        by_type.setdefault(item.query_type, {k: [] for k in ks})
        by_juris.setdefault(item.jurisdiction, {k: [] for k in ks})
        for k in ks:
            by_type[item.query_type][k].append(recall_at_k(ctx, item.gold_ids, k, mode=mode))
            by_juris[item.jurisdiction][k].append(recall_at_k(ctx, item.gold_ids, k, mode=mode))

    return BenchReport(
        n=len(pairs),
        recall={k: _mean(recall_acc[k]) for k in ks},
        mrr={k: _mean(mrr_acc[k]) for k in ks},
        ndcg={k: _mean(ndcg_acc[k]) for k in ks},
        attribution=attribution,
        by_query_type={t: {k: _mean(v[k]) for k in ks} for t, v in by_type.items()},
        by_jurisdiction={j: {k: _mean(v[k]) for k in ks} for j, v in by_juris.items()},
        details=details,
    )


def evaluate_retrieval(
    labeled: Sequence[LabeledQuery],
    retrieve_fn: RetrieveFn,
    ks: Sequence[int] = (5, 10, 20, 40),
) -> BenchReport:
    """Run a SYNC ``retrieve_fn`` over every labeled query and aggregate. Used with
    a stub in tests; the live pipeline is async, see ``evaluate_retrieval_async``."""
    return _aggregate([(item, retrieve_fn(item.query)) for item in labeled], ks)


async def evaluate_retrieval_async(
    labeled: Sequence[LabeledQuery],
    retrieve_fn: AsyncRetrieveFn,
    ks: Sequence[int] = (5, 10, 20, 40),
    concurrency: int = 1,
) -> BenchReport:
    """Run an ASYNC ``retrieve_fn`` (the real pipeline adapter) over every labeled
    query and aggregate.

    ``concurrency`` bounds parallel retrievals. A slow retriever makes a sequential
    run long; a handful in flight cuts it while staying inside provider rate limits. A query that raises
    is dropped from the report (logged by the caller) rather than killing the run.
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(item: LabeledQuery) -> tuple[LabeledQuery, StageIds] | None:
        async with sem:
            try:
                return item, await retrieve_fn(item.query)
            except Exception:  # noqa: BLE001 - a failing query is dropped, not fatal
                return None

    results = await asyncio.gather(*[_one(i) for i in labeled])
    pairs = [r for r in results if r is not None]
    return _aggregate(pairs, ks)


# ── Labeled-set IO (JSONL: one LabeledQuery per line) ──────────────────────


def load_labeled_set(path: str | Path) -> list[LabeledQuery]:
    """Read a JSONL labeled set. Each line: {query, gold_ids, jurisdiction?,
    query_type?, corpus_snapshot?, source?}. Blank lines are skipped."""
    out: list[LabeledQuery] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(
            LabeledQuery(
                query=str(rec["query"]),
                gold_ids=frozenset(str(g) for g in rec.get("gold_ids", [])),
                jurisdiction=str(rec.get("jurisdiction", "US")),
                query_type=str(rec.get("query_type", "semantic")),
                corpus_snapshot=str(rec.get("corpus_snapshot", "")),
                source=str(rec.get("source", "curated")),
                gold_mode=str(rec.get("gold_mode", "all")),
            )
        )
    return out


def write_bench_results(path: str | Path, report: BenchReport) -> dict[str, str]:
    """Persist a run for auditing: a summary JSON plus one JSONL line per query.

    The aggregate alone cannot be audited. The JSONL carries, for every query, the
    gold, exactly what came back (with labels), the hit rank, and where gold was
    lost, so a human can judge whether retrieval was actually any good.
    Returns the two paths written.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = base.with_suffix(".summary.json")
    details_path = base.with_suffix(".details.jsonl")
    summary_path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as fh:
        for d in report.details:
            fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    return {"summary": str(summary_path), "details": str(details_path)}
