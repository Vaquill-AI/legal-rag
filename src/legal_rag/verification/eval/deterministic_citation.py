"""Deterministic citation-correctness scorer for the answer-quality benchmark.

The LLM judge's citation dimension is too noisy to trust at the per-item level:
scored on identical answer text it swung 0.20 vs 0.70. Any citation or retrieval
fix worth ~0.05-0.2 is invisible under that noise. This module replaces the
judge for the ONE dimension a rule can settle: does each [N] marker point to a
source whose content actually supports the adjacent claim.

Same input always yields the same output, so a fix that moves it moved a real
thing. It reuses the same content-overlap idea the product's citation grounder
uses (word bigrams + distinctive unigrams + title-token overlap), reimplemented
self-contained so the metric stays stable if the grounder changes.

Per marker [N]:
  - out_of_range: N < 1 or N > len(sources)  -> a fabricated index, unsupported.
  - supported:    source N overlaps the claim window at or above SUPPORT_FLOOR.
  - misattributed: source N is weak but another source clearly supports the claim
                   (best_score >= SUPPORT_FLOOR and source N << best).
  - weak_window:  no source strongly matches (a generic sentence); not counted
                  against the answer because there is nothing to attribute to.

Reported per answer:
  - support_rate: supported / (supported + out_of_range + misattributed)
    the deterministic stand-in for citation_correctness.
  - best_attribution_rate: fraction of in-range markers that cite the single
    best-overlapping source (a stricter "did it cite the right one").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITE = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[^\s\[\]\(\)\{\}<>,.;:!?\"'/\\|@#$%^&*+=~`-]+", re.UNICODE)

# A citation is supported if its source overlaps the claim window at or above
# this floor. We deliberately do NOT rank against the best-overlapping source:
# a correct concise statute citation must not be marked wrong just because a
# verbose regulation in the panel happens to share more words. A citation counts
# as misattributed only when its own source is below the floor AND a source above
# the floor existed (the model cited a weak source when a supporting one was
# available). Calibrated on captured answers.
SUPPORT_FLOOR = 4.0

_STOP = frozenset(
    [
        "the",
        "of",
        "and",
        "in",
        "to",
        "is",
        "was",
        "that",
        "for",
        "on",
        "with",
        "as",
        "at",
        "by",
        "from",
        "or",
        "an",
        "be",
        "this",
        "which",
        "not",
        "are",
        "but",
        "had",
        "has",
        "have",
        "been",
        "were",
        "would",
        "shall",
        "may",
        "can",
        "will",
        "it",
        "its",
        "also",
        "such",
        "under",
        "said",
        "any",
        "than",
        "other",
        "into",
        "upon",
        "between",
        "after",
        "before",
        "above",
        "below",
        "all",
        "each",
        "every",
        "no",
        "if",
        "so",
        "more",
        "most",
        "only",
        "very",
        "same",
        "own",
        "through",
        "during",
        "about",
        "against",
        "over",
        "when",
        "where",
        "while",
        "case",
        "court",
        "section",
        "act",
        "order",
        "matter",
        "law",
        "rule",
        "federal",
        "united",
        "states",
        "u.s",
        "united-states",
        "general",
    ]
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 2 and w not in _STOP]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _window(text: str, start: int, end: int) -> str:
    lo = max(0, start - 400)
    nl_before = text.rfind("\n", lo, start)
    s = nl_before + 1 if nl_before > lo else lo
    hi = min(len(text), end + 200)
    nl_after = text.find("\n", end, hi)
    e = nl_after if nl_after != -1 else hi
    return text[s:e]


@dataclass
class CitationReport:
    total_markers: int
    in_range: int
    out_of_range: int
    supported: int
    misattributed: int
    weak_window: int
    best_attributions: int

    @property
    def support_rate(self) -> float:
        judged = self.supported + self.out_of_range + self.misattributed
        return round(self.supported / judged, 4) if judged else 1.0

    @property
    def best_attribution_rate(self) -> float:
        return round(self.best_attributions / self.in_range, 4) if self.in_range else 1.0

    @property
    def out_of_range_rate(self) -> float:
        return round(self.out_of_range / self.total_markers, 4) if self.total_markers else 0.0


def score_citations(answer: str, sources: list[dict]) -> CitationReport:
    """Deterministically score every [N] marker in `answer` against `sources`."""
    n_src = len(sources)

    # Source fingerprints
    src_bigrams: list[set] = []
    src_unigrams: list[set] = []
    src_titles: list[set] = []
    for s in sources:
        toks = _tokenize(s.get("text") or "")
        src_bigrams.append(_bigrams(toks))
        src_unigrams.append(set(toks))
        src_titles.append(set(_tokenize(s.get("title") or s.get("filename") or "")))
    word_count: dict[str, int] = {}
    for u in src_unigrams:
        for w in u:
            word_count[w] = word_count.get(w, 0) + 1
    distinctive = {w for w, c in word_count.items() if c <= 2}

    def overlap(window_bg, window_uni, window_dist, idx) -> float:
        bg = len(window_bg & src_bigrams[idx])
        di = len(window_dist & src_unigrams[idx])
        ti = len(window_uni & src_titles[idx]) if src_titles[idx] else 0
        ti = ti if (bg or di) else 0
        return bg + di * 3.0 + ti * 5.0

    total = in_range = oor = supported = misattr = weak = best_attr = 0
    seen: set[int] = set()
    for m in _CITE.finditer(answer or ""):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        total += 1
        if n < 1 or n > n_src:
            oor += 1
            continue
        in_range += 1
        window = _window(answer, m.start(), m.end())
        w_toks = _tokenize(window)
        w_bg, w_uni, w_dist = _bigrams(w_toks), set(w_toks), set(w_toks) & distinctive
        scores = [overlap(w_bg, w_uni, w_dist, i) for i in range(n_src)]
        score_n = scores[n - 1]
        best = max(scores) if scores else 0.0
        if best > 0 and abs(score_n - best) < 1e-9:
            best_attr += 1
        if score_n >= SUPPORT_FLOOR:
            supported += 1
        elif best >= SUPPORT_FLOOR:
            misattr += 1
        else:
            weak += 1
    return CitationReport(total, in_range, oor, supported, misattr, weak, best_attr)


__all__ = ["SUPPORT_FLOOR", "CitationReport", "score_citations"]
