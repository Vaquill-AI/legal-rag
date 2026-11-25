"""Standard-library fuzzy string ratios on a 0 to 100 scale.

A small stand-in for the ``ratio`` and ``partial_ratio`` functions of common fuzzy-matching packages, built on ``difflib`` so no extra dependency is needed.
"""

from __future__ import annotations

from difflib import SequenceMatcher


def ratio(a: str, b: str) -> int:
    """Similarity of two strings, 0 to 100."""
    if not a and not b:
        return 100
    return round(100 * SequenceMatcher(None, a, b).ratio())


def partial_ratio(a: str, b: str) -> int:
    """Best similarity of the shorter string against any equal-length window of the longer, 0 to 100."""
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if not shorter:
        return 100 if not longer else 0
    best = 0.0
    for start_in_short, start_in_long, _ in SequenceMatcher(
        None, shorter, longer
    ).get_matching_blocks():
        begin = max(start_in_long - start_in_short, 0)
        window = longer[begin : begin + len(shorter)]
        best = max(best, SequenceMatcher(None, shorter, window).ratio())
        if best > 0.995:
            break
    return round(100 * best)
