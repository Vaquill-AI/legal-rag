"""Script-aware token counting for routing heuristics.

Every "is this query short?" heuristic in the codebase used `len(text.split())`.
That silently misfires on every language without inter-word spaces: Chinese,
Japanese, Thai, Khmer, Lao, and Burmese all tokenize to **1 word** no matter how
long the sentence is.

Prod consequence (chat 32e51137, a Chinese-speaking US immigration attorney): a
58-character Chinese question counted as 1 word, so it tripped every `< 5 words`
short-query test, was force-condensed for the wrong reason, lost its explicit
"translate into Chinese" instruction, retrieved unrelated case law, and came back
in English about the wrong area of law. The user left a thumbs_down.

`count_words` is the drop-in replacement. It is a routing heuristic, not a
tokenizer: it exists to answer "is this utterance short?" consistently across
scripts, and deliberately does not try to match any model's BPE.
"""

from __future__ import annotations

import re

# Scripts written without spaces between words. A run of these characters carries
# roughly one word per 1-2 codepoints, so counting codepoints slightly OVER-counts
# words. That bias is the safe direction: it stops a long CJK sentence from being
# mistaken for a two-word fragment, and no heuristic here treats "long" as risky.
_SPACELESS_SCRIPT_RE = re.compile(
    r"[぀-ヿ"  # Hiragana + Katakana
    r"㐀-䶿"  # CJK Unified Ideographs Extension A
    r"一-鿿"  # CJK Unified Ideographs
    r"豈-﫿"  # CJK Compatibility Ideographs
    r"가-힯"  # Hangul syllables
    r"฀-๿"  # Thai
    r"຀-໿"  # Lao
    r"ក-៿"  # Khmer
    r"က-႟"  # Myanmar
    r"]"
)

# Average codepoints per word in spaceless scripts. Chinese words run 1-2
# characters; 1.5 keeps the estimate close without under-counting short queries.
_CHARS_PER_SPACELESS_WORD = 1.5


def count_words(text: str | None) -> int:
    """Approximate word count that works for spaceless scripts.

    Whitespace-delimited tokens are counted directly. Characters from spaceless
    scripts are counted separately and converted at `_CHARS_PER_SPACELESS_WORD`,
    so a Chinese sentence no longer collapses to a single "word".

    Returns 0 for empty or None input.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0

    spaceless_chars = len(_SPACELESS_SCRIPT_RE.findall(stripped))
    if not spaceless_chars:
        return len(stripped.split())

    # Count only the whitespace tokens that still hold non-spaceless content, so
    # a mixed query ("请解释 18 U.S.C. 1001") counts both halves once.
    remainder = _SPACELESS_SCRIPT_RE.sub(" ", stripped)
    latin_words = len(remainder.split())

    return latin_words + max(1, round(spaceless_chars / _CHARS_PER_SPACELESS_WORD))


def is_spaceless_script(text: str | None) -> bool:
    """True when the text contains characters from a script written without spaces.

    Use this to skip heuristics that are structurally meaningless for such text
    (ASCII pronoun sets, English phrase lists) rather than letting them silently
    return False.
    """
    return bool(text) and bool(_SPACELESS_SCRIPT_RE.search(text))
