"""
Content-based extraction-quality assessment.

A parser can report ``success`` and still hand back bytes that are not usable
human text. This happens whenever the source carries glyphs but no recoverable
character mapping, so the extracted "text" is really raw glyph indices, control
bytes, or mis-decoded code units. Every one of these renders fine on screen,
which is why they fool naive success checks.

Failure modes this module is designed to catch, regardless of the root cause:

1. Glyph-index noise. Type3 fonts with no ``ToUnicode`` (the classic
   "Print to PDF via PostScript/Distiller" artifact), subsetted TrueType/Type1
   with custom non-AGL encodings, and CID/Type0 ``Identity-H`` fonts with no
   ``ToUnicode``. Extraction yields control bytes in the ``\\x00``-``\\x1f``
   range, so the printable ratio collapses.
2. Mojibake. Wrong charset applied to a text/CSV file, or a broken byte->char
   path, producing a high density of U+FFFD replacement characters.
3. Private-Use-Area mapping. Symbolic fonts (Wingdings-style) and some legacy
   encoders map glyphs into the Unicode PUA. These are ``str.isprintable()`` so
   the printable check alone misses them.
4. Font-level DRM / glyph scrambling. Some commercial legal databases map
   real letters to scrambled codepoints. Output is alphanumeric (high printable
   ratio) but has almost no whitespace and absurdly long "words".

The assessment is deliberately content-based and font-agnostic: it works on the
extracted string, so it keeps catching new encoding variants we have never seen
without needing a per-font allowlist. Structural PDF inspection (Type3 +
no-ToUnicode) lives separately and is used only to enrich diagnostics, never as
the sole gate.

Note on legacy non-Unicode Indic fonts (Krutidev, DevLys, Shree, etc.): those
map to printable ASCII, so they read as "high printable ratio" gibberish and are
NOT caught here. They are handled by font-name detection in the PDF ingestor.
"""

from dataclasses import dataclass
from enum import Enum

# Sample the first N characters for all ratio checks. Matches the historical
# readability gate so behavior on already-handled documents is unchanged.
SAMPLE_SIZE = 2000

# Below this many non-whitespace chars there is nothing to judge.
MIN_CHARS_TO_JUDGE = 50

# Fraction of sampled chars that must be printable or whitespace. Real text in
# any script (Latin, CJK, Indic, Arabic, Hebrew, emoji) sits at ~1.0; glyph
# index / control-byte noise sits far below this.
MIN_PRINTABLE_RATIO = 0.80

# Max fraction of U+FFFD replacement characters before we call it mojibake.
# A handful can appear in otherwise-fine text, so the bar is generous.
MAX_REPLACEMENT_RATIO = 0.10

# Max fraction of Private-Use-Area codepoints before we call it symbol-mapped.
MAX_PUA_RATIO = 0.30

# DRM / glyph-scramble heuristics (preserved from the original readability gate).
# Known font-encryption schemes (some commercial legal databases) scramble Latin letters,
# so the output is dense ASCII with no whitespace. The "no spaces / long words"
# signals are therefore only meaningful when the text is mostly ASCII letters;
# applying them blindly would misfire on scriptio-continua scripts (Chinese,
# Japanese, Thai, Khmer, Lao) that legitimately omit inter-word spaces.
MIN_SPACE_RATIO = 0.02
MAX_AVG_WORD_LEN = 30.0
MIN_ASCII_ALPHA_RATIO_FOR_DRM = 0.50

_REPLACEMENT_CHAR = "�"


class TextQualityVerdict(str, Enum):
    """Why a piece of extracted text was accepted or rejected."""

    READABLE = "readable"
    TOO_SHORT = "too_short"  # accepted: not enough text to judge
    GLYPH_NOISE = "glyph_noise"  # control bytes / unmapped glyph indices
    MOJIBAKE = "mojibake"  # high U+FFFD density
    SYMBOL_MAPPED = "symbol_mapped"  # high Private-Use-Area density
    DRM_SCRAMBLED = "drm_scrambled"  # alphanumeric but no spaces / long words


@dataclass(frozen=True)
class TextQuality:
    """Result of a content-based extraction-quality assessment."""

    verdict: TextQualityVerdict
    is_readable: bool
    printable_ratio: float
    replacement_ratio: float
    pua_ratio: float
    space_ratio: float
    avg_word_len: float

    @property
    def reason(self) -> str:
        """Human-readable one-liner for logs."""
        return {
            TextQualityVerdict.READABLE: "extracted text looks readable",
            TextQualityVerdict.TOO_SHORT: "too little text to judge",
            TextQualityVerdict.GLYPH_NOISE: (
                f"only {self.printable_ratio:.0%} printable characters "
                "(unmapped glyph indices / control bytes, e.g. a Type3 or CID "
                "font with no ToUnicode mapping)"
            ),
            TextQualityVerdict.MOJIBAKE: (
                f"{self.replacement_ratio:.0%} replacement characters "
                "(wrong text encoding / mojibake)"
            ),
            TextQualityVerdict.SYMBOL_MAPPED: (
                f"{self.pua_ratio:.0%} Private-Use-Area characters "
                "(symbol font with no Unicode mapping)"
            ),
            TextQualityVerdict.DRM_SCRAMBLED: (
                "alphanumeric stream with almost no whitespace "
                "(copy-protected / font-encrypted PDF)"
            ),
        }[self.verdict]


def _is_pua(code_point: int) -> bool:
    """True if the code point is in any Unicode Private Use Area."""
    return (
        0xE000 <= code_point <= 0xF8FF
        or 0xF0000 <= code_point <= 0xFFFFD
        or 0x100000 <= code_point <= 0x10FFFD
    )


def assess_extraction_quality(text: str) -> TextQuality:
    """
    Assess whether extracted text is usable human-readable content.

    The checks run cheapest-first on a bounded sample. The first failing check
    wins, so the verdict names the dominant failure mode.

    Args:
        text: Extracted text from any parser.

    Returns:
        A :class:`TextQuality` with the verdict, a readability bool, and the
        individual ratios for observability.
    """
    stripped = text.strip() if text else ""

    # Compute ratios up-front so callers always get observability, even for the
    # short-circuit verdicts.
    sample = stripped[:SAMPLE_SIZE]
    sample_len = len(sample)

    if sample_len == 0 or len(stripped) < MIN_CHARS_TO_JUDGE:
        return TextQuality(
            verdict=TextQualityVerdict.TOO_SHORT,
            is_readable=True,
            printable_ratio=1.0,
            replacement_ratio=0.0,
            pua_ratio=0.0,
            space_ratio=0.0,
            avg_word_len=0.0,
        )

    # Private-Use-Area glyphs are category Co, which str.isprintable() reports
    # as non-printable. They ARE rendered glyphs though, so count them as
    # printable here and let the dedicated PUA check below classify them; that
    # keeps control-byte noise (Cc) and symbol-font mapping (Co) as distinct
    # verdicts instead of collapsing both into "glyph noise".
    pua = sum(1 for c in sample if _is_pua(ord(c)))
    printable = sum(1 for c in sample if c.isprintable() or c.isspace() or _is_pua(ord(c)))
    replacement = sample.count(_REPLACEMENT_CHAR)
    spaces = sum(1 for c in sample if c.isspace())
    ascii_alpha = sum(1 for c in sample if c.isascii() and c.isalpha())

    printable_ratio = printable / sample_len
    replacement_ratio = replacement / sample_len
    pua_ratio = pua / sample_len
    space_ratio = spaces / sample_len
    ascii_alpha_ratio = ascii_alpha / sample_len

    words = sample.split()
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0

    def _result(verdict: TextQualityVerdict) -> TextQuality:
        return TextQuality(
            verdict=verdict,
            is_readable=(verdict == TextQualityVerdict.READABLE),
            printable_ratio=printable_ratio,
            replacement_ratio=replacement_ratio,
            pua_ratio=pua_ratio,
            space_ratio=space_ratio,
            avg_word_len=avg_word_len,
        )

    # 1. Glyph-index noise: control bytes / unmapped glyphs. These often contain
    #    whitespace bytes (0x09/0x0a/0x20) by coincidence, which is exactly why
    #    the space-ratio and word-length checks below let them through.
    if printable_ratio < MIN_PRINTABLE_RATIO:
        return _result(TextQualityVerdict.GLYPH_NOISE)

    # 2. Mojibake: dense replacement characters.
    if replacement_ratio > MAX_REPLACEMENT_RATIO:
        return _result(TextQualityVerdict.MOJIBAKE)

    # 3. Private-Use-Area symbol mapping (printable but not real text).
    if pua_ratio > MAX_PUA_RATIO:
        return _result(TextQualityVerdict.SYMBOL_MAPPED)

    # 4. DRM / glyph-scramble: dense ASCII letters, almost no whitespace, long
    #    "words". Gated on ASCII dominance so scriptio-continua scripts
    #    (Chinese, Japanese, Thai) that legitimately omit spaces are not flagged.
    if ascii_alpha_ratio >= MIN_ASCII_ALPHA_RATIO_FOR_DRM:
        if space_ratio < MIN_SPACE_RATIO:
            return _result(TextQualityVerdict.DRM_SCRAMBLED)
        if avg_word_len > MAX_AVG_WORD_LEN:
            return _result(TextQualityVerdict.DRM_SCRAMBLED)

    return _result(TextQualityVerdict.READABLE)


def is_text_readable(text: str) -> bool:
    """Convenience wrapper: True if extracted text is usable human content."""
    return assess_extraction_quality(text).is_readable
