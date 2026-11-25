"""
Script-based language detection.

Detects the language of text, with special handling for:
- Non-Latin scripts (Devanagari, Tamil, Telugu, Arabic and others)
- Code-mixed text (two scripts in one passage)
- Romanized text (a non-English language written in the Latin script)
- Legal domain vocabulary

Methods:
1. Script-based detection (fast, accurate for native scripts)
2. Statistical detection using langdetect/fasttext
3. Domain-specific keyword detection
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from legal_rag.chunking._logging import get_logger

logger = get_logger("language_detector")

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "languages.json"


def _load_language_data() -> dict[str, Any]:
    with _DATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


_DATA = _load_language_data()


class ScriptType(str, Enum):
    """Writing systems the detector can recognise."""

    DEVANAGARI = "devanagari"  # Hindi, Marathi, Sanskrit, Nepali
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    GUJARATI = "gujarati"
    BENGALI = "bengali"
    GURMUKHI = "gurmukhi"  # Punjabi
    ODIA = "odia"
    ARABIC = "arabic"  # Arabic, Urdu
    LATIN = "latin"  # English, Transliterated text
    MIXED = "mixed"


@dataclass
class LanguageInfo:
    """
    Detected language information.

    Attributes:
        primary_language: Most likely language code
        primary_language_name: Full language name
        confidence: Detection confidence (0-1)
        script: Detected script type
        is_code_mixed: Whether text mixes multiple languages
        secondary_languages: Other languages detected
        is_transliterated: Whether native language is in Roman script
        word_count: Total words analyzed
    """

    primary_language: str
    primary_language_name: str
    confidence: float
    script: ScriptType
    is_code_mixed: bool = False
    secondary_languages: list[str] = field(default_factory=list)
    is_transliterated: bool = False
    word_count: int = 0
    script_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "primary_language_name": self.primary_language_name,
            "confidence": round(self.confidence, 3),
            "script": self.script.value,
            "is_code_mixed": self.is_code_mixed,
            "secondary_languages": self.secondary_languages,
            "is_transliterated": self.is_transliterated,
            "word_count": self.word_count,
            "script_distribution": {k: round(v, 3) for k, v in self.script_distribution.items()},
        }


class LanguageDetector:
    """
    Script-based language detector.

    Handles:
    - Native script detection (Devanagari, Tamil, Arabic and others)
    - Code-mixed text
    - Romanized text detection
    - Legal domain vocabulary

    Example:
        detector = LanguageDetector()

        info = detector.detect("This is a test")
        print(info.primary_language)  # "en"

        # A passage in a non-Latin script
        info = detector.detect("\u0647\u0630\u0627 \u0627\u062e\u062a\u0628\u0627\u0631")
        print(info.script)  # ScriptType.ARABIC
    """

    # Language metadata (data/languages.json)
    LANGUAGES: ClassVar[dict[str, dict[str, str | ScriptType]]] = {
        code: {"name": row["name"], "script": ScriptType(row["script"])}
        for code, row in _DATA["languages"].items()
    }

    # Unicode ranges for script detection
    SCRIPT_RANGES: ClassVar[dict[ScriptType, list[tuple[int, int]]]] = {
        ScriptType.DEVANAGARI: [
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF),
        ],  # Devanagari + Extended
        ScriptType.TAMIL: [(0x0B80, 0x0BFF)],
        ScriptType.TELUGU: [(0x0C00, 0x0C7F)],
        ScriptType.KANNADA: [(0x0C80, 0x0CFF)],
        ScriptType.MALAYALAM: [(0x0D00, 0x0D7F)],
        ScriptType.GUJARATI: [(0x0A80, 0x0AFF)],
        ScriptType.BENGALI: [(0x0980, 0x09FF)],
        ScriptType.GURMUKHI: [(0x0A00, 0x0A7F)],
        ScriptType.ODIA: [(0x0B00, 0x0B7F)],
        ScriptType.ARABIC: [
            (0x0600, 0x06FF),
            (0xFB50, 0xFDFF),
        ],  # Arabic + Arabic Presentation
        ScriptType.LATIN: [
            (0x0041, 0x007A),
            (0x00C0, 0x024F),
        ],  # Basic Latin + Extended
    }

    # Common words of a language written in Latin script (for romanization detection)
    ROMANIZED_WORDS: ClassVar[dict[str, set[str]]] = {
        lang: set(words) for lang, words in _DATA["romanized_words"].items()
    }

    # Legal terms that indicate language context
    LEGAL_TERMS_BY_LANGUAGE: ClassVar[dict[str, list[str]]] = _DATA["legal_terms"]

    def __init__(self, use_fasttext: bool = False):
        """
        Initialize language detector.

        Args:
            use_fasttext: Whether to use fasttext for statistical detection
        """
        self.use_fasttext = use_fasttext
        self._fasttext_model = None

    def detect(self, text: str) -> LanguageInfo:
        """
        Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageInfo with detection results
        """
        if not text or not text.strip():
            return LanguageInfo(
                primary_language="unknown",
                primary_language_name="Unknown",
                confidence=0.0,
                script=ScriptType.LATIN,
            )

        # Step 1: Analyze script distribution
        script_dist = self._analyze_scripts(text)

        # Step 2: Determine primary script
        primary_script = self._get_primary_script(script_dist)

        # Step 3: Check for code-mixing
        is_code_mixed = self._is_code_mixed(script_dist)

        # Step 4: Detect language from script + content
        primary_lang, confidence = self._detect_language(text, primary_script, script_dist)

        # Step 5: Check for transliteration
        is_transliterated = False
        if primary_script == ScriptType.LATIN:
            is_transliterated = self._is_transliterated(text)
            if is_transliterated:
                # Re-detect considering transliteration
                transliterated_lang = self._detect_transliterated_language(text)
                if transliterated_lang:
                    primary_lang = transliterated_lang
                    confidence = min(confidence, 0.7)  # Lower confidence for transliteration

        # Step 6: Detect secondary languages
        secondary_langs = self._detect_secondary_languages(text, primary_lang, script_dist)

        # Get language name
        lang_info = self.LANGUAGES.get(
            primary_lang, {"name": "Unknown", "script": ScriptType.LATIN}
        )

        words = text.split()

        return LanguageInfo(
            primary_language=primary_lang,
            primary_language_name=lang_info["name"],
            confidence=confidence,
            script=primary_script,
            is_code_mixed=is_code_mixed,
            secondary_languages=secondary_langs,
            is_transliterated=is_transliterated,
            word_count=len(words),
            script_distribution=script_dist,
        )

    def _analyze_scripts(self, text: str) -> dict[str, float]:
        """Analyze script distribution in text."""
        script_counts = {script.value: 0 for script in ScriptType}
        total_chars = 0

        for char in text:
            if char.isspace() or char in ".,!?;:()[]{}\"'-":
                continue

            code_point = ord(char)
            total_chars += 1

            for script, ranges in self.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script.value] += 1
                        break

        # Convert to percentages
        if total_chars > 0:
            return {k: v / total_chars for k, v in script_counts.items() if v > 0}
        return {}

    def _get_primary_script(self, script_dist: dict[str, float]) -> ScriptType:
        """Determine primary script from distribution."""
        if not script_dist:
            return ScriptType.LATIN

        primary = max(script_dist, key=lambda k: script_dist[k])
        return ScriptType(primary)

    def _is_code_mixed(self, script_dist: dict[str, float]) -> bool:
        """Check if text contains multiple scripts (code-mixing)."""
        significant_scripts = [s for s, pct in script_dist.items() if pct >= 0.1]
        return len(significant_scripts) > 1

    def _detect_language(
        self,
        text: str,
        script: ScriptType,
        script_dist: dict[str, float],
    ) -> tuple[str, float]:
        """Detect language from script and content."""
        # Script-based detection for native scripts
        script_to_lang = {
            ScriptType(name): code for name, code in _DATA["script_languages"].items()
        }

        if script in script_to_lang:
            return script_to_lang[script], 0.95

        # Devanagari is shared by several languages
        if script == ScriptType.DEVANAGARI:
            lang = self._distinguish_devanagari_language(text)
            return lang, 0.85

        # Arabic script: Urdu when Urdu-only letters appear, otherwise Arabic
        if script == ScriptType.ARABIC:
            if any(ch in text for ch in _DATA["urdu_markers"]):
                return "ur", 0.85
            return "ar", 0.85

        # Latin script - check for English vs transliterated
        if script == ScriptType.LATIN:
            if self._is_transliterated(text):
                return self._detect_transliterated_language(text) or "en", 0.6
            return "en", 0.9

        return "unknown", 0.0

    def _distinguish_devanagari_language(self, text: str) -> str:
        """Distinguish between the languages written in Devanagari."""
        # Indicators are matched directly against the original text (no case folding).
        counts = {
            lang: sum(1 for word in words if word in text)
            for lang, words in _DATA["devanagari_indicators"].items()
        }
        best = max(counts, key=lambda lang: counts[lang], default=None)
        if best is not None and counts[best] > 0:
            ties = [lang for lang, n in counts.items() if n == counts[best]]
            if len(ties) == 1:
                return best
            if "sa" in ties and counts["sa"] > 0 and len(ties) > 1:
                return "sa"
        return _DATA["default_devanagari_language"]

    def _is_transliterated(self, text: str) -> bool:
        """Check if Latin-script text is a romanized non-English language."""
        words = text.lower().split()
        if not words:
            return False
        return any(
            sum(1 for word in words if word in vocab) / len(words) >= 0.15
            for vocab in self.ROMANIZED_WORDS.values()
        )

    def _detect_transliterated_language(self, text: str) -> str | None:
        """Detect which language is romanized."""
        words = set(text.lower().split())
        best: tuple[int, str] | None = None
        for lang, vocab in self.ROMANIZED_WORDS.items():
            matches = len(words & vocab)
            if matches >= 3 and (best is None or matches > best[0]):
                best = (matches, lang)
        return best[1] if best else None

    def _detect_secondary_languages(
        self,
        text: str,
        primary_lang: str,
        script_dist: dict[str, float],
    ) -> list[str]:
        """Detect secondary languages in code-mixed text."""
        secondary = []

        for script_name, percentage in script_dist.items():
            if percentage >= 0.1:
                script = ScriptType(script_name)

                # Map script to language
                if script == ScriptType.LATIN and primary_lang != "en":
                    secondary.append("en")
                elif script == ScriptType.DEVANAGARI and primary_lang not in [
                    lang
                    for lang, row in self.LANGUAGES.items()
                    if row["script"] == ScriptType.DEVANAGARI
                ]:
                    secondary.append(_DATA["default_devanagari_language"])

        return list(set(secondary) - {primary_lang})

    def detect_batch(self, texts: list[str]) -> list[LanguageInfo]:
        """
        Detect languages for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of LanguageInfo results
        """
        return [self.detect(text) for text in texts]

    def get_supported_languages(self) -> dict[str, str]:
        """Get dictionary of supported languages."""
        return {code: info["name"] for code, info in self.LANGUAGES.items()}


# Singleton instance
_default_detector: LanguageDetector | None = None


def get_language_detector() -> LanguageDetector:
    """Get or create default language detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = LanguageDetector()
    return _default_detector


def detect_language(text: str) -> LanguageInfo:
    """
    Convenience function to detect language.

    Args:
        text: Text to analyze

    Returns:
        LanguageInfo with detection results
    """
    detector = get_language_detector()
    return detector.detect(text)
