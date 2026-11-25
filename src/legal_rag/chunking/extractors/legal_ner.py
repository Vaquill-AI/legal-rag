"""
Legal Named Entity Recognition (NER) Extractor.

Uses transformer-based models (Legal-BERT) for accurate extraction of:
- Case names (Smith v. Jones)
- Statute references (17 U.S.C. § 107)
- Legal concepts (negligence, due process)
- Court names, judges, parties

Superior to regex patterns for handling:
- Malformed citations (OCR errors, spacing issues)
- Context-dependent entities
- Multi-word entity spans
- Legal terminology variations

References:
- Legal-BERT: https://huggingface.co/nlpaueb/legal-bert-base-uncased
- LegNER Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12631292/
"""

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Protocol

from legal_rag.chunking._logging import get_logger

logger = get_logger("legal_ner")


class TokenClassifier(Protocol):
    """A token-classification client (for example a hosted Legal-BERT NER model).

    ``token_classification`` returns one dict per entity with the keys
    ``entity_group`` (``PER``, ``ORG``, ...), ``word`` and ``end`` (character offset).
    """

    async def token_classification(self, text: str) -> list[dict[str, Any]]: ...


# ============================================
# Configuration
# ============================================

# Legal concepts dictionary (common legal terminology)
_CONCEPTS_FILE = Path(__file__).resolve().parent.parent / "data" / "legal_concepts.json"


def load_legal_concepts(path: Path | None = None) -> frozenset[str]:
    """Load the legal concept vocabulary from JSON (``{"concepts": [...]}``)."""
    with (path or _CONCEPTS_FILE).open(encoding="utf-8") as handle:
        return frozenset(json.load(handle)["concepts"])


LEGAL_CONCEPTS = load_legal_concepts()

# Names that begin with a word that would otherwise look like a signal word.
# These are real captions, so they are returned untouched.
_PROTECTED_PREFIX_RE = re.compile(
    r"^(?:In\s+the\s+Matter\s+of|In\s+re|Ex\s+parte|Ex\s+rel\.?|Matter\s+of"
    r"|Estate\s+of|Petition\s+of|Application\s+of)\b",
    re.IGNORECASE,
)

# Citation signals and sentence openers that precede a case name in running text.
_LEADING_SIGNAL_RE = re.compile(
    r"^(?:"
    r"As\s+(?:held|stated|noted|decided|explained|recognized)\s+(?:in|by)"
    r"|(?:See|Cf|Compare|Accord|Contra|But|Also|In|Under|Per|By|Following|Citing)\.?,?"
    r")\s+"
)


def strip_leading_signal_words(name: str) -> str:
    """Drop citation signals and prepositions that precede a party name.

    ``"In Smith"`` becomes ``"Smith"`` and ``"See also Smith"`` becomes ``"Smith"``.
    Captions that legitimately start with such a word (``In re Gault``,
    ``Ex parte Young``, ``Matter of Smith``) are returned unchanged, and a name
    made only of a signal word (the surname ``See``) is kept as it is.

    Known limit: a party name that genuinely begins with a signal word followed by more
    words (``Under Armour v. X``) loses that word, because the two cases cannot be told
    apart from the text alone.
    """
    stripped = name.strip()
    if _PROTECTED_PREFIX_RE.match(stripped):
        return stripped
    current = stripped
    while True:
        remainder = _LEADING_SIGNAL_RE.sub("", current, count=1)
        if remainder == current or not remainder.strip():
            break
        if _PROTECTED_PREFIX_RE.match(remainder):
            current = remainder
            break
        current = remainder
    return current.strip()


# ============================================
# Hybrid Legal NER Extractor
# ============================================


class HybridLegalNERExtractor:
    """
    Hybrid approach combining regex patterns with transformer-based NER.

    Strategy:
    1. Use regex patterns for high-precision pattern matching (case names, statutes)
    2. Fallback to concept dictionary for legal terminology
    3. (Future) Add Legal-BERT NER for context-dependent extraction

    This provides immediate value while setting foundation for transformer upgrade.
    """

    # Case name patterns: "Smith v. Jones", "Brown v. Board of Education"
    CASE_NAME_PATTERNS: ClassVar[list[str]] = [
        # Standard "v." format with proper capitalization
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
        # "versus" format
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+versus\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
        # With year in parentheses
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+\((\d{4})\)",
        # With reporter citation
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s+\d+\s+[\w.]+\s+\d+",
    ]

    # Statute patterns: "17 U.S.C. § 107", "42 USC 1983"
    STATUTE_PATTERNS: ClassVar[list[str]] = [
        # U.S. Code with section symbol
        r"\b(\d+)\s+U\.?S\.?C\.?\s+§\s*(\d+(?:[a-z])?(?:-\d+)?)\b",
        # U.S. Code without section symbol
        r"\b(\d+)\s+USC\s+(\d+(?:[a-z])?)\b",
        # Code of Federal Regulations
        r"\b(\d+)\s+C\.?F\.?R\.?\s+§?\s*(\d+\.\d+)\b",
        # State statutes: "Cal. Penal Code § 187"
        r"\b((?:[A-Z][a-z]+\.?\s+)+(?:Penal|Civil|Criminal|Revenue|Labor|Education)\s+Code)\s+§\s*(\d+(?:\.\d+)?)\b",
        # Section references: "Section 1983", "§ 107"
        r"\b(?:Section|§)\s+(\d+(?:[a-z])?(?:-\d+)?)\b",
        # Public Law: "Pub. L. 115-123"
        r"\bPub\.\s*L\.\s*(\d+-\d+)\b",
    ]

    def __init__(
        self,
        token_classifier: TokenClassifier | None = None,
        legal_concepts: frozenset[str] | set[str] | None = None,
    ):
        """
        Initialize hybrid NER extractor.

        Args:
            token_classifier: Optional model client. When given, case names are
                extracted with it first and regexes are the fallback.
            legal_concepts: Concept vocabulary. Defaults to ``data/legal_concepts.json``.
        """
        self.legal_concepts = set(legal_concepts) if legal_concepts is not None else LEGAL_CONCEPTS
        self.hf_client = token_classifier
        self.use_transformer_ner = token_classifier is not None

        logger.debug(
            "hybrid_legal_ner_initialized",
            concept_count=len(self.legal_concepts),
            transformer_enabled=self.use_transformer_ner,
        )

    async def extract_case_names(self, text: str) -> list[str]:
        """
        Extract case names using Legal-BERT NER (if available) or regex fallback.

        Args:
            text: Text to analyze

        Returns:
            List of normalized case names (e.g., ["Smith v. Jones"])

        Examples:
            >>> extractor.extract_case_names("In Smith v. Jones, the court held...")
            ['Smith v. Jones']
            >>> extractor.extract_case_names("Brown v. Board of Education (1954)")
            ['Brown v. Board of Education']
        """
        # Try BERT NER if available (uses dslim/bert-base-NER which works with Inference API)
        if self.use_transformer_ner and self.hf_client is not None:
            try:
                entities = await self.hf_client.token_classification(text)

                # Build case names from consecutive PER (person) or ORG entities around " v. "
                # General NER tags: PER, ORG, LOC, MISC
                # We need to reconstruct case names from parts
                case_names = []

                # Look for pattern: [ENTITY] v. [ENTITY]
                i = 0
                while i < len(entities):
                    entity = entities[i]
                    word = entity.get("word", "")

                    # Check if this might be start of a case name
                    if entity.get("entity_group") in ["PER", "ORG"]:
                        # Look ahead for " v. " pattern in text
                        end_pos = entity.get("end", 0)
                        # Check if " v. " appears within next 10 characters
                        text_segment = text[end_pos : end_pos + 10]
                        if " v. " in text_segment.lower() or " vs. " in text_segment.lower():
                            # Found potential case name, collect both sides
                            party1 = strip_leading_signal_words(word)
                            # Find next entity after " v. "
                            if i + 1 < len(entities):
                                next_entity = entities[i + 1]
                                if next_entity.get("entity_group") in ["PER", "ORG"]:
                                    party2 = next_entity.get("word", "")
                                    case_name = f"{party1} v. {party2}"
                                    case_names.append(case_name)

                    i += 1

                if case_names:
                    logger.debug(
                        "case_names_extracted_transformer", count=len(case_names), method="bert_ner"
                    )
                    return list(dict.fromkeys(case_names))  # Deduplicate

            except Exception as e:  # noqa: BLE001 - degrade gracefully by design
                logger.warning("bert_ner_failed", error=str(e), fallback="regex")
                # Fall through to regex fallback

        # Regex fallback
        case_names = []

        for pattern in self.CASE_NAME_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()

                # Basic format: "Party1 v. Party2"
                if len(groups) >= 2:
                    party1 = strip_leading_signal_words(groups[0])
                    party2 = groups[1].strip()

                    # Normalize to standard format
                    case_name = f"{party1} v. {party2}"
                    case_names.append(case_name)

        # Deduplicate while preserving case sensitivity for proper names
        unique_cases = list(dict.fromkeys(case_names))

        logger.debug("case_names_extracted_regex", count=len(unique_cases), text_length=len(text))

        return unique_cases

    def extract_statute_references(self, text: str) -> list[str]:
        """
        Extract statute references using regex patterns.

        Args:
            text: Text to analyze

        Returns:
            List of normalized statute references

        Examples:
            >>> extractor.extract_statute_references("17 USC 107 provides...")
            ['17 U.S.C. § 107']
            >>> extractor.extract_statute_references("42 C.F.R. § 447.10")
            ['42 C.F.R. § 447.10']
        """
        statutes = set()

        for pattern in self.STATUTE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                statute = match.group(0).strip()

                # Normalize format
                statute = self._normalize_statute(statute)
                statutes.add(statute)

        statute_list = sorted(statutes)

        logger.debug("statutes_extracted", count=len(statute_list), text_length=len(text))

        return statute_list

    def _normalize_statute(self, statute: str) -> str:
        """
        Normalize statute to standard format.

        Args:
            statute: Raw statute reference

        Returns:
            Normalized statute reference

        Examples:
            >>> self._normalize_statute("17 USC 107")
            '17 U.S.C. § 107'
            >>> self._normalize_statute("42 CFR 447.10")
            '42 C.F.R. § 447.10'
        """
        # Standardize USC format
        statute = re.sub(r"\b(\d+)\s+USC\s+(\d+)", r"\1 U.S.C. § \2", statute, flags=re.IGNORECASE)

        # Standardize CFR format
        statute = re.sub(
            r"\b(\d+)\s+CFR\s+§?\s*(\d+\.\d+)", r"\1 C.F.R. § \2", statute, flags=re.IGNORECASE
        )

        # Add section symbol if missing
        if "U.S.C." in statute and "§" not in statute:
            statute = re.sub(r"U\.S\.C\.\s+(\d+)", r"U.S.C. § \1", statute)

        if "C.F.R." in statute and "§" not in statute:
            statute = re.sub(r"C\.F\.R\.\s+(\d+\.\d+)", r"C.F.R. § \1", statute)

        return statute.strip()

    def extract_legal_concepts(self, text: str) -> list[str]:
        """
        Extract legal concepts using dictionary matching.

        Args:
            text: Text to analyze

        Returns:
            List of legal concepts found

        Examples:
            >>> extractor.extract_legal_concepts("The defendant showed negligence...")
            ['negligence']
            >>> extractor.extract_legal_concepts("Due process requires notice...")
            ['due process']
        """
        text_lower = text.lower()
        concepts = []

        for concept in self.legal_concepts:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(concept) + r"\b"
            if re.search(pattern, text_lower):
                concepts.append(concept)

        logger.debug("legal_concepts_extracted", count=len(concepts), text_length=len(text))

        return sorted(concepts)

    async def extract_all_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extract all legal entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with keys: 'cases', 'statutes', 'concepts'

        Example:
            >>> entities = await extractor.extract_all_entities(document_text)
            >>> print(entities['cases'])
            ['Smith v. Jones', 'Doe v. Roe']
            >>> print(entities['statutes'])
            ['17 U.S.C. § 107', '42 U.S.C. § 1983']
        """
        return {
            "cases": await self.extract_case_names(text),
            "statutes": self.extract_statute_references(text),
            "concepts": self.extract_legal_concepts(text),
        }


# ============================================
# Factory Function
# ============================================

_legal_ner_extractor: HybridLegalNERExtractor | None = None


def get_legal_ner_extractor() -> HybridLegalNERExtractor:
    """Get or create the global legal NER extractor instance."""
    global _legal_ner_extractor
    if _legal_ner_extractor is None:
        _legal_ner_extractor = HybridLegalNERExtractor()
    return _legal_ner_extractor
