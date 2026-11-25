"""
Unified Hallucination Detector using rule-based pattern matching.

Provides both sync and async interfaces for hallucination detection using
rule-based claim extraction and key term matching for legal documents.

Two public interfaces are provided:
1. detect() - Sync interface (hallucination_score: 0=good, 1=bad)
2. check() - Async interface (score: 0=bad, 1=good)

The detector is rule-based and needs no ML model or GPU dependencies.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from legal_rag.verification._log import get_logger

logger = get_logger("legal_rag.verification.detector")


def fingerprint(text: str, n: int = 12) -> str:
    """Create fingerprint for correlation without storing content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


# ============================================================================
# Enums and Data Classes - Interface 1 (sync interface)
# ============================================================================


class HallucinationLevel(str, Enum):
    """Severity of hallucination detected."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class GroundingLevel(str, Enum):
    """Level of grounding for a claim (sync interface)."""

    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partial"
    UNGROUNDED = "ungrounded"
    CONTRADICTED = "contradicted"


@dataclass
class ClaimGrounding:
    """Grounding analysis for a single claim (sync interface)."""

    claim: str
    grounding_level: GroundingLevel
    supporting_evidence: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "grounding_level": self.grounding_level.value,
            "supporting_evidence": self.supporting_evidence,
            "source_chunks": self.source_chunks,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }


@dataclass
class HallucinationResult:
    """
    Result from hallucination detection (sync interface).

    Attributes:
        hallucination_score: 0.0 = no hallucination, 1.0 = fully hallucinated
    """

    is_hallucinated: bool
    hallucination_score: float  # 0=good, 1=bad (inverted from factual score)
    claims_analyzed: int = 0
    grounded_claims: int = 0
    partially_grounded_claims: int = 0
    ungrounded_claims: int = 0
    contradicted_claims: int = 0
    claim_analyses: list[ClaimGrounding] = field(default_factory=list)
    entities_found: set[str] = field(default_factory=set)
    entities_ungrounded: set[str] = field(default_factory=set)
    requires_review: bool = False
    suggested_action: str = ""
    detection_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_hallucinated": self.is_hallucinated,
            "hallucination_score": round(self.hallucination_score, 3),
            "claims_analyzed": self.claims_analyzed,
            "grounded_claims": self.grounded_claims,
            "partially_grounded_claims": self.partially_grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "contradicted_claims": self.contradicted_claims,
            "claim_analyses": [c.to_dict() for c in self.claim_analyses],
            "entities_found": list(self.entities_found),
            "entities_ungrounded": list(self.entities_ungrounded),
            "requires_review": self.requires_review,
            "suggested_action": self.suggested_action,
        }


# ============================================================================
# Enums and Data Classes - Interface 2 (async interface)
# ============================================================================


class VerificationResult(str, Enum):
    """Result of claim verification (async interface)."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NEUTRAL = "neutral"


@dataclass
class ClaimVerification:
    """Verification result for a single claim (async interface)."""

    claim: str
    result: VerificationResult
    confidence: float
    supporting_source: str | None = None


@dataclass
class HallucinationCheckResult:
    """
    Complete hallucination check result (async interface).

    Attributes:
        score: 0.0 = fully hallucinated, 1.0 = fully factual (inverted from hallucination_score)
    """

    score: float  # 0=bad, 1=good (factual grounding score)
    verified: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: list[str]
    claim_details: list[ClaimVerification]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection."""

    # Detection thresholds
    hallucination_threshold: float = 0.3  # Score above which response is hallucinated
    review_threshold: float = 0.15  # Score above which human review is recommended
    severe_threshold: float = 0.8
    high_threshold: float = 0.5
    medium_threshold: float = 0.2

    # Claim extraction
    min_claim_length: int = 10
    max_claims_to_check: int = 20

    # Matching
    min_overlap_ratio: float = 0.4
    enable_fuzzy_matching: bool = True


# ============================================================================
# Claim splitting helpers
# ============================================================================

# Periods inside these are not sentence boundaries. They are swapped for a
# placeholder before claims are split and restored afterwards.
_PERIOD_PLACEHOLDER = "\ue000"
_ABBREVIATION_WORDS = (
    "v|vs|Inc|Corp|Co|Ltd|No|Nos|Cir|Dist|Ct|Supp|App|St|Mr|Mrs|Ms|Dr|Jr|Sr|Cf|Id|Ex|Rel|Dep|"
    "Dept|Sec|Art|Cal|Mass|Ill|Pa|Tex|Fla|Mich|Wash|Conn|Colo|Ariz|Minn|Wis|Tenn|Ala|Okla|Nev|"
    "Ga|Va|Md|Ky"
)
_PROTECTED_PERIOD = re.compile(
    r"\b(?:[A-Za-z]\.){2,}"  # U.S., U.S.C., N.Y., D.C.
    r"|\b[A-Z]\.(?=\s+[A-Z0-9])"  # S. Ct., J. Smith
    rf"|\b(?:{_ABBREVIATION_WORDS})\."  # v., Inc., Corp., Cir., Cal.
    r"|(?<=\d)\.(?=\d)"  # section numbers such as 12.5
)


def _protect_abbreviation_periods(text: str) -> str:
    """Hide the periods of common legal abbreviations so they do not split a claim."""
    return _PROTECTED_PERIOD.sub(lambda m: m.group(0).replace(".", _PERIOD_PLACEHOLDER), text)


# ============================================================================
# Unified Hallucination Detector
# ============================================================================


class HallucinationDetector:
    """
    Unified Hallucination Detector using rule-based pattern matching.

    Provides two interfaces for backward compatibility:
    1. detect() - Sync interface (hallucination_score: 0=good, 1=bad)
    2. check() - Async interface (score: 0=bad, 1=good)

    Scoring rule (deliberate, lexical): a claim is graded by the share of its first
    ten key terms found in the context. At ``min_overlap_ratio`` (0.4) or more it
    counts as supported for the score, and from 0.8 it is ``grounded``, between
    the two it is ``partial``. Below 0.4 it is ``ungrounded`` and lowers the score
    (the score is supported claims over claims checked). So one grounded claim plus
    one ``partial`` claim scores 1.0 and ``verified=True`` while ``supported_claims``
    (grounded only) is 1 of 2, and ``unsupported_claims`` lists only ungrounded claims.
    Only sentences that match a claim pattern are checked once any sentence matches one.
    No claim is ever ``contradicted`` here.

    Example (sync):
        detector = get_hallucination_detector()
        result = detector.detect(response, context, citations)
        if result.is_hallucinated:
            print(f"Hallucination score: {result.hallucination_score}")

    Example (async):
        checker = get_hallucination_checker()
        result = await checker.check(response, sources)
        if not result.verified:
            print(f"Unsupported: {result.unsupported_claims}")
    """

    # Legal domain claim patterns
    CLAIM_PATTERNS: ClassVar[list[str]] = [
        r"([A-Z][^.!?]*(?:Section|Article|Rule|Regulation)\s+\d+[^.!?]*[.!?])",
        r"([A-Z][^.!?]*(?:is|are|was|were|has|have|provides|states|holds)[^.!?]*[.!?])",
        r"([A-Z][^.!?]*(?:shall|must|may|can|cannot|should|requires|mandates|prohibits)[^.!?]*[.!?])",
        r"([A-Z][^.!?]*(?:punishment|penalty|liable|imprisonment|fine)[^.!?]*[.!?])",
    ]

    STOPWORDS: ClassVar[set[str]] = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "as",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "to",
        "from",
        "up",
        "down",
    }

    def __init__(self, config: HallucinationConfig | None = None):
        """Initialize hallucination detector."""
        self.config = config or HallucinationConfig()

        self.stats = {
            "detected": 0,
            "rule_detections": 0,
            "avg_detection_time_ms": 0.0,
        }

    # ========================================================================
    # Interface 1: Sync detect()
    # ========================================================================

    def detect(
        self,
        response: str,
        context: str,
        citations: list[Any] | None = None,
    ) -> HallucinationResult:
        """
        Detect hallucinations (sync interface).

        Returns HallucinationResult with hallucination_score where 0=good, 1=bad.

        Args:
            response: Generated response to check
            context: Retrieved context (combined chunks)
            citations: Optional list of citations (unused, for API compatibility)

        Returns:
            HallucinationResult with hallucination_score and claim analyses
        """
        start_time = time.time()

        if not response or not response.strip():
            return HallucinationResult(
                is_hallucinated=False,
                hallucination_score=0.0,
                suggested_action="Empty response",
            )

        if not context or not context.strip():
            return HallucinationResult(
                is_hallucinated=True,
                hallucination_score=1.0,
                requires_review=True,
                suggested_action="No context available to verify response",
            )

        # Split context into list for rule-based detection
        contexts = [context] if isinstance(context, str) else context

        # Use rule-based detection
        factual_score, claim_analyses = self._detect_with_rules(response, contexts)
        self.stats["rule_detections"] += 1

        # Invert score: hallucination_score = 1 - factual_score
        hallucination_score = 1.0 - factual_score

        # Count claims by grounding level
        grounded = sum(1 for c in claim_analyses if c.grounding_level == GroundingLevel.GROUNDED)
        partial = sum(
            1 for c in claim_analyses if c.grounding_level == GroundingLevel.PARTIALLY_GROUNDED
        )
        ungrounded = sum(
            1 for c in claim_analyses if c.grounding_level == GroundingLevel.UNGROUNDED
        )
        contradicted = sum(
            1 for c in claim_analyses if c.grounding_level == GroundingLevel.CONTRADICTED
        )

        # Determine if hallucinated
        is_hallucinated = hallucination_score >= self.config.hallucination_threshold
        requires_review = hallucination_score >= self.config.review_threshold

        detection_time = (time.time() - start_time) * 1000
        self.stats["detected"] += 1
        self.stats["avg_detection_time_ms"] = (
            self.stats["avg_detection_time_ms"] * (self.stats["detected"] - 1) + detection_time
        ) / self.stats["detected"]

        logger.debug(
            "hallucination_detection_complete",
            hallucination_score=round(hallucination_score, 3),
            is_hallucinated=is_hallucinated,
            claims_analyzed=len(claim_analyses),
            detection_time_ms=round(detection_time, 2),
        )

        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            hallucination_score=hallucination_score,
            claims_analyzed=len(claim_analyses),
            grounded_claims=grounded,
            partially_grounded_claims=partial,
            ungrounded_claims=ungrounded,
            contradicted_claims=contradicted,
            claim_analyses=claim_analyses,
            requires_review=requires_review,
            suggested_action=self._get_suggested_action(
                hallucination_score, ungrounded, contradicted
            ),
            detection_time_ms=detection_time,
            metadata={"detection_method": "rule_based"},
        )

    # ========================================================================
    # Interface 2: Async check()
    # ========================================================================

    async def check(
        self,
        response: str,
        sources: list[dict],
    ) -> HallucinationCheckResult:
        """
        Check response for hallucinations (async interface).

        Returns HallucinationCheckResult with score where 0=bad, 1=good.

        Args:
            response: Generated response text to check
            sources: List of source chunks (dicts with 'text' field)

        Returns:
            HallucinationCheckResult with score and claim details
        """
        # Extract source texts
        source_texts = [s.get("text", "") for s in sources if s.get("text")]

        if not source_texts:
            return HallucinationCheckResult(
                score=0.5,
                verified=False,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                claim_details=[],
            )

        if not response or not response.strip():
            return HallucinationCheckResult(
                score=1.0,
                verified=True,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                claim_details=[],
            )

        # Use sync detection internally
        sync_result = self.detect(
            response=response,
            context=" ".join(source_texts),
            citations=None,
        )

        # Convert ClaimGrounding to ClaimVerification
        claim_details = []
        unsupported_claims = []

        for claim in sync_result.claim_analyses:
            if claim.grounding_level == GroundingLevel.GROUNDED:
                verification_result = VerificationResult.SUPPORTED
            elif claim.grounding_level == GroundingLevel.PARTIALLY_GROUNDED:
                verification_result = VerificationResult.NEUTRAL
            else:
                verification_result = VerificationResult.NOT_SUPPORTED
                unsupported_claims.append(claim.claim)

            claim_details.append(
                ClaimVerification(
                    claim=claim.claim,
                    result=verification_result,
                    confidence=claim.confidence,
                    supporting_source=claim.supporting_evidence[0]
                    if claim.supporting_evidence
                    else None,
                )
            )

        # Score is factual score (inverted from hallucination_score)
        score = 1.0 - sync_result.hallucination_score
        verified = score >= (1.0 - self.config.hallucination_threshold)

        return HallucinationCheckResult(
            score=score,
            verified=verified,
            total_claims=sync_result.claims_analyzed,
            supported_claims=sync_result.grounded_claims,
            unsupported_claims=unsupported_claims,
            claim_details=claim_details,
        )

    # ========================================================================
    # Internal Detection Methods
    # ========================================================================

    def _detect_with_rules(
        self,
        answer: str,
        contexts: list[str],
    ) -> tuple[float, list[ClaimGrounding]]:
        """Use rule-based pattern matching for detection."""
        claims = self._extract_claims(answer)
        combined_context = " ".join(contexts).lower()

        claim_analyses = []

        for claim in claims[: self.config.max_claims_to_check]:
            is_supported, confidence = self._check_claim_support(claim, combined_context)

            if is_supported:
                if confidence >= 0.8:
                    grounding_level = GroundingLevel.GROUNDED
                else:
                    grounding_level = GroundingLevel.PARTIALLY_GROUNDED
            else:
                grounding_level = GroundingLevel.UNGROUNDED

            claim_analyses.append(
                ClaimGrounding(
                    claim=claim,
                    grounding_level=grounding_level,
                    confidence=confidence,
                    explanation="Rule-based key term matching",
                )
            )

        # Calculate factual score
        supported = sum(
            1
            for c in claim_analyses
            if c.grounding_level in [GroundingLevel.GROUNDED, GroundingLevel.PARTIALLY_GROUNDED]
        )
        factual_score = supported / len(claim_analyses) if claim_analyses else 1.0

        return factual_score, claim_analyses

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text.

        Periods inside common legal abbreviations (``U.S.``, ``v.``, ``Inc.``,
        ``Cir.``, ``N.Y.``, ``S. Ct.``, numbers such as ``12.5``) do not split a claim.
        A sentence that ends in one of them is not split at that period either.
        """
        claims = []
        text = _protect_abbreviation_periods(text)

        for pattern in self.CLAIM_PATTERNS:
            matches = re.findall(pattern, text)
            claims.extend(matches)

        if not claims:
            sentences = re.split(r"[.!?]\s+", text)
            claims = [
                s.strip() + "." for s in sentences if len(s.strip()) >= self.config.min_claim_length
            ]

        claims = [c.replace(_PERIOD_PLACEHOLDER, ".") for c in claims]

        # Deduplicate
        seen = set()
        unique_claims = []
        for claim in claims:
            normalized = claim.lower().strip()
            if normalized not in seen and len(claim) >= self.config.min_claim_length:
                seen.add(normalized)
                unique_claims.append(claim)

        return unique_claims[: self.config.max_claims_to_check]

    def _check_claim_support(self, claim: str, context: str) -> tuple[bool, float]:
        """Check if claim is supported by context."""
        claim_terms = self._extract_key_terms(claim)

        if not claim_terms:
            return True, 1.0

        found_terms = sum(1 for term in claim_terms if term in context)
        overlap_ratio = found_terms / len(claim_terms)

        is_supported = overlap_ratio >= self.config.min_overlap_ratio
        return is_supported, overlap_ratio

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms for matching."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        key_terms = [w for w in words if w not in self.STOPWORDS]

        numbers = re.findall(r"\b\d+\b", text)
        key_terms.extend(numbers)

        legal_refs = re.findall(r"(?:Section|Article|Rule)\s+(\d+[A-Z]?)", text, re.IGNORECASE)
        key_terms.extend(legal_refs)

        return key_terms[:10]

    def _get_suggested_action(
        self,
        hallucination_score: float,
        ungrounded_claims: int,
        contradicted_claims: int,
    ) -> str:
        """Generate suggested action based on analysis."""
        if contradicted_claims > 0:
            return (
                f"CRITICAL: {contradicted_claims} claim(s) contradict source documents. "
                "Response should not be used without correction."
            )

        if hallucination_score >= 0.5:
            return (
                f"HIGH RISK: Significant hallucination detected (score: {hallucination_score:.2f}). "
                "Recommend regenerating with stricter grounding."
            )

        if hallucination_score >= self.config.hallucination_threshold:
            return (
                f"WARNING: Potential hallucination detected. "
                f"{ungrounded_claims} claim(s) could not be verified."
            )

        if hallucination_score >= self.config.review_threshold:
            return (
                "REVIEW RECOMMENDED: Some claims may require verification. "
                "Cross-reference with primary sources."
            )

        return "Response appears well-grounded in source documents."

    def get_stats(self) -> dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detected": self.stats["detected"],
            "rule_detections": self.stats["rule_detections"],
            "avg_detection_time_ms": round(self.stats["avg_detection_time_ms"], 2),
            "detection_method": "rule_based",
        }


# ============================================================================
# Singleton Instances
# ============================================================================

_detector: HallucinationDetector | None = None
_checker: HallucinationDetector | None = None


def get_hallucination_detector(
    config: HallucinationConfig | None = None,
) -> HallucinationDetector:
    """
    Get hallucination detector singleton (for sync interface).

    Example:
        detector = get_hallucination_detector()
        result = detector.detect(response, context, citations)
    """
    global _detector
    if _detector is None:
        _detector = HallucinationDetector(config)
    return _detector


def get_hallucination_checker(
    threshold: float = 0.8,
) -> HallucinationDetector:
    """
    Get hallucination checker singleton (for async interface).

    Example:
        checker = get_hallucination_checker()
        result = await checker.check(response, sources)
    """
    global _checker
    if _checker is None:
        config = HallucinationConfig(hallucination_threshold=1.0 - threshold)
        _checker = HallucinationDetector(config)
    return _checker
