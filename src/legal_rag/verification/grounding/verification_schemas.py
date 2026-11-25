"""
Pydantic models for RAG-native hallucination verification.

This module defines schemas for verifying AI-generated claims against
the actual source chunks that were fed to the LLM. This is "RAG-native"
verification - checking if the AI accurately represented what's in the
user's documents, NOT fact-checking against external sources.

Why RAG-Native Verification:
- When users choose RAG-only mode (web search OFF), they want answers from THEIR documents
- Verifying via web search would contradict this intent
- The source chunks stored with each message ARE the ground truth
- We answer: "Did the AI accurately quote/cite my documents?" not "Is this true?"
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ============================================
# Enums
# ============================================


class ClaimType(str, Enum):
    """Types of claims that can be extracted from AI responses."""

    DIRECT_QUOTE = "direct_quote"  # Exact text in quotes
    CITATION_REFERENCE = "citation_reference"  # [1], [2] style references
    PARAPHRASE = "paraphrase"  # Restated content from sources
    FACTUAL_ASSERTION = "factual_assertion"  # Specific facts, dates, numbers
    LEGAL_ENTITY = "legal_entity"  # Case names, statute references
    SUMMARY = "summary"  # Summarized content from multiple sources


class VerificationStatus(str, Enum):
    """Result of verifying a single claim."""

    VERIFIED = "verified"  # Claim matches source content
    PARTIALLY_VERIFIED = "partially_verified"  # Some aspects match, others don't
    UNVERIFIED = "unverified"  # No matching source found
    CONTRADICTED = "contradicted"  # Source says something different
    UNABLE_TO_VERIFY = "unable_to_verify"  # Can't determine (e.g., too vague)


class OverallVerificationStatus(str, Enum):
    """Overall verification result for a message."""

    FULLY_VERIFIED = "fully_verified"  # All claims verified
    MOSTLY_VERIFIED = "mostly_verified"  # >80% claims verified
    PARTIALLY_VERIFIED = "partially_verified"  # 50-80% claims verified
    MOSTLY_UNVERIFIED = "mostly_unverified"  # <50% claims verified
    NO_CLAIMS = "no_claims"  # No verifiable claims found
    WEB_SOURCES_ONLY = "web_sources_only"  # Sources are web-only; RAG verification not applicable
    ERROR = "error"  # Verification failed


# ============================================
# Claim Models
# ============================================


class ExtractedClaim(BaseModel):
    """
    A claim extracted from an AI response that can be verified.

    Claims are atomic statements that can be checked against source chunks.
    """

    model_config = ConfigDict(populate_by_name=True)

    claim_id: str = Field(
        ...,
        description="Unique identifier for this claim",
        validation_alias="claimId",
        serialization_alias="claimId",
    )
    claim_type: ClaimType = Field(
        ...,
        description="Type of claim (quote, citation, paraphrase, etc.)",
        validation_alias="claimType",
        serialization_alias="claimType",
    )
    text: str = Field(
        ...,
        description="The actual claim text extracted from the response",
        max_length=2000,
    )
    source_citation: str | None = Field(
        None,
        description="Citation reference if present (e.g., '[1]', '42 U.S.C. 1983')",
        validation_alias="sourceCitation",
        serialization_alias="sourceCitation",
    )
    position_start: int = Field(
        ...,
        description="Start position in the original response text",
        validation_alias="positionStart",
        serialization_alias="positionStart",
    )
    position_end: int = Field(
        ...,
        description="End position in the original response text",
        validation_alias="positionEnd",
        serialization_alias="positionEnd",
    )


class SourceMatch(BaseModel):
    """
    A source chunk that matches (or contradicts) a claim.
    """

    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(
        ...,
        description="ID of the source document",
        validation_alias="documentId",
        serialization_alias="documentId",
    )
    chunk_id: str | None = Field(
        None,
        description="ID of the specific chunk",
        validation_alias="chunkId",
        serialization_alias="chunkId",
    )
    filename: str = Field(
        ...,
        description="Name of the source file",
    )
    matched_text: str = Field(
        ...,
        description="The text from the source that matches/contradicts the claim",
        validation_alias="matchedText",
        serialization_alias="matchedText",
        max_length=2000,
    )
    similarity_score: float = Field(
        ...,
        description="Similarity score (0-1) between claim and source text",
        ge=0.0,
        le=1.0,
        validation_alias="similarityScore",
        serialization_alias="similarityScore",
    )
    match_type: str = Field(
        ...,
        description="Type of match: 'exact', 'semantic', 'partial', 'contradiction'",
        validation_alias="matchType",
        serialization_alias="matchType",
    )


class ConsistencyMetrics(BaseModel):
    """
    Multi-sample consistency metrics for detecting unreliable outputs.

    High variance across multiple samples indicates the model is uncertain
    about the answer, which correlates with hallucination risk.
    """

    model_config = ConfigDict(populate_by_name=True)

    sample_count: int = Field(
        ...,
        description="Number of samples generated for consistency check",
        validation_alias="sampleCount",
        serialization_alias="sampleCount",
    )
    variance_score: float = Field(
        ...,
        description="Variance across samples (0-1, lower is more consistent)",
        ge=0.0,
        le=1.0,
        validation_alias="varianceScore",
        serialization_alias="varianceScore",
    )
    is_consistent: bool = Field(
        ...,
        description="Whether the responses are consistent (variance below threshold)",
        validation_alias="isConsistent",
        serialization_alias="isConsistent",
    )
    agreement_rate: float = Field(
        ...,
        description="Rate of agreement across samples (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="agreementRate",
        serialization_alias="agreementRate",
    )


class GroundednessScore(BaseModel):
    """
    Sentence-level groundedness scoring for fine-grained verification.

    Measures how well each sentence in the response is grounded in
    the source documents, providing more granular feedback than claim-level.
    """

    model_config = ConfigDict(populate_by_name=True)

    overall_score: float = Field(
        ...,
        description="Overall groundedness score (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="overallScore",
        serialization_alias="overallScore",
    )
    sentence_scores: list[float] = Field(
        default_factory=list,
        description="Groundedness score for each sentence in the response",
        validation_alias="sentenceScores",
        serialization_alias="sentenceScores",
    )
    ungrounded_sentences: list[int] = Field(
        default_factory=list,
        description="Indices of sentences below groundedness threshold",
        validation_alias="ungroundedSentences",
        serialization_alias="ungroundedSentences",
    )
    is_well_grounded: bool = Field(
        ...,
        description="Whether the response meets the groundedness threshold",
        validation_alias="isWellGrounded",
        serialization_alias="isWellGrounded",
    )


class ClaimVerification(BaseModel):
    """
    Verification result for a single claim.

    Includes multiple verification signals for comprehensive hallucination detection:
    - Fuzzy/exact text matching (fast, basic)
    - LLM semantic verification (catches semantic errors)
    """

    model_config = ConfigDict(populate_by_name=True)

    claim: ExtractedClaim = Field(
        ...,
        description="The claim being verified",
    )
    status: VerificationStatus = Field(
        ...,
        description="Verification result",
    )
    confidence: float = Field(
        ...,
        description="Confidence in the verification result (0-1)",
        ge=0.0,
        le=1.0,
    )
    calibrated_confidence: float | None = Field(
        None,
        description="Calibrated confidence score adjusted for model overconfidence (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="calibratedConfidence",
        serialization_alias="calibratedConfidence",
    )
    source_matches: list[SourceMatch] = Field(
        default_factory=list,
        description="Source chunks that match this claim",
        validation_alias="sourceMatches",
        serialization_alias="sourceMatches",
    )
    explanation: str | None = Field(
        None,
        description="Human-readable explanation of the verification result",
        max_length=2000,
    )
    issue: str | None = Field(
        None,
        description="Specific issue identified by LLM verification (e.g., wrong number, missing qualifier)",
        max_length=1000,
    )
    verification_methods: list[str] = Field(
        default_factory=list,
        description="Methods used to verify this claim (e.g., 'fuzzy', 'llm')",
        validation_alias="verificationMethods",
        serialization_alias="verificationMethods",
    )


# ============================================
# Request/Response Models
# ============================================


class VerificationRequest(BaseModel):
    """
    Request to verify claims in an AI response.
    """

    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(
        ...,
        description="ID of the message to verify",
        serialization_alias="messageId",
    )
    response_text: str = Field(
        ...,
        description="The AI response text to verify",
        serialization_alias="responseText",
        max_length=50000,
    )
    sources: list[dict[str, Any]] = Field(
        ...,
        description="Source chunks that were used to generate the response",
        min_length=1,
    )


class VerificationResponse(BaseModel):
    """
    Complete verification result for an AI response.

    Includes comprehensive hallucination detection metrics:
    - Claim-level verification with multiple methods
    - Multi-sample consistency metrics
    - Sentence-level groundedness scores
    """

    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(
        ...,
        description="ID of the verified message",
        validation_alias="messageId",
        serialization_alias="messageId",
    )
    overall_status: OverallVerificationStatus = Field(
        ...,
        description="Overall verification status",
        validation_alias="overallStatus",
        serialization_alias="overallStatus",
    )
    verification_score: float = Field(
        ...,
        description="Overall verification score (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="verificationScore",
        serialization_alias="verificationScore",
    )
    calibrated_score: float | None = Field(
        None,
        description="Calibrated verification score adjusted for model overconfidence (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="calibratedScore",
        serialization_alias="calibratedScore",
    )
    total_claims: int = Field(
        ...,
        description="Total number of claims found",
        validation_alias="totalClaims",
        serialization_alias="totalClaims",
    )
    verified_claims: int = Field(
        ...,
        description="Number of verified claims",
        validation_alias="verifiedClaims",
        serialization_alias="verifiedClaims",
    )
    unverified_claims: int = Field(
        ...,
        description="Number of unverified claims",
        validation_alias="unverifiedClaims",
        serialization_alias="unverifiedClaims",
    )
    contradicted_claims: int = Field(
        ...,
        description="Number of contradicted claims",
        validation_alias="contradictedClaims",
        serialization_alias="contradictedClaims",
    )
    claim_verifications: list[ClaimVerification] = Field(
        default_factory=list,
        description="Detailed verification results for each claim",
        validation_alias="claimVerifications",
        serialization_alias="claimVerifications",
    )

    # Advanced verification metrics
    groundedness: GroundednessScore | None = Field(
        None,
        description="Sentence-level groundedness scoring for fine-grained verification",
    )
    consistency: ConsistencyMetrics | None = Field(
        None,
        description="Multi-sample consistency metrics for detecting unreliable outputs",
    )
    cross_reference_validation: dict[str, Any] | None = Field(
        None,
        description="Cross-reference citation validation results (index validity, quote matching)",
        validation_alias="crossReferenceValidation",
        serialization_alias="crossReferenceValidation",
    )
    verification_methods_used: list[str] = Field(
        default_factory=list,
        description="All verification methods used in this verification",
        validation_alias="verificationMethodsUsed",
        serialization_alias="verificationMethodsUsed",
    )
    citation_chunk_map: dict[int, str | None] = Field(
        default_factory=dict,
        description="Maps citation index (1, 2, 3...) to the source chunk_id that was cited. "
        "None if chunk_id is unavailable for that citation.",
        validation_alias="citationChunkMap",
        serialization_alias="citationChunkMap",
    )

    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of verification",
        validation_alias="verifiedAt",
        serialization_alias="verifiedAt",
    )
    processing_time_ms: int = Field(
        ...,
        description="Time taken to verify in milliseconds",
        validation_alias="processingTimeMs",
        serialization_alias="processingTimeMs",
    )


class VerificationSummary(BaseModel):
    """
    Lightweight verification summary for display in UI badges.
    """

    model_config = ConfigDict(populate_by_name=True)

    overall_status: OverallVerificationStatus = Field(
        ...,
        description="Overall verification status",
        serialization_alias="overallStatus",
    )
    verification_score: float = Field(
        ...,
        description="Overall verification score (0-1)",
        ge=0.0,
        le=1.0,
        serialization_alias="verificationScore",
    )
    total_claims: int = Field(
        ...,
        description="Total number of claims found",
        serialization_alias="totalClaims",
    )
    verified_claims: int = Field(
        ...,
        description="Number of verified claims",
        serialization_alias="verifiedClaims",
    )
    has_contradictions: bool = Field(
        ...,
        description="Whether any claims were contradicted by sources",
        serialization_alias="hasContradictions",
    )


# ============================================
# Exa Web Verification Models
# ============================================
# These models are used when verifying claims against web sources
# via Exa search. ONLY used when:
# 1. Matter Documents toggle is OFF (user not using their documents)
# 2. Web search is ON (user doing pure web research)
# ============================================


class ExaVerificationAssessment(str, Enum):
    """Assessment of a claim based on Exa web sources."""

    SUPPORTED = "supported"  # Web sources support the claim
    REFUTED = "refuted"  # Web sources contradict the claim
    PARTIALLY_SUPPORTED = "partially_supported"  # Some aspects supported, others not
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # No clear web evidence found
    OUTDATED = "outdated"  # Claim may be outdated based on newer web sources


class ExaSourceMatch(BaseModel):
    """
    A web source from Exa that matches or contradicts a claim.
    """

    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(
        ...,
        description="URL of the web source",
    )
    title: str = Field(
        ...,
        description="Title of the web page",
    )
    snippet: str = Field(
        ...,
        description="Relevant text snippet from the source",
        max_length=2000,
    )
    published_date: str | None = Field(
        None,
        description="Publication date of the source (if available)",
        validation_alias="publishedDate",
        serialization_alias="publishedDate",
    )
    author: str | None = Field(
        None,
        description="Author of the source (if available)",
    )
    relevance_score: float = Field(
        ...,
        description="Exa relevance score for this source (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="relevanceScore",
        serialization_alias="relevanceScore",
    )
    assessment: str = Field(
        ...,
        description="Whether this source supports, refutes, or is neutral on the claim",
    )


class ExaClaimVerification(BaseModel):
    """
    Verification result for a single claim against web sources.
    """

    model_config = ConfigDict(populate_by_name=True)

    claim: str = Field(
        ...,
        description="The claim being verified",
    )
    assessment: ExaVerificationAssessment = Field(
        ...,
        description="Overall assessment of the claim based on web sources",
    )
    confidence_score: float = Field(
        ...,
        description="Confidence in the assessment (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="confidenceScore",
        serialization_alias="confidenceScore",
    )
    supporting_sources: list[ExaSourceMatch] = Field(
        default_factory=list,
        description="Web sources that support the claim",
        validation_alias="supportingSources",
        serialization_alias="supportingSources",
    )
    refuting_sources: list[ExaSourceMatch] = Field(
        default_factory=list,
        description="Web sources that refute the claim",
        validation_alias="refutingSources",
        serialization_alias="refutingSources",
    )
    explanation: str = Field(
        "",
        description="Human-readable explanation of the verification result",
        max_length=2000,
    )
    search_query: str = Field(
        "",
        description="The search query used to find sources for this claim",
        validation_alias="searchQuery",
        serialization_alias="searchQuery",
    )


class ExaVerificationResult(BaseModel):
    """
    Complete Exa web verification result for all claims in a response.
    """

    model_config = ConfigDict(populate_by_name=True)

    total_claims: int = Field(
        ...,
        description="Total number of claims extracted and verified",
        validation_alias="totalClaims",
        serialization_alias="totalClaims",
    )
    verified_claims: int = Field(
        ...,
        description="Number of claims supported by web sources",
        validation_alias="verifiedClaims",
        serialization_alias="verifiedClaims",
    )
    refuted_claims: int = Field(
        ...,
        description="Number of claims refuted by web sources",
        validation_alias="refutedClaims",
        serialization_alias="refutedClaims",
    )
    insufficient_claims: int = Field(
        ...,
        description="Number of claims with insufficient web evidence",
        validation_alias="insufficientClaims",
        serialization_alias="insufficientClaims",
    )
    overall_confidence: float = Field(
        ...,
        description="Overall confidence in the verification (0-1)",
        ge=0.0,
        le=1.0,
        validation_alias="overallConfidence",
        serialization_alias="overallConfidence",
    )
    claim_verifications: list[ExaClaimVerification] = Field(
        default_factory=list,
        description="Detailed verification results for each claim",
        validation_alias="claimVerifications",
        serialization_alias="claimVerifications",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about potential issues (e.g., outdated info)",
    )
    processing_time_ms: int = Field(
        ...,
        description="Time taken to verify in milliseconds",
        validation_alias="processingTimeMs",
        serialization_alias="processingTimeMs",
    )


# ============================================
# Real-Time Verification Progress Models
# ============================================
# These models support streaming verification with
# real-time progress updates via SSE events.
# ============================================


class VerificationStep(str, Enum):
    """Steps in the verification pipeline."""

    STARTED = "started"  # Verification initiated
    EXTRACTING_CLAIMS = "extracting_claims"  # Extracting claims from response
    CLAIMS_EXTRACTED = "claims_extracted"  # Claims extraction complete
    VERIFYING_CLAIM = "verifying_claim"  # Verifying a specific claim
    CLAIM_VERIFIED = "claim_verified"  # Single claim verification complete
    EXACT_MATCHING = "exact_matching"  # Running exact text matching
    FUZZY_MATCHING = "fuzzy_matching"  # Running fuzzy/Levenshtein matching
    LLM_VERIFICATION = "llm_verification"  # Running LLM semantic verification

    CROSS_REFERENCE = "cross_reference"  # Running cross-reference validation
    CONSISTENCY_CHECK = "consistency_check"  # Running multi-sample consistency
    GROUNDEDNESS_SCORING = "groundedness_scoring"  # Computing groundedness scores
    COMPUTING_SCORES = "computing_scores"  # Calculating final scores
    EXA_VERIFICATION = "exa_verification"  # Running Exa web verification (when web search ON)
    # Exa-specific granular steps
    EXA_EXTRACTING_CLAIMS = "exa_extracting_claims"  # Extracting claims for web verification
    EXA_CLAIMS_EXTRACTED = "exa_claims_extracted"  # Claims extracted, showing count
    EXA_SEARCHING_CLAIM = "exa_searching_claim"  # Searching web for a specific claim
    EXA_VERIFYING_CLAIM = "exa_verifying_claim"  # Verifying claim against web sources
    EXA_CLAIM_VERIFIED = "exa_claim_verified"  # Single claim web verification complete
    COMPLETED = "completed"  # Verification complete
    ERROR = "error"  # Verification encountered an error


class VerificationProgressEvent(BaseModel):
    """
    Real-time progress event for streaming verification.

    Sent as SSE events to the frontend to show users exactly
    what's happening during verification.
    """

    model_config = ConfigDict(populate_by_name=True)

    step: VerificationStep = Field(
        ...,
        description="Current verification step",
    )
    message: str = Field(
        ...,
        description="Human-readable progress message",
    )
    progress_percent: int = Field(
        ...,
        description="Overall progress percentage (0-100)",
        ge=0,
        le=100,
        serialization_alias="progressPercent",
    )

    # Context about what's being processed
    current_claim_index: int | None = Field(
        None,
        description="Index of the claim currently being verified (0-based)",
        serialization_alias="currentClaimIndex",
    )
    total_claims: int | None = Field(
        None,
        description="Total number of claims to verify",
        serialization_alias="totalClaims",
    )
    current_claim_text: str | None = Field(
        None,
        description="Text of the claim being verified (truncated)",
        serialization_alias="currentClaimText",
        max_length=200,
    )
    current_claim_type: str | None = Field(
        None,
        description="Type of the current claim",
        serialization_alias="currentClaimType",
    )

    # Method-specific details
    current_method: str | None = Field(
        None,
        description="Current verification method being used",
        serialization_alias="currentMethod",
    )
    method_result: str | None = Field(
        None,
        description="Result from the current method (e.g., 'Match found: 92%')",
        serialization_alias="methodResult",
    )

    # Source being checked
    current_source_index: int | None = Field(
        None,
        description="Index of the source currently being checked",
        serialization_alias="currentSourceIndex",
    )
    total_sources: int | None = Field(
        None,
        description="Total number of sources to check",
        serialization_alias="totalSources",
    )
    current_source_filename: str | None = Field(
        None,
        description="Filename of the source being checked",
        serialization_alias="currentSourceFilename",
    )

    # Timing
    elapsed_ms: int = Field(
        0,
        description="Time elapsed since verification started (ms)",
        serialization_alias="elapsedMs",
    )

    # For claim_verified events
    claim_status: str | None = Field(
        None,
        description="Verification status of the completed claim",
        serialization_alias="claimStatus",
    )
    claim_confidence: float | None = Field(
        None,
        description="Confidence score of the completed claim",
        serialization_alias="claimConfidence",
    )

    # For error events
    error_message: str | None = Field(
        None,
        description="Error message if step is 'error'",
        serialization_alias="errorMessage",
    )


class VerificationProgressCallback:
    """
    Callback protocol for emitting verification progress events.

    Used by VerificationService to report progress during verification.
    The callback is typically connected to an SSE stream generator.
    """

    async def emit(self, event: VerificationProgressEvent) -> None:
        """Emit a progress event. Override in subclass."""


class VerificationStreamEvent(BaseModel):
    """
    SSE event wrapper for verification streaming.

    Wraps either a progress event or the final result.
    """

    model_config = ConfigDict(populate_by_name=True)

    event_type: str = Field(
        ...,
        description="Type of event: 'progress' or 'result' or 'error'",
        serialization_alias="eventType",
    )
    data: VerificationProgressEvent | VerificationResponse | dict = Field(
        ...,
        description="Event data (progress event, final result, or error)",
    )
