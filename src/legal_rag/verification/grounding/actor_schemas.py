"""
Pydantic schemas for Method Actor multi-stage response generation.

These models support the actor-based pipeline:
- Planning Actor: Strategizes response before generation
- Validation Actor: Pre-flight validation of plans
- Refinement Actor: Reviews and improves generated responses (Deep mode only)

Schema design informed by:
- Plan*RAG (Verma et al., 2024): DAG-based planning with atomic sub-problems
- LangGraph Plan-and-Execute: Simple step lists with dependency tracking
- Instructor library: Flat query plans with dependency IDs
- RAS framework: Post-retrieval planning with chunk-to-question mapping

"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Response structure types (deterministic heuristic + LLM fallback)
# ============================================================================

ResponseStructure = Literal[
    "IRAC",  # Issue, Rule, Application, Conclusion (legal reasoning)
    "Comparative",  # Side-by-side comparison with criteria
    "Chronological",  # Timeline-based analysis
    "Issue-by-Issue",  # Multiple distinct issues addressed separately
    "Direct",  # Factual lookup, straightforward answer
    "Procedural",  # Step-by-step process or procedure
]

# Validation recommendation types
ValidationRecommendation = Literal[
    "proceed",  # All sub-questions have good chunk support
    "retrieve_more",  # Chunks are incomplete, need additional retrieval
    "simplify",  # Plan too ambitious for available chunks
    "fallback_to_general",  # No relevant chunks, use general knowledge
]

# Legal query types for task-specific planning
LegalQueryType = Literal[
    "contract_analysis",
    "case_law_research",
    "statutory_interpretation",
    "comparative_analysis",
    "procedural_question",
    "compliance_review",
    "due_diligence",
    "general",
]


# ============================================================================
# Sub-question with chunk mapping (Plan*RAG / RAS pattern)
# ============================================================================


class SubQuestion(BaseModel):
    """
    A decomposed sub-question with evidence mapping.

    Maps each aspect of the query to specific retrieved chunks,
    so the generator knows exactly where to look for each answer.
    Inspired by Plan*RAG's atomic sub-problem decomposition.
    """

    question: str = Field(
        ...,
        description="Specific sub-question to address",
    )

    chunk_indices: list[int] = Field(
        default_factory=list,
        description="Indices of chunks that can answer this sub-question (0-based)",
    )

    answerable: bool = Field(
        default=True,
        description="Whether retrieved context can answer this sub-question",
    )

    @field_validator("answerable", mode="before")
    @classmethod
    def coerce_answerable(cls, v: object) -> bool:
        """Coerce non-boolean values from LLM output (e.g. 'partially' from GPT-5.2)."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower().strip() not in ("false", "no", "none", "0", "unanswerable")
        return bool(v)

    priority: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Priority: 1=must answer, 2=should answer, 3=nice to have",
    )


# ============================================================================
# Response Plan (post-retrieval, context-aware)
# ============================================================================


class ResponsePlan(BaseModel):
    """
    Plan produced by Planning Actor for how to answer a legal query.

    Generated AFTER retrieval so the plan is context-aware -- it knows
    what sources are available and maps sub-questions to specific chunks.

    Design principles:
    - Constrained types prevent LLM hallucination of invalid values
    - Chunk mapping enables focused generation (generator knows where to look)
    - Information gaps enable honest "insufficient context" responses
    - Deterministic structure selection for 70-80% of queries (LLM fallback for rest)
    """

    query_type: LegalQueryType = Field(
        default="general",
        description="Detected legal task type for task-specific prompt enhancement",
    )

    sub_questions: list[SubQuestion] = Field(
        ...,
        description="Decomposed sub-questions with chunk mappings",
        min_length=1,
    )

    required_sources: list[str] = Field(
        default_factory=list,
        description="Source names from retrieved context relevant to this query",
    )

    response_structure: ResponseStructure = Field(
        ...,
        description="Response structure selected by deterministic heuristic or LLM",
    )

    information_gaps: list[str] = Field(
        default_factory=list,
        description="Aspects of the query NOT covered by retrieved context",
    )

    risk_areas: list[str] = Field(
        default_factory=list,
        description="Areas requiring careful analysis (ambiguities, conflicts)",
    )

    estimated_complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Complexity score: 0.0=simple, 1.0=very complex. "
        "Used to gate refinement actor (skip if < 0.3 even in deep mode)",
    )

    strategy_notes: str | None = Field(
        None,
        description="Strategic guidance for the generator",
    )

    @field_validator("response_structure", mode="before")
    @classmethod
    def normalize_response_structure(cls, v: str) -> str:
        """Map common LLM variations to valid Literal values."""
        mapping = {
            "irac": "IRAC",
            "comparative": "Comparative",
            "comparison": "Comparative",
            "chronological": "Chronological",
            "timeline": "Chronological",
            "issue-by-issue": "Issue-by-Issue",
            "issue_by_issue": "Issue-by-Issue",
            "issues": "Issue-by-Issue",
            "direct": "Direct",
            "direct_answer": "Direct",
            "factual": "Direct",
            "procedural": "Procedural",
            "procedural_steps": "Procedural",
            "steps": "Procedural",
        }
        if isinstance(v, str):
            normalized = mapping.get(v.lower().strip(), v)
            # If still not a valid value after mapping, default to IRAC
            valid_values = {
                "IRAC",
                "Comparative",
                "Chronological",
                "Issue-by-Issue",
                "Direct",
                "Procedural",
            }
            if normalized not in valid_values:
                return "IRAC"
            return normalized
        return v

    @field_validator("query_type", mode="before")
    @classmethod
    def normalize_query_type(cls, v: str) -> str:
        """Map LLM variations to valid query type values."""
        if isinstance(v, str):
            v_lower = v.lower().strip().replace(" ", "_").replace("-", "_")
            valid_values = {
                "contract_analysis",
                "case_law_research",
                "statutory_interpretation",
                "comparative_analysis",
                "procedural_question",
                "compliance_review",
                "due_diligence",
                "general",
            }
            if v_lower in valid_values:
                return v_lower
            # Common mappings
            if "contract" in v_lower:
                return "contract_analysis"
            if "case" in v_lower or "judgment" in v_lower:
                return "case_law_research"
            if "statut" in v_lower or "section" in v_lower:
                return "statutory_interpretation"
            if "compar" in v_lower:
                return "comparative_analysis"
            if "procedur" in v_lower or "how_to" in v_lower or "process" in v_lower:
                return "procedural_question"
            if "complian" in v_lower:
                return "compliance_review"
            if "due_diligence" in v_lower:
                return "due_diligence"
            return "general"
        return v

    @field_validator("sub_questions", mode="before")
    @classmethod
    def normalize_sub_questions(cls, v: list) -> list:
        """Accept both SubQuestion objects and plain strings for backwards compat."""
        normalized = []
        for item in v:
            if isinstance(item, str):
                # Plain string -> wrap in SubQuestion
                normalized.append(SubQuestion(question=item))
            elif isinstance(item, dict):
                normalized.append(SubQuestion(**item))
            else:
                normalized.append(item)
        return normalized


class RefinementSuggestions(BaseModel):
    """
    Critique produced by Refinement Actor for improving generated responses.

    The Refinement Actor reviews the generated response against the original
    plan to identify weaknesses and suggest improvements.
    """

    missing_elements: list[str] = Field(
        default_factory=list,
        description="Unanswered sub-questions or missing components from plan",
    )

    citation_gaps: list[str] = Field(
        default_factory=list,
        description="Missing or weak citations",
    )

    clarity_issues: list[str] = Field(
        default_factory=list,
        description="Ambiguous language or unclear statements",
    )

    should_regenerate: bool = Field(
        ...,
        description="Whether the response should be regenerated with improvements",
    )

    suggested_improvements: str | None = Field(
        None,
        description="Specific guidance for regeneration if needed",
    )

    confidence_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence in the refinement analysis (0.0 = low, 1.0 = high)",
    )


class ValidationResult(BaseModel):
    """
    Validation result from pre-flight checks before response generation.

    The Validation Actor checks if the planned response can be fully grounded
    in the retrieved chunks before generation begins.
    """

    is_valid: bool = Field(
        ...,
        description="Whether the plan can be executed with available chunks",
    )

    missing_support: list[str] = Field(
        default_factory=list,
        description="Sub-questions or claims that lack supporting chunks",
    )

    recommendation: ValidationRecommendation = Field(
        ...,
        description="Action to take based on validation",
    )

    confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the validation assessment",
    )

    validation_notes: str | None = Field(
        None,
        description="Additional notes about validation results",
    )


class ActorMetrics(BaseModel):
    """
    Metrics for tracking actor performance and resource usage.

    Used for monitoring and optimization of the multi-stage pipeline.
    """

    actor_name: str = Field(
        ..., description="Name of the actor", examples=["planning", "generation", "refinement"]
    )

    execution_time_ms: float = Field(..., description="Execution time in milliseconds", ge=0.0)

    tokens_used: int = Field(default=0, description="Total tokens consumed", ge=0)

    llm_calls: int = Field(default=0, description="Number of LLM API calls made", ge=0)

    cache_hit: bool = Field(default=False, description="Whether result was retrieved from cache")

    error_occurred: bool = Field(default=False, description="Whether an error occurred")

    error_message: str | None = Field(None, description="Error message if error occurred")


class MultiStageResponse(BaseModel):
    """
    Complete multi-stage response with all intermediate artifacts.

    Captures the full actor pipeline execution for transparency and debugging.
    """

    plan: ResponsePlan | None = Field(None, description="Planning Actor output")

    response_text: str = Field(..., description="Final generated response")

    refinement: RefinementSuggestions | None = Field(
        None, description="Refinement Actor critique (Deep mode only)"
    )

    validation: ValidationResult | None = Field(None, description="Pre-flight validation result")

    metrics: list[ActorMetrics] = Field(
        default_factory=list, description="Performance metrics for each actor"
    )

    total_execution_time_ms: float = Field(
        default=0.0, description="Total pipeline execution time", ge=0.0
    )

    regenerated: bool = Field(
        default=False, description="Whether response was regenerated after refinement"
    )
