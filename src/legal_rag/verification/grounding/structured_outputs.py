"""
Pydantic Structured Output Models for LangChain Integration.

These models are used with LangChain's `with_structured_output()` method
to get type-safe, validated responses from LLMs instead of parsing strings.

Benefits:
- Eliminates regex/string parsing
- Type validation at runtime
- Better error handling
- Clear API contracts

Usage:
    from langchain_openai import ChatOpenAI
    from legal_rag.verification.grounding.structured_outputs import RelevanceScore

    llm = ChatOpenAI(model="gpt-5-mini")
    structured_llm = llm.with_structured_output(RelevanceScore)
    result: RelevanceScore = await structured_llm.ainvoke(messages)
    print(result.score)  # Typed float, guaranteed 0-1
"""

from typing import Literal

from pydantic import BaseModel, Field

# ============================================
# CRAG Corrector Models
# ============================================


class RelevanceScore(BaseModel):
    """Structured output for CRAG chunk relevance evaluation.

    Used by CRAGCorrector._evaluate_relevance() to assess how relevant
    a retrieved chunk is to the user's query.
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)",
    )


# ============================================
# Self-RAG Controller Models
# ============================================


class RetrievalAssessment(BaseModel):
    """Structured output for Self-RAG retrieval decision.

    Determines whether a query needs document retrieval or can be
    answered directly without context. Also extracts requested result
    count for dynamic top_k adjustment.
    """

    decision: Literal["RETRIEVE", "SKIP", "UNCERTAIN"] = Field(
        description="Whether to retrieve documents for this query"
    )
    requested_count: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of results explicitly requested by user (e.g., 'give me 10 cases' → 10). "
        "None if no specific count requested. Must distinguish count from years (e.g., '1995 cases' is a year, not count).",
    )


class AnswerCritique(BaseModel):
    """Structured output for Self-RAG answer quality evaluation.

    Assesses whether a generated answer is good enough or needs
    improvement/more context.
    """

    verdict: Literal["GOOD", "NEEDS_IMPROVEMENT", "NEEDS_MORE_CONTEXT"] = Field(
        description="Quality verdict for the generated answer"
    )


# ============================================
# Multi-Query Generator Models
# ============================================


class QueryVariations(BaseModel):
    """Structured list of query variations for multi-query retrieval.

    Contains alternative phrasings of the original query to improve
    retrieval coverage.
    """

    queries: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="List of query variations (max 5)",
    )


class QueryExpansions(BaseModel):
    """Structured list of expanded query variations.

    Used by QueryExpander to get type-safe expanded queries
    from the LLM instead of parsing numbered lists.
    """

    expansions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="List of expanded query variations (max 3)",
    )


# ============================================
# Query Condenser Models
# ============================================


class CondensedQueryOutput(BaseModel):
    """Structured output from the query condenser.

    Handles reference resolution, meta-instruction stripping,
    and multi-topic decomposition in a single LLM call.
    """

    primary_query: str = Field(
        description=(
            "Clean standalone query with all references resolved and "
            "meta-instructions stripped. Optimized for vector similarity search."
        ),
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Focused sub-queries when the primary query spans multiple distinct "
            "legal topics. Empty if query is single-topic."
        ),
    )
    resolved_references: bool = Field(
        default=False,
        description="True if structural references (Point 2, Part A) were resolved from chat history",
    )
    response_language: str | None = Field(
        default=None,
        description=(
            "The language the user explicitly asked the ANSWER to be written in "
            "(e.g. 'Chinese', 'Spanish'), when they said so in the message. Null "
            "when no output language was requested. This is rescued from the query "
            "before meta-instruction stripping discards it: the retrieval query "
            "should not carry it, but the answer must honor it."
        ),
    )


# ============================================
# Multi-Hop Retriever Models
# ============================================


class InformationGapAnalysis(BaseModel):
    """Analysis of what information is still needed after retrieval.

    Used by MultiHopRetriever to decide whether to perform another
    retrieval hop or stop.
    """

    is_sufficient: bool = Field(
        description="Whether current information is sufficient to answer the query"
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="List of information aspects still missing",
    )
    suggested_followup: str = Field(
        default="",
        description="Suggested follow-up query for next hop",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="Confidence in the sufficiency assessment",
    )


# ============================================
# Document Classifier Models
# ============================================


class ExtractedEntities(BaseModel):
    """Entities extracted from a legal document."""

    parties: list[str] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=list)
    acts_referenced: list[str] = Field(default_factory=list)
    sections_cited: list[str] = Field(default_factory=list)
    cases_cited: list[str] = Field(default_factory=list)


class ClassificationOutput(BaseModel):
    """Full document classification result.

    Structured output from the AI document classifier, replacing
    manual JSON.loads() parsing.
    """

    document_type: str = Field(
        description="Type: judgment, statute, contract, petition, order, notification, circular, article, other"
    )
    document_type_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in document type classification",
    )
    legal_domain: str = Field(
        description="Domain: constitutional, civil, criminal, commercial, tax, ip, labor, environmental, family, administrative"
    )
    legal_domain_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in legal domain classification",
    )
    practice_areas: list[str] = Field(
        default_factory=list,
        description="Specific practice areas covered",
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Key legal topics discussed",
    )
    entities: ExtractedEntities = Field(
        default_factory=ExtractedEntities,
        description="Extracted entities from the document",
    )
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall classification confidence",
    )


# ============================================
# Hallucination Checker Models
# ============================================


class ExtractedClaims(BaseModel):
    """List of extracted claims from a response for verification.

    Used by HallucinationChecker to get type-safe claim extraction
    from the LLM instead of parsing lines from string response.
    """

    claims: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="List of atomic, verifiable factual claims (max 20)",
    )


# ============================================
# Keyword Query Conversion Models
# ============================================


class KeywordQueryConversion(BaseModel):
    """Structured output for converting natural language to boolean keyword query.

    Used by the corpus search-natural endpoint to convert user queries
    into optimized boolean keyword search format.
    """

    keyword_query: str = Field(
        description=(
            "Boolean keyword search query using AND, OR, NOT operators, "
            'exact "phrases", and wildcards (* for 0+ chars). '
            "Example: 'murder AND \"beyond reasonable doubt\" AND Section 302'"
        ),
    )
