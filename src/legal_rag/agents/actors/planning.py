"""Planning actor: decide how to answer a question before any answer is written.

The actor reads the query and short previews of the retrieved chunks, then returns a
``ResponsePlan``: sub-questions mapped to chunk indices, a response structure, information gaps and
risk areas.
The response structure is chosen by keyword heuristics when they are confident, which saves the
model from reasoning about an obvious choice, and by the model otherwise.
If the model's reply cannot be validated, a plain single-question plan is returned instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from legal_rag.agents.actors.base import (
    ActorConfig,
    ActorOutputError,
    BaseActor,
    normalize_chunks,
    source_name,
)
from legal_rag.providers.protocols import LanguageModel
from legal_rag.types import Message, RetrievedChunk
from legal_rag.verification.grounding.actor_schemas import (
    ActorMetrics,
    ResponsePlan,
    ResponseStructure,
    SubQuestion,
)

#: Characters of each chunk shown to the planner. Enough to learn the topic, cheap in tokens.
CHUNK_PREVIEW_CHARS = 300

#: Sub-questions in a fallback plan point at no more than this many leading chunks.
FALLBACK_CHUNK_LIMIT = 10

#: Required source used when no chunk carries a source name.
GENERIC_SOURCE = "Retrieved context"

_STRUCTURE_PATTERNS: dict[ResponseStructure, list[re.Pattern[str]]] = {
    "Comparative": [
        re.compile(r"\bcompar(?:e|ing|ison)\b", re.IGNORECASE),
        re.compile(r"\bdifferen(?:ce|t|ces|tiate)\b", re.IGNORECASE),
        re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
        re.compile(r"\bcontrast\b", re.IGNORECASE),
        re.compile(r"\bwhich (?:is|one|option)\b", re.IGNORECASE),
        re.compile(r"\bbetween\b.*\band\b", re.IGNORECASE),
    ],
    "Chronological": [
        re.compile(r"\btimeline\b", re.IGNORECASE),
        re.compile(r"\bchronolog(?:y|ical)\b", re.IGNORECASE),
        re.compile(r"\bhistor(?:y|ical)\b", re.IGNORECASE),
        re.compile(r"\bevolution\b", re.IGNORECASE),
        re.compile(r"\bsequence of events\b", re.IGNORECASE),
        re.compile(r"\bhow (?:did|has) .+ (?:develop|evolve|change)\b", re.IGNORECASE),
    ],
    "Procedural": [
        re.compile(r"\bhow (?:to|do|can)\b", re.IGNORECASE),
        re.compile(r"\bsteps?\b.*\b(?:to|for|in)\b", re.IGNORECASE),
        re.compile(r"\bprocess\b.*\bfor\b", re.IGNORECASE),
        re.compile(r"\bprocedure\b", re.IGNORECASE),
        re.compile(r"\bfil(?:e|ing)\b", re.IGNORECASE),
        re.compile(r"\bappl(?:y|ication|ying)\b.*\bfor\b", re.IGNORECASE),
    ],
    "Direct": [
        re.compile(r"^(?:what is|define|explain)\b", re.IGNORECASE),
        re.compile(r"\bmeaning of\b", re.IGNORECASE),
        re.compile(r"\bdefinition\b", re.IGNORECASE),
        re.compile(r"\bis it (?:true|correct|valid)\b", re.IGNORECASE),
        re.compile(r"\byes or no\b", re.IGNORECASE),
    ],
    "Issue-by-Issue": [
        re.compile(r"\bissues?\b", re.IGNORECASE),
        re.compile(r"\bmultiple\b.*\b(?:claims?|grounds?|aspects?)\b", re.IGNORECASE),
        re.compile(r"\ball\b.*\b(?:provisions?|clauses?|sections?)\b", re.IGNORECASE),
        re.compile(r"\beach\b.*\b(?:provision|clause|section|point)\b", re.IGNORECASE),
    ],
    "IRAC": [
        re.compile(r"\banalyz(?:e|ing|sis)\b", re.IGNORECASE),
        re.compile(r"\blegal\b.*\b(?:position|opinion|analysis|assessment)\b", re.IGNORECASE),
        re.compile(r"\bliabilit(?:y|ies)\b", re.IGNORECASE),
        re.compile(r"\brights?\b.*\bunder\b", re.IGNORECASE),
        re.compile(r"\bvalidity\b", re.IGNORECASE),
        re.compile(r"\benforceab(?:le|ility)\b", re.IGNORECASE),
    ],
}

PLANNING_SYSTEM_PROMPT = """You are a senior legal research strategist.

Your role is to analyze a legal research question and plan how to answer it BEFORE any answer is written.

Produce a plan with these fields:
- query_type: one of contract_analysis, case_law_research, statutory_interpretation, comparative_analysis, procedural_question, compliance_review, due_diligence, general.
- sub_questions: a list of objects with question, chunk_indices (0-based), answerable (true or false) and priority (1, 2 or 3).
- required_sources: source names taken from the available sources.
- response_structure: one of IRAC, Comparative, Chronological, Issue-by-Issue, Direct, Procedural.
- information_gaps: aspects of the query that no retrieved chunk covers.
- risk_areas: areas that need careful analysis, such as ambiguities or conflicting authority.
- estimated_complexity: a number from 0.0 (trivial) to 1.0 (very complex).
- strategy_notes: optional guidance for the writer of the answer.

Rules:
1. sub_questions: break the query into 2 to 6 addressable sub-questions. For each, list the chunk_indices that contain relevant information. Set answerable to false if no chunk supports it. priority: 1 = must answer, 2 = should answer, 3 = nice to have.
2. required_sources: select ONLY from the names listed under available sources. Never invent a source name.
3. {structure_instruction}
4. information_gaps: list what the retrieved chunks do not cover. This keeps the answer honest, because the writer will acknowledge these gaps.
5. estimated_complexity: base it on the number of sources, the legal frameworks involved and any conflicts or ambiguities.
6. Reply with the plan only, with no other text."""

PLANNING_USER_PROMPT = """<query>{query}</query>

<query_context>
Query Type: {query_type}
Estimated Complexity: {complexity:.2f}
</query_context>

<available_sources>
Select required_sources ONLY from these:
{sources_list}
</available_sources>

<retrieved_chunks>
{chunk_count} chunks retrieved (0-indexed). Use chunk indices in sub_questions.chunk_indices:
{chunks_context}
</retrieved_chunks>

Create a strategic plan. Map each sub-question to specific chunk indices. Identify information gaps."""

_FIXED_STRUCTURE = 'response_structure MUST be "{structure}" (pre-determined from the query).'
_OPEN_STRUCTURE = (
    "Choose the best response_structure from: "
    "IRAC, Comparative, Chronological, Issue-by-Issue, Direct, Procedural."
)


def select_response_structure(
    query: str, query_type: str | None = None
) -> ResponseStructure | None:
    """Pick a response structure from keyword heuristics, or None when they are not confident.

    A structure wins with two or more matching patterns, or when it is the only one that matched.
    Several structures matching once each is ambiguous, so the model decides.
    ``query_type`` is accepted for callers that already know it and does not change the result.
    """
    scores: dict[ResponseStructure, int] = {}
    for structure, patterns in _STRUCTURE_PATTERNS.items():
        matches = sum(1 for pattern in patterns if pattern.search(query))
        if matches > 0:
            scores[structure] = matches

    if not scores:
        return None

    best = max(scores, key=lambda structure: scores[structure])
    if scores[best] >= 2 or len(scores) == 1:
        return best
    return None


def build_chunk_previews(chunks: Sequence[Mapping[str, Any]]) -> str:
    """One entry per chunk: index, source, score, page, section and a short text preview.

    Every chunk is listed, so the planner can map each sub-question to chunk indices.
    The answer writer receives the full text separately.
    """
    previews = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata") or {}
        header = [f"[{source_name(chunk) or 'Unknown'}]", f"score:{chunk.get('score', 0.0):.3f}"]
        if metadata.get("page_number"):
            header.append(f"p.{metadata['page_number']}")
        if metadata.get("section_title"):
            header.append(f"section {metadata['section_title']}")

        preview = content[:CHUNK_PREVIEW_CHARS].rstrip()
        if len(content) > CHUNK_PREVIEW_CHARS:
            preview += "..."
        previews.append(f"{index}. {' '.join(header)}\n{preview}")
    return "\n\n".join(previews)


def _named_sources(chunks: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({name for chunk in chunks if (name := source_name(chunk))})


class PlanningActor(BaseActor):
    """Produce a ``ResponsePlan`` for a query and its retrieved chunks."""

    DEFAULT_CONFIG = ActorConfig(max_tokens=2500, temperature=0.3)

    @property
    def actor_name(self) -> str:
        return "planning"

    def build_prompt(
        self,
        query: str,
        chunks: Sequence[Mapping[str, Any]],
        query_metadata: Mapping[str, Any] | None,
        deterministic_structure: str | None,
    ) -> list[Message]:
        """Build the planning messages."""
        names = _named_sources(chunks)
        sources_list = "\n".join(f"- {name}" for name in names) or "- No named sources available"
        metadata = query_metadata or {}
        structure_instruction = (
            _FIXED_STRUCTURE.format(structure=deterministic_structure)
            if deterministic_structure
            else _OPEN_STRUCTURE
        )
        return [
            Message(
                "system", PLANNING_SYSTEM_PROMPT.format(structure_instruction=structure_instruction)
            ),
            Message(
                "user",
                PLANNING_USER_PROMPT.format(
                    query=query,
                    query_type=metadata.get("query_type", "general"),
                    complexity=float(metadata.get("complexity", 0.5)),
                    sources_list=sources_list,
                    chunk_count=len(chunks),
                    chunks_context=build_chunk_previews(chunks),
                ),
            ),
        ]

    async def execute(self, **kwargs: Any) -> ResponsePlan:
        """Create a plan.

        Keyword arguments:
            query: the research question (required)
            chunks: retrieved chunks as ``RetrievedChunk`` or mappings (required)
            query_metadata: optional ``query_type`` and ``complexity`` from query analysis
        """
        query = kwargs.get("query")
        if not query:
            raise ValueError("query parameter is required")
        if not kwargs.get("chunks"):
            raise ValueError("chunks parameter is required")
        chunks = normalize_chunks(kwargs["chunks"])
        query_metadata = kwargs.get("query_metadata")

        query_type = (query_metadata or {}).get("query_type", "general")
        structure = select_response_structure(query, query_type)
        if structure:
            self.logger.info("structure_selected_deterministically", structure=structure)

        messages = self.build_prompt(query, chunks, query_metadata, structure)
        self.logger.info(
            "planning_started",
            chunks_count=len(chunks),
            deterministic_structure=structure,
        )

        try:
            plan = (await self.call_structured(messages, ResponsePlan)).model_copy(deep=True)
        except ActorOutputError as exc:
            self.logger.error("planning_output_invalid", error=str(exc))
            return self.fallback_plan(query, chunks, structure)

        if structure:
            plan.response_structure = structure

        last_index = len(chunks) - 1
        for sub_question in plan.sub_questions:
            sub_question.chunk_indices = [
                index for index in sub_question.chunk_indices if 0 <= index <= last_index
            ]

        if not plan.required_sources:
            plan.required_sources = _named_sources(chunks) or [GENERIC_SOURCE]

        self.logger.info(
            "planning_completed",
            sub_questions_count=len(plan.sub_questions),
            answerable_count=sum(1 for sq in plan.sub_questions if sq.answerable),
            response_structure=plan.response_structure,
            information_gaps_count=len(plan.information_gaps),
        )
        return plan

    def fallback_plan(
        self,
        query: str,
        chunks: Sequence[Mapping[str, Any]],
        structure: str | None,
    ) -> ResponsePlan:
        """A single-question plan over the leading chunks, used when the model's plan is unusable.

        ``strategy_notes`` stays empty so that no error text reaches the answer prompt.
        """
        return ResponsePlan(
            sub_questions=[
                SubQuestion(
                    question=query,
                    chunk_indices=list(range(min(len(chunks), FALLBACK_CHUNK_LIMIT))),
                    answerable=True,
                    priority=1,
                )
            ],
            required_sources=_named_sources(chunks) or [GENERIC_SOURCE],
            response_structure=structure or "IRAC",
            information_gaps=[],
            risk_areas=["Fallback plan: model planning unavailable"],
            estimated_complexity=0.5,
            strategy_notes=None,
        )


async def plan_response(
    model: LanguageModel,
    query: str,
    chunks: Sequence[RetrievedChunk | Mapping[str, Any]],
    query_metadata: Mapping[str, Any] | None = None,
    config: ActorConfig | None = None,
) -> tuple[ResponsePlan, ActorMetrics]:
    """Plan a response and return the plan with the actor's metrics."""
    actor = PlanningActor(model, config)
    return await actor.run_with_metrics(query=query, chunks=chunks, query_metadata=query_metadata)
