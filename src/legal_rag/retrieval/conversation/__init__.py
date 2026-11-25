"""Conversation handling: condensing, memory, intent routing and query analysis."""

from legal_rag.retrieval.conversation.anaphora import (
    AnaphoraResolver,
    get_anaphora_resolver,
)
from legal_rag.retrieval.conversation.condenser import (
    CondensationResult,
    CondensedQueryOutput,
    QueryCondenser,
    QueryType,
)
from legal_rag.retrieval.conversation.decision import (
    RetrievalAction,
    RetrievalDecision,
    RetrievalDecisionMaker,
    get_retrieval_decision_maker,
)
from legal_rag.retrieval.conversation.hyde import (
    HyDEResult,
    HyDETransformer,
    HyDETransformResult,
)
from legal_rag.retrieval.conversation.intent_router import (
    IntentClassification,
    IntentRouter,
    IntentRoutingResult,
    UserIntent,
    extract_instruction_and_content,
)
from legal_rag.retrieval.conversation.loop_detector import (
    LoopDetectionResult,
    RetrievalHistory,
    RetrievalLoopDetector,
    get_loop_detector,
)
from legal_rag.retrieval.conversation.memory import (
    ConversationManager,
    ConversationSession,
    ConversationTurn,
)
from legal_rag.retrieval.conversation.message_split import (
    has_contract_structure,
    looks_like_transcript,
    split_message,
)
from legal_rag.retrieval.conversation.prior_answer import (
    apply_prior_answer_context,
    build_prior_answer_block,
    is_reformat_request,
    last_assistant_message,
    references_prior_answer,
)
from legal_rag.retrieval.conversation.query_analyzer import (
    LegalTaskType,
    QueryAnalysisResult,
    QueryAnalyzer,
    QueryComplexity,
    QueryIntent,
    get_query_analyzer,
)
from legal_rag.retrieval.conversation.query_enricher import (
    QueryEnricher,
    QueryEnrichmentResult,
    get_query_enricher,
)
from legal_rag.retrieval.conversation.query_expander import (
    QueryExpander,
    QueryExpansionResult,
    QueryExpansions,
)
from legal_rag.retrieval.conversation.scoped_general_question import (
    is_general_doctrinal_question,
)
from legal_rag.retrieval.conversation.text_tokens import (
    count_words,
    is_spaceless_script,
)

__all__ = [
    "AnaphoraResolver",
    "CondensationResult",
    "CondensedQueryOutput",
    "ConversationManager",
    "ConversationSession",
    "ConversationTurn",
    "HyDEResult",
    "HyDETransformResult",
    "HyDETransformer",
    "IntentClassification",
    "IntentRouter",
    "IntentRoutingResult",
    "LegalTaskType",
    "LoopDetectionResult",
    "QueryAnalysisResult",
    "QueryAnalyzer",
    "QueryComplexity",
    "QueryCondenser",
    "QueryEnricher",
    "QueryEnrichmentResult",
    "QueryExpander",
    "QueryExpansionResult",
    "QueryExpansions",
    "QueryIntent",
    "QueryType",
    "RetrievalAction",
    "RetrievalDecision",
    "RetrievalDecisionMaker",
    "RetrievalHistory",
    "RetrievalLoopDetector",
    "UserIntent",
    "apply_prior_answer_context",
    "build_prior_answer_block",
    "count_words",
    "extract_instruction_and_content",
    "get_anaphora_resolver",
    "get_loop_detector",
    "get_query_analyzer",
    "get_query_enricher",
    "get_retrieval_decision_maker",
    "has_contract_structure",
    "is_general_doctrinal_question",
    "is_reformat_request",
    "is_spaceless_script",
    "last_assistant_message",
    "looks_like_transcript",
    "references_prior_answer",
    "split_message",
]
