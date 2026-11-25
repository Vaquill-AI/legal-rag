"""
Advanced Query Analysis for RAG Systems.

Implements industry best practices from 2024-2025 research:
- Multi-intent detection (decompose "hey, also what's the penalty for theft?")
- Query complexity scoring (adaptive retrieval decisions)
- Ambiguous query detection (expand "how are damages" -> "how are damages calculated in contract law")
- Out-of-domain detection (queries not matching legal domain)

Based on:
- Adaptive-RAG (KAIST 2024)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from legal_rag.retrieval.routing.structured_log import get_logger

logger = get_logger("rag.conversation.query_analyzer")


class QueryComplexity(Enum):
    """Query complexity levels for adaptive retrieval."""

    TRIVIAL = "trivial"  # Greetings, acknowledgments (skip retrieval)
    SIMPLE = "simple"  # Single-hop questions (standard retrieval)
    MODERATE = "moderate"  # Multi-aspect questions (enhanced retrieval)
    COMPLEX = "complex"  # Multi-hop reasoning (query decomposition + retrieval)


class QueryIntent(Enum):
    """Query intent classification."""

    CONVERSATIONAL = "conversational"  # Greetings, small talk
    FACTUAL = "factual"  # Single fact lookup
    ANALYTICAL = "analytical"  # Analysis, comparison, reasoning
    PROCEDURAL = "procedural"  # How-to, process questions
    MIXED = "mixed"  # Multiple intents combined


class LegalTaskType(Enum):
    """
    Legal task type classification for query-type-specific prompts.

    Implements Phase 5: Query-Type-Specific Prompts enhancement.
    Each task type gets specialized prompt guidance optimized for that legal domain.
    """

    CONTRACT_ANALYSIS = "contract_analysis"  # Contract provisions, obligations, risks
    STATUTORY_INTERPRETATION = "statutory_interpretation"  # Statutes, regulations, compliance
    CASE_LAW_RESEARCH = "case_law_research"  # Precedents, holdings, distinctions
    COMPARISON = "comparison"  # Multi-document or multi-provision comparison
    PROCEDURAL = "procedural"  # Court rules, filing deadlines, jurisdiction
    GENERAL = "general"  # General legal research (no specific task type)

    # Phase 4: Legal AI Tool task types (auto-detect in chat)
    CONTRACT_REVIEW = "contract_review"  # Clause analysis, redlines, playbook review
    NDA_TRIAGE = "nda_triage"  # NDA screening, classification, 10-criteria check
    RISK_ASSESSMENT = "risk_assessment"  # Legal risk matrix, mitigation strategies
    COMPLIANCE_CHECK = "compliance_check"  # GDPR/CCPA/HIPAA compliance analysis
    CANNED_RESPONSE = "canned_response"  # Templated legal responses, DSR, discovery hold


@dataclass
class QueryAnalysisResult:
    """Result of comprehensive query analysis."""

    # Intent and complexity
    primary_intent: QueryIntent
    complexity: QueryComplexity
    confidence: float

    # Multi-intent detection
    has_multiple_intents: bool
    intents_detected: list[QueryIntent]

    # Domain analysis
    is_in_domain: bool  # Is this a legal query?
    is_ambiguous: bool  # Needs expansion/clarification
    is_out_of_scope: bool  # Non-legal general knowledge

    # Recommendations
    should_retrieve: bool
    should_expand_query: bool
    should_decompose: bool

    # Metadata
    legal_entities_found: list[str]
    ambiguity_score: float
    processing_time_ms: float

    # Phase 5: Legal task type classification
    legal_task_type: LegalTaskType  # Task-specific prompt enhancement


class QueryAnalyzer:
    """
    Advanced query analyzer for RAG systems.

    Implements production-ready query analysis patterns from top AI companies.
    """

    # Legal domain indicators (strong signals for in-domain queries)
    LEGAL_ENTITIES: ClassVar[set[str]] = {
        # Statutory references
        "section",
        "article",
        "rule",
        "regulation",
        "clause",
        "sub-section",
        "proviso",
        "schedule",
        "amendment",
        "ordinance",
        "notification",
        # Case law
        "case",
        "judgment",
        "order",
        "appeal",
        "petition",
        "writ",
        "habeas corpus",
        "mandamus",
        "certiorari",
        "prohibition",
        "quo warranto",
        # Legal codes
        "evidence act",
        "constitution",
        # Parties and roles
        "defendant",
        "plaintiff",
        "respondent",
        "petitioner",
        "appellant",
        "complainant",
        # Courts
        "supreme court",
        "high court",
        "district court",
        "tribunal",
        "magistrate",
        # Legal concepts
        "damages",
        "liability",
        "breach",
        "contract",
        "tort",
        "negligence",
        "jurisdiction",
        "precedent",
        "ratio decidendi",
        "obiter dicta",
        "burden of proof",
        "mens rea",
        "actus reus",
    }

    # Out-of-scope patterns (general knowledge, not legal)
    OUT_OF_SCOPE_PATTERNS: ClassVar[set[str]] = {
        "weather in",
        "capital of",
        "president of",
        "prime minister",
        "population of",
        "what time",
        "current date",
        "today's date",
        "how many people",
        "recipe for",
        "movie about",
        "song lyrics",
        "sports score",
        "stock price",
    }

    # Ambiguity indicators (incomplete/unclear queries)
    AMBIGUITY_PATTERNS: ClassVar[set[str]] = {
        # Incomplete references
        r"^(what|which|how|why|when|where)\s+(about|is|are)\s+\w{1,4}(\s|$|\?)",  # "what about it"
        r"(this|that|it|they)\s+(one|thing|case|section|law)(\s|$|\?)",  # "that one"
        # Vague quantifiers
        r"\b(some|any|few|several|many)\b",
        # Missing specificity
        r"^(tell me|show me|explain|what)\s+(about|more)(\s|$)",  # "tell me about"
    }

    # Multi-intent indicators (query has multiple questions/requests)
    MULTI_INTENT_SEPARATORS: ClassVar[set[str]] = {
        "also",
        "and then",
        "additionally",
        "furthermore",
        "moreover",
        "plus",
        "as well as",
        "along with",
    }

    # Multi-hop reasoning indicators (requires chaining multiple retrievals)
    MULTI_HOP_PATTERNS: ClassVar[set[str]] = {
        "compare",
        "difference between",
        "similarity between",
        "contrast",
        "versus",
        "vs",
        "rather than",
        "instead of",
        "in relation to",
        "connected to",
    }

    # Phase 5: Legal task type classification patterns
    CONTRACT_PATTERNS: ClassVar[set[str]] = {
        "contract",
        "agreement",
        "provision",
        "clause",
        "terms",
        "obligations",
        "termination",
        "breach",
        "performance",
        "warranty",
        "indemnity",
        "liability",
        "force majeure",
        "assignment",
        "consideration",
    }

    STATUTORY_PATTERNS: ClassVar[set[str]] = {
        "section",
        "article",
        "statute",
        "regulation",
        "ordinance",
        "amendment",
        "act",
        "code",
        "compliance",
        "constitution",
        "notification",
        "schedule",
    }

    CASE_LAW_PATTERNS: ClassVar[set[str]] = {
        "case",
        "cases",
        "case law",
        "case laws",
        "judgment",
        "judgments",
        "judgement",
        "judgements",
        "precedent",
        "holding",
        "ratio",
        "obiter",
        "appeal",
        "petition",
        "writ",
        "court held",
        "supreme court",
        "high court",
        "overruled",
        "distinguished",
        "affirmed",
        "landmark",
        "leading case",
        "court decision",
        "tribunal",
        "bench",
        "verdict",
        "decree",
        "ruling",
        "citation",
    }

    # Strong phrases that unambiguously indicate case law discovery intent.
    # A single match from this set is enough to classify as CASE_LAW_RESEARCH
    # (bypasses the default 2-match threshold).
    CASE_LAW_STRONG_PHRASES: ClassVar[set[str]] = {
        "case law",
        "case laws",
        "landmark case",
        "landmark cases",
        "landmark judgment",
        "landmark judgement",
        "leading case",
        "leading cases",
        "recent judgment",
        "recent judgement",
        "recent case",
        "supreme court case",
        "high court case",
        "court held that",
        "find me case",
        "give me case",
        "list cases",
        "cases related to",
        "cases on",
        "judgments on",
        "judgements on",
        "relevant cases",
    }

    COMPARISON_PATTERNS: ClassVar[set[str]] = {
        "compare",
        "difference",
        "similarity",
        "contrast",
        "versus",
        "vs",
        "between",
        "distinguish",
        "relation",
    }

    PROCEDURAL_PATTERNS: ClassVar[set[str]] = {
        "procedure",
        "filing",
        "deadline",
        "jurisdiction",
        "court rules",
        "limitation",
        "appeals",
        "how to file",
        "process",
        "steps",
        "requirements",
    }

    # Phase 4: Legal tool detection patterns
    CONTRACT_REVIEW_PATTERNS: ClassVar[set[str]] = {
        "review this contract",
        "review the contract",
        "contract review",
        "review the agreement",
        "review this agreement",
        "analyze this contract",
        "redline",
        "red-line",
        "playbook",
        "clause analysis",
        "clause-by-clause",
        "flag risky clauses",
        "risky provisions",
        "review my contract",
        "contract analysis tool",
    }

    NDA_TRIAGE_PATTERNS: ClassVar[set[str]] = {
        "triage this nda",
        "nda triage",
        "screen this nda",
        "nda screening",
        "classify this nda",
        "nda classification",
        "non-disclosure agreement review",
        "nda review",
        "review this nda",
        "confidentiality agreement review",
        "nda check",
        "evaluate this nda",
    }

    RISK_ASSESSMENT_PATTERNS: ClassVar[set[str]] = {
        "risk assessment",
        "legal risk",
        "risk matrix",
        "risk analysis",
        "assess the risk",
        "evaluate risk",
        "risk score",
        "mitigation strategy",
        "risk mitigation",
        "likelihood and impact",
        "severity matrix",
        "legal exposure",
    }

    COMPLIANCE_CHECK_PATTERNS: ClassVar[set[str]] = {
        "compliance check",
        "compliance analysis",
        "gdpr compliance",
        "ccpa compliance",
        "hipaa compliance",
        "dpdp compliance",
        "data protection compliance",
        "privacy compliance",
        "regulatory compliance",
        "compliance audit",
        "compliance gap",
        "compliance review",
        "check compliance",
        "sox compliance",
    }

    CANNED_RESPONSE_PATTERNS: ClassVar[set[str]] = {
        "draft a response",
        "draft response",
        "templated response",
        "canned response",
        "respond to this",
        "generate a response",
        "data subject request",
        "discovery hold",
        "litigation hold",
        "privacy inquiry",
        "vendor question",
        "subpoena response",
        "insurance inquiry",
        "form response",
        "standard response",
    }

    def __init__(self):
        """Initialize QueryAnalyzer."""

    @staticmethod
    def _has_marker_match(query_lower: str, markers: set[str]) -> bool:
        """Check if any marker appears as whole word(s) in query.

        Uses word-set intersection for single words and word-boundary regex
        for multi-word phrases. Prevents false positives from substring
        matches (e.g., "hi" inside "this", "hey" inside "they").
        """
        query_words = set(query_lower.split())
        for marker in markers:
            if " " in marker:
                # Multi-word phrase: use word boundary regex
                if re.search(r"\b" + re.escape(marker) + r"\b", query_lower):
                    return True
            # Single word: check word set (O(1) lookup)
            elif marker in query_words:
                return True
        return False

    def analyze(
        self,
        query: str,
        chat_history: list[dict] | None = None,
        intent_routing=None,
    ) -> QueryAnalysisResult:
        """
        Perform comprehensive query analysis.

        Args:
            query: User query to analyze
            chat_history: Optional conversation history for context
            intent_routing: Optional IntentRoutingResult from intent router.
                When provided and pasted content was detected, domain detection
                runs on the instruction portion only (not the full 2000-word paste).

        Returns:
            QueryAnalysisResult with recommendations
        """
        import time

        start_time = time.time()

        query_lower = query.lower().strip()
        _words = query_lower.split()

        # Detect multi-intent
        has_multiple = self._detect_multi_intent(query)
        intents = self._detect_intents(query)

        # Detect domain: when intent routing extracted pasted content, check
        # both instruction and content. If either has legal signal, treat as
        # legal. This prevents false negatives when a user pastes a legal
        # notice and asks a plain-language question about it.
        domain_check_text = query
        if intent_routing and intent_routing.content_text is not None:
            domain_check_text = intent_routing.instruction_text

        is_legal = self._is_legal_domain(domain_check_text)
        # Also check the pasted content for legal signal
        if not is_legal and intent_routing and intent_routing.content_text is not None:
            is_legal = self._is_legal_domain(intent_routing.content_text)
        is_out_of_scope = self._is_out_of_scope(domain_check_text)

        # Detect ambiguity
        is_ambiguous, ambiguity_score = self._detect_ambiguity(query)

        # Extract legal entities
        legal_entities = self._extract_legal_entities(query)

        # Determine complexity
        complexity = self._assess_complexity(query, has_multiple, legal_entities)

        # Determine primary intent
        primary_intent = intents[0] if intents else QueryIntent.FACTUAL

        # Phase 5: Classify legal task type
        legal_task_type = self._classify_legal_task_type(query, is_legal)

        # Make recommendations
        should_retrieve = self._should_retrieve(
            primary_intent, complexity, is_legal, is_out_of_scope
        )
        should_expand = is_ambiguous and is_legal
        should_decompose = has_multiple or complexity == QueryComplexity.COMPLEX

        # Calculate confidence
        confidence = self._calculate_confidence(
            is_legal, is_out_of_scope, ambiguity_score, legal_entities
        )

        processing_time = (time.time() - start_time) * 1000

        return QueryAnalysisResult(
            primary_intent=primary_intent,
            complexity=complexity,
            confidence=confidence,
            has_multiple_intents=has_multiple,
            intents_detected=intents,
            is_in_domain=is_legal,
            is_ambiguous=is_ambiguous,
            is_out_of_scope=is_out_of_scope,
            should_retrieve=should_retrieve,
            should_expand_query=should_expand,
            should_decompose=should_decompose,
            legal_entities_found=legal_entities,
            ambiguity_score=ambiguity_score,
            processing_time_ms=processing_time,
            legal_task_type=legal_task_type,  # Phase 5: Task-specific prompt enhancement
        )

    def _detect_multi_intent(self, query: str) -> bool:
        """
        Detect if query has multiple intents.

        Examples:
        - "hey, also what's the penalty for theft?" -> True (greeting + legal question)
        - "explain Section 420 and also Article 21" -> True (multiple legal questions)
        """
        query_lower = query.lower()

        # Check for separators
        has_separator = any(sep in query_lower for sep in self.MULTI_INTENT_SEPARATORS)

        # Check for multiple question marks or conjunctions
        question_marks = query.count("?")
        has_multiple_questions = question_marks > 1

        # Check for "and" with different sentence structures
        and_count = query_lower.count(" and ")
        has_complex_and = and_count > 1

        return has_separator or has_multiple_questions or has_complex_and

    def _detect_intents(self, query: str) -> list[QueryIntent]:
        """Detect all intents present in query."""
        query_lower = query.lower()
        intents = []

        # Check conversational (word-boundary match to avoid "hi" in "this", "hey" in "they")
        conversational_markers = {"hello", "hi", "hey", "thanks", "bye", "how are you"}
        if self._has_marker_match(query_lower, conversational_markers):
            intents.append(QueryIntent.CONVERSATIONAL)

        # Check analytical (comparison, reasoning)
        analytical_markers = {"compare", "difference", "why", "analyze", "evaluate"}
        if self._has_marker_match(query_lower, analytical_markers):
            intents.append(QueryIntent.ANALYTICAL)

        # Check procedural (how-to)
        procedural_markers = {"how to", "how do", "process", "procedure", "steps"}
        if self._has_marker_match(query_lower, procedural_markers):
            intents.append(QueryIntent.PROCEDURAL)

        # Check factual (what, when, where - but not how/why)
        factual_markers = {"what is", "when", "where", "who is"}
        if self._has_marker_match(query_lower, factual_markers):
            intents.append(QueryIntent.FACTUAL)

        # If multiple intents detected, mark as mixed
        if len(intents) > 1:
            return [QueryIntent.MIXED, *intents]

        # Default to factual if no specific intent
        return intents if intents else [QueryIntent.FACTUAL]

    def _is_legal_domain(self, query: str) -> bool:
        """Check if query is in legal domain."""
        query_lower = query.lower()

        # Check for legal entities
        legal_found = sum(1 for entity in self.LEGAL_ENTITIES if entity in query_lower)

        # Threshold: at least 1 legal entity for in-domain
        return legal_found >= 1

    def _is_out_of_scope(self, query: str) -> bool:
        """Check if query is out of legal scope (general knowledge)."""
        query_lower = query.lower()

        return any(pattern in query_lower for pattern in self.OUT_OF_SCOPE_PATTERNS)

    def _detect_ambiguity(self, query: str) -> tuple[bool, float]:
        """
        Detect if query is ambiguous or incomplete.

        Returns:
            (is_ambiguous, ambiguity_score) where score is 0.0-1.0
        """
        query_lower = query.lower()
        score = 0.0

        # Check regex patterns
        for pattern in self.AMBIGUITY_PATTERNS:
            if re.search(pattern, query_lower):
                score += 0.3

        # Check for pronouns without clear antecedents
        pronouns = {"it", "this", "that", "they", "them", "those"}
        pronoun_count = sum(1 for word in query_lower.split() if word in pronouns)
        if pronoun_count > 0:
            score += pronoun_count * 0.2

        # Check for vague terms
        vague_terms = {"some", "any", "few", "several", "stuff", "thing", "things"}
        vague_count = sum(1 for word in query_lower.split() if word in vague_terms)
        if vague_count > 0:
            score += vague_count * 0.15

        # Very short queries tend to be ambiguous
        words = query_lower.split()
        if len(words) < 3 and not any(
            word in {"hello", "hi", "hey", "thanks", "bye"} for word in words
        ):
            score += 0.25

        # Cap at 1.0
        score = min(score, 1.0)

        # Threshold for ambiguity
        is_ambiguous = score > 0.4

        return is_ambiguous, score

    def _extract_legal_entities(self, query: str) -> list[str]:
        """
        Extract specific legal entities from query.

        Returns:
            List of legal entities found (case numbers, sections, etc.)
        """
        entities = []

        # Case numbers (1:23-cv-01239, 2024 WL 123)
        case_patterns = [
            r"\d+/\d{4}",  # 1239/2023 (year-numbered docket)
            r"\d{4}\s+\w+\s+\d+",  # 2024 WL 123
            r"Case\s+No\.?\s*[\w\d/-]+",  # Case No. XYZ
        ]
        for pattern in case_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend([m.strip() for m in matches])

        # Section references (Section 420, Article 21)
        section_patterns = [
            r"Section\s+\d+[\w]*(\s+\w+)?",  # Section 1983
            r"Article\s+\d+[\w]*",  # Article 21
            r"Rule\s+\d+[\w]*",  # Rule 37
        ]
        for pattern in section_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend([m.strip() for m in matches])

        return entities

    # Expanded analytical markers for complexity detection.
    # These indicate the user wants analysis, not a simple factual lookup.
    ANALYTICAL_MARKERS: ClassVar[set[str]] = {
        # Core analytical verbs
        "analyze",
        "evaluate",
        "assess",
        "impact",
        "effect",
        # Research and synthesis verbs
        "summarize",
        "summarise",
        "digest",
        "outline",
        "explain",
        "discuss",
        "examine",
        "critique",
        "review",
        "interpret",
        # Depth/scope indicators
        "comprehensive",
        "detailed",
        "thorough",
        "exhaustive",
        "in-depth",
        "key",
        # Validity and precedent analysis
        "validity",
        "current status",
        "still valid",
        "overruled",
        "distinguished",
        "ratio decidendi",
        "obiter dicta",
        "binding",
        "precedent",
        "stare decisis",
        # Multi-aspect signals (not multi-hop but still complex)
        "implications",
        "consequences",
        "considerations",
        "factors",
        "elements",
        "requirements",
        "conditions",
        "grounds",
    }

    # Legal concept markers that indicate domain richness in the query.
    # Unlike LEGAL_ENTITIES (case numbers, section refs), these are general
    # legal concepts that signal a substantive query even without explicit
    # case citations. Used for complexity classification, not domain detection.
    #
    # Coverage: US first, with common international terms,
    # Hong Kong, South Africa, international law, and common law doctrines.
    LEGAL_CONCEPT_MARKERS: ClassVar[set[str]] = {
        # ===================================================================
        # CROSS-JURISDICTIONAL LEGAL DOMAINS
        # ===================================================================
        # Financial crime / AML
        "money laundering",
        "anti-money laundering",
        "aml",
        "kyc",
        "know your customer",
        "terrorist financing",
        "beneficial ownership",
        "suspicious transaction",
        "financial crime",
        # Corporate / commercial
        "arbitration",
        "insolvency",
        "bankruptcy",
        "liquidation",
        "receivership",
        "winding up",
        "restructuring",
        "merger",
        "acquisition",
        "joint venture",
        "shareholders agreement",
        "corporate governance",
        "fiduciary",
        "director liability",
        "piercing the veil",
        "lifting the veil",
        # Tax
        "income tax",
        "capital gains",
        "transfer pricing",
        "tax avoidance",
        "tax evasion",
        "double taxation",
        "tax treaty",
        "withholding tax",
        "vat",
        "customs duty",
        "customs act",
        # Competition / antitrust
        "antitrust",
        "competition law",
        "cartel",
        "market dominance",
        "abuse of dominance",
        "price fixing",
        "bid rigging",
        "merger control",
        # Securities / capital markets
        "securities",
        "insider trading",
        "market manipulation",
        "prospectus",
        "ipo",
        "takeover",
        "delisting",
        # Intellectual property
        "intellectual property",
        "trademark",
        "patent",
        "copyright",
        "trade secret",
        "licensing",
        "infringement",
        "passing off",
        "fair use",
        "fair dealing",
        # Employment / labor
        "employment law",
        "wrongful termination",
        "unfair dismissal",
        "constructive dismissal",
        "workplace discrimination",
        "sexual harassment",
        "collective bargaining",
        "trade union",
        "non-compete",
        "restrictive covenant",
        "whistleblower",
        # Data / privacy
        "data protection",
        "privacy law",
        "data breach",
        "consent",
        "data subject",
        "data controller",
        "data processor",
        "cross-border transfer",
        "right to be forgotten",
        # Litigation / dispute
        "class action",
        "group litigation",
        "representative action",
        "contingency fee",
        "costs order",
        "damages",
        "injunction",
        "specific performance",
        "indemnity",
        "limitation of liability",
        "force majeure",
        # Criminal law
        "white collar",
        "fraud",
        "bribery",
        "corruption",
        "embezzlement",
        "forgery",
        "perjury",
        "contempt of court",
        "conspiracy",
        "money mule",
        # International / sanctions
        "sanctions",
        "extradition",
        "mutual legal assistance",
        "export control",
        "embargo",
        "ofac",
        "magnitsky",
        # Immigration
        "immigration",
        "asylum",
        "deportation",
        "visa",
        "work permit",
        "citizenship",
        "refugee",
        # Real estate / property
        "real estate",
        "conveyancing",
        "easement",
        "encumbrance",
        "mortgage",
        "foreclosure",
        "eminent domain",
        "compulsory acquisition",
        "adverse possession",
        "tenancy",
        "lease",
        "landlord",
        # Family law
        "divorce",
        "custody",
        "child support",
        "alimony",
        "maintenance",
        "adoption",
        "prenuptial",
        "matrimonial",
        # Environmental
        "environmental law",
        "pollution",
        "carbon emissions",
        "esg",
        "environmental impact",
        "climate litigation",
        # Maritime / admiralty
        "maritime law",
        "admiralty",
        "salvage",
        "general average",
        "bill of lading",
        "charter party",
        # Healthcare
        "medical malpractice",
        "clinical negligence",
        "informed consent",
        "patient rights",
        # Constitutional / public law
        "constitutional law",
        "bill of rights",
        "separation of powers",
        "federalism",
        "devolution",
        "parliamentary sovereignty",
        "prerogative",
        # ===================================================================
        # UNITED STATES
        # ===================================================================
        # Acts and bodies
        "rico",
        "sec",
        "ftc",
        "doj",
        "cfr",
        "usc",
        "dodd-frank",
        "sarbanes-oxley",
        "sox",
        "fcpa",
        "ada",
        "hipaa",
        "ccpa",
        "cpra",
        "ferpa",
        "coppa",
        "ecpa",
        "cfaa",
        "dmca",
        "foia",
        "erisa",
        "flsa",
        "fmla",
        "osha",
        "epa",
        "nlrb",
        "eeoc",
        "sherman act",
        "clayton act",
        "lanham act",
        "hatch act",
        "warn act",
        "cda section 230",
        "section 230",
        "title ix",
        "title vii",
        "upa",
        "ucc",
        "restatement",
        "model penal code",
        # Constitutional
        "first amendment",
        "second amendment",
        "fourth amendment",
        "fifth amendment",
        "sixth amendment",
        "eighth amendment",
        "fourteenth amendment",
        "due process clause",
        "equal protection",
        "commerce clause",
        "supremacy clause",
        "takings clause",
        "establishment clause",
        "free exercise clause",
        "qualified immunity",
        "sovereign immunity",
        # Courts and procedure
        "federal register",
        "supreme court of the united states",
        "scotus",
        "circuit court",
        "federal court",
        "bankruptcy court",
        "federal rules",
        "frcp",
        "daubert",
        "erie doctrine",
        "diversity jurisdiction",
        "federal question",
        "removal",
        "class certification",
        "multidistrict litigation",
        "mdl",
        # ===================================================================
        # UNITED KINGDOM
        # ===================================================================
        # Acts and bodies
        "sfo",
        "fca",
        "hmrc",
        "ofcom",
        "ico",
        "cma",
        "companies house",
        "uk gdpr",
        "human rights act",
        "equality act",
        "proceeds of crime act",
        "bribery act",
        "modern slavery act",
        "data protection act",
        "consumer rights act",
        "employment rights act",
        "insolvency act",
        "companies act 2006",
        "senior managers regime",
        "smcr",
        "financial services act",
        "defamation act",
        "misrepresentation act",
        "limitation act",
        "sale of goods act",
        "supply of goods",
        "unfair contract terms",
        "ucta",
        # Courts
        "house of lords",
        "privy council",
        "crown court",
        "magistrates court",
        "court of appeal",
        "uksc",
        "employment tribunal",
        "first-tier tribunal",
        "upper tribunal",
        "chancery division",
        "queens bench",
        "kings bench",
        "family division",
        # ===================================================================
        # EUROPEAN UNION
        # ===================================================================
        "gdpr",
        "eu directive",
        "eu regulation",
        "echr",
        "european convention",
        "european court of justice",
        "ecj",
        "cjeu",
        "european court of human rights",
        "ecthr",
        "brussels regulation",
        "rome regulation",
        "lugano convention",
        "eu competition",
        "state aid",
        "eu merger regulation",
        "mifid",
        "aifmd",
        "emir",
        "psd2",
        "dora",
        "digital services act",
        "digital markets act",
        "ai act",
        "eu ai act",
        "nis directive",
        "eu taxonomy",
        "csrd",
        "sfdr",
        "right to erasure",
        "dpa",
        "schrems",
        "standard contractual clauses",
        "adequacy decision",
        # ===================================================================
        # CANADA
        # ===================================================================
        "charter of rights",
        "canadian charter",
        "pipeda",
        "competition act",
        "criminal code of canada",
        "supreme court of canada",
        "federal court of canada",
        "provincial court",
        "superior court",
        "canadian human rights act",
        "employment standards act",
        "occupiers liability",
        "quebec civil code",
        "notwithstanding clause",
        "section 33",
        # ===================================================================
        # AUSTRALIA
        # ===================================================================
        "asic",
        "accc",
        "apra",
        "oaic",
        "fair work act",
        "fair work commission",
        "corporations act",
        "competition and consumer act",
        "privacy act",
        "australian consumer law",
        "acl",
        "family law act",
        "migration act",
        "high court of australia",
        "federal court of australia",
        "aaat",
        "robodebt",
        # ===================================================================
        # SINGAPORE
        # ===================================================================
        "mas",
        "monetary authority of singapore",
        "siac",
        "companies act singapore",
        "employment act singapore",
        "pdpa",
        "personal data protection act",
        "court of appeal singapore",
        "sicc",
        # ===================================================================
        # HONG KONG
        # ===================================================================
        "hkma",
        "sfc",
        "court of final appeal",
        "basic law",
        "hong kong arbitration",
        "hkiac",
        "anti-money laundering hong kong",
        # ===================================================================
        # SOUTH AFRICA
        # ===================================================================
        "constitutional court",
        "competition tribunal",
        "popia",
        "protection of personal information",
        "broad-based black economic empowerment",
        "bbbee",
        "labour relations act",
        "national credit act",
        # ===================================================================
        # INTERNATIONAL LAW
        # ===================================================================
        "international court of justice",
        "icj",
        "international criminal court",
        "icc",
        "uncitral",
        "unidroit",
        "hague convention",
        "geneva convention",
        "vienna convention",
        "new york convention",
        "wto",
        "icsid",
        "bilateral investment treaty",
        "bit",
        "investor-state",
        "isds",
        "international humanitarian law",
        "law of the sea",
        "unclos",
        "diplomatic immunity",
        "jus cogens",
        "customary international law",
        "treaty",
        "ratification",
        # ===================================================================
        # PROCEDURAL CONCEPTS (CROSS-JURISDICTIONAL)
        # ===================================================================
        "bail",
        "twin conditions",
        "predicate offence",
        "predicate offense",
        "proceeds of crime",
        "discovery",
        "disclosure",
        "summary judgment",
        "default judgment",
        "preliminary injunction",
        "temporary restraining order",
        "interlocutory",
        "habeas corpus",
        "mandamus",
        "certiorari",
        "plea bargain",
        "plea deal",
        "sentencing",
        "parole",
        "probation",
        "statute of limitations",
        "limitation period",
        "laches",
        "service of process",
        "forum non conveniens",
        "lis pendens",
        "subpoena",
        "deposition",
        "interrogatories",
        "cross-examination",
        "burden of proof",
        "standard of proof",
        "beyond reasonable doubt",
        "balance of probabilities",
        "preponderance of evidence",
        "prima facie",
        "suo motu",
        "amicus curiae",
        "pro bono",
        "stay of proceedings",
        "appeal",
        "writ",
        "motion to dismiss",
        "demurrer",
        "voir dire",
        "recusal",
        # ===================================================================
        # DOCTRINAL CONCEPTS (COMMON LAW + CIVIL LAW)
        # ===================================================================
        "fundamental rights",
        "natural justice",
        "due process",
        "judicial review",
        "ultra vires",
        "locus standi",
        "standing",
        "res judicata",
        "issue estoppel",
        "collateral estoppel",
        "estoppel",
        "stare decisis",
        "proportionality",
        "reasonableness",
        "wednesbury unreasonableness",
        "fiduciary duty",
        "duty of care",
        "proximate cause",
        "causation",
        "foreseeability",
        "remoteness",
        "contributory negligence",
        "comparative negligence",
        "assumption of risk",
        "strict liability",
        "vicarious liability",
        "joint and several liability",
        "crown immunity",
        "state immunity",
        "double jeopardy",
        "autrefois acquit",
        "autrefois convict",
        "mens rea",
        "actus reus",
        "good faith",
        "bona fide",
        "mala fide",
        "ab initio",
        "nemo judex",
        "audi alteram partem",
        "pacta sunt servanda",
        "rebus sic stantibus",
        "lex mercatoria",
        "lex loci",
        "choice of law",
        "conflict of laws",
        "public policy",
        "unconscionability",
        "consideration",
        "privity",
        "quantum meruit",
        "unjust enrichment",
        "equitable remedy",
        "constructive trust",
        "resulting trust",
        "tracing",
        "subrogation",
        "set-off",
        "waiver",
        "novation",
        "frustration",
        "impossibility",
        "impracticability",
        "mistake",
        "misrepresentation",
        "undue influence",
        "duress",
        "illegality",
    }

    def _assess_complexity(
        self,
        query: str,
        has_multiple_intents: bool,
        legal_entities: list[str],
    ) -> QueryComplexity:
        """
        Assess query complexity level.

        Factors:
        - Number of intents
        - Number of legal entities (case numbers, section refs)
        - Number of legal concept markers (PMLA, bail, etc.)
        - Multi-hop reasoning indicators
        - Analytical markers (summarize, evaluate, etc.)
        - Query length and structure
        """
        query_lower = query.lower()
        words = query_lower.split()

        # Trivial: Pure conversational (word-boundary match to avoid "hi" in "this")
        conversational_markers = {"hello", "hi", "hey", "thanks", "bye"}
        if self._has_marker_match(query_lower, conversational_markers) and not legal_entities:
            return QueryComplexity.TRIVIAL

        # Complex: Multi-hop reasoning
        multi_hop = self._has_marker_match(query_lower, self.MULTI_HOP_PATTERNS)
        if multi_hop or has_multiple_intents:
            return QueryComplexity.COMPLEX

        # Count legal concept markers present in the query
        concept_count = sum(
            1
            for marker in self.LEGAL_CONCEPT_MARKERS
            if (
                marker in set(words)
                if " " not in marker
                else re.search(r"\b" + re.escape(marker) + r"\b", query_lower)
            )
        )

        # Complex: Multiple legal concepts combined (e.g., "PMLA bail twin conditions")
        if concept_count >= 3:
            return QueryComplexity.COMPLEX

        # Moderate: Multiple legal entities or analytical query
        if len(legal_entities) > 1:
            return QueryComplexity.MODERATE

        # Moderate: Analytical markers present (expanded set)
        if self._has_marker_match(query_lower, self.ANALYTICAL_MARKERS):
            return QueryComplexity.MODERATE

        # Moderate: Legal concept markers present (even 1-2 indicate substantive query)
        if concept_count >= 1:
            return QueryComplexity.MODERATE

        # Moderate: Long queries (>15 words) with legal domain signal
        # Long queries are rarely simple lookups
        if len(words) > 15 and self._is_legal_domain(query):
            return QueryComplexity.MODERATE

        # Simple: Single entity lookup or short factual question
        return QueryComplexity.SIMPLE

    def _should_retrieve(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        is_legal: bool,
        is_out_of_scope: bool,
    ) -> bool:
        """
        Decide if retrieval should be performed.

        Industry best practice: Skip retrieval for trivial/conversational,
        always retrieve for legal, use web search for out-of-scope.
        """
        # Never retrieve for pure conversational
        if intent == QueryIntent.CONVERSATIONAL and complexity == QueryComplexity.TRIVIAL:
            return False

        # Always retrieve for legal queries
        if is_legal:
            return True

        # Don't retrieve for out-of-scope (use web search instead)
        if is_out_of_scope:
            return False

        # Default: retrieve for non-trivial queries
        return complexity != QueryComplexity.TRIVIAL

    def _calculate_confidence(
        self,
        is_legal: bool,
        is_out_of_scope: bool,
        ambiguity_score: float,
        legal_entities: list[str],
    ) -> float:
        """
        Calculate confidence in the analysis.

        Higher confidence for:
        - Clear legal queries with entities
        - Clear out-of-scope queries
        - Low ambiguity

        Lower confidence for:
        - Ambiguous queries
        - Borderline domain classification
        """
        confidence = 0.5  # Base confidence

        # Strong legal signals increase confidence
        if is_legal and legal_entities:
            confidence += 0.3
        elif is_legal:
            confidence += 0.15

        # Clear out-of-scope increases confidence
        if is_out_of_scope:
            confidence += 0.25

        # Ambiguity reduces confidence
        confidence -= ambiguity_score * 0.3

        # Specific legal entities increase confidence
        confidence += min(len(legal_entities) * 0.1, 0.2)

        return max(0.0, min(confidence, 1.0))

    def _classify_legal_task_type(self, query: str, is_legal: bool) -> LegalTaskType:
        """
        Classify query into specific legal task type for prompt enhancement.

        Phase 5: Query-Type-Specific Prompts + Phase 4: Legal AI Tool detection.

        Tool-specific patterns are checked FIRST (higher priority) because
        they are more specific and should trigger tool-specific prompts
        even when general legal patterns also match.

        Args:
            query: User query to classify
            is_legal: Whether query is in legal domain

        Returns:
            LegalTaskType enum value for task-specific prompt selection
        """
        query_lower = query.lower()

        # Phase 4: Check legal tool patterns FIRST (higher priority, more specific)
        # These can trigger even for non-legal domain queries since they're explicit tool requests
        tool_matches = {
            LegalTaskType.CONTRACT_REVIEW: sum(
                1 for p in self.CONTRACT_REVIEW_PATTERNS if p in query_lower
            ),
            LegalTaskType.NDA_TRIAGE: sum(1 for p in self.NDA_TRIAGE_PATTERNS if p in query_lower),
            LegalTaskType.RISK_ASSESSMENT: sum(
                1 for p in self.RISK_ASSESSMENT_PATTERNS if p in query_lower
            ),
            LegalTaskType.COMPLIANCE_CHECK: sum(
                1 for p in self.COMPLIANCE_CHECK_PATTERNS if p in query_lower
            ),
            LegalTaskType.CANNED_RESPONSE: sum(
                1 for p in self.CANNED_RESPONSE_PATTERNS if p in query_lower
            ),
        }

        # If any tool pattern matches, return the best match
        best_tool = max(tool_matches, key=tool_matches.get)  # type: ignore[arg-type]
        if tool_matches[best_tool] >= 1:
            return best_tool

        if not is_legal:
            return LegalTaskType.GENERAL

        # Count pattern matches for each task type
        contract_matches = sum(1 for p in self.CONTRACT_PATTERNS if p in query_lower)
        statutory_matches = sum(1 for p in self.STATUTORY_PATTERNS if p in query_lower)
        case_law_matches = sum(1 for p in self.CASE_LAW_PATTERNS if p in query_lower)
        comparison_matches = sum(1 for p in self.COMPARISON_PATTERNS if p in query_lower)
        procedural_matches = sum(1 for p in self.PROCEDURAL_PATTERNS if p in query_lower)

        # Strong case law phrases bypass the 2-match threshold - a single
        # "landmark case" or "cases related to" is enough for classification.
        has_strong_case_law = any(p in query_lower for p in self.CASE_LAW_STRONG_PHRASES)
        if has_strong_case_law:
            case_law_matches = max(case_law_matches, 2)

        # Comparison gets highest priority (multi-document analysis)
        if comparison_matches >= 2:
            return LegalTaskType.COMPARISON

        # Find the task type with most pattern matches
        max_matches = max(
            contract_matches,
            statutory_matches,
            case_law_matches,
            procedural_matches,
        )

        # Require at least 2 matches for confident classification
        if max_matches < 2:
            return LegalTaskType.GENERAL

        # Return task type with most matches
        if contract_matches == max_matches:
            return LegalTaskType.CONTRACT_ANALYSIS
        elif statutory_matches == max_matches:
            return LegalTaskType.STATUTORY_INTERPRETATION
        elif case_law_matches == max_matches:
            return LegalTaskType.CASE_LAW_RESEARCH
        elif procedural_matches == max_matches:
            return LegalTaskType.PROCEDURAL
        else:
            return LegalTaskType.GENERAL


# Singleton instance
_analyzer_instance: QueryAnalyzer | None = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get or create QueryAnalyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = QueryAnalyzer()
    return _analyzer_instance
