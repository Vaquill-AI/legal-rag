"""Detect a general legal-doctrine question so it is not locked to a scoped document.

When a user has a document locked into scope and asks a question about the document,
answering from that document alone is correct. When the locked question is a general
doctrinal question that never references the document, a document-only prompt makes the model
refuse ("I could not find that in <file>") instead of answering the law.

This detector is deliberately conservative: it fires only when the query reads like a general
explainer (framing cue), signals generality ("under the applicable law", "typically"), and does
not reference the scoped document. A false negative leaves the answer scoped (safe); a false
positive only adds general law while the scoped document still rides along.
"""

from __future__ import annotations

import re

# Reads like a general explainer / "how does X work" question rather than a command
# ("review this", "summarize") or a document-specific lookup.
_FRAMING_RE = re.compile(
    r"\b(?:explain|describe|what\s+is|what\s+are|what\s+does|what\s+do|"
    r"how\s+is|how\s+are|how\s+does|how\s+do|how\s+would|how\s+should|"
    r"under\s+what\s+circumstances|walk\s+me\s+through|tell\s+me\s+about)\b",
    re.IGNORECASE,
)

# Signals the answer is about the law in general, not about this specific document.
_GENERALITY_RE = re.compile(
    r"\b(?:under\s+(?:the\s+)?(?:applicable|governing|general|federal|state|us|u\.s\.)\s+law"
    r"|as\s+a\s+matter\s+of\s+law"
    r"|in\s+general\b|generally\b|typically\b|commonly\b|usually\b|ordinarily\b"
    r"|in\s+most\s+jurisdictions|across\s+jurisdictions|black[- ]letter)\b",
    re.IGNORECASE,
)

# References to THE scoped document. If present, the user is asking about the document,
# so document scoping must be preserved (do NOT relax). Note: a bare determiner +
# a concept noun (e.g. "an indemnity clause", "breach of contract") is NOT a document
# reference, so those general phrasings do not trip this guard.
_DOC_REFERENCE_RE = re.compile(
    r"\b(?:this|that|these|those|the|my|our|attached|uploaded|provided|above|foregoing)\s+"
    r"(?:document|doc|file|agreement|contract|nda|lease|policy|clause|section|"
    r"provision|exhibit|schedule|addendum|memo|memorandum|letter|deed|draft|"
    r"redline|pdf|paper|filing|instrument|excerpt)s?\b"
    r"|\bin\s+(?:this|the|my|our)\s+(?:document|doc|agreement|contract|nda|lease|file)\b",
    re.IGNORECASE,
)


def _mentions_pinned_doc(query: str, scoped_doc_names: list[str] | None) -> bool:
    """True if the query names one of the scoped documents (by a distinctive filename token)."""
    if not scoped_doc_names:
        return False
    q = query.lower()
    for name in scoped_doc_names:
        if not name:
            continue
        stem = re.split(r"\.[a-z0-9]{1,5}$", name.strip().lower())[0]
        for token in re.split(r"[^a-z0-9]+", stem):
            if len(token) >= 4 and token in q:
                return True
    return False


def is_general_doctrinal_question(query: str, scoped_doc_names: list[str] | None = None) -> bool:
    """True when a document-scoped query is a general legal-doctrine question that
    should be answered from legal knowledge rather than refused for being
    absent from the scoped document.

    Conservative by design: requires an explainer framing AND a generality cue AND no
    reference to the pinned document (neither a generic "this document" phrase nor the
    document filename).
    """
    if not query or not query.strip():
        return False
    if _DOC_REFERENCE_RE.search(query):
        return False
    if _mentions_pinned_doc(query, scoped_doc_names):
        return False
    return bool(_FRAMING_RE.search(query)) and bool(_GENERALITY_RE.search(query))
