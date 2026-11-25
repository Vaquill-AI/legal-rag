"""is_general_doctrinal_question: when a document-scoped query should be answered from the law."""

from __future__ import annotations

import pytest

from legal_rag.retrieval.conversation import is_general_doctrinal_question
from legal_rag.retrieval.conversation.scoped_general_question import (
    is_general_doctrinal_question as imported_from_module,
)

DOC_NAMES = ["Acme_Master_Services_Agreement.pdf"]


def test_package_export_is_the_module_function():
    assert is_general_doctrinal_question is imported_from_module


@pytest.mark.parametrize(
    "query",
    [
        "Explain how adverse possession works generally",
        "What is the statute of limitations for breach of contract under applicable law?",
        "How does an indemnity clause typically work",
        "EXPLAIN IN GENERAL HOW LIENS WORK",
        "Walk me through what courts commonly require for a valid waiver",
    ],
)
def test_general_explainers_are_released_from_the_document(query):
    assert is_general_doctrinal_question(query) is True


@pytest.mark.parametrize(
    "query",
    [
        # No explainer framing, even though it reads as general.
        "Generally speaking, liens are risky",
        # Explainer framing but no generality cue: a document lookup.
        "What is the termination date?",
        # A command, not an explainer.
        "Summarize this agreement",
    ],
)
def test_queries_missing_a_framing_or_generality_cue_stay_scoped(query):
    assert is_general_doctrinal_question(query) is False


@pytest.mark.parametrize(
    "query",
    [
        "Explain generally what this agreement means",
        "Explain typically how termination works in the contract",
        "What does the uploaded lease usually say about subletting",
        "Describe in general the attached exhibit",
    ],
)
def test_reference_to_the_document_keeps_the_scope(query):
    assert is_general_doctrinal_question(query) is False


def test_naming_the_pinned_file_keeps_the_scope():
    query = "Explain generally how Acme handles termination"
    assert is_general_doctrinal_question(query, DOC_NAMES) is False
    assert is_general_doctrinal_question(query) is True


def test_the_file_extension_is_not_a_distinctive_token():
    query = "Explain generally how a pdf is signed under applicable law"
    assert is_general_doctrinal_question(query, DOC_NAMES) is True


def test_short_filename_tokens_do_not_count_as_a_reference():
    query = "explain generally how nda terms work"
    assert is_general_doctrinal_question(query, ["NDA_v2.pdf"]) is True


def test_blank_filenames_are_ignored():
    assert is_general_doctrinal_question("Explain how liens work generally", [""]) is True
    assert is_general_doctrinal_question("Explain how liens work", [""]) is False


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_queries_are_never_general(query):
    assert is_general_doctrinal_question(query) is False
    assert is_general_doctrinal_question(query, DOC_NAMES) is False
