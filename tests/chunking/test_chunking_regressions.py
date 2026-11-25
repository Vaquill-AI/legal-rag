"""Regression tests for five chunking defects: separator stripping, zero options,
the optional semchunk import, heading titles and the contextual cache key."""

import subprocess
import sys

import pytest

from legal_rag.chunking import (
    DocumentType,
    RecursiveLegalChunker,
    SemchunkLegalChunker,
    get_universal_section_detector,
)
from legal_rag.chunking.contextual_retrieval import ContextualRetrievalEnhancer
from legal_rag.chunking.multilingual import CrossLingualSearcher
from legal_rag.chunking.multilingual.cross_lingual import SearchDocument
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeLanguageModel

FILLER = " and on" * 40


def _squash(text: str) -> str:
    return "".join(text.split())


# Defect 1: separator removal must strip a suffix, not a character set.


@pytest.mark.parametrize(
    ("separator", "tail_word"),
    [
        ("\n\nSection", "consent"),
        ("\n\nARTICLE", "PARTICLE"),
        ("\n\nWHEREAS", "SHEREAS"),
    ],
)
def test_split_does_not_eat_letters_that_occur_in_the_separator(separator, tail_word):
    chunker = RecursiveLegalChunker(chunk_size=12, chunk_overlap=0, min_chunk_size=1)
    heading = separator.strip()
    text = f"The party gave {tail_word}{separator} 2{FILLER}{separator} 3{FILLER}"

    pieces = chunker._split_text_recursive(text, [separator, "\n", " ", ""])

    assert pieces[0] == f"The party gave {tail_word}"
    assert pieces[1].startswith(f"{heading} 2")
    assert sum(piece.count(heading) for piece in pieces) == text.count(heading)
    assert _squash("".join(pieces)) == _squash(text)


async def test_default_separators_keep_heading_words_and_chunk_endings():
    text = (
        "Section 101. Definitions.\n"
        "The controller needs the consumer's consent\n\n"
        f"Section 102. Rights.{FILLER}\n\n"
        f"Section 103. Duties.{FILLER}\n"
    )
    chunker = RecursiveLegalChunker(chunk_size=40, chunk_overlap=0, min_chunk_size=1)

    result = await chunker.chunk(text, "doc-1", document_type=DocumentType.STATUTE)
    joined = "\n".join(chunk.content for chunk in result.chunks)

    for heading in ("Section 101.", "Section 102.", "Section 103."):
        assert heading in joined
    assert "consumer's consent" in joined


# Defect 2: an explicit 0 is a value, not "use the default".


def test_recursive_chunker_accepts_zero_overlap_and_zero_minimum():
    chunker = RecursiveLegalChunker(chunk_overlap=0, min_chunk_size=0)
    assert chunker.chunk_overlap == 0
    assert chunker.min_chunk_size == 0
    default = RecursiveLegalChunker()
    assert (default.chunk_overlap, default.min_chunk_size) == (100, 50)


def test_zero_overlap_adds_no_overlap_text():
    text = " ".join(f"word{i}" for i in range(300))
    no_overlap = RecursiveLegalChunker(chunk_size=40, chunk_overlap=0, min_chunk_size=1)
    pieces = no_overlap._split_text_recursive(text, [" ", ""])
    assert len(pieces) > 1
    assert _squash("".join(pieces)) == _squash(text)


def test_semchunk_chunker_accepts_zero_overlap_and_zero_minimum():
    pytest.importorskip("semchunk")
    chunker = SemchunkLegalChunker(chunk_overlap=0, min_chunk_size=0)
    assert chunker.chunk_overlap == 0
    assert chunker.min_chunk_size == 0
    assert SemchunkLegalChunker().chunk_overlap == 100


async def test_cross_lingual_search_treats_zero_min_score_as_a_value():
    searcher = CrossLingualSearcher(embedder=FakeEmbedder(dimension=8))
    await searcher.initialize()
    documents = [SearchDocument(id=str(i), text=f"doc {i}", language="en") for i in range(30)]
    await searcher.index_documents(documents)
    assert searcher.config.similarity_threshold > 0

    default = await searcher.search("doc 0", top_k=30)
    zero = await searcher.search("doc 0", top_k=30, min_score=0.0)

    assert len(zero.results) > len(default.results)


# Defect 3: semchunk is an optional extra.

NO_SEMCHUNK = """
import asyncio, sys
sys.modules["semchunk"] = None

import legal_rag.chunking as chunking
from legal_rag.chunking import RecursiveLegalChunker, EmailChunker

result = asyncio.run(RecursiveLegalChunker().chunk("ARTICLE I TERM\\n\\nThe term is two years.", "d"))
assert result.total_chunks >= 1
try:
    chunking.SemchunkLegalChunker()
except ImportError as exc:
    assert 'pip install "legal-rag[chunking]"' in str(exc), str(exc)
    print("ok")
else:
    raise SystemExit("SemchunkLegalChunker built without semchunk")
"""


def test_package_imports_without_semchunk_and_the_semchunk_chunker_names_the_extra():
    completed = subprocess.run(
        [sys.executable, "-c", NO_SEMCHUNK],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


# Defect 4: the title is the whole heading token, not the matched prefix.


@pytest.mark.parametrize(
    ("text", "document_type", "title"),
    [
        ("Section 12-101. Definitions\nIn this part.\n", DocumentType.STATUTE, "Section 12-101."),
        ("Sec. 3.4(b) Notice\nA notice is required.\n", DocumentType.STATUTE, "Sec. 3.4(b)"),
        ("Section 101. Definitions\nIn this part.\n", DocumentType.STATUTE, "Section 101."),
        ("§ 12-101 Definitions\nIn this part.\n", DocumentType.STATUTE, "§ 12-101"),
        ("ARTICLE IV\nThe term is two years.\n", DocumentType.UNKNOWN, "ARTICLE IV"),
    ],
)
def test_section_title_is_the_full_heading_token(text, document_type, title):
    sections = get_universal_section_detector().detect(text, document_type)
    assert sections[0].title == title


async def test_chunk_section_title_is_the_full_heading_token():
    text = "Section 12-101. Definitions\nIn this part, a person means an individual.\n"
    result = await RecursiveLegalChunker(min_chunk_size=1).chunk(
        text, "doc-1", document_type=DocumentType.STATUTE
    )
    assert result.chunks[0].section_title == "Section 12-101."


# Defect 5: the context cache is keyed on content, not on the random chunk id.

CONTRACT = (
    "ARTICLE I TERM\n\nThe Receiving Party shall protect the information for two years.\n\n"
    "ARTICLE II LAW\n\nThis Agreement is governed by the laws of Delaware.\n"
)


async def _chunks(text: str):
    chunker = RecursiveLegalChunker(chunk_size=20, chunk_overlap=3, min_chunk_size=3)
    return list((await chunker.chunk(text, "nda-1")).chunks)


async def test_rechunking_the_same_text_hits_the_context_cache():
    llm = FakeLanguageModel(reply="Article I, a term clause.")
    cache = FakeCache()
    enhancer = ContextualRetrievalEnhancer(llm, cache)

    first_chunks = await _chunks(CONTRACT)
    first = await enhancer.enhance_chunks(first_chunks, CONTRACT, "nda-1")
    calls_after_first = len(llm.calls)
    assert calls_after_first >= 2

    second_chunks = await _chunks(CONTRACT)
    assert {c.id for c in first_chunks}.isdisjoint({c.id for c in second_chunks})
    second = await enhancer.enhance_chunks(second_chunks, CONTRACT, "nda-1")

    assert len(llm.calls) == calls_after_first
    assert second == first
    assert all(chunk.id not in key for chunk in first_chunks for key in cache.data)


async def test_context_cache_misses_when_the_document_changes():
    llm = FakeLanguageModel(reply="Article I, a term clause.")
    enhancer = ContextualRetrievalEnhancer(llm, FakeCache())
    chunks = await _chunks(CONTRACT)
    await enhancer.enhance_chunks(chunks, CONTRACT, "nda-1")
    calls = len(llm.calls)

    await enhancer.enhance_chunks(chunks, CONTRACT + "\nARTICLE III NOTICE\n\nIn writing.", "nda-1")

    assert len(llm.calls) == calls + len(chunks)


def test_a_sentence_separator_keeps_its_period_and_adds_none():
    """Trimming ". " removes only the space, and the last split gets no separator after it."""
    sentence = "The tenant shall pay rent monthly and keep the premises in good order. "
    text = sentence * 30
    chunker = RecursiveLegalChunker(chunk_size=60, chunk_overlap=0, min_chunk_size=1)

    chunks = chunker._split_text_recursive(text, [". ", " ", ""])

    assert len(chunks) > 1
    assert all(chunk.rstrip().endswith("order.") for chunk in chunks)
    assert "".join(chunks).count(".") == text.count(".")
