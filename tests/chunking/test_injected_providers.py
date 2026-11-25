"""Components that take a language model, embedder, cache or classifier."""

import json

from legal_rag.chunking import Chunk
from legal_rag.chunking.contextual_retrieval import (
    ContextualRetrievalEnhancer,
    SimplifiedContextEnhancer,
    build_context_prompt,
)
from legal_rag.chunking.extractors.legal_ner import HybridLegalNERExtractor
from legal_rag.chunking.extractors.us_legal_metadata import (
    CaptionMetadata,
    extract_us_legal_metadata,
)
from legal_rag.chunking.multilingual import (
    CrossLingualSearcher,
    EmbedderType,
    MultilingualPipeline,
)
from legal_rag.chunking.multilingual.cross_lingual import SearchDocument
from legal_rag.providers.fakes import FakeCache, FakeEmbedder, FakeLanguageModel


class FailingModel:
    async def generate(self, messages, *, max_tokens=None, temperature=None):
        raise RuntimeError("model down")

    async def generate_structured(self, messages, schema, *, max_tokens=None):
        raise RuntimeError("model down")


def _chunks():
    return [
        Chunk(id="c1", content="The Receiving Party shall protect it."),
        Chunk(id="c2", content="Term is two years."),
    ]


def test_prompt_is_built_from_data_file():
    prompt = build_context_prompt("chunk text", "document text")
    assert "definition | obligation" in prompt
    assert "chunk text" in prompt and "document text" in prompt


async def test_contextual_enhancer_uses_model_and_cache():
    llm = FakeLanguageModel(reply="Article 5 obligation.")
    cache = FakeCache()
    enhancer = ContextualRetrievalEnhancer(llm, cache)
    first = await enhancer.enhance_chunks(_chunks(), "full document", "doc")
    assert first[0].startswith("Article 5 obligation.\n\nThe Receiving Party")
    assert len(llm.calls) == 2
    assert len(cache.data) == 2
    second = await enhancer.enhance_chunks(_chunks(), "full document", "doc")
    assert second == first
    assert len(llm.calls) == 2


async def test_contextual_enhancer_falls_back_when_model_fails():
    enhancer = ContextualRetrievalEnhancer(FailingModel())
    out = await enhancer.enhance_chunks(_chunks(), "full document", "doc")
    assert out[0].endswith("The Receiving Party shall protect it.")
    assert out[0].startswith("[Section:")


def test_rule_based_enhancer_adds_header():
    out = SimplifiedContextEnhancer().enhance_chunks(
        _chunks(), {"title": "NDA"}, ["Article 1", ""], ["Receiving Party"]
    )
    assert out[0].startswith("[Document: NDA]")
    assert "[References: Receiving Party]" in out[0]
    assert out[1] == "[Document: NDA]\n\nTerm is two years."


async def test_us_metadata_regex_only_without_model():
    result = await extract_us_legal_metadata("Case No. 2:25-cv-01234", llm=None)
    assert result["case_numbers"] == ["CV-2025-1234"]
    assert result["document_type"] is None


async def test_us_metadata_with_fake_model():
    canned = CaptionMetadata(
        document_type="complaint",
        primary_case_number="2025-CA-2720",
        court="Circuit Court",
        court_level="trial",
        jurisdiction="Florida",
        plaintiffs=["A"],
        defendants=["B"],
    )
    llm = FakeLanguageModel(structured={"CaptionMetadata": canned})
    result = await extract_us_legal_metadata("IN THE CIRCUIT COURT\nCase 2025-CA-2720", llm=llm)
    assert result["document_type"] == "complaint"
    assert result["case_number_key"] == "CA-2025-2720"
    assert result["plaintiffs"] == ["A"]


async def test_us_metadata_degrades_on_model_failure_and_bad_enums():
    failed = await extract_us_legal_metadata("Case 2:25-cv-01234", llm=FailingModel())
    assert failed["case_numbers"] == ["CV-2025-1234"]
    odd = CaptionMetadata(document_type="not_a_type", court_level="mars")
    llm = FakeLanguageModel(structured={"CaptionMetadata": odd})
    result = await extract_us_legal_metadata("some text", llm=llm)
    assert result["document_type"] == "other"
    assert result["court_level"] == "unknown"


async def test_ner_regex_and_injected_classifier():
    text = (
        "As held in Smith v. Jones, 410 U.S. 113, the court cited 42 U.S.C. § 1983 on negligence."
    )
    plain = HybridLegalNERExtractor()
    assert plain.use_transformer_ner is False
    assert "Smith v. Jones" in await plain.extract_case_names(text)
    assert plain.extract_statute_references(text)
    assert "negligence" in plain.extract_legal_concepts(text)

    class Classifier:
        async def token_classification(self, text):
            return [
                {"entity_group": "ORG", "word": "Acme", "end": 4},
                {"entity_group": "ORG", "word": "Beta", "end": 14},
            ]

    model = HybridLegalNERExtractor(token_classifier=Classifier())
    assert await model.extract_case_names("Acme v. Beta was decided.") == ["Acme v. Beta"]


async def test_multilingual_pipeline_uses_injected_embedder():
    embedder = FakeEmbedder(dimension=6)
    pipeline = MultilingualPipeline(embedder)
    single = await pipeline.embed("The court affirmed.")
    assert single.dimension == 6
    assert single.language == "en"
    assert single.embedder_used == EmbedderType.DEFAULT
    batch = await pipeline.embed_batch(["one text", "another text"])
    assert [b.text for b in batch] == ["one text", "another text"]
    assert "en" in pipeline.get_supported_languages()


async def test_cross_lingual_search_with_injected_embedder():
    searcher = CrossLingualSearcher(embedder=FakeEmbedder(dimension=8))
    assert await searcher.initialize() is True
    await searcher.index_documents([SearchDocument(id="1", text="alpha beta", language="en")])
    result = await searcher.search("alpha beta", min_score=0.0)
    assert result.results and result.results[0]["id"] == "1"
    assert json.dumps(result.to_dict())
