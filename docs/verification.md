# Verification

Verification checks whether the citations and claims in a generated legal answer are supported by the sources that were retrieved for it.
Every check is deterministic and runs without a model or a network call.
Most checks that cannot run say so (for example `verified=False`) instead of passing.
Two are the exception: `detect_fabricated_cases` and `unbacked_case_citations` return an empty list when they have no sources, and only log a warning (see "Known limits").

## Install

```bash
pip install "legal-rag[verification]"
```

The `verification` extra has no dependencies beyond the base install, and the package needs Python 3.11 or newer.

## The checks

None of these calls a language model.
Where a check needs outside knowledge, such as whether a case exists, you inject it through a protocol (see "Supplying a CaseResolver").

| Check | Entry point | Needs a language model |
| --- | --- | --- |
| Citation extraction | `legal_rag.verification.extract`, `legal_rag.verification.citations.extract_unique` | No |
| Citation grounding | `legal_rag.verification.grounding.ungrounded_legal_citations`, `ungrounded_case_citations`, and `legal_rag.verification.citations.unbacked_case_citations` | No |
| Fabricated-case detection | `legal_rag.verification.grounding.detect_fabricated_cases`, `detect_state_mismatches`, `evaluate_named_case_guard`, `evaluate_named_case_guard_with_resolver` | No (the resolver is yours) |
| Deterministic grounding | `legal_rag.verification.eval.score_citations`, and in `legal_rag.verification.citations.validator`: `quick_citation_validation`, `strip_fabricated_citations`, `CrossReferenceCitationValidator` | No |
| Claim checks | `legal_rag.verification.grounding.HallucinationDetector` (`detect` and `check`) | No |
| RAGAS-style evaluation | `legal_rag.verification.eval.RAGASEvaluator`, `LegalEvaluator` | No |
| Retrieval metrics | `legal_rag.verification.eval.evaluate_retrieval`, `evaluate_retrieval_async`, `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, `stage_attribution` | No |

The fakes in `legal_rag.providers.fakes` are not needed by any check, but they are handy for building a test `CaseResolver`.

## Quick start

This script extracts the citations in an answer, flags a case that no retrieved source supports, and scores the answer with `HallucinationDetector`.
It needs no keys and no network.

```python
import asyncio

from legal_rag.verification import extract
from legal_rag.verification.grounding import HallucinationDetector, detect_fabricated_cases

ANSWER = (
    "Courts may strike down statutes that conflict with the Constitution. "
    "See Marbury v. Madison, 5 U.S. 137 (1803). "
    "The Court later extended this reasoning to state agencies in "
    "Hartwell v. Blackstone Holdings, 482 U.S. 911 (1987). "
    "Standing is governed by 28 U.S.C. § 1331."
)

# What retrieval returned: one case passage.
SOURCES = [
    {
        "case_name": "Marbury v. Madison",
        "citation": "5 U.S. 137",
        "text": "Marbury v. Madison, 5 U.S. 137 (1803). Courts may strike down statutes "
        "that conflict with the Constitution.",
    }
]

# 1. Extract every citation-shaped string. A match is a shape, not proof the case exists.
for c in extract(ANSWER):
    print(f"{c.kind.value:<5} {c.raw!r} -> {c.normalized}")

# 2. Flag cases that no retrieved source supports.
for flag in detect_fabricated_cases(ANSWER, SOURCES):
    print("FLAGGED:", flag.case_name, "|", flag.citation)

# 3. Score the answer against the retrieved text. No model is called.
detector = HallucinationDetector()
context = " ".join(s["text"] for s in SOURCES)
result = detector.detect(ANSWER, context)
print("hallucination_score:", round(result.hallucination_score, 2))
print("is_hallucinated:", result.is_hallucinated, "| requires_review:", result.requires_review)
for claim in result.claim_analyses:
    print(f"  {claim.grounding_level.value:<10} {claim.confidence:.2f}  {claim.claim[:60]}")


checked = asyncio.run(detector.check(ANSWER, SOURCES))
print("check score:", round(checked.score, 2), "| verified:", checked.verified)
print("unsupported:", len(checked.unsupported_claims), "of", checked.total_claims)
```

Output:

```text
case  '5 U.S. 137' -> 5 U.S. 137
case  '482 U.S. 911' -> 482 U.S. 911
usc   '28 U.S.C. § 1331' -> 28 U.S.C. § 1331
FLAGGED: Hartwell v. Blackstone Holdings | 482 U.S. 911
hallucination_score: 0.5
is_hallucinated: True | requires_review: True
  grounded   0.83  See Marbury v. Madison, 5 U.S. 137 (1803).
  ungrounded 0.10  The Court later extended this reasoning to state agencies in
  ungrounded 0.00  Standing is governed by 28 U.S.C. § 1331.
  grounded   1.00  Courts may strike down statutes that conflict with the Const
check score: 0.5 | verified: False
unsupported: 2 of 4
```

Claim splitting keeps citations such as `5 U.S. 137` and `28 U.S.C.` whole (see "Known limits").
A runnable variant that also resolves each citation is in `examples/verification_basic.py`.

## Supplying a CaseResolver

`CaseResolver` is a `runtime_checkable` protocol in `legal_rag.verification`.
It has one method, `async def resolve(self, query: str) -> ResolvedCase | None`, which returns the matching case or `None`.
`ResolvedCase` is a frozen dataclass with `case_name`, `citation`, `court`, `year` and `metadata`.
Back it with your own case-law index, API client or database.

`evaluate_named_case_guard_with_resolver(query, sources, resolver)` is the one component that takes a resolver.
When the sources do not match a case the user asked about, it asks the resolver with each reporter citation and then with the `A v. B` name.
It stops declining as soon as one probe returns a case.
Citation probes are passed verbatim, and the name probe is the case name as written in the query (`Marbury v. Madison`, with its case kept), trimmed to the name.
A name typed in all lowercase has no capitals to find its edges, so it is cut to at most three words on each side and you should match loosely.
The tests build a resolver on `FakeEmbedder` and `FakeVectorStore` from `legal_rag.providers.fakes`.
This example uses a dict, which is enough to show the shape.

```python
import asyncio

from legal_rag.verification import CaseResolver, ResolvedCase
from legal_rag.verification.grounding import evaluate_named_case_guard_with_resolver


class DictResolver:
    """A CaseResolver over a dict. Swap in your own case-law index or API client."""

    CASE = ResolvedCase("Marbury v. Madison", "5 U.S. 137", "U.S.", 1803)
    KNOWN = {"marbury v. madison": CASE, "5 u.s. 137": CASE}

    async def resolve(self, query: str) -> ResolvedCase | None:
        return self.KNOWN.get(query.lower())


resolver = DictResolver()
print("satisfies protocol:", isinstance(resolver, CaseResolver))
for query in ("Marbury v. Madison", "Hartwell v. Blackstone Holdings"):
    result = asyncio.run(evaluate_named_case_guard_with_resolver(query, [], resolver))
    print(query, "-> should_decline:", result.should_decline)
```

Output:

```text
satisfies protocol: True
Marbury v. Madison -> should_decline: False
Hartwell v. Blackstone Holdings -> should_decline: True
```

`isinstance(x, CaseResolver)` only checks that a `resolve` attribute exists, not its signature.

## Reading the results

| Object | Field | Meaning |
| --- | --- | --- |
| `Citation` | `raw`, `start`, `end` | The exact matched text, where `text[start:end] == raw`. |
| `Citation` | `normalized` | A canonical form, so two spellings of one cite compare equal. |
| `Citation` | `kind` | A `CitationKind`: `usc`, `cfr`, `federal_rule`, `public_law`, `federal_register`, `state_statute` or `case`. |
| `FlaggedCase` | `case_name`, `citation`, `surrounding_text` | A case named in the answer that no source supports, with the text around it. |
| `HallucinationResult` (from `detect`) | `hallucination_score` | 0.0 is good and 1.0 is bad. |
| | `is_hallucinated` | True when the score is at least 0.3 (`HallucinationConfig.hallucination_threshold`). |
| | `requires_review` | True when the score is at least 0.15 (`review_threshold`). |
| | `claim_analyses` | One `ClaimGrounding` per claim, graded `grounded` (confidence 0.8 or more), `partial` or `ungrounded`. |
| `HallucinationCheckResult` (from `check`) | `score` | 1.0 is good and 0.0 is bad, the opposite direction of `detect`. |
| | `verified` | True when `score >= 1 - hallucination_threshold`. |
| | `supported_claims` | Grounded claims only. A `partial` claim is scored as supported but is not counted here. |
| | `unsupported_claims` | Ungrounded claims only. |
| `CitationReport` (from `score_citations`) | `support_rate` | Supported `[N]` markers over supported plus out-of-range plus misattributed, or 1.0 when none were judged. |
| | `out_of_range`, `misattributed`, `weak_window` | A marker past the source count, a weak source when a supporting one existed, and no source strong enough to judge. |
| `EvaluationMetrics` (from `RAGASEvaluator`) | `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` | Each runs from 0 to 1, where 1 is best. |
| | `hallucination_rate` | Ungrounded claims over claims, so lower is better. |
| `EvaluationResult`, `LegalEvaluationResult` | `passed` | True when the weighted overall score reaches `quality_threshold` (default 0.7). |
| `LegalEvaluationResult` | `citation_accuracy`, `jurisdiction_consistency`, `temporal_validity`, `legal_term_precision` | Four sub-results, each with a `to_dict()`. |
| `BenchReport` | `recall`, `mrr`, `ndcg` | Averages keyed by k. |
| `BenchReport` | `attribution` | A count of queries by the first stage that lost the gold passage, or `retrieved`. |

Inputs that cannot be checked are reported, not passed.
`detect` with an empty context returns score 1.0 and `requires_review=True`.
`check` with no sources returns `score=0.5`, `verified=False` and zero claims.
`RAGASEvaluator.evaluate_batch([])` returns `samples_evaluated` 0 and an empty `individual_results`, with `pass_rate` and `average_metrics` set to `None` because nothing was scored.

## Other examples

Index checks for `[N]` markers, a provenance check, and the deterministic citation score:

```python
from legal_rag.verification.citations import unbacked_case_citations
from legal_rag.verification.citations.validator import (
    CrossReferenceCitationValidator,
    quick_citation_validation,
    strip_fabricated_citations,
)
from legal_rag.verification.eval import score_citations

sources = [{"text": "42 U.S.C. § 1983. Civil action for deprivation of rights under color of state law."}]
print(quick_citation_validation("Claim one [1]. Claim two [5].", len(sources)))
print(strip_fabricated_citations("Liability applies [1] and more [7].", len(sources)))
print(CrossReferenceCitationValidator().validate("A claim [9].", sources).invalid_indices)
report = score_citations("Liability arises under 42 U.S.C. 1983 for state actors [1]. A second claim [4].", sources)
print(report.total_markers, report.supported, report.out_of_range, report.support_rate)

opinions = [{"source_type": "case", "title": "Marbury v. Madison"}]
print(unbacked_case_citations("See Marbury v. Madison and also Smith v. Acme Holdings.", opinions))
```

```text
(False, [5])
('Liability applies [1] and more.', [7])
[9]
2 1 1 0.5
['Smith v. Acme Holdings']
```

The evaluators are async.
`LegalEvaluator` takes an optional `precedent_checker` (a `PrecedentChecker` with one method, `check_validity`) and an optional `jurisdiction_patterns` mapping.

```python
import asyncio

from legal_rag.verification.eval import LegalEvaluator, PrecedentStatus, PrecedentValidity, RAGASEvaluator

CONTEXT = "42 U.S.C. § 1983. Civil action for deprivation of rights under color of state law."


class Citator:
    async def check_validity(self, citation: str, *, context: str = "") -> PrecedentValidity:
        overruled = "5 U.S. 137" in citation
        return PrecedentValidity(status=PrecedentStatus.OVERRULED if overruled else PrecedentStatus.VALID)


answer = "42 U.S.C. 1983 provides a civil action for deprivation of rights under color of state law."
ragas = asyncio.run(RAGASEvaluator().evaluate("What does 42 U.S.C. 1983 provide?", answer, [CONTEXT]))
print(ragas.metrics.to_dict(), ragas.passed)
legal = asyncio.run(
    LegalEvaluator(precedent_checker=Citator()).evaluate(
        "q", "Marbury v. Madison, 5 U.S. 137 (1803), and 99 F.3d 1 agree.", "5 U.S. 137"
    )
)
print(round(legal.overall_legal_score, 3), legal.citation_accuracy.fabricated_citations)
print(legal.temporal_validity.overruled_citations)
```

```text
{'faithfulness': 1.0, 'answer_relevancy': 0.25, 'context_precision': 1.0, 'context_recall': 0.9, 'hallucination_rate': 0.0, 'overall_score': 0.798} True
0.77 ['99 F.3d 1']
['5 U.S. 137']
```

For a labeled retrieval benchmark, pass `LabeledQuery` items and a retriever returning `StageIds` to `evaluate_retrieval` (sync) or `evaluate_retrieval_async`.
Both return a `BenchReport`, and `load_labeled_set` reads the queries from JSONL.

## Known limits

- A citation match means the text has the shape of a citation.
  It never means the authority exists.
- `detect_fabricated_cases` returns an empty list when there are no sources, and `unbacked_case_citations` does the same when no retrieved source is an opinion.
  An empty list then means nothing could be checked, not that nothing was fabricated, and the return type cannot tell the two apart.
  Each call logs one `WARNING` in that case, from the logger `legal_rag.verification.grounding.fabricated_case` or `legal_rag.verification.citations.provenance`, and the docstrings say so.
  Test for empty sources yourself, or watch for the warning.
- `detect_fabricated_cases` flags a case only when its name is absent from the sources, so a real name with a wrong citation passes.
  It skips bare names with no reporter cite, `id.` and `supra`.
- `ungrounded_legal_citations` treats a citation as grounded when its core identifier appears anywhere in the source text.
  It does not cover statutes, regulations or Federal Register cites.
- `HallucinationDetector` and both evaluators are lexical.
  They compare up to ten key terms per claim and do not read meaning, and `detect` never assigns the `contradicted` level.
- Claim splitting cuts at `.`, `!` and `?`, except inside a short list of legal abbreviations: dotted forms such as `U.S.`, `N.Y.` and `S. Ct.`, `v.`, `Inc.`, `Corp.`, `Co.`, `No.`, `Cir.`, `Dist.`, `Ct.`, state names such as `Cal.`, and numbers such as `12.5`.
  A sentence that ends in one of those is not split there, and any abbreviation not on the list still splits a claim.
  A claim with no key terms counts as grounded, and the `citations` argument of `detect` is unused.
- A claim counts against the score only when it is `ungrounded`, meaning fewer than 0.4 of its first ten key terms (`HallucinationConfig.min_overlap_ratio`) appear in the context.
  A `partial` claim (0.4 up to 0.8) is scored as supported, so one grounded claim plus one partial claim gives `score=1.0` and `verified=True` while `supported_claims` is 1 of `total_claims` 2.
  This is deliberate, not a bug.
  Read `claim_details` (a partial claim is `NEUTRAL`) when you need to see partial support.
- Once any sentence matches one of the four claim patterns, sentences that match none of them are not checked at all.
  An unsupported sentence worded without any pattern word (`is`, `are`, `may`, `must`, `shall` and so on) can therefore leave the score at 1.0.
- `RAGASEvaluator` is RAGAS-style, not the `ragas` library.
  `answer_relevancy` is key-term overlap with the query, plus 0.2 for an answer over 100 characters.
  Without `ground_truth`, `context_recall` measures how much of the answer appears in the context.
- `LegalEvaluator` counts every extracted citation as valid in format, so `invalid_citations` stays 0.
  Without a `precedent_checker` every citation counts as valid, and `UNKNOWN` status also counts as valid.
  Without `jurisdiction_patterns` the jurisdiction check finds nothing to flag.
- `score_citations` judges only `[N]` markers, against a fixed support floor (`SUPPORT_FLOOR`).
- `detect_state_mismatches` knows only the prefixes in its bundled data file and never flags federal reporters.
- `get_hallucination_detector`, `get_hallucination_checker`, `get_ragas_evaluator` and `get_legal_evaluator` are singletons.
  Arguments on a later call are ignored, so construct the class directly when you need a different configuration.
- Only US citation formats are recognized.

## Tests

The tests live in `tests/verification/`, with one file each for extraction, citation checks, grounding and evaluation.
Run them with `python -m pytest tests/verification`.
