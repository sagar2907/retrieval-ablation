"""Measure how much the single-gold-passage design deflates the reported metrics.

THE PROBLEM, FROM THE LITERATURE

nDCG treats any retrieved document without a relevance judgement as non-relevant.
That is safe on a pooled test collection where every plausible candidate has been
judged, and unsafe here, where each query carries exactly one labelled gold row.
The IR literature calls this pooling bias, and warns specifically that "missing
relevant but not judged documents yield no gain in evaluation" and that systems
differing in how many unjudged documents they return cannot be compared fairly.

WHY IT BITES THIS CORPUS PARTICULARLY HARD

A filing reports the same figure in several places. Research and development
expense for a fiscal year appears in the income statement, again in the
Management's Discussion and Analysis prose, and often again in a segment or
five-year-summary table. The eval set labels one of those rows as gold, because
that is the one the fact was extracted from. A retriever that returns the MD&A
sentence stating the identical number has answered the question, and is scored as
having failed.

WHAT THIS MODULE DOES

It quantifies the effect instead of arguing about it. For each query it counts how
many of the top-k retrieved chunks contain the query's expected value *in the
correct document* but are not the labelled gold chunk. Those are
unjudged-but-plausibly-relevant.

A LENIENT VARIANT, AND WHY IT IS NOT AN UPPER BOUND

`lenient_qrels` widens the judgements so any chunk from the gold document
containing the expected figure counts as relevant, graded 1 against the labelled
gold's 2.

**The obvious reading of that variant is wrong, and measuring it is what showed
so.** It was built expecting lenient nDCG and Recall to come out *higher* than
strict, bracketing the true value from above. They come out lower: on the full
corpus, lenient nDCG@10 fell from 0.1912 to 0.1830 and Recall@50 from 0.5324 to
0.3623.

That is not evidence retrieval is worse. It is arithmetic, and it follows directly
from a decision made much earlier in this project. IDCG is computed from the
*complete* judgement set rather than from the retrieved list (see
`metrics.retrieval`, where a regression test pins that). Widening the judgements
adds relevant documents the retriever mostly did not return, so the ideal ranking
gets better while the actual one barely moves, and the ratio falls. Recall behaves
the same way for a simpler reason: its denominator is the count of relevant
documents, and that count grew from one to several.

Worked through concretely, with the gold at rank 3 of 5 and four duplicates that
exist in the filing but are not retrieved:

    strict   nDCG@10 = 0.5000   Recall = 1.0000   MRR = 0.3333
    lenient  nDCG@10 = 0.3031   Recall = 0.2000   MRR = 0.3333

So **only MRR is a valid signal here**, because it depends solely on the rank of
the first relevant document and cannot be diluted by relevant documents that were
never retrieved. Under the lenient judgements MRR rises from 0.1745 to 0.2396 on
structure-aware chunks, and that increase *is* the evidence that the strict labels
under-credit retrieval.

Publishing lenient nDCG or Recall as an upper bound would have been a real
methodological error, arrived at by reasoning that sounded right. The numbers are
still reported, clearly labelled as not an upper bound, because the direction of
the change is itself informative about how incomplete the judgements are.

Restricting to the gold document is deliberate and load-bearing. The corpus holds
four consecutive annual reports per company, and a figure reported in fiscal 2025
is frequently restated as a prior-year comparative in the fiscal 2026 filing.
Counting those would reward exactly the year confusion the corpus was built to
punish, and would turn a lenient bound into a meaningless one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..evalset.relevance import Chunk
from ..evalset.schema import EvalQuery
from ..generation.score import contains_expected_value


@dataclass(frozen=True, slots=True)
class UnjudgedReport:
    """How much relevant material the strict judgements are missing."""

    n_queries: int
    #: Queries where at least one unjudged top-k chunk carries the expected value.
    n_with_unjudged_relevant: int
    #: Total such chunks across all queries.
    n_unjudged_relevant_chunks: int
    #: Queries the strict labels score as a complete miss at k, but where an
    #: unjudged chunk in the gold document did carry the answer. These are the
    #: cases where the reported metric is most clearly wrong.
    n_strict_miss_but_answer_present: int
    top_k: int

    @property
    def fraction_affected(self) -> float:
        return self.n_with_unjudged_relevant / self.n_queries if self.n_queries else 0.0

    @property
    def fraction_falsely_scored_zero(self) -> float:
        return self.n_strict_miss_but_answer_present / self.n_queries if self.n_queries else 0.0

    def __str__(self) -> str:
        return (
            f"{self.n_with_unjudged_relevant}/{self.n_queries} queries "
            f"({self.fraction_affected:.1%}) have an unjudged chunk carrying the answer "
            f"in the top {self.top_k}; {self.n_strict_miss_but_answer_present} "
            f"({self.fraction_falsely_scored_zero:.1%}) are scored as complete misses "
            f"despite the answer being retrieved"
        )


def _expected(query: EvalQuery) -> str:
    return query.metadata.get("expected_value", "")


def measure_unjudged(
    queries: Sequence[EvalQuery],
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    chunks: Mapping[str, Chunk],
    top_k: int = 10,
) -> UnjudgedReport:
    """Count retrieved-but-unlabelled chunks that carry the expected figure."""
    affected = 0
    total_chunks = 0
    false_zeros = 0
    scored = 0

    for query in queries:
        expected = _expected(query)
        ranking = run.get(query.query_id)
        if not expected or not ranking:
            continue
        gold_ids = {c for c, g in qrels.get(query.query_id, {}).items() if g > 0}
        if not gold_ids:
            continue
        scored += 1

        gold_docs = {g.doc_id for g in query.gold}
        window = list(ranking[:top_k])
        found_gold = bool(gold_ids & set(window))

        unjudged_hits = 0
        for chunk_id in window:
            if chunk_id in gold_ids:
                continue
            chunk = chunks.get(chunk_id)
            if chunk is None or chunk.doc_id not in gold_docs:
                continue
            if contains_expected_value(chunk.text, expected):
                unjudged_hits += 1

        if unjudged_hits:
            affected += 1
            total_chunks += unjudged_hits
            if not found_gold:
                false_zeros += 1

    return UnjudgedReport(
        n_queries=scored,
        n_with_unjudged_relevant=affected,
        n_unjudged_relevant_chunks=total_chunks,
        n_strict_miss_but_answer_present=false_zeros,
        top_k=top_k,
    )


def lenient_qrels(
    queries: Sequence[EvalQuery],
    qrels: Mapping[str, Mapping[str, int]],
    chunks: Sequence[Chunk],
) -> dict[str, dict[str, int]]:
    """Strict judgements widened to any same-document chunk holding the figure.

    Duplicates are graded 1 against the labelled gold's 2. They answer the
    question but are not the passage the fact was extracted from, and nDCG's
    graded gain can express that distinction where a binary widening could not.

    Scanning is restricted to the gold document, which keeps a prior-year
    comparative in the *next* year's filing from counting as relevant. Without
    that restriction the lenient bound would reward the year confusion the corpus
    exists to test.
    """
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    out: dict[str, dict[str, int]] = {}
    for query in queries:
        judgements = dict(qrels.get(query.query_id, {}))
        expected = _expected(query)
        if expected:
            for gold in query.gold:
                for chunk in by_doc.get(gold.doc_id, ()):
                    if chunk.chunk_id in judgements:
                        continue
                    if contains_expected_value(chunk.text, expected):
                        judgements[chunk.chunk_id] = 1
        out[query.query_id] = judgements
    return out
