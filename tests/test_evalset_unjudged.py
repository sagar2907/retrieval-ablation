"""Tests for the unjudged-relevance diagnostic and its counter-intuitive arithmetic."""

from __future__ import annotations

from typing import ClassVar

import pytest

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.relevance import Chunk
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind
from retrieval_ablation.evalset.unjudged import lenient_qrels, measure_unjudged
from retrieval_ablation.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank


def chunk(chunk_id: str, text: str, doc_id: str = "d1") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, span=Span(0, len(text)), text=text)


def query(expected: str = "34,550", doc_id: str = "d1") -> EvalQuery:
    return EvalQuery(
        query_id="q1",
        text="What was research and development expense for 2025?",
        gold=(GoldPassage("p1", doc_id, Span(0, 40)),),
        kind=QueryKind.TABLE_LOOKUP,
        metadata={"expected_value": expected},
    )


class TestMeasureUnjudged:
    def test_counts_a_retrieved_duplicate_as_unjudged_relevant(self):
        """The MD&A restating the income-statement figure is relevant and unlabelled."""
        chunks = {
            "gold": chunk("gold", "| Research and development | 34,550 |"),
            "dup": chunk("dup", "R&D expense rose to 34,550 million in fiscal 2025."),
        }
        report = measure_unjudged(
            [query()],
            {"q1": ["dup", "gold"]},
            {"q1": {"gold": 2}},
            chunks,
            top_k=10,
        )
        assert report.n_with_unjudged_relevant == 1
        assert report.n_unjudged_relevant_chunks == 1

    def test_flags_a_query_scored_zero_although_the_answer_was_retrieved(self):
        """The case where the reported metric is most clearly wrong."""
        chunks = {
            "dup": chunk("dup", "R&D expense was 34,550 million."),
            "noise": chunk("noise", "Unrelated text about leases."),
        }
        report = measure_unjudged(
            [query()],
            {"q1": ["dup", "noise"]},  # gold itself never retrieved
            {"q1": {"gold": 2}},
            chunks,
            top_k=10,
        )
        assert report.n_strict_miss_but_answer_present == 1
        assert report.fraction_falsely_scored_zero == pytest.approx(1.0)

    def test_ignores_the_same_figure_in_a_different_filing(self):
        """A prior-year comparative in next year's filing is not a valid duplicate.

        The corpus holds four consecutive annual reports per company and figures
        are restated as comparatives. Counting those would reward exactly the year
        confusion the corpus was built to punish.
        """
        chunks = {
            "other_year": chunk("other_year", "R&D was 34,550 million.", doc_id="d2"),
        }
        report = measure_unjudged(
            [query()], {"q1": ["other_year"]}, {"q1": {"gold": 2}}, chunks, top_k=10
        )
        assert report.n_with_unjudged_relevant == 0

    def test_query_without_an_expected_value_is_skipped(self):
        report = measure_unjudged(
            [query(expected="")], {"q1": ["x"]}, {"q1": {"gold": 2}}, {}, top_k=10
        )
        assert report.n_queries == 0
        assert report.fraction_affected == 0.0

    def test_only_the_top_k_window_is_examined(self):
        chunks = {
            "noise": chunk("noise", "nothing"),
            "dup": chunk("dup", "R&D was 34,550 million."),
        }
        report = measure_unjudged(
            [query()], {"q1": ["noise", "dup"]}, {"q1": {"gold": 2}}, chunks, top_k=1
        )
        assert report.n_with_unjudged_relevant == 0


class TestLenientQrels:
    def test_adds_same_document_duplicates_at_a_lower_grade(self):
        chunks = [
            chunk("gold", "| Research and development | 34,550 |"),
            chunk("dup", "R&D expense was 34,550 million."),
            chunk("other", "Leases are accounted for as follows."),
        ]
        widened = lenient_qrels([query()], {"q1": {"gold": 2}}, chunks)
        assert widened["q1"]["gold"] == 2
        assert widened["q1"]["dup"] == 1
        assert "other" not in widened["q1"]

    def test_does_not_cross_document_boundaries(self):
        chunks = [chunk("elsewhere", "R&D was 34,550 million.", doc_id="d2")]
        widened = lenient_qrels([query()], {"q1": {"gold": 2}}, chunks)
        assert "elsewhere" not in widened["q1"]

    def test_strict_judgements_are_preserved(self):
        widened = lenient_qrels([query()], {"q1": {"gold": 2}}, [])
        assert widened["q1"] == {"gold": 2}


class TestWideningLowersNdcgAndRecall:
    """Regression on a conclusion that was initially reasoned out backwards.

    The lenient variant was built expecting it to bound the true score from
    *above*. Measured on the full corpus it does the opposite: nDCG@10 fell from
    0.1912 to 0.1830 and Recall@50 from 0.5324 to 0.3623.

    That is arithmetic, not a retrieval result. IDCG is computed from the complete
    judgement set, so adding relevant documents the retriever did not return
    improves the ideal ranking while the actual ranking barely moves. Recall falls
    because its denominator grew. Only MRR is immune, because it depends solely on
    the rank of the first relevant document.

    These assertions exist so nobody later "fixes" the lenient variant into an
    upper bound and publishes it as one.
    """

    RANKING: ClassVar = ["a", "b", "gold", "c", "d"]
    STRICT: ClassVar = {"gold": 2}
    # Duplicates that exist in the filing but were not retrieved -- the common case.
    LENIENT: ClassVar = {"gold": 2, "far1": 1, "far2": 1, "far3": 1, "far4": 1}

    def test_ndcg_falls_when_judgements_widen(self):
        strict = ndcg_at_k(self.RANKING, self.STRICT, 10)
        lenient = ndcg_at_k(self.RANKING, self.LENIENT, 10)
        assert strict == pytest.approx(0.5000, abs=1e-4)
        assert lenient == pytest.approx(0.3031, abs=1e-4)
        assert lenient < strict

    def test_recall_falls_because_its_denominator_grows(self):
        assert recall_at_k(self.RANKING, self.STRICT, 50) == pytest.approx(1.0)
        assert recall_at_k(self.RANKING, self.LENIENT, 50) == pytest.approx(0.2)

    def test_mrr_is_unaffected_by_unretrieved_duplicates(self):
        """Which is why MRR is the only valid signal from this comparison."""
        assert reciprocal_rank(self.RANKING, self.STRICT) == pytest.approx(1 / 3)
        assert reciprocal_rank(self.RANKING, self.LENIENT) == pytest.approx(1 / 3)

    def test_mrr_rises_when_a_duplicate_outranks_the_gold(self):
        """The signal that shows strict labels under-credit retrieval."""
        ranking = ["a", "far1", "gold", "c", "d"]
        assert reciprocal_rank(ranking, self.STRICT) == pytest.approx(1 / 3)
        assert reciprocal_rank(ranking, self.LENIENT) == pytest.approx(0.5)
