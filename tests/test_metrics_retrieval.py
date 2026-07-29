"""Tests for rank-based retrieval metrics.

Every expected value here is computed by hand from the TREC definitions and
written as a literal, so the test would catch a change in the implementation
rather than restating it.
"""

from __future__ import annotations

import math

import pytest

from retrieval_ablation.metrics.retrieval import (
    aggregate,
    dcg_at_k,
    ideal_dcg_at_k,
    ndcg_at_k,
    per_query_scores,
    recall_at_k,
    reciprocal_rank,
    score_run,
)


class TestDCG:
    def test_hand_computed_binary(self):
        # relevant at positions 1 and 3:
        #   1/log2(2) + 1/log2(4) = 1.0 + 0.5
        assert dcg_at_k(["a", "b", "c"], {"a": 1, "c": 1}, 3) == pytest.approx(1.5)

    def test_graded_gain_uses_exponential_form(self):
        # gain 3 at position 1 -> (2^3 - 1)/log2(2) = 7.0
        assert dcg_at_k(["a"], {"a": 3}, 1) == pytest.approx(7.0)

    def test_truncation_at_k_excludes_later_hits(self):
        assert dcg_at_k(["x", "a"], {"a": 1}, 1) == pytest.approx(0.0)

    def test_ideal_dcg_ignores_the_ranking_entirely(self):
        # Two relevant documents -> 1/log2(2) + 1/log2(3)
        expected = 1.0 + 1.0 / math.log2(3)
        assert ideal_dcg_at_k({"a": 1, "z": 1}, 10) == pytest.approx(expected)


class TestNDCG:
    def test_hand_computed(self):
        # DCG = 1.5, IDCG = 1 + 1/log2(3) = 1.630929...
        assert ndcg_at_k(["a", "b", "c"], {"a": 1, "c": 1}, 3) == pytest.approx(0.91972, abs=1e-5)

    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(1.0)

    def test_no_relevant_retrieved_is_zero(self):
        assert ndcg_at_k(["x", "y"], {"a": 1}, 10) == pytest.approx(0.0)

    def test_order_matters(self):
        good = ndcg_at_k(["a", "x", "y"], {"a": 1}, 10)
        bad = ndcg_at_k(["x", "y", "a"], {"a": 1}, 10)
        assert good is not None and bad is not None
        assert good > bad

    def test_graded_relevance_prefers_higher_gain_first(self):
        high_first = ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 10)
        low_first = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 10)
        assert high_first == pytest.approx(1.0)
        assert low_first is not None and low_first < 1.0

    def test_idcg_is_built_from_full_qrels_not_the_returned_list(self):
        """Regression: IDCG built from retrieved docs scores a 1-of-3 hit as 1.0.

        A tempting simplification is to normalise by the DCG of the documents
        the system returned. Under that definition a system that finds one of
        three relevant documents and stops gets a perfect nDCG, because the
        single hit it returned is also the best possible ordering *of that
        list*. The metric then rewards returning less, which inverts what it is
        supposed to measure.

        Correct value: DCG = 1.0, IDCG@10 over three relevant documents is
        1 + 1/log2(3) + 1/log2(4) = 2.130929..., so nDCG = 0.469277...
        """
        value = ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 10)
        assert value == pytest.approx(0.46928, abs=1e-5)
        assert value != pytest.approx(1.0)


class TestRecall:
    def test_hand_computed(self):
        # top-3 contains a and c; three documents are relevant
        assert recall_at_k(["a", "b", "c", "d"], {"a": 1, "c": 1, "z": 1}, 3) == pytest.approx(
            2.0 / 3.0
        )

    def test_unretrievable_relevant_doc_caps_recall(self):
        assert recall_at_k(["a"], {"a": 1, "missing": 1}, 50) == pytest.approx(0.5)

    def test_zero_gain_entries_are_not_relevant(self):
        # An explicit judgement of 0 means "judged, not relevant" and must not
        # be counted in the denominator.
        assert recall_at_k(["a"], {"a": 1, "b": 0}, 10) == pytest.approx(1.0)

    def test_k_larger_than_ranking_is_not_an_error(self):
        assert recall_at_k(["a"], {"a": 1}, 1000) == pytest.approx(1.0)


class TestReciprocalRank:
    def test_first_relevant_at_position_two(self):
        assert reciprocal_rank(["x", "a", "b"], {"a": 1}) == pytest.approx(0.5)

    def test_only_the_first_hit_counts(self):
        assert reciprocal_rank(["a", "b"], {"a": 1, "b": 1}) == pytest.approx(1.0)

    def test_no_hit_is_a_real_zero_not_none(self):
        # Distinct from the unjudged case: the system genuinely failed.
        assert reciprocal_rank(["x"], {"a": 1}) == 0.0


class TestUnmeasurableReturnsNone:
    """The project must never report an invented number. These pin that."""

    def test_ndcg_none_when_no_relevant_documents_exist(self):
        assert ndcg_at_k(["a"], {}, 10) is None

    def test_recall_none_when_no_relevant_documents_exist(self):
        assert recall_at_k(["a"], {}, 10) is None

    def test_mrr_none_when_no_relevant_documents_exist(self):
        assert reciprocal_rank(["a"], {}) is None

    def test_all_zero_gain_qrels_count_as_unjudged(self):
        assert ndcg_at_k(["a"], {"a": 0, "b": 0}, 10) is None
        assert recall_at_k(["a"], {"a": 0}, 10) is None
        assert reciprocal_rank(["a"], {"a": 0}) is None


class TestValidation:
    def test_duplicate_passage_ids_are_rejected(self):
        """A repeated document would double-count in Recall and pad MRR."""
        with pytest.raises(ValueError, match="duplicate"):
            ndcg_at_k(["a", "a"], {"a": 1}, 10)

    def test_negative_gain_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ndcg_at_k(["a"], {"a": -1}, 10)


class TestAggregate:
    def test_mean_over_scored_queries_only(self):
        summary = aggregate("m", {"q1": 1.0, "q2": 0.0, "q3": None})
        assert summary.value == pytest.approx(0.5)
        assert summary.n_scored == 2
        assert summary.n_skipped == 1
        assert summary.measured

    def test_unjudged_queries_do_not_drag_the_mean_to_zero(self):
        """Regression: treating None as 0.0 reports a measurement not taken."""
        summary = aggregate("m", {"q1": 1.0, "q2": None, "q3": None})
        assert summary.value == pytest.approx(1.0)  # not 1/3
        assert summary.n_skipped == 2

    def test_nothing_scored_yields_none_not_zero(self):
        summary = aggregate("m", {"q1": None, "q2": None})
        assert summary.value is None
        assert not summary.measured
        assert summary.n_scored == 0
        assert "not measured" in str(summary)


class TestScoreRun:
    def test_end_to_end_shape_and_keys(self):
        run = {"q1": ["a", "b"], "q2": ["x", "c"]}
        qrels = {"q1": {"a": 1}, "q2": {"c": 1}}
        out = score_run(run, qrels, ndcg_k=10, recall_k=50)

        assert set(out) == {"ndcg@10", "recall@50", "mrr"}
        assert out["recall@50"].value == pytest.approx(1.0)
        # q1 hits at rank 1, q2 at rank 2 -> MRR = (1.0 + 0.5) / 2
        assert out["mrr"].value == pytest.approx(0.75)

    def test_query_missing_from_qrels_is_skipped_not_scored_zero(self):
        run = {"q1": ["a"], "unlabelled": ["z"]}
        qrels = {"q1": {"a": 1}}
        out = score_run(run, qrels)
        assert out["ndcg@10"].value == pytest.approx(1.0)
        assert out["ndcg@10"].n_scored == 1
        assert out["ndcg@10"].n_skipped == 1

    def test_per_query_scores_omits_unjudged(self):
        run = {"q1": ["a"], "q2": ["z"]}
        qrels = {"q1": {"a": 1}}
        scores = per_query_scores(run, qrels, "ndcg", 10)
        assert set(scores) == {"q1"}
