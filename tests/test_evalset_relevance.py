"""Tests for translating chunker-independent gold spans into qrels."""

from __future__ import annotations

import pytest

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.relevance import (
    Chunk,
    build_qrels,
    common_judgeable_queries,
    judgeable_queries,
    reachability,
    relevant_chunk_ids,
)


def chunk(chunk_id: str, start: int, end: int, doc_id: str = "d1") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, span=Span(start, end), text="x" * (end - start))


class TestRelevantChunkIds:
    def test_fully_containing_chunk_is_relevant(self):
        gold = GoldPassage("p1", "d1", Span(100, 200))
        assert relevant_chunk_ids(gold, [chunk("c1", 0, 500)]) == ["c1"]

    def test_chunk_below_threshold_is_not_relevant(self):
        # Chunk covers 100 of the gold span's 400 characters -> 0.25 < 0.5
        gold = GoldPassage("p1", "d1", Span(100, 500))
        assert relevant_chunk_ids(gold, [chunk("c1", 0, 200)]) == []

    def test_chunk_exactly_at_threshold_is_relevant(self):
        gold = GoldPassage("p1", "d1", Span(0, 100))
        assert relevant_chunk_ids(gold, [chunk("c1", 0, 50)], min_coverage=0.5) == ["c1"]

    def test_all_qualifying_overlapping_chunks_are_relevant(self):
        """Overlapping windows must all count, or recall is understated.

        A sliding-window chunker emits several chunks containing the same
        answer. Retrieving any one of them is a success, so marking only the
        single best chunk relevant would make an overlapping chunker look worse
        than a non-overlapping one for no real reason.
        """
        gold = GoldPassage("p1", "d1", Span(100, 200))
        chunks = [
            chunk("c1", 0, 300),  # contains the gold span entirely -> 1.00
            chunk("c2", 180, 450),  # catches only the tail -> 0.20, excluded
            chunk("c3", 50, 250),  # contains it entirely -> 1.00
        ]
        assert relevant_chunk_ids(gold, chunks) == ["c1", "c3"]

    def test_chunks_from_other_documents_are_ignored(self):
        gold = GoldPassage("p1", "d1", Span(0, 100))
        assert relevant_chunk_ids(gold, [chunk("c1", 0, 500, doc_id="d2")]) == []

    def test_stricter_threshold_shrinks_the_relevant_set(self):
        gold = GoldPassage("p1", "d1", Span(0, 100))
        chunks = [chunk("c1", 0, 60), chunk("c2", 0, 100)]
        assert relevant_chunk_ids(gold, chunks, min_coverage=0.5) == ["c1", "c2"]
        assert relevant_chunk_ids(gold, chunks, min_coverage=1.0) == ["c2"]

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_threshold_rejected(self, bad: float):
        gold = GoldPassage("p1", "d1", Span(0, 10))
        with pytest.raises(ValueError, match="min_coverage"):
            relevant_chunk_ids(gold, [], min_coverage=bad)


class TestBuildQrels:
    def test_basic_mapping(self):
        gold_by_query = {"q1": [GoldPassage("p1", "d1", Span(100, 200))]}
        chunks = [chunk("c1", 0, 300), chunk("c2", 400, 700)]
        assert build_qrels(gold_by_query, chunks) == {"q1": {"c1": 2}}

    def test_grade_is_carried_from_the_gold_passage(self):
        gold_by_query = {"q1": [GoldPassage("p1", "d1", Span(0, 100), gain=1)]}
        assert build_qrels(gold_by_query, [chunk("c1", 0, 100)]) == {"q1": {"c1": 1}}

    def test_chunk_covering_two_golds_takes_the_higher_grade(self):
        gold_by_query = {
            "q1": [
                GoldPassage("p1", "d1", Span(0, 50), gain=1),
                GoldPassage("p2", "d1", Span(60, 90), gain=2),
            ]
        }
        assert build_qrels(gold_by_query, [chunk("c1", 0, 100)]) == {"q1": {"c1": 2}}

    def test_unreachable_query_is_present_but_empty(self):
        """Distinguishes "judged, nothing reachable" from "not in the eval set".

        The metrics treat both as unmeasurable, but only the first indicates a
        chunking failure, and conflating them would hide it.
        """
        gold_by_query = {"q1": [GoldPassage("p1", "d1", Span(0, 1000))]}
        qrels = build_qrels(gold_by_query, [chunk("c1", 0, 100)])
        assert qrels == {"q1": {}}
        assert "q1" in qrels

    def test_multiple_queries_are_independent(self):
        gold_by_query = {
            "q1": [GoldPassage("p1", "d1", Span(0, 50))],
            "q2": [GoldPassage("p2", "d1", Span(500, 550))],
        }
        qrels = build_qrels(gold_by_query, [chunk("c1", 0, 100), chunk("c2", 480, 600)])
        assert qrels == {"q1": {"c1": 2}, "q2": {"c2": 2}}


class TestTheChunkingComparabilityTrap:
    """The failure mode `relevance.py` exists to surface.

    A long gold span is unreachable by small chunks. Left unreported, the
    affected queries vanish from that configuration's average, and the
    configuration that fails hardest on them stops being scored on them --
    which looks identical to an improvement.
    """

    @pytest.fixture
    def gold_by_query(self):
        return {
            "short_answer": [GoldPassage("p_short", "d1", Span(0, 100))],
            "long_table_answer": [GoldPassage("p_long", "d1", Span(1000, 3000))],
        }

    def test_small_chunks_cannot_reach_a_long_gold_span(self, gold_by_query):
        small = [chunk(f"s{i}", i * 250, (i + 1) * 250) for i in range(16)]
        report = reachability("fixed-250", gold_by_query, small)
        assert report.unreachable_passage_ids == ("p_long",)
        assert report.n_reachable == 1
        assert report.fraction_reachable == pytest.approx(0.5)

    def test_large_chunks_reach_both(self, gold_by_query):
        large = [chunk("l0", 0, 2000), chunk("l1", 900, 3200)]
        report = reachability("structure-aware", gold_by_query, large)
        assert report.unreachable_passage_ids == ()
        assert report.fraction_reachable == pytest.approx(1.0)

    def test_the_two_configs_would_otherwise_be_scored_on_different_queries(self, gold_by_query):
        small = [chunk(f"s{i}", i * 250, (i + 1) * 250) for i in range(16)]
        large = [chunk("l0", 0, 2000), chunk("l1", 900, 3200)]

        qrels_small = build_qrels(gold_by_query, small)
        qrels_large = build_qrels(gold_by_query, large)

        assert judgeable_queries(qrels_small) == {"short_answer"}
        assert judgeable_queries(qrels_large) == {"short_answer", "long_table_answer"}

        # The shared subset is what a fair headline comparison must use.
        common = common_judgeable_queries({"small": qrels_small, "large": qrels_large})
        assert common == {"short_answer"}

    def test_common_subset_of_a_single_config_is_its_own(self):
        gold = {"q1": [GoldPassage("p1", "d1", Span(0, 50))]}
        qrels = build_qrels(gold, [chunk("c1", 0, 100)])
        assert common_judgeable_queries({"only": qrels}) == {"q1"}

    def test_common_subset_of_nothing_is_empty(self):
        assert common_judgeable_queries({}) == set()

    def test_config_that_reaches_nothing_empties_the_common_subset(self):
        gold = {"q1": [GoldPassage("p1", "d1", Span(0, 500))]}
        good = build_qrels(gold, [chunk("c1", 0, 500)])
        useless = build_qrels(gold, [chunk("c2", 0, 10)])
        assert common_judgeable_queries({"good": good, "useless": useless}) == set()


class TestReachabilityReport:
    def test_empty_eval_set_does_not_divide_by_zero(self):
        report = reachability("cfg", {}, [chunk("c1", 0, 10)])
        assert report.n_gold == 0
        assert report.fraction_reachable == 0.0

    def test_str_is_human_readable(self):
        gold = {"q1": [GoldPassage("p1", "d1", Span(0, 50))]}
        assert "1/1 gold passages reachable (100.0%)" in str(
            reachability("cfg", gold, [chunk("c1", 0, 100)])
        )
