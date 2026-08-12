"""Tests for spotting queries no retriever can answer as posed.

Offline: pure function over EvalQuery objects.
"""

from __future__ import annotations

from retrieval_ablation.ablation.runner import ambiguous_queries
from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind, Verification


def q(qid: str, text: str, passage_id: str) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        text=text,
        gold=(
            GoldPassage(
                passage_id=passage_id,
                doc_id=passage_id.split(":", maxsplit=1)[0],
                span=Span(0, 5),
            ),
        ),
        kind=QueryKind.TABLE_LOOKUP,
        verification=Verification.GENERATED,
        lexical_overlap=0.5,
        metadata={},
    )


class TestAmbiguousQueries:
    def test_same_text_different_gold_is_flagged(self):
        """Retrieval is a function of the text, so one of these must score zero.

        A figure reported in two consecutive filings yields the same question
        twice, labelled against different documents, and both labels are correct.
        Every configuration sees one string, returns one ranking, and is marked
        wrong for whichever of the pair it did not return.
        """
        flagged = ambiguous_queries(
            [
                q("a", "What was the tax credit in 2025?", "wmt-2025:1-9"),
                q("b", "What was the tax credit in 2025?", "wmt-2026:1-9"),
                q("c", "Something else entirely?", "nke-2025:1-9"),
            ]
        )

        assert flagged == ["a", "b"]

    def test_same_text_and_same_gold_is_not_flagged(self):
        """A genuine duplicate is redundant, not unanswerable.

        Both copies want the same passage, so a retriever that finds it scores on
        both. Nothing is lost and nothing needs reporting.
        """
        assert (
            ambiguous_queries(
                [
                    q("a", "Identical question?", "wmt-2025:1-9"),
                    q("b", "Identical question?", "wmt-2025:1-9"),
                ]
            )
            == []
        )

    def test_distinct_texts_are_never_flagged(self):
        assert (
            ambiguous_queries(
                [
                    q("a", "First question?", "wmt-2025:1-9"),
                    q("b", "Second question?", "wmt-2026:1-9"),
                ]
            )
            == []
        )

    def test_three_way_collision_reports_every_member(self):
        flagged = ambiguous_queries(
            [
                q("a", "Same?", "d1:1-9"),
                q("b", "Same?", "d2:1-9"),
                q("c", "Same?", "d3:1-9"),
            ]
        )

        assert flagged == ["a", "b", "c"]
