"""Tests for growing the eval set without discarding what is attached to it.

Offline: `merge_preserving_existing` is pure, which is where the risk lives.
"""

from __future__ import annotations

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.build import merge_preserving_existing
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind, Verification


def q(qid: str, verification=Verification.GENERATED, checked_by=None) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        text=f"question {qid}",
        gold=(GoldPassage(passage_id=f"p-{qid}", doc_id="d1", span=Span(0, 10)),),
        kind=QueryKind.TABLE_LOOKUP,
        verification=verification,
        lexical_overlap=0.5,
        metadata={},
        checked_by=checked_by,
    )


class TestMergePreservingExisting:
    def test_audit_verdicts_survive_the_extension(self):
        """Regression: regenerating would have wiped 216 model-checked labels.

        A query's text and gold span are a pure function of the corpus and can be
        rebuilt at will. Its verification state cannot -- it is the record of an
        audit that cost 216 model calls, and the 44 REJECTED verdicts are what the
        accepted-subset robustness check rests on. `write_eval_set` overwrites
        unconditionally and rebuilt queries carry `GENERATED` with no checker, so
        growing the set by re-running the builder would have replaced every
        audited label with an unaudited one bearing the same id.
        """
        existing = [
            q("a", Verification.REJECTED, "gemini-3.5-flash-lite"),
            q("b", Verification.MODEL_CHECKED, "gemini-3.5-flash-lite"),
        ]
        rebuilt = [q("a"), q("b"), q("c")]

        merged, orphaned = merge_preserving_existing(existing, rebuilt)

        by_id = {x.query_id: x for x in merged}
        assert by_id["a"].verification is Verification.REJECTED
        assert by_id["b"].verification is Verification.MODEL_CHECKED
        assert by_id["a"].checked_by == "gemini-3.5-flash-lite"
        assert orphaned == []

    def test_new_queries_are_appended(self):
        merged, _ = merge_preserving_existing([q("a")], [q("a"), q("b"), q("c")])

        assert [x.query_id for x in merged] == ["a", "b", "c"]

    def test_existing_order_is_preserved(self):
        """Committed ids keep their positions so diffs stay readable."""
        merged, _ = merge_preserving_existing([q("b"), q("a")], [q("a"), q("b"), q("c")])

        assert [x.query_id for x in merged][:2] == ["b", "a"]

    def test_a_committed_query_the_corpus_no_longer_makes_is_kept_and_reported(self):
        """Dropping it silently would shrink the benchmark without saying so.

        An id the current corpus does not regenerate usually means the corpus
        moved under the labels -- which happened here twice, once to a parser bug
        and once to a company filing a new annual report. That is worth surfacing
        and never worth discarding quietly.
        """
        merged, orphaned = merge_preserving_existing([q("gone"), q("a")], [q("a"), q("b")])

        assert orphaned == ["gone"]
        assert "gone" in {x.query_id for x in merged}

    def test_extending_with_nothing_new_changes_nothing(self):
        existing = [q("a", Verification.REJECTED), q("b")]

        merged, orphaned = merge_preserving_existing(existing, [q("a"), q("b")])

        assert merged == existing
        assert orphaned == []
