"""Tests for query paraphrasing.

Offline: no client is constructed. `check_rewrite` and `apply_results` are pure,
which is why the risky logic lives in them rather than inside the API loop.
"""

from __future__ import annotations

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.paraphrase import (
    ParaphraseResult,
    _company_tokens,
    apply_results,
    check_rewrite,
)
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind, Verification

PASSAGE = "Research and development | 34,550 | 31,370 | Apple Inc. fiscal 2023 results."


def query(text: str = "What was Apple's research and development expense in fiscal 2023?"):
    return EvalQuery(
        query_id="q-1",
        text=text,
        gold=(GoldPassage(passage_id="p-1", doc_id="aapl", span=Span(0, len(PASSAGE))),),
        kind=QueryKind.TABLE_LOOKUP,
        verification=Verification.GENERATED,
        lexical_overlap=0.9,
        metadata={"ticker": "AAPL", "period": "2023", "report_date": "2023-09-30"},
    )


class TestCompanyTokens:
    def test_ignores_corporate_suffixes(self):
        """ "The Southern Company" and "Southern Co" name the same company.

        Rejecting a rewrite for dropping "Company" or "Inc" would refuse good
        paraphrases in bulk -- those words carry no identifying information.
        """
        assert _company_tokens("The Southern Company") == _company_tokens("Southern Co")

    def test_keeps_distinctive_words(self):
        assert "apple" in _company_tokens("Apple Inc.")


class TestCheckRewrite:
    def test_accepts_a_rewrite_that_keeps_company_and_year(self):
        assert (
            check_rewrite(
                query(), "How much did Apple spend developing new products in 2023?", "Apple Inc."
            )
            is None
        )

    def test_rejects_a_rewrite_that_drops_the_company(self):
        """Without the company the question has thirty correct answers.

        The gold span still points at one row in one filing, so such a query is
        scored wrong whatever a retriever returns -- it would depress every
        configuration equally for a reason that has nothing to do with retrieval.
        """
        reason = check_rewrite(
            query(), "How much was spent developing new products in 2023?", "Apple Inc."
        )
        assert reason is not None
        assert "company" in reason

    def test_rejects_a_rewrite_that_drops_the_fiscal_year(self):
        """The corpus holds four consecutive years per company by design."""
        reason = check_rewrite(
            query(), "How much did Apple spend developing new products?", "Apple Inc."
        )
        assert reason is not None
        assert "period" in reason

    def test_company_check_is_actually_reached(self):
        """Regression: the guard read a metadata key that does not exist.

        `EvalQuery.metadata` carries `ticker`, never `company`. The first version
        of this function looked up "company", got the empty-string default, found
        no tokens to require, and returned None for every input -- passing its own
        tests while checking nothing. The company is now passed in explicitly.

        This asserts the failure is reachable at all: a rewrite that keeps the
        year but drops the name must be refused when a real company is supplied.
        """
        assert query().metadata.get("company") is None
        assert (
            check_rewrite(query(), "What was spent on product development in 2023?", "Apple Inc.")
            is not None
        )

    def test_empty_company_does_not_invent_a_failure(self):
        """A missing name means nothing to check, not something to reject."""
        assert check_rewrite(query(), "What was spent on product development in 2023?", "") is None

    def test_rejects_empty_and_overlong_output(self):
        assert check_rewrite(query(), "", "Apple Inc.") is not None
        assert check_rewrite(query(), "x" * 500, "Apple Inc.") is not None


class TestApplyResults:
    def test_refused_rewrites_keep_their_original_text(self):
        """Both files must hold the same query ids.

        The comparison being run is paraphrased-vs-original on the *same* queries.
        Dropping refused ones would change the shared subset between the two runs
        and quietly make the two sets of numbers incomparable.
        """
        q = query()
        refused = ParaphraseResult("q-1", q.text, None, "dropped the company name", 0.9, None)

        out = apply_results([q], {"q-1": refused}, {"q-1": PASSAGE})

        assert len(out) == 1
        assert out[0].text == q.text
        assert out[0].paraphrase_source is None

    def test_accepted_rewrite_replaces_text_and_recomputes_overlap(self):
        q = query()
        new = "How much did Apple spend developing new products in 2023?"
        kept = ParaphraseResult("q-1", q.text, new, None, 0.9, 0.2)

        out = apply_results([q], {"q-1": kept}, {"q-1": PASSAGE})

        assert out[0].text == new
        assert out[0].paraphrase_source is not None
        # Recomputed from the passage, not copied from the result object.
        assert out[0].lexical_overlap is not None
        assert out[0].lexical_overlap < q.lexical_overlap

    def test_gold_spans_are_never_modified(self):
        """The labels are the benchmark. Paraphrasing rewrites questions only.

        If a rewrite could move gold, the paraphrased run would be measuring a
        different benchmark and the comparison against the original would mean
        nothing.
        """
        q = query()
        kept = ParaphraseResult(
            "q-1", q.text, "How much did Apple spend on R&D in 2023?", None, 0.9, 0.3
        )

        out = apply_results([q], {"q-1": kept}, {"q-1": PASSAGE})

        assert out[0].gold == q.gold
        assert out[0].query_id == q.query_id
        assert out[0].verification == q.verification
