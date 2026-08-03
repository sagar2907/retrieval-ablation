"""Tests for model-assisted label checking. No API call."""

from __future__ import annotations

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind, Verification
from retrieval_ablation.evalset.verify import CheckResult, apply_results, parse_verdict


def query(qid: str = "q1", verification: Verification = Verification.GENERATED) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        text="What was research and development expense for 2025?",
        gold=(GoldPassage("p1", "d1", Span(0, 40)),),
        kind=QueryKind.TABLE_LOOKUP,
        verification=verification,
        metadata={"expected_value": "34,550"},
    )


class TestParseVerdict:
    def test_accepts(self):
        assert parse_verdict("VERDICT: OK — passage states the figure") == (
            True,
            "passage states the figure",
        )

    def test_rejects(self):
        accepted, reason = parse_verdict("VERDICT: REJECT — subject is a company name")
        assert accepted is False
        assert "company name" in reason

    def test_tolerates_surrounding_text(self):
        accepted, _ = parse_verdict("Let me think.\n\nVERDICT: OK - fine\n\nDone.")
        assert accepted is True

    def test_case_insensitive(self):
        assert parse_verdict("verdict: reject - nope")[0] is False

    def test_plain_colon_separator(self):
        assert parse_verdict("VERDICT: OK: looks right")[0] is True

    def test_unparseable_returns_none_not_a_rejection(self):
        """A failed read must not inflate the rejection rate.

        The rejection rate is the entire output of this module. Counting the
        judge's formatting mistakes as rejections would corrupt the one number it
        exists to produce.
        """
        accepted, reason = parse_verdict("I'm not sure about this one.")
        assert accepted is None
        assert reason


class TestApplyResults:
    def test_accepted_becomes_model_checked_not_human_verified(self):
        """The distinction this whole module rests on.

        A model checking programmatically generated labels is not an independent
        second opinion: both can be satisfied by a query that is grammatical and
        meaningless. Recording its verdict as HUMAN_VERIFIED would misrepresent
        the only field telling a reader how far to trust the benchmark.
        """
        out = apply_results(
            [query()], {"q1": CheckResult("q1", True, "states the figure", "judge-model")}
        )
        assert out[0].verification is Verification.MODEL_CHECKED
        assert out[0].verification is not Verification.HUMAN_VERIFIED
        assert out[0].checked_by == "judge-model"
        assert out[0].check_reason == "states the figure"

    def test_rejected_is_retained_not_deleted(self):
        """Deleting rejects would hide how large the discarded set was."""
        out = apply_results(
            [query()], {"q1": CheckResult("q1", False, "subject is a place name", "m")}
        )
        assert len(out) == 1
        assert out[0].verification is Verification.REJECTED
        assert "place name" in out[0].check_reason

    def test_unparseable_verdict_leaves_the_label_untouched(self):
        out = apply_results([query()], {"q1": CheckResult("q1", None, "garbled", "m")})
        assert out[0].verification is Verification.GENERATED
        assert out[0].checked_by is None

    def test_a_model_never_overrules_a_human(self):
        """If a person has already judged a label, a model does not get a vote."""
        human = query(verification=Verification.HUMAN_VERIFIED)
        out = apply_results([human], {"q1": CheckResult("q1", False, "disagree", "m")})
        assert out[0].verification is Verification.HUMAN_VERIFIED

    def test_unchecked_queries_pass_through(self):
        out = apply_results([query("q1"), query("q2")], {"q1": CheckResult("q1", True, "ok", "m")})
        assert out[0].verification is Verification.MODEL_CHECKED
        assert out[1].verification is Verification.GENERATED

    def test_preserves_everything_else(self):
        original = query()
        out = apply_results([original], {"q1": CheckResult("q1", True, "ok", "m")})[0]
        assert out.query_id == original.query_id
        assert out.text == original.text
        assert out.gold == original.gold
        assert out.metadata == original.metadata

    def test_empty_results_change_nothing(self):
        assert apply_results([query()], {})[0].verification is Verification.GENERATED


class TestVerificationStates:
    def test_model_checked_is_distinct_from_human_verified(self):
        assert Verification.MODEL_CHECKED != Verification.HUMAN_VERIFIED

    def test_round_trips_through_json(self):
        checked = apply_results(
            [query()], {"q1": CheckResult("q1", True, "states the figure", "judge")}
        )[0]
        restored = EvalQuery.from_json(checked.to_json())
        assert restored == checked
        assert restored.verification is Verification.MODEL_CHECKED
        assert restored.checked_by == "judge"
