"""Tests for answer parsing and scoring. No API call, no key, fully deterministic."""

from __future__ import annotations

import pytest

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset.schema import EvalQuery, QueryKind
from retrieval_ablation.generation.answer import (
    NOT_IN_CONTEXT,
    GeneratedAnswer,
    build_prompt,
    parse_citations,
)
from retrieval_ablation.generation.score import (
    aggregate,
    aggregate_by_arm,
    compare_arms,
    contains_expected_value,
    latency_stats,
    normalise_number,
    score_answer,
    token_cost,
)


def make_answer(
    text: str,
    context_ids=("c1", "c2", "c3"),
    cited=(),
    invalid=(),
    refused=False,
    arm="retrieval",
    prompt_tokens=5000,
    output_tokens=100,
    latency=1.5,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query_id="q1",
        question="What was research and development expense for 2025?",
        answer=text,
        context_ids=tuple(context_ids),
        cited_ids=tuple(cited),
        invalid_citations=tuple(invalid),
        refused=refused,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        from_cache=False,
        model="test",
        arm=arm,
    )


def make_query(expected: str = "34,550") -> EvalQuery:
    return EvalQuery(
        query_id="q1",
        text="What was research and development expense for 2025?",
        gold=(GoldPassage("p1", "d1", Span(0, 50)),),
        kind=QueryKind.TABLE_LOOKUP,
        metadata={"expected_value": expected},
    )


class TestBuildPrompt:
    def test_numbers_passages_from_one(self):
        prompt = build_prompt("Q?", ["alpha", "beta"])
        assert "[1] alpha" in prompt
        assert "[2] beta" in prompt

    def test_offers_a_refusal_option(self):
        """Without an explicit escape the model invents an answer.

        Roughly half these queries have their gold passage outside the top 10, so
        if refusal were unavailable the faithfulness metric would measure the
        prompt's pressure to answer rather than retrieval quality.
        """
        assert NOT_IN_CONTEXT in build_prompt("Q?", ["alpha"])

    def test_includes_the_question(self):
        assert "How much revenue?" in build_prompt("How much revenue?", ["x"])


class TestParseCitations:
    def test_extracts_bracket_numbers(self):
        assert parse_citations("The figure was 34,550 [2].", 3) == ([2], [])

    def test_multiple_citations(self):
        assert parse_citations("See [1] and [3].", 3) == ([1, 3], [])

    def test_out_of_range_citation_is_invalid(self):
        """Citing a passage that was never supplied is a hallucinated source.

        Distinct from citing the wrong real passage, and more serious, so the two
        are tracked separately.
        """
        assert parse_citations("As shown in [12].", 3) == ([], [12])

    def test_duplicates_collapse(self):
        # A repeated citation must not inflate any count computed from this list.
        assert parse_citations("[2] and again [2].", 3) == ([2], [])

    def test_no_citations(self):
        assert parse_citations("34,550 million.", 3) == ([], [])

    def test_zero_is_invalid(self):
        assert parse_citations("See [0].", 3) == ([], [0])


class TestNormaliseNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("34,550", "34550"),
            ("$ 34,550", "34550"),
            ("34550", "34550"),
            ("16.1 %", "16.1"),
            ("(9,415)", "-9415"),
            ("2.20", "2.2"),
        ],
    )
    def test_canonicalises(self, raw: str, expected: str):
        assert normalise_number(raw) == expected

    def test_no_number_returns_none(self):
        assert normalise_number("not applicable") is None

    def test_empty_returns_none(self):
        assert normalise_number("") is None


class TestContainsExpectedValue:
    def test_matches_differently_formatted_figure(self):
        """A correct answer written in a different valid style is still correct."""
        assert contains_expected_value("R&D was 34550 million.", "34,550")
        assert contains_expected_value("R&D was $34,550 million.", "34550")

    def test_scans_every_number_not_just_the_first(self):
        """A correct reply usually restates the question first.

        "In 2025, R&D was 34,550" leads with the year; comparing only the first
        number would score it wrong.
        """
        assert contains_expected_value("In 2025, R&D was 34,550 million.", "34,550")

    def test_wrong_figure_fails(self):
        assert not contains_expected_value("R&D was 31,370 million.", "34,550")

    def test_no_number_in_answer_fails(self):
        assert not contains_expected_value("The filing does not say.", "34,550")

    def test_negative_in_parentheses(self):
        assert contains_expected_value("Net cash was (9,415).", "(9,415)")


class TestScoreAnswer:
    def test_correct_value_and_correct_citation(self):
        answer = make_answer("R&D was 34,550 million [1].", cited=("c1",))
        score = score_answer(answer, make_query(), gold_chunk_ids=["c1"])
        assert score.value_correct is True
        assert score.citation_precision == pytest.approx(1.0)
        assert score.citation_recall is True

    def test_citation_precision_penalises_extra_citations(self):
        answer = make_answer("R&D was 34,550 [1][2].", cited=("c1", "c2"))
        score = score_answer(answer, make_query(), gold_chunk_ids=["c1"])
        assert score.citation_precision == pytest.approx(0.5)
        assert score.citation_recall is True

    def test_answering_without_citing_scores_zero_precision(self):
        answer = make_answer("R&D was 34,550 million.")
        score = score_answer(answer, make_query(), gold_chunk_ids=["c1"])
        assert score.citation_precision == pytest.approx(0.0)
        assert score.citation_recall is False

    def test_refusal_is_not_a_wrong_value(self):
        """Scoring abstention as incorrect would reward hallucination.

        A model that says NOT IN CONTEXT when the passages genuinely lack the
        answer behaved correctly; folding that into accuracy would push models
        toward guessing.
        """
        answer = make_answer(NOT_IN_CONTEXT, refused=True)
        score = score_answer(answer, make_query(), gold_chunk_ids=["c1"])
        assert score.value_correct is None
        assert score.refused is True

    def test_hallucinated_citation_is_flagged(self):
        answer = make_answer("R&D was 34,550 [9].", invalid=(9,))
        score = score_answer(answer, make_query(), gold_chunk_ids=["c1"])
        assert score.cited_a_hallucinated_passage is True

    def test_query_without_expected_value_is_unscorable(self):
        answer = make_answer("Something.")
        score = score_answer(answer, make_query(expected=""), gold_chunk_ids=["c1"])
        assert score.value_correct is None

    def test_faithfulness_starts_unmeasured(self):
        score = score_answer(make_answer("x"), make_query(), gold_chunk_ids=["c1"])
        assert score.faithfulness is None

    def test_citation_metrics_are_none_when_not_applicable(self):
        """Regression: an undefined metric was being reported as a real 0.0.

        The long-context arm receives the whole filing as one pseudo-chunk, so its
        citations can never match a gold chunk id however well it cites. Passing
        an empty gold list made precision compute as 0.0, and the published table
        showed long-context at citation precision 0.000 next to retrieval's 0.567
        -- which reads as "long context cites badly" when the metric simply does
        not apply. None is the only honest value.
        """
        answer = make_answer("R&D was 34,550 [1].", cited=("doc#fulldoc",), arm="long_context")
        score = score_answer(answer, make_query(), gold_chunk_ids=None)
        assert score.citation_precision is None
        assert score.citation_recall is None
        # The value metric still applies and must survive.
        assert score.value_correct is True


class TestAggregate:
    def test_reports_both_accuracy_denominators(self):
        """Neither denominator alone is honest.

        Accuracy over answered queries ignores refusals; accuracy over all
        queries punishes correct abstention. Both are reported with their counts.
        """
        scores = [
            score_answer(make_answer("34,550 [1]", cited=("c1",)), make_query(), ["c1"]),
            score_answer(make_answer(NOT_IN_CONTEXT, refused=True), make_query(), ["c1"]),
        ]
        summary = aggregate(scores)
        assert summary["value_accuracy_of_answered"] == pytest.approx(1.0)
        assert summary["value_accuracy_of_all"] == pytest.approx(0.5)
        assert summary["n_value_scored"] == 1
        assert summary["refusal_rate"] == pytest.approx(0.5)

    def test_faithfulness_is_none_when_never_judged(self):
        scores = [score_answer(make_answer("x [1]", cited=("c1",)), make_query(), ["c1"])]
        summary = aggregate(scores)
        assert summary["faithfulness"] is None
        assert summary["n_faithfulness_judged"] == 0

    def test_empty_input_does_not_crash(self):
        summary = aggregate([])
        assert summary["n_answers"] == 0
        assert summary["refusal_rate"] is None

    def test_groups_by_arm(self):
        scores = [
            score_answer(
                make_answer("34,550 [1]", cited=("c1",), arm="retrieval"), make_query(), ["c1"]
            ),
            score_answer(
                make_answer("31,370 [1]", cited=("c1",), arm="long_context"), make_query(), ["c1"]
            ),
        ]
        by_arm = aggregate_by_arm(scores)
        assert set(by_arm) == {"retrieval", "long_context"}
        assert by_arm["retrieval"]["value_accuracy_of_answered"] == pytest.approx(1.0)
        assert by_arm["long_context"]["value_accuracy_of_answered"] == pytest.approx(0.0)


class TestTokenCost:
    def test_uses_reported_tokens(self):
        answers = [make_answer("x", prompt_tokens=1_000_000, output_tokens=0)]
        cost = token_cost(answers, input_price_per_million=1.5, output_price_per_million=7.5)
        assert cost["total_cost_usd"] == pytest.approx(1.5)
        assert cost["cost_per_query_usd"] == pytest.approx(1.5)

    def test_includes_output_tokens(self):
        answers = [make_answer("x", prompt_tokens=0, output_tokens=1_000_000)]
        cost = token_cost(answers, 1.5, 7.5)
        assert cost["total_cost_usd"] == pytest.approx(7.5)

    def test_empty(self):
        assert token_cost([], 1.5, 7.5)["n"] == 0


class TestLatency:
    def test_percentiles(self):
        answers = [make_answer("x", latency=float(i)) for i in range(1, 101)]
        stats = latency_stats(answers)
        assert stats["n_live"] == 100
        assert stats["p95"] == pytest.approx(96.0, abs=2.0)

    def test_all_cached_reports_no_latency_rather_than_zero(self):
        """A cached run's latency describes the disk, not the API.

        Reporting 0.0 would be a fabricated measurement of API performance.
        """
        cached = [
            GeneratedAnswer(
                query_id="q",
                question="q",
                answer="a",
                context_ids=(),
                cited_ids=(),
                invalid_citations=(),
                refused=False,
                prompt_tokens=1,
                output_tokens=1,
                latency_seconds=None,
                from_cache=True,
                model="m",
                arm="retrieval",
            )
        ]
        stats = latency_stats(cached)
        assert stats["p95"] is None
        assert "cache" in stats["note"]


class TestCompareArms:
    def test_reports_the_measured_ratio(self):
        cost = {
            "retrieval": {"cost_per_query_usd": 0.0105, "mean_prompt_tokens": 5000},
            "long_context": {"cost_per_query_usd": 0.303, "mean_prompt_tokens": 200_000},
        }
        latency = {"retrieval": {"p95": 2.0}, "long_context": {"p95": 20.0}}
        out = compare_arms(cost, latency)
        assert out["measured"] is True
        assert out["cost_ratio_long_context_over_retrieval"] == pytest.approx(28.9, abs=0.1)
        assert out["latency_ratio"] == pytest.approx(10.0)

    def test_unmeasured_when_an_arm_is_missing(self):
        """The comparison must not be reported from one arm plus an assumption."""
        out = compare_arms({"retrieval": {"cost_per_query_usd": 0.01}}, {})
        assert out["measured"] is False
        assert "reason" in out


class TestLatencyExcludesCache:
    """Regression: cached timings were averaged in as if freshly measured."""

    @staticmethod
    def answer(latency: float | None, from_cache: bool):
        from retrieval_ablation.generation.answer import GeneratedAnswer

        return GeneratedAnswer(
            query_id="q",
            question="?",
            answer="a",
            context_ids=("c",),
            cited_ids=(),
            invalid_citations=(),
            refused=False,
            prompt_tokens=1,
            output_tokens=1,
            latency_seconds=latency,
            from_cache=from_cache,
            model="m",
            arm="retrieval",
        )

    def test_a_cached_answer_is_not_counted(self):
        """The client writes the measured latency into the cached body.

        So a cache hit carries a real, non-None number from whenever that call was
        first made. Filtering on "has a latency" therefore excluded nothing, while
        the docstring claimed cached answers were dropped -- a documented behaviour
        the code never had.
        """
        from retrieval_ablation.generation.score import latency_stats

        stats = latency_stats([self.answer(99.0, from_cache=True)])

        assert stats["n_live"] == 0
        assert stats["p95"] is None
        assert "not comparable" in stats["note"] or "earlier session" in stats["note"]

    def test_live_answers_are_counted(self):
        from retrieval_ablation.generation.score import latency_stats

        stats = latency_stats([self.answer(2.0, False), self.answer(4.0, False)])

        assert stats["n_live"] == 2
        assert stats["p95"] is not None

    def test_a_mixed_set_uses_only_the_live_answers(self):
        from retrieval_ablation.generation.score import latency_stats

        stats = latency_stats([self.answer(2.0, False), self.answer(500.0, True)])

        assert stats["n_live"] == 1
        assert stats["max"] == 2.0

    def test_an_arm_without_live_calls_yields_no_ratio_and_says_why(self):
        """Never report a comparison the data cannot support.

        A run whose long-context answers were all cached and whose retrieval
        answers were made live under throttling reported long context as 2.5x
        faster. Both numbers were real; the comparison was between a quiet session
        and a congested one.
        """
        from retrieval_ablation.generation.score import compare_arms

        got = compare_arms(
            {
                "retrieval": {"cost_per_query_usd": 0.01},
                "long_context": {"cost_per_query_usd": 0.2},
            },
            {"retrieval": {"p95": 3.0}, "long_context": {"p95": None}},
        )

        assert got["measured"] is True  # cost is still comparable
        assert got["cost_ratio_long_context_over_retrieval"] == 20.0
        assert got["latency_ratio"] is None
        assert got["latency_note"]
