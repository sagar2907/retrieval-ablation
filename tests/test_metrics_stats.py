"""Tests for interval estimation and paired significance testing.

Determinism is tested explicitly. Any test here that depended on wall-clock
time or an unseeded generator would be flaky by construction, so every
stochastic function is called with an explicit seed and asserted to reproduce.
"""

from __future__ import annotations

import pytest

from retrieval_ablation.metrics.stats import (
    PairedTest,
    bootstrap_ci,
    holm_bonferroni,
    paired_randomization_test,
)


def _stub(p_value: float) -> PairedTest:
    """A PairedTest carrying only a p-value, for testing the correction alone."""
    return PairedTest(
        baseline_mean=0.0,
        system_mean=0.0,
        delta=0.0,
        p_value=p_value,
        n_pairs=10,
        n_permutations=100,
    )


class TestBootstrapCI:
    def test_interval_brackets_the_point_estimate(self):
        values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
        ci = bootstrap_ci(values, n_resamples=2000)
        assert ci is not None
        assert ci.low <= ci.point <= ci.high
        assert ci.point == pytest.approx(sum(values) / len(values))

    def test_identical_values_give_a_degenerate_interval(self):
        ci = bootstrap_ci([0.5] * 20, n_resamples=500)
        assert ci is not None
        assert ci.low == pytest.approx(0.5)
        assert ci.high == pytest.approx(0.5)

    def test_same_seed_reproduces_exactly(self):
        values = [0.1, 0.9, 0.3, 0.7, 0.5]
        first = bootstrap_ci(values, n_resamples=1000, seed=7)
        second = bootstrap_ci(values, n_resamples=1000, seed=7)
        assert first == second

    def test_different_seed_changes_the_interval(self):
        # Confirms the generator is actually consumed rather than the seed being
        # accepted and ignored.
        values = [0.1, 0.9, 0.3, 0.7, 0.5]
        a = bootstrap_ci(values, n_resamples=1000, seed=1)
        b = bootstrap_ci(values, n_resamples=1000, seed=2)
        assert a is not None and b is not None
        assert (a.low, a.high) != (b.low, b.high)

    def test_wider_interval_for_higher_variance(self):
        tight = bootstrap_ci([0.5, 0.51, 0.49, 0.5, 0.5], n_resamples=2000, seed=3)
        loose = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.5], n_resamples=2000, seed=3)
        assert tight is not None and loose is not None
        assert (loose.high - loose.low) > (tight.high - tight.low)

    def test_empty_input_returns_none(self):
        assert bootstrap_ci([]) is None

    def test_single_observation_does_not_claim_certainty(self):
        ci = bootstrap_ci([0.42])
        assert ci is not None
        assert ci.n == 1


class TestPairedRandomizationTest:
    def test_identical_systems_are_not_significant(self):
        scores = {f"q{i}": i / 20 for i in range(20)}
        result = paired_randomization_test(scores, dict(scores), n_permutations=2000)
        assert result is not None
        assert result.delta == pytest.approx(0.0)
        assert result.p_value > 0.9

    def test_uniformly_better_system_is_significant(self):
        baseline = {f"q{i}": 0.30 for i in range(40)}
        system = {f"q{i}": 0.55 for i in range(40)}
        result = paired_randomization_test(baseline, system, n_permutations=5000)
        assert result is not None
        assert result.delta == pytest.approx(0.25)
        assert result.p_value < 0.01
        assert result.significant()

    def test_small_mean_gain_from_inconsistent_wins_is_not_significant(self):
        """A +0.01 mean built from large wins and near-equal losses is noise.

        This encodes what a *paired* test actually measures: consistency of the
        per-query difference, not the size of the mean. Ten queries improve by
        0.20 and ten degrade by 0.18, netting a mean gain of exactly +0.01. The
        per-query spread (~0.19) dwarfs the mean, so sign-flipping easily
        produces differences this large and the result must not be reported as a
        win.

        Contrast with `test_tiny_but_perfectly_consistent_gain_is_significant`,
        which has the *same* mean and the opposite verdict. A test that judged
        on effect size alone could not distinguish the two.
        """
        baseline = {f"q{i}": 0.50 for i in range(20)}
        system = {f"q{i}": 0.70 if i < 10 else 0.32 for i in range(20)}
        result = paired_randomization_test(baseline, system, n_permutations=5000, seed=13)
        assert result is not None
        assert result.delta == pytest.approx(0.01)
        assert not result.significant(alpha=0.05)

    def test_tiny_but_perfectly_consistent_gain_is_significant(self):
        """+0.01 on every single query is a real effect, and should be called one.

        Written down because it is counter-intuitive and was initially got
        wrong: a difference this small looks like noise next to typical nDCG
        standard errors, but a paired test conditions on query difficulty, and
        20 independent coin flips all landing the same way is p < 0.001.
        """
        baseline = {f"q{i}": 0.40 + (i % 5) * 0.1 for i in range(20)}
        system = {q: v + 0.01 for q, v in baseline.items()}
        result = paired_randomization_test(baseline, system, n_permutations=5000, seed=13)
        assert result is not None
        assert result.delta == pytest.approx(0.01)
        assert result.significant(alpha=0.05)

    def test_p_value_is_never_exactly_zero(self):
        """Regression: a finite permutation sample cannot justify p = 0.

        Without the +1 correction on numerator and denominator, a large,
        perfectly consistent effect yields zero extreme permutations and the
        function would report p = 0.0 -- a certainty the sampling procedure
        cannot support.
        """
        baseline = {f"q{i}": 0.0 for i in range(50)}
        system = {f"q{i}": 1.0 for i in range(50)}
        result = paired_randomization_test(baseline, system, n_permutations=1000)
        assert result is not None
        assert result.p_value > 0.0
        assert result.p_value == pytest.approx(1.0 / 1001.0)

    def test_only_shared_queries_are_paired(self):
        baseline = {"q1": 0.5, "q2": 0.5, "only_baseline": 0.0}
        system = {"q1": 0.6, "q2": 0.6, "only_system": 1.0}
        result = paired_randomization_test(baseline, system, n_permutations=500)
        assert result is not None
        assert result.n_pairs == 2
        assert result.baseline_mean == pytest.approx(0.5)
        assert result.system_mean == pytest.approx(0.6)

    def test_no_overlap_returns_none(self):
        assert paired_randomization_test({"a": 1.0}, {"b": 1.0}) is None

    def test_same_seed_reproduces_exactly(self):
        baseline = {f"q{i}": (i * 7 % 11) / 11 for i in range(25)}
        system = {f"q{i}": (i * 5 % 11) / 11 for i in range(25)}
        first = paired_randomization_test(baseline, system, n_permutations=1000, seed=11)
        second = paired_randomization_test(baseline, system, n_permutations=1000, seed=11)
        assert first is not None and second is not None
        assert first.p_value == second.p_value

    def test_is_two_sided(self):
        # Reversing which system is which must not change the p-value.
        baseline = {f"q{i}": 0.2 for i in range(30)}
        system = {f"q{i}": 0.6 for i in range(30)}
        forward = paired_randomization_test(baseline, system, n_permutations=2000, seed=5)
        reverse = paired_randomization_test(system, baseline, n_permutations=2000, seed=5)
        assert forward is not None and reverse is not None
        assert forward.p_value == pytest.approx(reverse.p_value)
        assert forward.delta == pytest.approx(-reverse.delta)


class TestHolmBonferroni:
    def test_adjusted_p_is_never_below_raw_p(self):
        family = {name: _stub(p) for name, p in {"a": 0.001, "b": 0.02, "c": 0.30}.items()}
        out = holm_bonferroni(family)
        for name, test in out.items():
            assert test.p_adjusted is not None
            assert test.p_adjusted >= family[name].p_value

    def test_step_down_multipliers(self):
        # m=3: smallest p x3, next x2, largest x1
        out = holm_bonferroni({"a": _stub(0.001), "b": _stub(0.02), "c": _stub(0.30)})
        assert out["a"].p_adjusted == pytest.approx(0.003)
        assert out["b"].p_adjusted == pytest.approx(0.04)
        assert out["c"].p_adjusted == pytest.approx(0.30)

    def test_adjusted_values_are_monotonic_in_rank(self):
        """Required for step-down control of the family-wise error rate.

        Raw p-values 0.020/0.021/0.022/0.023 with multipliers 4/3/2/1 give
        0.080/0.063/0.044/0.023 before enforcement -- decreasing, which would
        let a larger raw p be reported as more significant than a smaller one.
        The running maximum fixes the order.
        """
        out = holm_bonferroni(
            {"a": _stub(0.020), "b": _stub(0.021), "c": _stub(0.022), "d": _stub(0.023)}
        )
        ordered = sorted(out.values(), key=lambda t: t.p_value)
        adjusted = [t.p_adjusted for t in ordered]
        assert all(a is not None for a in adjusted)
        assert adjusted == sorted(adjusted)
        assert adjusted[0] == pytest.approx(0.08)
        # All collapse to the first value because the running max dominates.
        assert adjusted[-1] == pytest.approx(0.08)

    def test_correction_can_overturn_an_uncorrected_win(self):
        """The whole point of the correction, pinned as a behaviour.

        With 14 configurations compared against one baseline, a raw p of 0.02
        looks significant at alpha=0.05 but is expected to occur by chance
        somewhere in a family that size.
        """
        family = {f"cfg{i}": _stub(0.02 + i * 0.05) for i in range(14)}
        out = holm_bonferroni(family)
        assert family["cfg0"].significant(alpha=0.05)
        assert not out["cfg0"].significant(alpha=0.05)

    def test_strong_result_survives_correction(self):
        # The correction must not be so blunt that nothing can ever pass.
        family = {"winner": _stub(0.0001)} | {f"cfg{i}": _stub(0.4 + i * 0.02) for i in range(13)}
        out = holm_bonferroni(family)
        assert out["winner"].significant(alpha=0.05)

    def test_capped_at_one(self):
        out = holm_bonferroni({f"c{i}": _stub(0.9) for i in range(5)})
        for test in out.values():
            assert test.p_adjusted == pytest.approx(1.0)

    def test_empty_family_is_empty(self):
        assert holm_bonferroni({}) == {}

    def test_single_comparison_is_unchanged(self):
        out = holm_bonferroni({"only": _stub(0.04)})
        assert out["only"].p_adjusted == pytest.approx(0.04)

    def test_significant_prefers_adjusted_over_raw(self):
        raw = _stub(0.01)
        assert raw.significant(alpha=0.05)
        corrected = holm_bonferroni({f"c{i}": _stub(0.01) for i in range(10)})
        assert not corrected["c0"].significant(alpha=0.05)
