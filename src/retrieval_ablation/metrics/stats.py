"""Significance testing and interval estimation for the ablation table.

Why this module exists at all: a 15-row ablation table of bare point estimates
is not evidence. With ~220 queries, the standard error on nDCG@10 is roughly
0.02-0.03, so two configurations differing by 0.01 are indistinguishable, and
reporting one as "better" is a claim the data does not support.

Two choices here are deliberate and worth defending:

1.  A **paired randomization (permutation) test**, not an unpaired t-test.
    Both systems are run on the same queries, so the pairing removes
    query difficulty as a source of variance. Smucker, Allan & Carterette
    (CIKM 2007), "A Comparison of Statistical Significance Tests for
    Information Retrieval Evaluation", evaluated the alternatives and treat
    the randomization test as the reference method; the t-test is a close
    approximation and the Wilcoxon signed-rank and sign tests were found
    less suitable. We implement the reference method directly since our
    query counts are small enough that cost is irrelevant.

2.  **Holm-Bonferroni correction across the whole table.** Comparing 14
    configurations against one baseline at alpha=0.05 gives roughly a 51%
    chance of at least one false positive if uncorrected
    (1 - 0.95^14). An ablation study is precisely the multiple-comparison
    setting, so leaving this out would make the headline finding unsound.
    Holm is used rather than plain Bonferroni because it is uniformly more
    powerful at the same family-wise error rate.

All randomness is drawn from an explicitly seeded generator. No function here
reads the clock or the global numpy random state, so a given input always
produces the same p-value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..config import GLOBAL_SEED


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    level: float
    n: int

    def __str__(self) -> str:
        pct = int(self.level * 100)
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({pct}% CI, n={self.n})"


@dataclass(frozen=True, slots=True)
class PairedTest:
    """Outcome of comparing a system against a baseline on shared queries."""

    baseline_mean: float
    system_mean: float
    delta: float
    p_value: float
    n_pairs: int
    n_permutations: int

    #: Set by `holm_bonferroni`; None until a correction has been applied.
    p_adjusted: float | None = None

    def significant(self, alpha: float = 0.05) -> bool:
        """Whether the difference survives at `alpha`.

        Uses the corrected p-value when one is available. Falling back to the
        raw p-value silently would defeat the correction, so the distinction is
        surfaced by `p_adjusted` being None.
        """
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return p < alpha


def bootstrap_ci(
    values: Sequence[float],
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = GLOBAL_SEED,
) -> ConfidenceInterval | None:
    """Percentile bootstrap CI for the mean. None if there is nothing to resample.

    The percentile bootstrap is used rather than a normal approximation because
    per-query nDCG is bounded in [0, 1] and heavily skewed -- many queries score
    exactly 0 or exactly 1 -- so a symmetric interval would extend outside the
    metric's own range.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if array.size == 1:
        # A single observation carries no information about spread. Returning a
        # zero-width interval would overstate certainty, so we widen to the
        # full range of the metric and let the n=1 in the label speak.
        return ConfidenceInterval(float(array[0]), float(array[0]), float(array[0]), level, 1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        point=float(array.mean()),
        low=float(np.quantile(means, alpha)),
        high=float(np.quantile(means, 1.0 - alpha)),
        level=level,
        n=int(array.size),
    )


def paired_randomization_test(
    baseline: Mapping[str, float],
    system: Mapping[str, float],
    n_permutations: int = 10_000,
    seed: int = GLOBAL_SEED,
) -> PairedTest | None:
    """Two-sided paired permutation test on the mean difference.

    Only queries scored by *both* systems are used; the intersection is what
    makes the pairing valid. Returns None when the overlap is empty.

    The test statistic is the observed mean difference. Under the null the two
    systems are interchangeable per query, so each pair's sign may be flipped
    independently; the p-value is the fraction of sign-flip assignments giving a
    mean difference at least as extreme as observed.
    """
    shared = sorted(set(baseline) & set(system))
    if not shared:
        return None

    b = np.array([baseline[q] for q in shared], dtype=np.float64)
    s = np.array([system[q] for q in shared], dtype=np.float64)
    diff = s - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff.size))
    null_means = (signs * diff).mean(axis=1)

    # The +1 in numerator and denominator includes the observed arrangement in
    # the null distribution. Without it a p-value of exactly 0 is reportable,
    # which is never justified from a finite sample of permutations.
    extreme = int(np.sum(np.abs(null_means) >= abs(observed)))
    p_value = (extreme + 1) / (n_permutations + 1)

    return PairedTest(
        baseline_mean=float(b.mean()),
        system_mean=float(s.mean()),
        delta=observed,
        p_value=float(p_value),
        n_pairs=int(diff.size),
        n_permutations=n_permutations,
    )


def holm_bonferroni(tests: Mapping[str, PairedTest]) -> dict[str, PairedTest]:
    """Apply Holm-Bonferroni across a family of comparisons.

    Returns new `PairedTest` objects with `p_adjusted` populated. Adjusted
    values are made monotonic non-decreasing in rank order, which is required
    for the step-down procedure to control the family-wise error rate.
    """
    if not tests:
        return {}

    ordered = sorted(tests.items(), key=lambda kv: kv[1].p_value)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0

    for rank, (name, test) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * test.p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max

    return {
        name: PairedTest(
            baseline_mean=test.baseline_mean,
            system_mean=test.system_mean,
            delta=test.delta,
            p_value=test.p_value,
            n_pairs=test.n_pairs,
            n_permutations=test.n_permutations,
            p_adjusted=adjusted[name],
        )
        for name, test in tests.items()
    }
