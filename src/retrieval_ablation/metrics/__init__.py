"""Retrieval and generation metrics, plus the statistics used to report them."""

from .retrieval import (
    MetricSummary,
    aggregate,
    ndcg_at_k,
    per_query_scores,
    recall_at_k,
    reciprocal_rank,
    score_run,
)
from .stats import (
    ConfidenceInterval,
    PairedTest,
    bootstrap_ci,
    holm_bonferroni,
    paired_randomization_test,
)

__all__ = [
    "ConfidenceInterval",
    "MetricSummary",
    "PairedTest",
    "aggregate",
    "bootstrap_ci",
    "holm_bonferroni",
    "ndcg_at_k",
    "paired_randomization_test",
    "per_query_scores",
    "recall_at_k",
    "reciprocal_rank",
    "score_run",
]
