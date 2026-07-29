"""Rank-based retrieval metrics: nDCG@k, Recall@k, MRR.

Definitions follow TREC convention so numbers are comparable to published work:

    DCG@k  = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)
    IDCG@k = DCG@k of the *ideal* ranking built from the full relevance
             judgements, not from the retrieved list
    nDCG@k = DCG@k / IDCG@k

The IDCG detail is load-bearing and is the most common way an implementation
silently inflates its own scores. If IDCG is computed from only the documents
the system happened to return, then a system that returns one relevant document
and nothing else scores nDCG = 1.0. Building IDCG from the complete qrels is
what makes the metric measure *recall of the relevant set*, not just the
internal ordering of whatever came back. There is a regression test pinning
this exact failure.

Unjudged queries are excluded from aggregates and counted, never scored as
zero. A query with no known relevant document cannot distinguish a good system
from a bad one, so averaging a 0.0 into the mean would report a measurement
that was not taken.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Relevance judgements for one query: passage id -> graded gain (>= 0).
# Binary judgements are the special case where every gain is 1.
Qrels = Mapping[str, int]


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """An aggregate metric over a query set.

    `value` is None when nothing could be measured. That is a deliberate,
    tested contract: callers must not be able to confuse "we measured zero"
    with "we could not measure this".
    """

    name: str
    value: float | None
    n_scored: int
    n_skipped: int

    @property
    def measured(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        if self.value is None:
            return f"{self.name}=not measured (0 of {self.n_skipped} queries judged)"
        return f"{self.name}={self.value:.4f} (n={self.n_scored}, skipped={self.n_skipped})"


def _validate(ranking: Sequence[str], qrels: Qrels) -> None:
    if any(gain < 0 for gain in qrels.values()):
        raise ValueError("relevance gains must be non-negative")
    if len(set(ranking)) != len(ranking):
        # A duplicated document would be counted twice by Recall and would let a
        # system game MRR by padding. Callers must de-duplicate before scoring,
        # and we refuse to guess which copy was intended.
        raise ValueError("ranking contains duplicate passage ids")


def dcg_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float:
    """Discounted cumulative gain of `ranking` truncated at k."""
    total = 0.0
    for position, passage_id in enumerate(ranking[:k], start=1):
        gain = qrels.get(passage_id, 0)
        if gain:
            total += (2.0**gain - 1.0) / math.log2(position + 1)
    return total


def ideal_dcg_at_k(qrels: Qrels, k: int) -> float:
    """DCG of the best ranking achievable given the full judgement set."""
    best = sorted((g for g in qrels.values() if g > 0), reverse=True)
    return sum(
        (2.0**gain - 1.0) / math.log2(position + 1)
        for position, gain in enumerate(best[:k], start=1)
    )


def ndcg_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float | None:
    """nDCG@k, or None when the query has no relevant documents."""
    _validate(ranking, qrels)
    idcg = ideal_dcg_at_k(qrels, k)
    if idcg == 0.0:
        return None
    return dcg_at_k(ranking, qrels, k) / idcg


def recall_at_k(ranking: Sequence[str], qrels: Qrels, k: int) -> float | None:
    """Fraction of all relevant documents appearing in the top k."""
    _validate(ranking, qrels)
    relevant = {pid for pid, gain in qrels.items() if gain > 0}
    if not relevant:
        return None
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], qrels: Qrels, k: int | None = None) -> float | None:
    """1 / rank of the first relevant document; 0.0 if none appears within k.

    Unlike nDCG and Recall, a genuine 0.0 here is meaningful (the system
    returned nothing useful), so only the unjudged case returns None.
    """
    _validate(ranking, qrels)
    if not any(gain > 0 for gain in qrels.values()):
        return None
    window = ranking if k is None else ranking[:k]
    for position, passage_id in enumerate(window, start=1):
        if qrels.get(passage_id, 0) > 0:
            return 1.0 / position
    return 0.0


def aggregate(
    name: str,
    per_query: Mapping[str, float | None],
) -> MetricSummary:
    """Mean over queries that could be scored, carrying the skipped count."""
    scored = [v for v in per_query.values() if v is not None]
    skipped = len(per_query) - len(scored)
    if not scored:
        return MetricSummary(name=name, value=None, n_scored=0, n_skipped=skipped)
    return MetricSummary(
        name=name,
        value=sum(scored) / len(scored),
        n_scored=len(scored),
        n_skipped=skipped,
    )


def score_run(
    run: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Qrels],
    ndcg_k: int = 10,
    recall_k: int = 50,
) -> dict[str, MetricSummary]:
    """Score a full run.

    `run` maps query id -> ranked passage ids. Queries present in `run` but
    absent from `qrels_by_query` are treated as unjudged (skipped), which is why
    an eval set with missing labels cannot quietly drag a score toward zero.
    """
    ndcg: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}
    rr: dict[str, float | None] = {}

    for query_id, ranking in run.items():
        qrels = qrels_by_query.get(query_id, {})
        ndcg[query_id] = ndcg_at_k(ranking, qrels, ndcg_k)
        recall[query_id] = recall_at_k(ranking, qrels, recall_k)
        rr[query_id] = reciprocal_rank(ranking, qrels)

    return {
        f"ndcg@{ndcg_k}": aggregate(f"ndcg@{ndcg_k}", ndcg),
        f"recall@{recall_k}": aggregate(f"recall@{recall_k}", recall),
        "mrr": aggregate("mrr", rr),
    }


def per_query_scores(
    run: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Qrels],
    metric: str,
    k: int,
) -> dict[str, float]:
    """Per-query values for one metric, unjudged queries omitted.

    Needed by the significance tests, which must compare two systems on exactly
    the same query subset for the pairing to be valid.
    """
    fn = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": reciprocal_rank}[metric]
    out: dict[str, float] = {}
    for query_id, ranking in run.items():
        value = fn(ranking, qrels_by_query.get(query_id, {}), k)
        if value is not None:
            out[query_id] = value
    return out
