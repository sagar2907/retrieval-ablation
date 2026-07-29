"""Turn chunker-independent gold spans into per-configuration qrels.

Gold labels are character spans (see `corpus.models`). Metrics need judgements
keyed by the ids of whatever units the system actually retrieved. This module
performs that translation, once per chunking configuration.

A chunk is relevant to a gold passage when the chunk covers at least
`min_coverage` of the gold span. Note the direction: coverage of the *gold* by
the *chunk*. Asking the reverse ("how much of the chunk is gold?") would punish
a large chunk that fully contains a short answer, which is a successful
retrieval.

THE TRAP THIS MODULE EXISTS TO MAKE VISIBLE
-------------------------------------------
If a gold span is longer than any chunk a configuration produces, no chunk can
reach `min_coverage`, the query has no relevant unit, and the metrics correctly
return None -- so the query is silently dropped from that configuration's
average.

That is a measurement disaster hiding as a reasonable default. Fixed-size
256-token chunking would drop every query whose answer is a long table, while
structure-aware chunking keeps them. The two configurations would then be
compared on *different query sets*, and fixed-size chunking would look better
than it is, because the queries it fails at hardest are exactly the ones it
stops being scored on.

Two defences, both implemented here:

  1. `reachability` reports, per configuration, which gold passages are
     unreachable. This is published alongside the ablation table, not buried.
  2. `common_judgeable_queries` computes the intersection of queries that every
     configuration can score, and the ablation reports its headline comparison
     on that shared subset. Per-configuration numbers on the full set are also
     reported, clearly labelled, since the difference between the two is itself
     an informative result about chunking.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..corpus.models import GoldPassage, Span

#: Fraction of a gold span a chunk must contain to be judged relevant. 0.5 means
#: "most of the answer is here". Chosen as a documented default rather than a
#: discovered one; the ablation includes a sensitivity check across
#: {0.3, 0.5, 0.7, 1.0} so this constant is a reported parameter, not a hidden
#: assumption.
DEFAULT_MIN_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced by a chunker.

    `span` locates it in the source document's canonical text, which is what
    permits scoring against chunker-independent gold labels.
    """

    chunk_id: str
    doc_id: str
    span: Span
    text: str
    section_path: tuple[str, ...] = ()
    contains_table: bool = False


@dataclass(frozen=True, slots=True)
class ReachabilityReport:
    """Which gold passages a chunking configuration can represent at all."""

    config_name: str
    n_gold: int
    n_reachable: int
    unreachable_passage_ids: tuple[str, ...]

    @property
    def fraction_reachable(self) -> float:
        if self.n_gold == 0:
            return 0.0
        return self.n_reachable / self.n_gold

    def __str__(self) -> str:
        return (
            f"{self.config_name}: {self.n_reachable}/{self.n_gold} gold passages "
            f"reachable ({self.fraction_reachable:.1%})"
        )


def relevant_chunk_ids(
    gold: GoldPassage,
    chunks: Sequence[Chunk],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> list[str]:
    """Ids of chunks covering at least `min_coverage` of `gold`.

    More than one chunk may qualify when a chunker emits overlapping windows.
    All of them are relevant: retrieving any one is a success, and marking only
    a single "best" chunk would understate recall for overlapping chunkers.
    """
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    return [
        chunk.chunk_id
        for chunk in chunks
        if chunk.doc_id == gold.doc_id and chunk.span.coverage_of(gold.span) >= min_coverage
    ]


def build_qrels(
    gold_by_query: Mapping[str, Sequence[GoldPassage]],
    chunks: Sequence[Chunk],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, dict[str, int]]:
    """Per-query chunk-level relevance judgements for one chunking configuration.

    Queries whose gold passages are all unreachable appear with an empty dict
    rather than being omitted, so downstream code can tell "judged, nothing
    reachable" apart from "query not in the eval set". The metrics treat both as
    unmeasurable, but only the first is a chunking failure worth reporting.
    """
    # Group chunks by document so each gold passage compares against a short
    # list instead of the whole corpus. With ~22k chunks and ~220 gold passages
    # the naive version is 5M span comparisons per configuration, which is slow
    # enough to discourage re-running the ablation -- and an ablation you avoid
    # re-running is an ablation that goes stale.
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    qrels: dict[str, dict[str, int]] = {}
    for query_id, golds in gold_by_query.items():
        judgements: dict[str, int] = {}
        for gold in golds:
            for chunk_id in relevant_chunk_ids(gold, by_doc.get(gold.doc_id, ()), min_coverage):
                # A chunk covering two gold passages of different grades takes
                # the higher grade: its best justification is what matters.
                judgements[chunk_id] = max(judgements.get(chunk_id, 0), gold.gain)
        qrels[query_id] = judgements
    return qrels


def reachability(
    config_name: str,
    gold_by_query: Mapping[str, Sequence[GoldPassage]],
    chunks: Sequence[Chunk],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> ReachabilityReport:
    """How much of the eval set this chunking configuration can express."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    all_gold: list[GoldPassage] = [g for golds in gold_by_query.values() for g in golds]
    unreachable = tuple(
        gold.passage_id
        for gold in all_gold
        if not relevant_chunk_ids(gold, by_doc.get(gold.doc_id, ()), min_coverage)
    )
    return ReachabilityReport(
        config_name=config_name,
        n_gold=len(all_gold),
        n_reachable=len(all_gold) - len(unreachable),
        unreachable_passage_ids=unreachable,
    )


def judgeable_queries(qrels: Mapping[str, Mapping[str, int]]) -> set[str]:
    """Queries with at least one relevant chunk under this configuration."""
    return {qid for qid, judgements in qrels.items() if any(g > 0 for g in judgements.values())}


def common_judgeable_queries(
    qrels_by_config: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> set[str]:
    """Queries every configuration can score.

    The headline ablation comparison runs on this subset. Comparing
    configurations on their own individually-judgeable subsets would let a
    configuration improve its average by failing to represent hard queries at
    all -- an artefact that looks exactly like a real improvement.
    """
    if not qrels_by_config:
        return set()
    sets: Iterable[set[str]] = (judgeable_queries(q) for q in qrels_by_config.values())
    return set.intersection(*sets)
