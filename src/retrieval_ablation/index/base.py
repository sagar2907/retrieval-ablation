"""The retriever interface and the result type every arm of the ablation returns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..evalset.relevance import Chunk


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk with the score that put it there.

    `score` is only comparable within a single retriever. BM25 scores are
    unbounded sums of term weights, cosine similarities live in [-1, 1], and
    reciprocal-rank-fusion scores are small reciprocals. Comparing them across
    retrievers is meaningless, which is exactly why fusion works on ranks rather
    than on scores.
    """

    chunk_id: str
    score: float
    #: Which component produced this hit, for the score-inspection UI and for
    #: debugging a fused ranking by hand.
    source: str = ""


class Retriever(ABC):
    """Ranks chunks against a query.

    Implementations must be deterministic: the same index and the same query
    produce the same ordering, including ties. Non-deterministic tie-breaking
    would make nDCG@10 jitter between runs and any small measured difference
    unreproducible.
    """

    name: str

    @abstractmethod
    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        """Return up to `top_k` hits, best first."""

    def search_many(self, queries: Mapping[str, str], top_k: int = 50) -> dict[str, list[Hit]]:
        return {qid: self.search(text, top_k) for qid, text in queries.items()}

    def run(self, queries: Mapping[str, str], top_k: int = 50) -> dict[str, list[str]]:
        """Shape results for the metrics: query id -> ranked chunk ids."""
        return {
            qid: [hit.chunk_id for hit in hits]
            for qid, hits in self.search_many(queries, top_k).items()
        }


def stable_rank(hits: Sequence[Hit]) -> list[Hit]:
    """Sort by descending score, breaking ties by chunk id.

    Ties are common and not rare edge cases: BM25 gives identical scores to
    chunks containing the same query terms at the same frequencies, which happens
    constantly in a corpus of four near-identical annual reports per company.
    Without a deterministic tiebreak the ranking would depend on dictionary
    ordering and the measured metrics would drift between runs.
    """
    return sorted(hits, key=lambda h: (-h.score, h.chunk_id))


def chunks_by_id(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}
