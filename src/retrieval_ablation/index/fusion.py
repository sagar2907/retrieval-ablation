"""Reciprocal Rank Fusion: combine rankings without comparing incomparable scores.

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

Cormack, Clarke & Buettcher (SIGIR 2009), "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", which introduced it and used
k = 60.

Why fuse on ranks rather than on scores, which is the obvious alternative and is
wrong here: BM25 produces unbounded sums of term weights, cosine similarity lives
in [-1, 1], and the two have no common unit. Any score-level combination needs a
normalisation step, and every available choice is unprincipled -- min-max
normalisation makes a document's score depend on which *other* documents happened
to be retrieved alongside it, so adding an irrelevant low-scoring result changes
the score of the top hit. z-scoring assumes a distribution shape that neither
retriever has. Ranks discard magnitude, which is precisely the information that
cannot be compared.

The cost of discarding magnitude is real and worth stating: RRF cannot tell a
confident first place from a marginal one. A retriever that is certain about its
top hit contributes exactly the same 1/(k+1) as one that barely preferred it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .base import Hit, Retriever, stable_rank

#: From the original paper. Large enough that differences among the first few
#: ranks are gentle -- 1/61 versus 1/62 rather than 1/1 versus 1/2 -- so a single
#: retriever's top hit cannot dominate the fused ranking on its own.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Hit]],
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    top_k: int = 50,
) -> list[Hit]:
    """Fuse several rankings into one.

    `rankings` maps a retriever name to its ordered hits. A document absent from a
    retriever's list contributes nothing from that retriever -- deliberately not a
    penalty. Treating absence as "ranked last" would require knowing the corpus
    size and would make the fused score depend on how deep each retriever was
    asked to go, so asking for top-100 instead of top-50 would silently change the
    fused ordering of the top 10.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    contributions: dict[str, list[str]] = {}

    for source, hits in rankings.items():
        weight = 1.0 if weights is None else weights.get(source, 1.0)
        if weight == 0.0:
            continue
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            contributions.setdefault(hit.chunk_id, []).append(source)

    fused = [
        Hit(chunk_id=chunk_id, score=score, source="+".join(sorted(contributions[chunk_id])))
        for chunk_id, score in scores.items()
    ]
    return stable_rank(fused)[:top_k]


class HybridRetriever(Retriever):
    """Fuses several retrievers with RRF.

    `candidate_k` is how deep each component is asked to go before fusion, and it
    is separate from the `top_k` returned. Fusing only each component's top 10
    would discard documents that one retriever ranked 30th and the other ranked
    12th -- exactly the complementary cases fusion exists to recover.
    """

    def __init__(
        self,
        components: Sequence[Retriever],
        k: int = DEFAULT_RRF_K,
        weights: Mapping[str, float] | None = None,
        candidate_k: int = 100,
        name: str | None = None,
    ) -> None:
        if not components:
            raise ValueError("HybridRetriever needs at least one component")
        self.components = list(components)
        self.k = k
        self.weights = dict(weights) if weights else None
        self.candidate_k = candidate_k
        self.name = name or "rrf(" + "+".join(c.name for c in self.components) + ")"

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        depth = max(self.candidate_k, top_k)
        rankings = {c.name: c.search(query, depth) for c in self.components}
        return reciprocal_rank_fusion(rankings, k=self.k, weights=self.weights, top_k=top_k)
