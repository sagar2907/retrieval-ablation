"""Dense vector retrieval by exact search.

EXACT, NOT APPROXIMATE, AND THAT IS THE POINT.

The obvious choice for a vector index is an approximate one (HNSW, IVF, Qdrant's
default). It is the wrong choice for the measurement half of this project, because
an ANN index has its own recall error: it misses some true nearest neighbours. That
error lands in the same number the ablation is trying to read. A configuration
could then appear better or worse than another because of how the graph happened
to be built, and no amount of repetition would separate that from a real retrieval
difference -- the confound is inside the metric.

So the ablation uses brute-force cosine search, which is exact by construction.
The cost is affordable and was checked rather than assumed: 37,000 chunks at 1,024
dimensions in float32 is 152 MB, and one query is a single matrix-vector product.

Approximate search still matters for the served API, where per-query latency is
the thing being optimised rather than measured. That is `service/` territory, and
the honest way to report it is to measure the ANN index's recall *against this
exact index* and publish the gap, rather than pretending it is free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..embed.base import Embedder, l2_normalize
from ..evalset.relevance import Chunk
from .base import Hit, Retriever, stable_rank


class DenseIndex(Retriever):
    """Brute-force cosine similarity over unit-normalised chunk vectors."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        embedder: Embedder,
        name: str | None = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(f"{len(chunk_ids)} chunk ids but {vectors.shape[0]} vectors")
        self.chunk_ids = list(chunk_ids)
        # Re-normalised defensively. Vectors may arrive from a cache written by an
        # earlier version, and a non-unit vector turns the dot product into
        # something that is not cosine similarity -- silently, since the ranking
        # still looks plausible.
        self.vectors = l2_normalize(vectors)
        self.embedder = embedder
        self.name = name or f"dense-{embedder.name}"

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        name: str | None = None,
    ) -> DenseIndex:
        vectors = embedder.encode_passages([c.text for c in chunks])
        return cls([c.chunk_id for c in chunks], vectors, embedder, name)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        if not self.chunk_ids:
            return []
        # Normalised on this side too. The passage side is normalised defensively,
        # with a comment explaining that a non-unit vector turns the dot product
        # into something that is not cosine -- and the query side then asserted the
        # same property instead of enforcing it. Every embedder used here happens to
        # return unit vectors, so the published scores are genuine cosine, but the
        # guarantee was one-sided and the class documents cosine.
        query_vector = l2_normalize(self.embedder.encode_queries([query]))[0]
        scores = self.vectors @ query_vector

        k = min(top_k, scores.shape[0])
        # argpartition is O(n) against a full O(n log n) sort; at 37,000 chunks
        # and 220 queries per configuration the difference is minutes.
        candidates = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(k)

        hits = [
            Hit(chunk_id=self.chunk_ids[int(i)], score=float(scores[i]), source=self.name)
            for i in candidates
        ]
        return stable_rank(hits)[:top_k]

    def search_batch(self, queries: Sequence[str], top_k: int = 50) -> list[list[Hit]]:
        """Search many queries in one matrix multiply.

        Materially faster than looping: one (n_queries x dim) @ (dim x n_chunks)
        product replaces n_queries separate passes over the vector matrix, and the
        embedding model is called once with a full batch instead of once per query.
        """
        if not self.chunk_ids or not queries:
            return [[] for _ in queries]

        query_vectors = l2_normalize(self.embedder.encode_queries(list(queries)))
        scores = query_vectors @ self.vectors.T

        out: list[list[Hit]] = []
        k = min(top_k, scores.shape[1])
        for row in range(scores.shape[0]):
            row_scores = scores[row]
            candidates = (
                np.argpartition(-row_scores, k - 1)[:k]
                if k < row_scores.shape[0]
                else np.arange(row_scores.shape[0])
            )
            hits = [
                Hit(
                    chunk_id=self.chunk_ids[int(i)],
                    score=float(row_scores[i]),
                    source=self.name,
                )
                for i in candidates
            ]
            out.append(stable_rank(hits)[:top_k])
        return out

    def save(self, path) -> None:
        """Persist vectors and ids together.

        One file rather than two, because an id list and a vector matrix that
        drift out of sync produce a working index that returns confidently wrong
        chunk ids -- the hardest possible failure to notice.
        """
        np.savez_compressed(
            path,
            vectors=self.vectors,
            chunk_ids=np.array(self.chunk_ids, dtype=object),
            embedder=self.embedder.name,
        )

    @classmethod
    def load(cls, path, embedder: Embedder, name: str | None = None) -> DenseIndex:
        payload = np.load(path, allow_pickle=True)
        stored = str(payload["embedder"])
        if stored != embedder.name:
            raise ValueError(
                f"index was built with embedder {stored!r} but {embedder.name!r} "
                f"was supplied; query and passage vectors must come from the same model"
            )
        return cls(
            [str(c) for c in payload["chunk_ids"]],
            payload["vectors"],
            embedder,
            name,
        )
