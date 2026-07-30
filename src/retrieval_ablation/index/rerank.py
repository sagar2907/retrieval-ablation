"""Cross-encoder reranking, and the candidate-set size that governs its value.

A bi-encoder embeds query and passage independently, so it can never model how
they interact -- it compares two summaries written without knowledge of each other.
A cross-encoder reads the pair jointly and scores it directly, which is strictly
more expressive and correspondingly more expensive: cost is one forward pass per
candidate rather than one lookup, so it can only ever run over a shortlist.

That makes candidate-set size an axis of the ablation rather than a tuning detail,
and the trade-off runs in both directions:

- Too small, and reranking cannot help. A cross-encoder can only reorder what it
  is given; if the gold passage is not in the first-stage top-k, no amount of
  reranking recovers it. The ceiling on reranked nDCG is the first stage's
  Recall@k.
- Too large, and it costs proportionally more with diminishing returns, since deep
  first-stage results are mostly irrelevant.

Reranking is widely described as the highest-value component in production
retrieval. This module exists so that claim can be measured on this corpus and at
several candidate sizes, rather than assumed.

The reranker is loaded lazily and released explicitly. On the 6 GB development
card it cannot be resident at the same time as an embedding model.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from ..config import MODEL_DIR
from .base import Hit, Retriever, stable_rank

log = logging.getLogger(__name__)

#: bge-reranker-v2-m3: the cross-encoder counterpart to the BGE-M3 embedder, so
#: the comparison is between architectures rather than between training corpora.
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.name = model_name.rsplit("/", maxsplit=1)[-1]
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._fp16 = fp16
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from sentence_transformers import CrossEncoder

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading reranker %s on %s", self.model_name, device)
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=device,
            cache_folder=str(MODEL_DIR),
        )
        if self._fp16 and device == "cuda":
            self._model.model = self._model.model.half()

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Relevance score per passage. Higher is more relevant."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).ravel()

    def release(self) -> None:
        if self._model is None:
            return
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class RerankingRetriever(Retriever):
    """A first-stage retriever followed by cross-encoder reordering.

    `candidate_k` is the shortlist depth and is the parameter the ablation varies.
    """

    def __init__(
        self,
        first_stage: Retriever,
        reranker: CrossEncoderReranker,
        chunk_texts: Mapping[str, str],
        candidate_k: int = 100,
        name: str | None = None,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.first_stage = first_stage
        self.reranker = reranker
        self.chunk_texts = chunk_texts
        self.candidate_k = candidate_k
        self.name = name or f"{first_stage.name}+rerank{candidate_k}"

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        # The shortlist is always at least as deep as the requested output;
        # otherwise asking for top_k=50 with candidate_k=10 would silently return
        # 10 results and the missing 40 would look like a retrieval failure.
        depth = max(self.candidate_k, top_k)
        candidates = self.first_stage.search(query, depth)
        if not candidates:
            return []

        # Only rerank the configured shortlist even if the first stage returned
        # more, so candidate_k means what it says as a cost parameter.
        shortlist = candidates[: self.candidate_k]
        texts = [self.chunk_texts.get(hit.chunk_id, "") for hit in shortlist]
        scores = self.reranker.score(query, texts)

        reranked = [
            Hit(chunk_id=hit.chunk_id, score=float(score), source=self.name)
            for hit, score in zip(shortlist, scores, strict=True)
        ]
        ordered = stable_rank(reranked)

        # Anything beyond the shortlist keeps its first-stage order and is appended
        # below every reranked hit. Its scores are on a different scale and must
        # not be interleaved: doing so would mix cross-encoder logits with BM25
        # sums in one ranking, which is exactly the incomparable-scores error that
        # RRF exists to avoid.
        if len(candidates) > self.candidate_k and top_k > self.candidate_k:
            tail = candidates[self.candidate_k : top_k]
            ordered = ordered + [
                Hit(chunk_id=h.chunk_id, score=float("-inf"), source=f"{self.name}:unreranked")
                for h in tail
            ]
        return ordered[:top_k]


def recall_ceiling(
    first_stage_run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    candidate_k: int,
) -> float | None:
    """Fraction of queries whose gold passage appears in the shortlist.

    This is the hard upper bound on what reranking at `candidate_k` can achieve,
    and reporting it beside the reranked scores is what distinguishes "the
    cross-encoder is weak" from "the cross-encoder never saw the answer". Without
    it, a disappointing reranking result is uninterpretable.
    """
    scored = 0
    found = 0
    for query_id, ranking in first_stage_run.items():
        relevant = {c for c, gain in qrels.get(query_id, {}).items() if gain > 0}
        if not relevant:
            continue
        scored += 1
        if relevant & set(ranking[:candidate_k]):
            found += 1
    return found / scored if scored else None
