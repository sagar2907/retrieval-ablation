"""Load GPU-produced vectors and reranker scores, aligning them by chunk id.

WHY ALIGNMENT IS BY ID AND NOT BY POSITION

The obvious way to consume a vector file is to assume row *i* corresponds to
chunk *i* of the locally rebuilt corpus. That worked for 42,214 of 42,215 chunks
and would have silently mis-assigned every vector after the one that did not.

One filing parsed 360 characters longer on the GPU worker than it did locally.
Because chunk ids encode character spans, the final chunk of that document got a
different id, and every subsequent row would have been offset by one document
boundary under positional alignment -- assigning Southern Company vectors to
Walmart chunks, with no error anywhere and entirely plausible-looking metrics
afterwards.

The first explanation recorded here was that SEC had re-posted the filing. That
was wrong, and worth stating plainly: re-fetching both documents showed their raw
bytes byte-identical to the committed manifest, last modified in 2023. The two
machines disagreed because the *parser* disagreed -- libxml2's document size
ceiling and its handling of C1 numeric character references both vary by version.
See `corpus/html_parse.py`, which now pins both so the parse is version-stable.

Aligning by id makes that class of failure impossible regardless, and turns a
silent corruption into a counted, reported discrepancy. It is kept precisely
because the next source of disagreement will not be one anybody predicted.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RESULTS_DIR
from ..embed.base import Embedder
from ..evalset.relevance import Chunk
from .base import Hit, Retriever, stable_rank
from .dense import DenseIndex

log = logging.getLogger(__name__)


class PrecomputedEmbedder(Embedder):
    """Stands in for a real model when vectors already exist for every chunk.

    Passage vectors are looked up rather than computed. Query vectors cannot be:
    a query was never embedded on the GPU worker, so `encode` raises rather than
    returning a zero vector. Returning zeros would make every cosine similarity
    zero, every ranking arbitrary, and the resulting metrics meaningless while
    still looking like numbers.
    """

    #: False, and checked by callers *before* a search is attempted.
    #:
    #: The first version relied on `encode` raising, which is correct but too
    #: late: the exception surfaced mid-run and aborted the whole 15-row grid,
    #: breaking the rule that one unavailable configuration must never take the
    #: others down with it. A capability flag lets the runner record a skip with
    #: a reason and carry on, which is what every other unavailable component
    #: already does.
    can_embed_queries = False

    def __init__(self, name: str, dimension: int, query_vectors: dict | None = None) -> None:
        self.name = name
        self.dimension = dimension
        self._query_vectors = query_vectors or {}
        if self._query_vectors:
            self.can_embed_queries = True

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query and self._query_vectors:
            missing = [t for t in texts if t not in self._query_vectors]
            if missing:
                raise KeyError(
                    f"{len(missing)} queries have no precomputed vector for {self.name}; "
                    f"first: {missing[0][:80]!r}"
                )
            return np.asarray([self._query_vectors[t] for t in texts], dtype=np.float32)
        raise NotImplementedError(
            f"{self.name} vectors are precomputed; queries cannot be embedded locally. "
            f"Query embeddings must come from the same GPU run as the passage vectors."
        )


@dataclass(frozen=True, slots=True)
class LoadedVectors:
    """Vectors aligned to the local corpus, plus what did not line up."""

    embedder_name: str
    dimension: int
    chunk_ids: list[str]
    vectors: np.ndarray
    #: Local chunks with no vector, because the remote corpus differed.
    n_missing_locally: int
    #: Remote vectors whose chunk is absent locally.
    n_unmatched_remotely: int

    @property
    def coverage(self) -> float:
        total = len(self.chunk_ids) + self.n_missing_locally
        return len(self.chunk_ids) / total if total else 0.0


def load_vectors(path: Path, chunks: Sequence[Chunk]) -> LoadedVectors:
    """Load an .npz and reorder it to match `chunks`, by id."""
    payload = np.load(path, allow_pickle=True)
    remote_ids = [str(c) for c in payload["chunk_ids"]]
    matrix = np.asarray(payload["vectors"], dtype=np.float32)
    if matrix.shape[0] != len(remote_ids):
        raise ValueError(f"{path.name}: {matrix.shape[0]} vectors but {len(remote_ids)} ids")

    row_of = {cid: i for i, cid in enumerate(remote_ids)}
    kept_ids: list[str] = []
    rows: list[int] = []
    for chunk in chunks:
        row = row_of.get(chunk.chunk_id)
        if row is not None:
            kept_ids.append(chunk.chunk_id)
            rows.append(row)

    missing = len(chunks) - len(kept_ids)
    unmatched = len(remote_ids) - len(kept_ids)
    if missing or unmatched:
        log.warning(
            "%s: %d local chunks have no vector, %d vectors have no local chunk",
            path.name,
            missing,
            unmatched,
        )

    return LoadedVectors(
        embedder_name=str(payload["embedder"]),
        dimension=int(matrix.shape[1]),
        chunk_ids=kept_ids,
        vectors=matrix[rows],
        n_missing_locally=missing,
        n_unmatched_remotely=unmatched,
    )


def load_query_vectors(
    embedder_name: str,
    query_text_by_id: Mapping[str, str],
    directory: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Query vectors from the same GPU run, keyed by query *text*.

    Keyed by text because that is what the `Retriever` interface receives. None
    when the file is absent, which is a normal state rather than an error: a GPU
    run that produced only passage vectors leaves the dense arm unrunnable, and
    the caller reports that as a skip.

    THE TEXT MUST BE CHECKED, NOT ASSUMED

    Query ids are stable across a rewrite of the query text -- that is exactly
    what makes the paraphrased and original eval sets comparable. It also means an
    id is not evidence that a vector belongs to the text now filed under it. The
    first version of this function looked the id up and stored the row under
    whatever text the caller currently held, so running the paraphrased eval set
    served every dense arm the vector of the *original* wording. The failure was
    invisible in every way that matters: no exception, full coverage, plausible
    metrics -- and nDCG@10 identical to the original run at four decimal places,
    which is the only reason it was caught.

    So the artifact records the text it embedded, and a row whose text no longer
    matches is dropped rather than served. An artifact written before that field
    existed cannot be checked, and is refused outright: a dense arm reported as
    unmeasured costs a re-run, while one silently scored against stale vectors
    costs the credibility of every number beside it.
    """
    root = directory or RESULTS_DIR

    # Every artifact for this model is a candidate, and the recorded text decides
    # which one belongs to the queries being scored. Selecting by filename instead
    # would mean one eval set's vectors overwriting another's, or a naming
    # convention that has to be kept in step by hand across a notebook, a loader
    # and whoever copies files out of a Kaggle session. The name is a hint; the
    # text is the authority.
    candidates = sorted(root.glob(f"queryvectors-{embedder_name}.npz")) + sorted(
        root.glob(f"queryvectors-{embedder_name}-*.npz")
    )
    if not candidates:
        return None

    best: dict[str, np.ndarray] = {}
    best_name = ""
    unverifiable: list[str] = []
    for path in candidates:
        payload = np.load(path, allow_pickle=True)
        if "query_texts" not in payload:
            unverifiable.append(path.name)
            continue

        ids = [str(q) for q in payload["query_ids"]]
        embedded = [str(t) for t in payload["query_texts"]]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)

        matched: dict[str, np.ndarray] = {}
        for query_id, embedded_text, row in zip(ids, embedded, vectors, strict=True):
            current = query_text_by_id.get(query_id)
            if current is not None and current == embedded_text:
                matched[current] = row
        if len(matched) > len(best):
            best, best_name = matched, path.name

    if unverifiable:
        log.warning(
            "ignoring %s: no query_texts recorded, so the vectors cannot be shown to "
            "match the queries being scored. Re-run the GPU notebook.",
            ", ".join(unverifiable),
        )
    if not best:
        log.warning(
            "none of %s was embedded from the query text being scored (%d queries). "
            "The dense arm needs a GPU run against this eval set.",
            ", ".join(p.name for p in candidates),
            len(query_text_by_id),
        )
        return None
    if len(best) < len(query_text_by_id):
        log.warning(
            "%s covers %d of %d queries; the rest were embedded from different text "
            "and were dropped.",
            best_name,
            len(best),
            len(query_text_by_id),
        )
    return best


def dense_index_from_artifact(
    path: Path,
    chunks: Sequence[Chunk],
    query_vectors: dict[str, np.ndarray] | None = None,
) -> DenseIndex:
    loaded = load_vectors(path, chunks)
    return DenseIndex(
        loaded.chunk_ids,
        loaded.vectors,
        PrecomputedEmbedder(loaded.embedder_name, loaded.dimension, query_vectors),
        name=f"dense-{loaded.embedder_name}",
    )


class PrecomputedReranker(Retriever):
    """Applies cross-encoder scores computed elsewhere.

    Only reorders candidates that were actually scored. A candidate the GPU run
    never saw keeps its first-stage position *below* every reranked hit rather
    than being dropped or given a default score -- inventing a score would put an
    unscored passage into the ranking on the strength of a number nobody
    computed.
    """

    def __init__(
        self,
        first_stage: Retriever,
        scores: Mapping[str, Mapping[str, float]],
        query_ids: Mapping[str, str],
        candidate_k: int = 100,
        name: str | None = None,
    ) -> None:
        self.first_stage = first_stage
        self.scores = scores
        #: query text -> query id, because the retriever interface takes text.
        self.query_ids = query_ids
        self.candidate_k = candidate_k
        self.name = name or f"{first_stage.name}+rerank{candidate_k}"

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        depth = max(self.candidate_k, top_k)
        candidates = self.first_stage.search(query, depth)
        if not candidates:
            return []

        query_id = self.query_ids.get(query)
        table = self.scores.get(query_id) if query_id else None
        if not table:
            # No scores for this query: return the first stage untouched rather
            # than pretending a reranking happened.
            return candidates[:top_k]

        shortlist = candidates[: self.candidate_k]
        scored = [
            Hit(chunk_id=h.chunk_id, score=float(table[h.chunk_id]), source=self.name)
            for h in shortlist
            if h.chunk_id in table
        ]
        unscored = [h for h in shortlist if h.chunk_id not in table]
        ordered = stable_rank(scored)

        tail = unscored + list(candidates[self.candidate_k :])
        ordered += [
            Hit(chunk_id=h.chunk_id, score=float("-inf"), source=f"{self.name}:unscored")
            for h in tail
        ]
        return ordered[:top_k]


def load_rerank_scores(path: Path) -> dict[str, dict[str, float]]:
    """Read cross-encoder scores, gzipped or plain."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def available_artifacts(directory: Path | None = None) -> dict[str, Path]:
    """Map artifact kind to path for whatever the GPU run produced."""
    root = directory or RESULTS_DIR
    found: dict[str, Path] = {}
    for path in sorted(root.glob("vectors-*.npz")):
        # vectors-<embedder>-<chunker>.npz
        stem = path.stem.removeprefix("vectors-")
        found[f"vectors:{stem}"] = path
    for pattern in ("rerank-scores-*.json", "rerank-scores-*.json.gz"):
        for path in sorted(root.glob(pattern)):
            found[f"rerank:{path.name}"] = path
    return found
