"""BM25 lexical retrieval, implemented directly rather than taken from a library.

Written by hand for two reasons, both about this corpus specifically.

**Tokenisation has to understand money.** A filing says "34,550" and a question
may say "34550"; a filing says "$1.2 billion" and a question says "1.2 billion".
The usual `text.lower().split()` tokeniser treats those as unrelated strings, and
on a corpus whose answers are almost entirely numbers that is not a small loss.
`tokenize` emits a comma-stripped variant alongside each formatted number so both
spellings match, and it keeps decimal points and percent signs, which a
punctuation-stripping tokeniser would destroy.

**The lexical arm is the baseline the whole study leans on.** The headline claim
is that hybrid retrieval plus reranking beats lexical search. A weak or
misconfigured BM25 would manufacture that result, so it is better to control the
implementation than to inherit an opaque one. `rank_bm25` was the alternative
considered and rejected: it is pure-Python per-query loops over every document,
which at 37,000 chunks and 220 queries across 15 configurations is slow enough to
discourage re-running the ablation.

Scoring uses the Robertson/Sparck Jones BM25 with the standard non-negative IDF
floor. Sparse matrices come from scipy, so a query touches only the postings of
its own terms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from ..evalset.relevance import Chunk
from .base import Hit, Retriever, stable_rank

#: Word, number (with thousands separators and decimals), or percent-bearing
#: token. Deliberately keeps digits, commas and periods inside a token.
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d[\d,]*(?:\.\d+)?%?")

_THOUSANDS_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")

#: Standard BM25 parameters. k1 controls term-frequency saturation, b controls
#: length normalisation. Left at the widely used defaults rather than tuned,
#: because tuning the baseline on the eval set it is measured against would leak
#: the test set into the baseline and inflate its score.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, with a comma-stripped duplicate for formatted numbers.

    "Revenue rose to $34,550" yields ["revenue", "rose", "to", "34,550", "34550"].
    The duplicate is what lets a query written either way match, and it costs one
    extra posting per formatted number.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        if "," in token and _THOUSANDS_RE.match(token):
            tokens.append(token.replace(",", ""))
    return tokens


class BM25Index(Retriever):
    """A sparse BM25 index over chunk texts."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        name: str = "bm25",
    ) -> None:
        self.name = name
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = [c.chunk_id for c in chunks]

        vocabulary: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        lengths = np.zeros(len(chunks), dtype=np.float64)

        for doc_index, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            lengths[doc_index] = len(tokens)
            local: dict[int, int] = {}
            for token in tokens:
                term = vocabulary.get(token)
                if term is None:
                    term = len(vocabulary)
                    vocabulary[token] = term
                local[term] = local.get(term, 0) + 1
            for term, count in local.items():
                rows.append(doc_index)
                cols.append(term)
                counts.append(count)

        self.vocabulary = vocabulary
        n_docs = len(chunks)
        n_terms = len(vocabulary)
        self.n_docs = n_docs

        # CSC so a query can slice whole columns (postings for one term) cheaply.
        self._tf = sp.csc_matrix(
            (np.array(counts, dtype=np.float64), (rows, cols)),
            shape=(max(n_docs, 1), max(n_terms, 1)),
        )
        self._lengths = lengths
        self._avg_length = float(lengths.mean()) if n_docs else 0.0

        document_frequency = np.asarray((self._tf > 0).sum(axis=0)).ravel()
        # The +0.5 smoothing and the max(0, ...) floor are the standard BM25 IDF.
        # Without the floor, a term appearing in more than half the corpus gets a
        # negative weight and its presence actively demotes a chunk -- which on a
        # corpus where every document contains "company" and "fiscal" would push
        # genuinely relevant chunks down the ranking.
        with np.errstate(divide="ignore", invalid="ignore"):
            idf = np.log(1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5))
        self._idf = np.maximum(idf, 0.0)

    def __len__(self) -> int:
        return self.n_docs

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every chunk. Zero for chunks sharing no query term."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return scores

        # Length normalisation denominator, precomputable per document.
        norm = self.k1 * (1.0 - self.b + self.b * self._lengths / (self._avg_length or 1.0))

        for token in tokenize(query):
            term = self.vocabulary.get(token)
            if term is None:
                continue
            column = self._tf.getcol(term)
            if column.nnz == 0:
                continue
            doc_indices = column.indices
            frequencies = column.data
            contribution = (
                self._idf[term] * frequencies * (self.k1 + 1.0) / (frequencies + norm[doc_indices])
            )
            scores[doc_indices] += contribution
        return scores

    def search(self, query: str, top_k: int = 50) -> list[Hit]:
        scores = self.scores(query)
        if not scores.size:
            return []

        # Only chunks that matched at least one term are candidates. Returning
        # zero-scoring chunks would pad the ranking with arbitrary documents and
        # make Recall@50 depend on chunk ordering rather than on retrieval.
        nonzero = np.flatnonzero(scores > 0.0)
        if nonzero.size == 0:
            return []

        if nonzero.size > top_k:
            # argpartition is O(n); a full sort of 37,000 chunks per query is
            # wasted work when only the top 50 are ever used.
            keep = nonzero[np.argpartition(-scores[nonzero], top_k)[:top_k]]
        else:
            keep = nonzero

        hits = [
            Hit(chunk_id=self.chunk_ids[int(i)], score=float(scores[i]), source=self.name)
            for i in keep
        ]
        return stable_rank(hits)[:top_k]


def idf_report(index: BM25Index, top_n: int = 15) -> list[tuple[str, float]]:
    """Highest-IDF terms, for sanity-checking that the index is not dominated by noise."""
    inverse = {term: token for token, term in index.vocabulary.items()}
    order = np.argsort(-index._idf)[:top_n]
    return [(inverse[int(t)], float(index._idf[t])) for t in order]
