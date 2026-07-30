"""The embedding interface, plus a deterministic fake for offline tests.

Real embedding models are an optional dependency (`.[gpu]`) and are never
imported by the test suite. Everything downstream of this interface is therefore
testable without a GPU, a download, or a network call, which is what keeps CI
honest about being offline.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """Turns text into vectors.

    Implementations must be deterministic for a fixed model and input. A model
    run in a non-deterministic mode would make the dense arm of the ablation
    unreproducible, and any measured difference under 0.01 nDCG meaningless.
    """

    name: str
    dimension: int

    @abstractmethod
    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return an array of shape (len(texts), dimension).

        `is_query` exists because several strong retrieval models are asymmetric:
        E5 requires the literal prefixes "query: " and "passage: ", and omitting
        them costs a large amount of accuracy for no visible error. Making the
        distinction part of the interface means a caller cannot forget it.
        """

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, is_query=True)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, is_query=False)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise, leaving zero rows as zero rather than NaN.

    A zero vector is not hypothetical: an embedding model handed whitespace can
    return one, and dividing by its norm produces NaN, which then propagates
    silently through cosine similarity into a ranking of all-NaN scores that sorts
    arbitrarily instead of raising.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder(Embedder):
    """A deterministic, dependency-free stand-in for a real model.

    Hashed bag of words. It has no semantic ability whatsoever and must never be
    used for a reported number -- its purpose is to exercise index construction,
    search, fusion and reranking wiring in tests, where the requirement is
    determinism rather than quality.

    It is a genuine embedder in the ways the pipeline cares about: fixed
    dimension, deterministic, and asymmetric handling of queries is a no-op that
    still goes through the same code path.
    """

    def __init__(self, dimension: int = 64, name: str = "hashing") -> None:
        self.dimension = dimension
        self.name = name

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:  # noqa: ARG002
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                # blake2b rather than the builtin hash(): PYTHONHASHSEED
                # randomises str hashing per process, so a builtin-hash embedder
                # would produce different vectors on every run and quietly break
                # reproducibility of anything built on it.
                digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                out[row, int.from_bytes(digest, "big") % self.dimension] += 1.0
        return l2_normalize(out)
