"""Tests for loading precomputed GPU artifacts back onto the local corpus.

Offline: vector files are written to a tmp_path with numpy, no model involved.

These exist because of a real divergence. The GPU worker rebuilds the corpus
from EDGAR rather than shipping 68 MB of text to a notebook, so two copies of
the corpus exist and they are not guaranteed to agree. The run that produced
`results/vectors-*.npz` held a copy of one document that parsed 360 characters
longer than the committed one, which changed the character offsets encoded in
that document's final chunk id.

The cause was a version-dependent parser, not a changed filing -- see
`corpus/html_parse.py` -- and it is fixed there. These tests stay because the
loader is the last line of defence and must not depend on the parser being
right: any future disagreement has to surface as a counted discrepancy rather
than as vectors quietly attached to the wrong passages.
"""

from __future__ import annotations

import numpy as np

from retrieval_ablation.corpus.models import Span
from retrieval_ablation.evalset.relevance import Chunk
from retrieval_ablation.index.artifacts import load_vectors


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d1", span=Span(0, len(text)), text=text)


def write_vectors(path, chunk_ids: list[str], dim: int = 4) -> None:
    """Write an .npz shaped like the GPU worker's output.

    Row i is filled with the value i, so a misassigned vector is obvious from
    its contents rather than only from a count.
    """
    matrix = np.array([[float(i)] * dim for i in range(len(chunk_ids))], dtype=np.float32)
    np.savez(path, vectors=matrix, chunk_ids=np.array(chunk_ids, dtype=object), embedder="fake")


class TestLoadVectors:
    def test_reorders_by_id_not_by_row_position(self, tmp_path):
        """Vectors follow their chunk id even when the two orderings disagree.

        The obvious implementation -- trusting that row i belongs to chunk i --
        is correct only while both sides enumerate the corpus identically. It
        fails silently the moment they do not: every chunk still gets a vector,
        the shapes still match, nothing raises, and the vectors belong to the
        wrong passages.
        """
        path = tmp_path / "v.npz"
        write_vectors(path, ["c2", "c0", "c1"])
        chunks = [chunk("c0", "a"), chunk("c1", "b"), chunk("c2", "c")]

        loaded = load_vectors(path, chunks)

        assert loaded.chunk_ids == ["c0", "c1", "c2"]
        # c0 was written at row 1, c1 at row 2, c2 at row 0.
        assert [v[0] for v in loaded.vectors] == [1.0, 2.0, 0.0]
        assert loaded.n_missing_locally == 0
        assert loaded.n_unmatched_remotely == 0

    def test_drifted_chunk_id_is_reported_on_both_sides(self, tmp_path):
        """A document that changed length is counted, not absorbed.

        This is the observed failure in miniature. The remote corpus had a
        longer copy of one document, so its final chunk id carried a different
        end offset. Both ids refer to the same passage, but nothing in the id
        says so, and guessing that they match would be worse than dropping it.
        The load must therefore report one local chunk without a vector AND one
        vector without a chunk -- the pair is the signature of drift, as opposed
        to a merely truncated or extended run.
        """
        path = tmp_path / "v.npz"
        write_vectors(path, ["d#000-100", "d#100-460"])
        chunks = [chunk("d#000-100", "a"), chunk("d#100-200", "b")]

        loaded = load_vectors(path, chunks)

        assert loaded.chunk_ids == ["d#000-100"]
        assert loaded.n_missing_locally == 1
        assert loaded.n_unmatched_remotely == 1
        assert loaded.coverage == 0.5

    def test_coverage_is_zero_for_a_completely_foreign_file(self, tmp_path):
        """Loading vectors for the wrong chunker yields no rows, not a crash.

        Every id differs, so there is nothing to align. Returning an empty set
        with coverage 0.0 lets the caller decide the arm is unavailable; raising
        here would abort the whole ablation grid over one bad artifact.
        """
        path = tmp_path / "v.npz"
        write_vectors(path, ["other#0-10", "other#10-20"])
        chunks = [chunk("d#000-100", "a")]

        loaded = load_vectors(path, chunks)

        assert loaded.chunk_ids == []
        assert loaded.vectors.shape[0] == 0
        assert loaded.coverage == 0.0
        assert loaded.n_missing_locally == 1
        assert loaded.n_unmatched_remotely == 2
