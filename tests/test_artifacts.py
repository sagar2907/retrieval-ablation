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
from retrieval_ablation.index.artifacts import load_query_vectors, load_vectors


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


def write_query_vectors(path, ids: list[str], texts: list[str] | None, dim: int = 4) -> None:
    """Write an .npz shaped like the GPU worker's query output.

    `texts=None` reproduces the older artifacts, which recorded ids but not the
    text each vector was built from.
    """
    matrix = np.array([[float(i)] * dim for i in range(len(ids))], dtype=np.float32)
    payload = {
        "vectors": matrix,
        "query_ids": np.array(ids, dtype=object),
        "embedder": "fake",
    }
    if texts is not None:
        payload["query_texts"] = np.array(texts, dtype=object)
    np.savez(path, **payload)


class TestLoadQueryVectors:
    """Query ids survive a rewrite of the query text. Vectors must not."""

    def test_vectors_load_when_the_text_still_matches(self, tmp_path):
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1", "q2"], ["ask a", "ask b"])

        got = load_query_vectors("fake", {"q1": "ask a", "q2": "ask b"}, directory=tmp_path)

        assert got is not None
        assert set(got) == {"ask a", "ask b"}

    def test_a_rewritten_query_does_not_get_the_old_wordings_vector(self, tmp_path):
        """Regression: the paraphrased eval set was scored with original vectors.

        Paraphrasing rewrites `text` and deliberately keeps `query_id`, which is
        what makes the two eval sets comparable. The loader used to look up the id
        and file the row under whatever text the caller currently held, so every
        dense arm received the vector of the wording it was no longer being asked
        about. Nothing raised, coverage was total, and the metrics looked ordinary
        -- the only visible symptom was nDCG@10 matching the original run to four
        decimal places.
        """
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1"], ["what was revenue"])

        got = load_query_vectors("fake", {"q1": "how much money came in"}, directory=tmp_path)

        assert got is None

    def test_partially_rewritten_sets_keep_only_the_untouched_queries(self, tmp_path):
        """Dropping the stale ones beats serving them or failing the whole arm."""
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1", "q2"], ["ask a", "ask b"])

        got = load_query_vectors("fake", {"q1": "ask a", "q2": "rewritten"}, directory=tmp_path)

        assert got is not None
        assert set(got) == {"ask a"}

    def test_an_artifact_without_query_texts_is_refused(self, tmp_path):
        """Unprovable is treated as unusable.

        An artifact predating the field is not necessarily wrong, but it cannot
        show which text it embedded. A dense arm reported unmeasured costs a
        re-run; one silently scored against stale vectors costs the credibility of
        every number printed beside it.
        """
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1"], None)

        assert load_query_vectors("fake", {"q1": "ask a"}, directory=tmp_path) is None

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert load_query_vectors("absent", {"q1": "ask a"}, directory=tmp_path) is None

    def test_the_matching_artifact_wins_regardless_of_filename(self, tmp_path):
        """Two eval sets can have vectors side by side; content picks the right one.

        Selecting by filename would need a naming convention kept in step by hand
        across the notebook, the loader, and whoever copies files out of a Kaggle
        session. The recorded text already decides correctness, so it decides
        selection too.
        """
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1"], ["original wording"])
        write_query_vectors(
            tmp_path / "queryvectors-fake-paraphrased.npz", ["q1"], ["rewritten wording"]
        )

        original = load_query_vectors("fake", {"q1": "original wording"}, directory=tmp_path)
        rewritten = load_query_vectors("fake", {"q1": "rewritten wording"}, directory=tmp_path)

        assert original is not None and set(original) == {"original wording"}
        assert rewritten is not None and set(rewritten) == {"rewritten wording"}

    def test_an_unverifiable_artifact_does_not_block_a_good_one(self, tmp_path):
        """A legacy file sitting beside a valid one must not veto it."""
        write_query_vectors(tmp_path / "queryvectors-fake.npz", ["q1"], None)
        write_query_vectors(tmp_path / "queryvectors-fake-new.npz", ["q1"], ["ask a"])

        got = load_query_vectors("fake", {"q1": "ask a"}, directory=tmp_path)

        assert got is not None
        assert set(got) == {"ask a"}


class TestQueryVectorCoverage:
    """One answer to "does this artifact cover these queries", shared by two callers.

    The same mistake was made twice -- once in the ablation runner, once in the
    candidate export -- because each place asked that question of the same artifact
    and answered it separately. Both compared the mapping's size against the number
    of queries, and the mapping is keyed by query *text*.
    """

    @staticmethod
    def queries(texts):
        from types import SimpleNamespace

        return [SimpleNamespace(text=t, query_id=f"q{i}") for i, t in enumerate(texts)]

    def test_duplicate_query_texts_do_not_look_like_missing_vectors(self):
        """Regression: a complete artifact was reported as "582 of 586".

        Two queries wording the same question collapse to one entry, because text is
        what a Retriever is handed. Comparing against the query count therefore
        showed a shortfall for an artifact covering everything, and the arm was
        skipped -- a measurable configuration reported as unmeasured, twice.
        """
        from retrieval_ablation.index.artifacts import (
            covers_every_query,
            query_vector_coverage,
        )

        queries = self.queries(["what was revenue?", "what was revenue?", "and profit?"])
        vectors = {"what was revenue?": object(), "and profit?": object()}

        covered, wanted = query_vector_coverage(vectors, queries)

        assert (covered, wanted) == (2, 2)
        assert covers_every_query(vectors, queries) is True

    def test_a_genuinely_partial_artifact_is_still_refused(self):
        """The check must not become permissive in fixing the false alarm.

        Growing the eval set reaches this: the artifact covers the queries that
        existed when the GPU last ran, and every query added since has no vector.
        """
        from retrieval_ablation.index.artifacts import (
            covers_every_query,
            query_vector_coverage,
        )

        queries = self.queries(["a", "b", "c"])
        vectors = {"a": object(), "b": object()}

        assert query_vector_coverage(vectors, queries) == (2, 3)
        assert covers_every_query(vectors, queries) is False

    def test_a_missing_artifact_covers_nothing(self):
        from retrieval_ablation.index.artifacts import (
            covers_every_query,
            query_vector_coverage,
        )

        queries = self.queries(["a", "b"])

        assert query_vector_coverage(None, queries) == (0, 2)
        assert covers_every_query(None, queries) is False

    def test_an_empty_query_set_is_covered_by_anything(self):
        """Nothing to serve is not a shortfall, and must not read as one."""
        from retrieval_ablation.index.artifacts import covers_every_query

        assert covers_every_query({}, []) is True

    def test_extra_vectors_are_not_a_problem(self):
        """An artifact built against a larger eval set still serves a smaller one."""
        from retrieval_ablation.index.artifacts import covers_every_query

        queries = self.queries(["a"])

        assert covers_every_query({"a": object(), "b": object()}, queries) is True
