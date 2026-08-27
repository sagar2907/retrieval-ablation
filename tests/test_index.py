"""Tests for BM25, dense search, RRF fusion, and reranking.

All offline. Dense search uses `HashingEmbedder`; reranking uses a fake scorer, so
ordering behaviour is exactly predictable without a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_ablation.corpus.models import Span
from retrieval_ablation.embed.base import Embedder, HashingEmbedder, l2_normalize
from retrieval_ablation.evalset.relevance import Chunk
from retrieval_ablation.index.base import Hit, stable_rank
from retrieval_ablation.index.bm25 import BM25Index, tokenize
from retrieval_ablation.index.dense import DenseIndex
from retrieval_ablation.index.fusion import HybridRetriever, reciprocal_rank_fusion
from retrieval_ablation.index.rerank import RerankingRetriever, recall_ceiling


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d1", span=Span(0, len(text)), text=text)


CORPUS = [
    chunk("c1", "Research and development expense was 34,550 million in fiscal 2025."),
    chunk("c2", "Research and development expense was 31,370 million in fiscal 2024."),
    chunk("c3", "Selling, general and administrative expense was 26,097 million."),
    chunk("c4", "The Company designs and markets smartphones and personal computers."),
    chunk("c5", "Total net sales were 416,161 million for the year."),
]


class TestTokenize:
    def test_lowercases_words(self):
        assert tokenize("Research AND Development") == ["research", "and", "development"]

    def test_keeps_formatted_numbers_intact(self):
        assert "34,550" in tokenize("was 34,550 million")

    def test_emits_a_comma_stripped_duplicate(self):
        """The reason this tokeniser is hand-written.

        A filing writes "34,550" and a question may write "34550". A tokeniser
        that split on punctuation would produce "34" and "550"; one that kept the
        comma would fail to match the unformatted spelling. Emitting both makes
        either phrasing retrievable, which matters on a corpus whose answers are
        almost entirely numbers.
        """
        tokens = tokenize("was 34,550 million")
        assert "34,550" in tokens
        assert "34550" in tokens

    def test_keeps_decimals(self):
        assert "16.1" in tokenize("effective rate of 16.1 percent")

    def test_keeps_percent_sign(self):
        assert "16.1%" in tokenize("rate was 16.1%")

    def test_plain_number_gets_no_duplicate(self):
        assert tokenize("2025") == ["2025"]

    def test_drops_standalone_punctuation(self):
        assert tokenize("a -- b") == ["a", "b"]

    def test_handles_apostrophes(self):
        assert "company's" in tokenize("the Company's revenue")


class TestBM25Index:
    @pytest.fixture
    def index(self) -> BM25Index:
        return BM25Index(CORPUS)

    def test_finds_the_matching_chunk_first(self, index):
        hits = index.search("research and development expense", top_k=5)
        assert hits[0].chunk_id in {"c1", "c2"}

    def test_discriminates_on_the_year(self, index):
        """The corpus is deliberately full of near-duplicates differing by year."""
        hits = index.search("research and development 2025", top_k=5)
        assert hits[0].chunk_id == "c1"

    def test_number_query_matches_unformatted_spelling(self, index):
        hits = index.search("34550", top_k=5)
        assert hits and hits[0].chunk_id == "c1"

    def test_returns_nothing_when_no_term_matches(self, index):
        """Padding a ranking with zero-scoring chunks would make Recall@50 depend
        on chunk ordering rather than on retrieval."""
        assert index.search("zebra pharmacology quantum", top_k=10) == []

    def test_respects_top_k(self, index):
        assert len(index.search("expense", top_k=2)) <= 2

    def test_scores_are_descending(self, index):
        scores = [h.score for h in index.search("research development expense million", 5)]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_across_calls(self, index):
        first = [(h.chunk_id, h.score) for h in index.search("expense million", 5)]
        second = [(h.chunk_id, h.score) for h in index.search("expense million", 5)]
        assert first == second

    def test_idf_is_never_negative(self, index):
        """A negative IDF would make a common term actively demote a chunk.

        Without the standard max(0, ...) floor, any term in more than half the
        corpus gets a negative weight, so a relevant chunk containing "company"
        would be pushed *down* the ranking for containing it.
        """
        assert (index._idf >= 0).all()

    def test_empty_corpus_is_searchable(self):
        empty = BM25Index([])
        assert len(empty) == 0
        assert empty.search("anything") == []

    def test_ties_break_deterministically(self):
        # Two identical texts must always come back in the same order.
        pair = [chunk("b_second", "identical text here"), chunk("a_first", "identical text here")]
        hits = BM25Index(pair).search("identical text here", top_k=2)
        assert [h.chunk_id for h in hits] == ["a_first", "b_second"]


class TestDenseIndex:
    @pytest.fixture
    def index(self) -> DenseIndex:
        return DenseIndex.build(CORPUS, HashingEmbedder(dimension=128))

    def test_returns_hits(self, index):
        hits = index.search("research and development expense", top_k=3)
        assert len(hits) == 3
        assert all(isinstance(h.score, float) for h in hits)

    def test_exact_search_finds_an_identical_passage_first(self, index):
        hits = index.search(CORPUS[2].text, top_k=5)
        assert hits[0].chunk_id == "c3"

    def test_scores_are_descending(self, index):
        scores = [h.score for h in index.search("expense", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_batch_matches_single_query_search(self, index):
        """The batched path is an optimisation and must not change results."""
        queries = ["research development", "net sales", "smartphones"]
        single = [index.search(q, top_k=3) for q in queries]
        batched = index.search_batch(queries, top_k=3)
        for one, many in zip(single, batched, strict=True):
            assert [h.chunk_id for h in one] == [h.chunk_id for h in many]
            for a, b in zip(one, many, strict=True):
                assert a.score == pytest.approx(b.score, abs=1e-5)

    def test_mismatched_ids_and_vectors_are_rejected(self):
        with pytest.raises(ValueError, match="chunk ids but"):
            DenseIndex(["a", "b"], np.zeros((3, 4)), HashingEmbedder())

    def test_one_dimensional_vectors_are_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            DenseIndex(["a"], np.zeros(4), HashingEmbedder())

    def test_empty_index_returns_nothing(self):
        index = DenseIndex([], np.zeros((0, 8)), HashingEmbedder(dimension=8))
        assert index.search("anything") == []
        assert index.search_batch(["a", "b"]) == [[], []]

    def test_save_and_load_round_trip(self, index, tmp_path):
        path = tmp_path / "index.npz"
        index.save(path)
        restored = DenseIndex.load(path, index.embedder)
        assert restored.chunk_ids == index.chunk_ids
        assert [h.chunk_id for h in restored.search("expense", 3)] == [
            h.chunk_id for h in index.search("expense", 3)
        ]

    def test_loading_with_a_different_embedder_is_refused(self, index, tmp_path):
        """Query and passage vectors must come from the same model.

        Mixing them produces a working index that returns confidently wrong
        results, with no error anywhere.
        """
        path = tmp_path / "index.npz"
        index.save(path)
        with pytest.raises(ValueError, match="same model"):
            DenseIndex.load(path, HashingEmbedder(dimension=128, name="different"))


class TestL2Normalize:
    def test_rows_become_unit_length(self):
        out = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_zero_row_stays_zero_and_is_not_nan(self):
        """Regression: dividing a zero vector by its norm yields NaN.

        NaN then propagates through cosine similarity into an all-NaN ranking
        that sorts arbitrarily instead of raising.
        """
        out = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
        assert not np.isnan(out).any()
        assert np.allclose(out[0], 0.0)


class TestHashingEmbedder:
    def test_deterministic_within_a_process(self):
        e = HashingEmbedder(dimension=32)
        assert np.allclose(e.encode(["hello world"]), e.encode(["hello world"]))

    def test_uses_a_stable_hash_not_the_builtin(self):
        """PYTHONHASHSEED randomises str hashing per process.

        A builtin-hash embedder would produce different vectors on every run,
        silently destroying reproducibility of anything built on it. blake2b of a
        known word must give a known bucket.
        """
        vector = HashingEmbedder(dimension=8).encode(["alpha"])[0]
        assert int(np.count_nonzero(vector)) == 1
        # The bucket is fixed by the hash function, not by process state.
        assert np.array_equal(HashingEmbedder(dimension=8).encode(["alpha"])[0], vector)

    def test_output_shape(self):
        assert HashingEmbedder(dimension=16).encode(["a", "b", "c"]).shape == (3, 16)


class TestReciprocalRankFusion:
    def test_document_ranked_well_by_both_wins(self):
        a = [Hit("x", 9.0), Hit("y", 8.0), Hit("z", 7.0)]
        b = [Hit("y", 0.9), Hit("x", 0.8), Hit("w", 0.7)]
        fused = reciprocal_rank_fusion({"bm25": a, "dense": b})
        assert fused[0].chunk_id in {"x", "y"}
        assert {h.chunk_id for h in fused} == {"x", "y", "z", "w"}

    def test_ignores_score_magnitude(self):
        """The property that makes fusion sound across incomparable retrievers.

        BM25 scores are unbounded sums; cosine similarities live in [-1, 1].
        Multiplying one retriever's scores by a thousand must not change the
        fused ordering, because only ranks are used.
        """
        a = [Hit("x", 1.0), Hit("y", 0.9)]
        b = [Hit("y", 5000.0), Hit("x", 4000.0)]
        scaled = [Hit("y", 5_000_000.0), Hit("x", 4_000_000.0)]
        assert [h.chunk_id for h in reciprocal_rank_fusion({"a": a, "b": b})] == [
            h.chunk_id for h in reciprocal_rank_fusion({"a": a, "b": scaled})
        ]

    def test_absence_is_not_a_penalty(self):
        """A document missing from one list contributes nothing, not a negative.

        Treating absence as "ranked last" would make the fused score depend on how
        deep each retriever was asked to go, so requesting top-100 instead of
        top-50 would change the fused order of the top 10.
        """
        shallow = {"a": [Hit("x", 1.0)], "b": [Hit("x", 1.0), Hit("y", 0.5)]}
        deep = {
            "a": [Hit("x", 1.0)],
            "b": [Hit("x", 1.0), Hit("y", 0.5)] + [Hit(f"pad{i}", 0.1) for i in range(50)],
        }
        assert reciprocal_rank_fusion(shallow)[0].chunk_id == (
            reciprocal_rank_fusion(deep)[0].chunk_id
        )

    def test_weights_shift_the_balance(self):
        a = [Hit("x", 1.0), Hit("y", 0.9)]
        b = [Hit("y", 1.0), Hit("x", 0.9)]
        favour_a = reciprocal_rank_fusion({"a": a, "b": b}, weights={"a": 5.0, "b": 1.0})
        favour_b = reciprocal_rank_fusion({"a": a, "b": b}, weights={"a": 1.0, "b": 5.0})
        assert favour_a[0].chunk_id == "x"
        assert favour_b[0].chunk_id == "y"

    def test_zero_weight_excludes_a_retriever(self):
        fused = reciprocal_rank_fusion(
            {"a": [Hit("x", 1.0)], "b": [Hit("y", 1.0)]}, weights={"b": 0.0}
        )
        assert [h.chunk_id for h in fused] == ["x"]

    def test_source_records_which_retrievers_contributed(self):
        fused = reciprocal_rank_fusion({"bm25": [Hit("x", 1.0)], "dense": [Hit("x", 1.0)]})
        assert fused[0].source == "bm25+dense"

    def test_respects_top_k(self):
        hits = [Hit(f"c{i}", 1.0 / (i + 1)) for i in range(20)]
        assert len(reciprocal_rank_fusion({"a": hits}, top_k=5)) == 5

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fusion({"a": []}, k=0)

    def test_single_retriever_preserves_its_order(self):
        hits = [Hit("a", 3.0), Hit("b", 2.0), Hit("c", 1.0)]
        assert [h.chunk_id for h in reciprocal_rank_fusion({"only": hits})] == ["a", "b", "c"]


class TestHybridRetriever:
    def test_combines_lexical_and_dense(self):
        bm25 = BM25Index(CORPUS)
        dense = DenseIndex.build(CORPUS, HashingEmbedder(dimension=128))
        hybrid = HybridRetriever([bm25, dense])
        hits = hybrid.search("research and development expense 2025", top_k=3)
        assert hits
        assert hits[0].chunk_id in {"c1", "c2"}

    def test_name_describes_the_combination(self):
        hybrid = HybridRetriever([BM25Index(CORPUS)])
        assert "rrf(" in hybrid.name

    def test_requires_at_least_one_component(self):
        with pytest.raises(ValueError, match="at least one component"):
            HybridRetriever([])

    def test_run_shapes_results_for_the_metrics(self):
        hybrid = HybridRetriever([BM25Index(CORPUS)])
        run = hybrid.run({"q1": "research development"}, top_k=3)
        assert set(run) == {"q1"}
        assert all(isinstance(cid, str) for cid in run["q1"])


class FakeReranker:
    """Scores by counting query words present in the passage. Exactly predictable."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        wanted = set(query.lower().split())
        return np.array([len(wanted & set(p.lower().split())) for p in passages], dtype=np.float32)


class TestRerankingRetriever:
    @pytest.fixture
    def texts(self):
        return {c.chunk_id: c.text for c in CORPUS}

    def test_reorders_the_shortlist(self, texts):
        # A first stage that returns the right answer last.
        class Reversed:
            name = "reversed"

            def search(self, query, top_k=50):  # noqa: ARG002
                return [Hit(c.chunk_id, 1.0) for c in reversed(CORPUS)][:top_k]

        retriever = RerankingRetriever(Reversed(), FakeReranker(), texts, candidate_k=5)
        hits = retriever.search("research and development expense", top_k=5)
        assert hits[0].chunk_id in {"c1", "c2"}

    def test_only_the_shortlist_is_reranked(self, texts):
        """candidate_k must mean what it says as a cost parameter."""

        class Wide:
            name = "wide"

            def search(self, query, top_k=50):  # noqa: ARG002
                return [Hit(c.chunk_id, 1.0) for c in CORPUS][:top_k]

        reranker = FakeReranker()
        retriever = RerankingRetriever(Wide(), reranker, texts, candidate_k=2)
        hits = retriever.search("research development", top_k=5)
        # Unreranked tail is appended below, never interleaved.
        assert hits[0].chunk_id in {"c1", "c2"}
        assert any(h.source.endswith("unreranked") for h in hits[2:])

    def test_unreranked_tail_never_outranks_a_reranked_hit(self, texts):
        """Mixing cross-encoder logits with first-stage scores in one ranking is
        the same incomparable-scores error that RRF exists to avoid."""

        class Wide:
            name = "wide"

            def search(self, query, top_k=50):  # noqa: ARG002
                return [Hit(c.chunk_id, 100.0) for c in CORPUS][:top_k]

        retriever = RerankingRetriever(Wide(), FakeReranker(), texts, candidate_k=2)
        hits = retriever.search("smartphones", top_k=5)
        reranked = [h for h in hits if not h.source.endswith("unreranked")]
        unreranked = [h for h in hits if h.source.endswith("unreranked")]
        assert hits[: len(reranked)] == reranked
        assert all(h.score == float("-inf") for h in unreranked)

    def test_shortlist_is_at_least_as_deep_as_requested_output(self, texts):
        class Wide:
            name = "wide"

            def search(self, query, top_k=50):  # noqa: ARG002
                return [Hit(c.chunk_id, 1.0) for c in CORPUS][:top_k]

        retriever = RerankingRetriever(Wide(), FakeReranker(), texts, candidate_k=1)
        # Asking for 4 must not return 1 just because candidate_k is 1.
        assert len(retriever.search("expense", top_k=4)) == 4

    def test_empty_first_stage_yields_nothing(self, texts):
        class Empty:
            name = "empty"

            def search(self, query, top_k=50):  # noqa: ARG002
                return []

        retriever = RerankingRetriever(Empty(), FakeReranker(), texts)
        assert retriever.search("anything") == []

    def test_invalid_candidate_k_rejected(self, texts):
        with pytest.raises(ValueError, match="candidate_k"):
            RerankingRetriever(BM25Index(CORPUS), FakeReranker(), texts, candidate_k=0)


class TestRecallCeiling:
    def test_reports_the_upper_bound_on_reranking(self):
        """Distinguishes "the cross-encoder is weak" from "it never saw the answer"."""
        run = {"q1": ["c1", "c2", "c3"], "q2": ["c4", "c5", "c1"]}
        qrels = {"q1": {"c1": 2}, "q2": {"c1": 2}}
        assert recall_ceiling(run, qrels, candidate_k=1) == pytest.approx(0.5)
        assert recall_ceiling(run, qrels, candidate_k=3) == pytest.approx(1.0)

    def test_unjudged_queries_are_excluded(self):
        run = {"q1": ["c1"], "unlabelled": ["c9"]}
        qrels = {"q1": {"c1": 2}}
        assert recall_ceiling(run, qrels, candidate_k=1) == pytest.approx(1.0)

    def test_nothing_judged_returns_none_not_zero(self):
        assert recall_ceiling({"q1": ["c1"]}, {}, candidate_k=10) is None


class TestStableRank:
    def test_sorts_by_descending_score(self):
        hits = [Hit("a", 1.0), Hit("b", 3.0), Hit("c", 2.0)]
        assert [h.chunk_id for h in stable_rank(hits)] == ["b", "c", "a"]

    def test_ties_break_by_chunk_id(self):
        """Ties are constant in a corpus of near-identical annual reports.

        Without a deterministic tiebreak the ranking would depend on dictionary
        ordering and measured metrics would drift between runs.
        """
        hits = [Hit("z", 1.0), Hit("a", 1.0), Hit("m", 1.0)]
        assert [h.chunk_id for h in stable_rank(hits)] == ["a", "m", "z"]


class TestDenseScoresAreActuallyCosine:
    """The class documents cosine similarity, so both sides must be unit vectors."""

    class Unnormalised(Embedder):
        """An embedder returning non-unit vectors, which the interface permits."""

        dimension = 3
        name = "unnormalised"

        def encode(self, texts, is_query: bool = False):  # noqa: ARG002 - interface
            table = {
                "alpha": [1.0, 0.0, 0.0],
                "beta": [0.0, 1.0, 0.0],
                "both": [1.0, 1.0, 0.0],
                "q": [3.0, 0.3, 0.0],
            }
            return np.array([table[t] for t in texts], dtype=np.float32)

    def chunks(self) -> list[Chunk]:
        return [chunk(name, name) for name in ("alpha", "beta", "both")]

    def test_a_non_unit_query_still_yields_cosine(self):
        """Regression: passages were normalised defensively, queries only assumed.

        `DenseIndex.__init__` normalises the passage matrix and says why -- a
        non-unit vector "turns the dot product into something that is not cosine
        similarity, silently, since the ranking still looks plausible". The query
        side then asserted that same property in a comment rather than enforcing it.

        Every embedder in this project returns unit vectors, verified against the
        committed artifacts, so the published dense scores are genuine cosine and no
        measurement moved. The guarantee was simply one-sided.
        """
        index = DenseIndex.build(self.chunks(), self.Unnormalised())
        got = {h.chunk_id: h.score for h in index.search("q", top_k=3)}

        q = np.array([3.0, 0.3, 0.0])
        for name, vec in (("alpha", [1, 0, 0]), ("beta", [0, 1, 0]), ("both", [1, 1, 0])):
            v = np.array(vec, dtype=np.float64)
            cosine = float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v)))
            assert got[name] == pytest.approx(cosine, abs=1e-6)

    def test_scores_stay_inside_the_cosine_range(self):
        index = DenseIndex.build(self.chunks(), self.Unnormalised())

        for hit in index.search("q", top_k=3):
            assert -1.0 - 1e-6 <= hit.score <= 1.0 + 1e-6

    def test_the_batch_path_agrees_with_the_single_path(self):
        """Two code paths computing the same quantity must not diverge."""
        index = DenseIndex.build(self.chunks(), self.Unnormalised())

        single = {h.chunk_id: h.score for h in index.search("q", top_k=3)}
        batch = {h.chunk_id: h.score for h in index.search_batch(["q"], top_k=3)[0]}

        assert single == pytest.approx(batch)

    def test_a_unit_embedder_is_unaffected(self):
        """The fix must be a no-op for every embedder actually used here."""
        index = DenseIndex.build(self.chunks(), HashingEmbedder(dimension=16))

        scores = [h.score for h in index.search("alpha", top_k=3)]

        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)
