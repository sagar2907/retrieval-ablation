"""Tests for choosing which cross-encoder scores belong to a configuration.

Offline: only filenames are involved, so no artifact is read.
"""

from __future__ import annotations

from retrieval_ablation.ablation.configs import build_grid
from retrieval_ablation.ablation.runner import _artifact_rerank_scores, _candidates_stem


def reranking_configs():
    return [c for c in build_grid() if c.reranker is not None]


class TestCandidatesStem:
    def test_depth_variants_share_the_first_stage_shortlist(self):
        """rerank-candidates-25/50/200 differ only in how deep they rerank.

        export_candidates groups by (chunker, first stage) and exports the deepest
        shortlist once, so all three read the same file. If this mapping drifted
        from the exporter's, a configuration would look for a file nobody wrote.
        """
        by_name = {c.name: c for c in reranking_configs()}
        shared = {
            _candidates_stem(by_name[n])
            for n in (
                "rerank-bm25-100",
                "rerank-candidates-25",
                "rerank-candidates-50",
                "rerank-candidates-200",
            )
        }

        assert shared == {"candidates-rerank-bm25-100"}

    def test_a_hybrid_first_stage_has_its_own_shortlist(self):
        by_name = {c.name: c for c in reranking_configs()}

        assert _candidates_stem(by_name["hybrid-plus-rerank"]) == "candidates-hybrid-plus-rerank"


class TestArtifactRerankScores:
    def test_a_config_only_sees_scores_for_its_own_shortlist(self, tmp_path, monkeypatch):
        """Regression: selection was by query coverage, so a tie picked by filename.

        The previous version returned every score file and kept whichever covered
        the most queries. With one file that worked. Add a hybrid run and both
        cover all 216, the tie falls to whichever sorts first, and every BM25
        reranking arm gets scored against a shortlist it never retrieved --
        roughly 46% of its candidates absent, each pushed below every scored hit.
        Four measured arms degraded, silently.
        """
        from retrieval_ablation.ablation import runner

        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
        for name in (
            "rerank-scores-candidates-hybrid-plus-rerank-paraphrased.json.gz",
            "rerank-scores-candidates-rerank-bm25-100-paraphrased.json.gz",
            "rerank-scores-candidates-rerank-bm25-100.json.gz",
        ):
            (tmp_path / name).write_bytes(b"")

        by_name = {c.name: c for c in reranking_configs()}
        bm25 = [p.name for p in _artifact_rerank_scores(by_name["rerank-candidates-50"])]
        hybrid = [p.name for p in _artifact_rerank_scores(by_name["hybrid-plus-rerank"])]

        assert all("rerank-bm25-100" in n for n in bm25)
        assert not any("hybrid" in n for n in bm25)
        assert all("hybrid" in n for n in hybrid)

    def test_both_wordings_are_offered_so_the_text_check_can_choose(self, tmp_path, monkeypatch):
        """Name selects the shortlist; recorded query text selects the wording."""
        from retrieval_ablation.ablation import runner

        monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
        (tmp_path / "rerank-scores-candidates-rerank-bm25-100.json.gz").write_bytes(b"")
        (tmp_path / "rerank-scores-candidates-rerank-bm25-100-paraphrased.json.gz").write_bytes(b"")

        by_name = {c.name: c for c in reranking_configs()}
        found = [p.name for p in _artifact_rerank_scores(by_name["rerank-bm25-100"])]

        assert len(found) == 2
