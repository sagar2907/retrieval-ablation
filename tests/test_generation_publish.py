"""Tests for not destroying a better run's results.

Offline: `publish` is pure file IO over a dict, which is why the guard lives
there rather than inline in the quota-bound loop it protects.
"""

from __future__ import annotations

import json

from retrieval_ablation.generation.run import publish


def payload(n_scores: int, reason: str | None = None) -> dict:
    return {
        "model": "m",
        "n_queries_sampled": n_scores,
        "n_queries_total": 216,
        "retrieval_top_k": 10,
        "value_accuracy_ci95": {},
        "incomplete_reason": reason,
        "by_arm": {},
        "cost": {},
        "latency": {},
        # compare_arms always returns a dict; it reports measured=False
        # rather than returning None, so the fixture mirrors that.
        "comparison": {"measured": False, "reason": "fixture"},
        "api_usage": {},
        "answers": [],
        "scores": [{"query_id": f"q{i}", "arm": "retrieval"} for i in range(n_scores)],
    }


class TestPublish:
    def test_writes_when_nothing_exists_yet(self, tmp_path):
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"

        assert publish(payload(5), results, table) is True
        assert len(json.loads(results.read_text(encoding="utf-8"))["scores"]) == 5

    def test_refuses_to_replace_a_more_complete_run(self, tmp_path):
        """Regression: a quota-limited re-run destroyed a finished one.

        This arm's reach is decided by whatever is left of a daily free-tier
        allowance, not by the code. A re-run asking for 30 queries exhausted its
        retries after one answer and overwrote a completed 12-query result with a
        1-query one -- real measurements lost, in a committed file, with no error
        and no prompt. Every partial run looks exactly like a complete one, which
        is precisely why nothing caught it.
        """
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(12), results, table)

        assert publish(payload(1, "quota exhausted"), results, table) is False
        assert len(json.loads(results.read_text(encoding="utf-8"))["scores"]) == 12

    def test_allows_an_equal_or_larger_run_through(self, tmp_path):
        """Re-running the same size must still work, or a rerun can never fix a bad file."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(12), results, table)

        assert publish(payload(12), results, table) is True
        assert publish(payload(30), results, table) is True
        assert len(json.loads(results.read_text(encoding="utf-8"))["scores"]) == 30

    def test_an_unreadable_existing_file_does_not_block_a_write(self, tmp_path):
        """Corrupt output is not evidence of measurements worth protecting."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        results.write_text("{ not json", encoding="utf-8")

        assert publish(payload(1), results, table) is True
