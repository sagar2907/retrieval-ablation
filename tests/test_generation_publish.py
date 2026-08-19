"""Tests for not destroying a better run's results.

Offline: `publish` is pure file IO over a dict, which is why the guard lives
there rather than inline in the quota-bound loop it protects.
"""

from __future__ import annotations

import json

from retrieval_ablation.generation.run import publish


def payload(
    n_scores: int,
    reason: str | None = None,
    n_verdicts: int = 0,
    arm: str = "retrieval",
    extra: list[dict] | None = None,
) -> dict:
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
        "scores": [
            {
                "query_id": f"q{i}",
                "arm": arm,
                "faithfulness": 1.0 if i < n_verdicts else None,
            }
            for i in range(n_scores)
        ]
        + list(extra or []),
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

    def test_refuses_to_drop_faithfulness_verdicts_at_equal_score_count(self, tmp_path):
        """Regression: the guard counted scores and ignored the verdicts inside them.

        Answers are cached, so repeating the same sample yields an identical score
        count. If the judge is rate-limited that day every verdict is null, the
        totals match, and a count-only comparison lets the write through --
        destroying the most expensive data in the file, which is precisely what
        this guard exists to prevent.
        """
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(12, n_verdicts=5), results, table)

        assert publish(payload(12, "judge rate-limited", n_verdicts=0), results, table) is False
        kept = json.loads(results.read_text(encoding="utf-8"))["scores"]
        assert sum(1 for s in kept if s["faithfulness"] is not None) == 5

    def test_more_verdicts_at_equal_score_count_still_writes(self, tmp_path):
        """Otherwise a successful judge pass could never be recorded."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(12, n_verdicts=2), results, table)

        assert publish(payload(12, n_verdicts=9), results, table) is True
        kept = json.loads(results.read_text(encoding="utf-8"))["scores"]
        assert sum(1 for s in kept if s["faithfulness"] is not None) == 9


class TestPerArmProtection:
    """Totals hide an arm disappearing entirely."""

    @staticmethod
    def long_context(n: int) -> list[dict]:
        return [
            {"query_id": f"lc{i}", "arm": "long_context", "faithfulness": None} for i in range(n)
        ]

    def test_refuses_a_run_that_drops_an_entire_arm(self, tmp_path):
        """Regression: a bigger run could delete a whole arm's measurements.

        The guard compared totals. A file holding 11 retrieval and 10
        long-context scores has 21; a --skip-long-context run of 40 retrieval
        queries has 40, so the totals said "more" and the write would have gone
        through, taking every long-context score with it -- including the
        retrieval-versus-long-context cost ratio quoted in the README. Nothing
        would have errored, and the file would have looked like a better run.
        """
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(11, extra=self.long_context(10)), results, table)

        assert publish(payload(40, "skipped the long-context arm"), results, table) is False
        kept = json.loads(results.read_text(encoding="utf-8"))["scores"]
        assert sum(1 for s in kept if s["arm"] == "long_context") == 10

    def test_growing_one_arm_while_keeping_the_other_writes(self, tmp_path):
        """Otherwise the guard would forbid the very improvement it wants."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(11, extra=self.long_context(10)), results, table)

        assert publish(payload(40, extra=self.long_context(10)), results, table) is True
        kept = json.loads(results.read_text(encoding="utf-8"))["scores"]
        assert len(kept) == 50

    def test_refuses_losing_verdicts_within_one_arm(self, tmp_path):
        """The per-arm counts carry verdicts too, not just totals."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(payload(11, n_verdicts=5, extra=self.long_context(10)), results, table)

        worse = payload(11, "judge rate-limited", n_verdicts=0, extra=self.long_context(30))
        assert publish(worse, results, table) is False


class TestResolveContext:
    """The judge must see what the model read, or nothing at all."""

    @staticmethod
    def _answer(arm: str, context_ids: tuple[str, ...]):
        from retrieval_ablation.generation.answer import GeneratedAnswer

        return GeneratedAnswer(
            query_id="q1",
            question="What was revenue?",
            answer="Revenue was 100.",
            context_ids=context_ids,
            cited_ids=(),
            invalid_citations=(),
            refused=False,
            prompt_tokens=1,
            output_tokens=1,
            latency_seconds=0.1,
            from_cache=False,
            model="m",
            arm=arm,
        )

    def test_retrieval_context_resolves_to_the_chunk_texts(self):
        from retrieval_ablation.corpus.models import Span
        from retrieval_ablation.evalset.relevance import Chunk
        from retrieval_ablation.generation.run import resolve_context

        chunks = {
            "c1": Chunk(chunk_id="c1", doc_id="d1", span=Span(0, 5), text="alpha"),
            "c2": Chunk(chunk_id="c2", doc_id="d1", span=Span(6, 10), text="beta"),
        }

        got = resolve_context(self._answer("retrieval", ("c1", "c2")), chunks, {})

        assert got == ["alpha", "beta"]

    def test_long_context_resolves_to_the_document_it_was_given(self):
        """Regression: this returned the literal string "(full document)".

        The long-context arm cites one `<doc_id>#fulldoc` pseudo-chunk that does
        not exist in the chunk map, and the old lookup substituted a placeholder
        for it. The judge would then have been asked whether a claim about revenue
        is supported by the words "(full document)" -- and whatever it answered
        would have been reported in the faithfulness column as a measurement.
        Nobody saw it because faithfulness never finished running.
        """
        from tests.test_chunking_replay import document

        from retrieval_ablation.generation.run import resolve_context

        doc = document("d1")
        got = resolve_context(self._answer("long_context", ("d1#fulldoc",)), {}, {"d1": doc})

        assert got is not None
        assert got[0].startswith("Sentence number 0")
        assert "(full document)" not in got[0]

    def test_an_unresolvable_id_yields_none_rather_than_a_placeholder(self):
        """Unknown context is unmeasurable, not judgeable against a stand-in."""
        from retrieval_ablation.generation.run import resolve_context

        assert resolve_context(self._answer("retrieval", ("missing",)), {}, {}) is None

    def test_an_answer_with_no_context_yields_none(self):
        from retrieval_ablation.generation.run import resolve_context

        assert resolve_context(self._answer("retrieval", ()), {}, {}) is None


class TestSampleQueries:
    """A larger run must contain the smaller run's queries, not a fresh draw."""

    @staticmethod
    def queries(n: int):
        from retrieval_ablation.corpus.models import GoldPassage, Span
        from retrieval_ablation.evalset.schema import EvalQuery, QueryKind, Verification

        return [
            EvalQuery(
                query_id=f"q{i:03d}",
                text=f"question {i}",
                gold=(GoldPassage(passage_id=f"p{i}", doc_id="d1", span=Span(0, 5)),),
                kind=QueryKind.TABLE_LOOKUP,
                verification=Verification.GENERATED,
                lexical_overlap=i / n,
                metadata={},
            )
            for i in range(n)
        ]

    def test_a_larger_sample_keeps_the_pinned_queries(self):
        """Regression: growing the sample re-drew it and voided every cached answer.

        The sample was a function of `n`, so asking for 40 after a 12-query run
        picked a different 12 among them. Every previous answer then missed the
        cache and the run paid to answer them again -- on an arm where one answer
        costs 134,000 prompt tokens and the daily allowance is what stops the
        evaluation. Pinning is what makes this arm growable at all.
        """
        from retrieval_ablation.generation.run import sample_queries

        qs = self.queries(100)
        first = sample_queries(qs, 12)
        pinned = [q.query_id for q in first]

        larger = sample_queries(qs, 40, pinned=pinned)

        assert len(larger) == 40
        assert set(pinned) <= {q.query_id for q in larger}
        # And the pinned ones come first, so a truncated long-context budget
        # covers exactly the queries already paid for.
        assert [q.query_id for q in larger[:12]] == pinned

    def test_pinning_more_than_requested_truncates_rather_than_overshooting(self):
        from retrieval_ablation.generation.run import sample_queries

        qs = self.queries(100)
        pinned = [q.query_id for q in qs[:20]]

        assert len(sample_queries(qs, 5, pinned=pinned)) == 5

    def test_an_unknown_pinned_id_is_ignored(self):
        """A results file naming a query the eval set no longer has must not crash."""
        from retrieval_ablation.generation.run import sample_queries

        got = sample_queries(self.queries(50), 10, pinned=["gone", "q001"])

        assert len(got) == 10
        assert got[0].query_id == "q001"

    def test_no_duplicates_when_a_pinned_id_would_also_be_drawn(self):
        from retrieval_ablation.generation.run import sample_queries

        qs = self.queries(30)
        got = sample_queries(qs, 30, pinned=["q000", "q015"])

        assert len(got) == len({q.query_id for q in got}) == 30


class TestPreviouslyAnswered:
    def test_reads_query_ids_in_order_without_duplicates(self, tmp_path):
        """Both arms answer the same query, so ids repeat in the answers list."""
        from retrieval_ablation.generation.run import previously_answered

        path = tmp_path / "generation.json"
        path.write_text(
            json.dumps(
                {
                    "answers": [
                        {"query_id": "a", "arm": "retrieval"},
                        {"query_id": "a", "arm": "long_context"},
                        {"query_id": "b", "arm": "retrieval"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        assert previously_answered(path) == ["a", "b"]

    def test_a_missing_or_corrupt_file_yields_nothing(self, tmp_path):
        """A first run has nothing to resume, and neither does a broken file."""
        from retrieval_ablation.generation.run import previously_answered

        assert previously_answered(tmp_path / "absent.json") == []
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        assert previously_answered(bad) == []


class TestLatencyProtection:
    """Latency is the one field a successful-looking re-run can silently destroy."""

    @staticmethod
    def with_latency(scores: int, retrieval_live: int, lc_live: int) -> dict:
        out = payload(
            scores,
            extra=[
                {"query_id": f"lc{i}", "arm": "long_context", "faithfulness": None}
                for i in range(10)
            ],
        )
        out["latency"] = {
            "retrieval": {"n_live": retrieval_live, "p95": 4.542 if retrieval_live else None},
            "long_context": {"n_live": lc_live, "p95": 16.908 if lc_live else None},
        }
        return out

    def test_a_cached_rerun_cannot_erase_a_live_latency_measurement(self, tmp_path):
        """Regression: a valid same-session comparison was replaced by "not measured".

        Every answer can come from cache, so a re-run reproduces the same score
        count and the same verdicts while making no live call at all. Cost survives
        that, because token counts do not depend on when a call was made; latency
        does not. An 11-and-10 live measurement was overwritten by a file reporting
        latency as unmeasured, and it was noticed only afterwards.
        """
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(self.with_latency(11, retrieval_live=11, lc_live=10), results, table)

        cached = self.with_latency(11, retrieval_live=0, lc_live=0)
        cached["incomplete_reason"] = "every answer served from cache"

        assert publish(cached, results, table) is False
        kept = json.loads(results.read_text(encoding="utf-8"))
        assert kept["latency"]["long_context"]["n_live"] == 10

    def test_a_run_with_more_live_calls_still_writes(self, tmp_path):
        """Otherwise re-measuring latency could never be recorded."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(self.with_latency(11, retrieval_live=5, lc_live=5), results, table)

        assert publish(self.with_latency(11, retrieval_live=11, lc_live=10), results, table) is True

    def test_a_payload_without_latency_at_all_is_still_handled(self, tmp_path):
        """The field is optional; its absence must not crash the guard."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"

        assert publish(payload(3), results, table) is True

    def test_the_advice_names_archiving_when_only_latency_regressed(self, tmp_path, capsys):
        """A run that gains scores and loses only latency must not be told to delete.

        That is the common case -- it is what happened the first time -- and
        deleting the file to let the better run through would throw away the gain in
        order to keep the smaller thing. Archiving keeps both.
        """
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(self.with_latency(11, retrieval_live=11, lc_live=10), results, table)

        better = self.with_latency(30, retrieval_live=0, lc_live=0)
        assert publish(better, results, table) is False

        printed = capsys.readouterr().out
        assert "archive" in printed
        assert "Only the live latency samples regressed" in printed

    def test_a_genuinely_smaller_run_is_told_to_re_run(self, tmp_path, capsys):
        """The other branch: fewer scores is not an archiving problem."""
        results, table = tmp_path / "generation.json", tmp_path / "generation.md"
        publish(self.with_latency(20, retrieval_live=5, lc_live=5), results, table)

        assert publish(self.with_latency(2, retrieval_live=5, lc_live=5), results, table) is False

        printed = capsys.readouterr().out
        assert "Re-run when quota allows" in printed
