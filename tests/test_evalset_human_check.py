"""Tests for reading a completed verification sample back.

Offline: `parse` reads a markdown file and `apply_verdicts` rewrites a JSONL file,
both under tmp_path. No network, no model, nothing that needs a key.
"""

from __future__ import annotations

import pytest

from retrieval_ablation.corpus.models import GoldPassage, Span
from retrieval_ablation.evalset import human_check
from retrieval_ablation.evalset.schema import (
    EvalQuery,
    QueryKind,
    Verification,
    read_eval_set,
    write_eval_set,
)

ENTRY = """## {n}. `{qid}`

**Query:** What was revenue in 2023?

- lexical overlap: `0.40`
- document: `d1`

**Labelled passage:**

```
| Revenue | 100 |
```

- [{ok}] ok
- [{reject}] reject &mdash; reason:{reason}
"""


def sample(tmp_path, entries):
    """Build a sample file. Each entry is (query_id, ok_ticked, reject_ticked, reason)."""
    blocks = ["# Verification sample", ""]
    for index, (qid, ok, reject, reason) in enumerate(entries, start=1):
        blocks.append(
            ENTRY.format(
                n=index,
                qid=qid,
                ok="x" if ok else " ",
                reject="x" if reject else " ",
                reason=f" {reason}" if reason else "",
            )
        )
    path = tmp_path / "verification_sample.md"
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def query(qid: str, verification: Verification = Verification.GENERATED) -> EvalQuery:
    return EvalQuery(
        query_id=qid,
        text="What was revenue in 2023?",
        gold=(GoldPassage(passage_id=f"p-{qid}", doc_id="d1", span=Span(0, 10)),),
        kind=QueryKind.TABLE_LOOKUP,
        verification=verification,
        lexical_overlap=0.4,
        metadata={"ticker": "X"},
    )


class TestParse:
    def test_an_unmarked_sample_reports_no_rate(self, tmp_path):
        """Regression in spirit: the rate must not be invented from zero ticks.

        The whole purpose of the sample is to replace a guess with a measurement.
        A number produced from nothing would defeat it, so the rate is None and the
        report says "not measured" -- the same discipline every unmeasurable metric
        in this project follows.
        """
        path = sample(tmp_path, [("q1", False, False, ""), ("q2", False, False, "")])

        summary = human_check.parse(path)

        assert summary.n_entries == 2
        assert summary.n_marked == 0
        assert summary.rejection_rate is None
        assert summary.complete is False
        assert "not measured" in human_check.report(summary)

    def test_a_partly_marked_sample_reports_only_what_is_marked(self, tmp_path):
        path = sample(
            tmp_path,
            [
                ("q1", True, False, ""),
                ("q2", False, True, "wrong row"),
                ("q3", False, False, ""),
            ],
        )

        summary = human_check.parse(path)

        assert (summary.n_marked, summary.n_accepted, summary.n_rejected) == (2, 1, 1)
        assert summary.rejection_rate == pytest.approx(0.5)
        assert summary.complete is False
        assert "not over the eval set" in human_check.report(summary)

    def test_a_fully_marked_sample_is_complete(self, tmp_path):
        path = sample(tmp_path, [("q1", True, False, ""), ("q2", True, False, "")])

        summary = human_check.parse(path)

        assert summary.complete is True
        assert summary.rejection_rate == 0.0
        assert "(incomplete)" not in human_check.report(summary)

    def test_both_boxes_ticked_is_skipped_rather_than_guessed(self, tmp_path):
        """Choosing one would invent a verdict the person did not give."""
        path = sample(tmp_path, [("q1", True, True, "unsure")])

        summary = human_check.parse(path)

        assert summary.n_marked == 0
        assert summary.n_marked_both == 1
        assert "skipped rather than guessed" in human_check.report(summary)

    def test_the_rejection_reason_is_captured(self, tmp_path):
        path = sample(tmp_path, [("q1", False, True, "the label is a place name")])

        summary = human_check.parse(path)

        assert summary.verdicts[0].reason == "the label is a place name"

    def test_an_uppercase_tick_counts(self, tmp_path):
        """A person typing X should not silently lose their answer."""
        path = tmp_path / "s.md"
        path.write_text("## 1. `q1`\n\n- [X] ok\n- [ ] reject\n", encoding="utf-8")

        assert human_check.parse(path).n_accepted == 1

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        summary = human_check.parse(tmp_path / "absent.md")

        assert summary.n_entries == 0
        assert "no verification sample" in human_check.report(summary)


class TestApplyVerdicts:
    def path_with(self, tmp_path, queries):
        path = tmp_path / "queries.jsonl"
        write_eval_set(queries, path)
        return path

    def test_marked_queries_take_the_human_status(self, tmp_path):
        path = self.path_with(tmp_path, [query("q1"), query("q2")])
        summary = human_check.parse(
            sample(tmp_path, [("q1", True, False, ""), ("q2", False, True, "bad")])
        )

        applied = human_check.apply_verdicts(summary, path)

        by_id = {q.query_id: q for q in read_eval_set(path)}
        assert applied.changed == 2
        assert applied.already_agreed == 0
        assert by_id["q1"].verification is Verification.HUMAN_VERIFIED
        assert by_id["q2"].verification is Verification.REJECTED
        assert by_id["q2"].metadata["human_reject_reason"] == "bad"

    def test_unmarked_queries_are_left_exactly_as_they_were(self, tmp_path):
        """A half-finished sample must not relabel the rest of the benchmark.

        Marking by omission is the failure worth guarding: the sample covers 40 of
        586 queries, so treating "not in the sample" as any kind of verdict would
        mislabel most of the eval set from a single afternoon's work.
        """
        path = self.path_with(
            tmp_path,
            [query("q1"), query("q2", Verification.MODEL_CHECKED), query("q3")],
        )
        summary = human_check.parse(sample(tmp_path, [("q1", True, False, "")]))

        human_check.apply_verdicts(summary, path)

        by_id = {q.query_id: q for q in read_eval_set(path)}
        assert by_id["q2"].verification is Verification.MODEL_CHECKED
        assert by_id["q3"].verification is Verification.GENERATED
        assert len(by_id) == 3

    def test_applying_twice_changes_nothing_the_second_time(self, tmp_path):
        path = self.path_with(tmp_path, [query("q1")])
        summary = human_check.parse(sample(tmp_path, [("q1", True, False, "")]))

        assert human_check.apply_verdicts(summary, path).changed == 1
        # Re-applying is not a lost verdict: the status already matches.
        second = human_check.apply_verdicts(summary, path)
        assert (second.changed, second.already_agreed) == (0, 1)

    def test_gold_and_text_are_never_altered(self, tmp_path):
        """Verification records an opinion about a label, not a new label."""
        original = query("q1")
        path = self.path_with(tmp_path, [original])
        summary = human_check.parse(sample(tmp_path, [("q1", False, True, "no")]))

        human_check.apply_verdicts(summary, path)

        after = read_eval_set(path)[0]
        assert after.gold == original.gold
        assert after.text == original.text
        assert after.lexical_overlap == original.lexical_overlap


class TestVerificationSampleSpread:
    """The sample claims to span the overlap range, so it has to."""

    @staticmethod
    def spread(total: int, n: int) -> list[int]:
        """The indices the sampler now selects, as a pure calculation."""
        if n >= total:
            return list(range(total))
        stride = total / n
        return [min(total - 1, int(i * stride)) for i in range(n)]

    def test_the_highest_overlap_queries_are_reachable(self):
        """Regression: the top 39 queries could never be sampled.

        `ordered[::step][:n]` strides and then truncates. At 586 queries and 40
        wanted the stride is 14, so the last index taken was 546 of 585 and the
        highest-overlap end of the range was unreachable -- while the file's own
        header told the reader it was "spread across the lexical-overlap range".

        It survived because the bias ran towards the queries this project cares
        about least: low-overlap queries are the interesting ones, so nobody missed
        the high-overlap tail.
        """
        old = list(range(0, 586, max(1, 586 // 40)))[:40]
        new = self.spread(586, 40)

        assert old[-1] == 546
        assert new[-1] > old[-1]
        assert new[-1] >= 585 - 40

    def test_the_sample_is_the_requested_size(self):
        assert len(self.spread(586, 40)) == 40

    def test_indices_are_strictly_increasing(self):
        """A repeated index would silently shrink the sample."""
        got = self.spread(586, 40)

        assert got == sorted(got)
        assert len(set(got)) == len(got)

    def test_asking_for_more_than_exists_returns_everything(self):
        assert self.spread(5, 40) == [0, 1, 2, 3, 4]

    def test_the_low_end_is_still_included(self):
        """The fix must not trade one end of the range for the other."""
        assert self.spread(586, 40)[0] == 0


class TestAgreementIsNotLoss:
    """A verdict matching the recorded status changes nothing, and must say so."""

    def test_agreeing_with_an_existing_rejection_is_counted_separately(self, tmp_path):
        """Regression in reporting: 7 verdicts reported as "6 changed".

        The model audit has already rejected 44 of the 586 queries. A human
        rejecting one of those agrees with it and changes nothing, so a bare change
        count is one lower than the number of ticks -- indistinguishable, to the
        person who did the ticking, from a verdict that was silently dropped.
        Found by dry-running the real sample file rather than a fixture.
        """
        path = tmp_path / "queries.jsonl"
        write_eval_set(
            [query("q1", Verification.REJECTED), query("q2", Verification.GENERATED)], path
        )
        summary = human_check.parse(
            sample(tmp_path, [("q1", False, True, "still bad"), ("q2", False, True, "also bad")])
        )

        applied = human_check.apply_verdicts(summary, path)

        assert applied.changed == 1
        assert applied.already_agreed == 1
        assert applied.total == 2 == summary.n_marked

    def test_total_accounts_for_every_marked_verdict(self, tmp_path):
        """changed + already_agreed must reconcile with what the person marked."""
        path = tmp_path / "queries.jsonl"
        write_eval_set([query(f"q{i}") for i in range(1, 4)], path)
        summary = human_check.parse(
            sample(
                tmp_path,
                [("q1", True, False, ""), ("q2", True, False, ""), ("q3", False, True, "no")],
            )
        )

        applied = human_check.apply_verdicts(summary, path)

        assert applied.total == summary.n_marked == 3
