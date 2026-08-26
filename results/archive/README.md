# Archived results: the 216-query eval set

These are the completed ablation runs against `data/eval/queries.jsonl` as it
stood at 216 queries, before it was extended to 586.

They are kept because they are a **complete** measurement and the current files
are not. Every GPU artifact — dense vectors, query vectors, cross-encoder scores
— was built against those 216 queries, so at 216 the grid measured 13 of 15
configurations on the original wording and 14 of 15 on the paraphrased one, with
full artifact coverage on every arm.

Extending the eval set does not invalidate any of that. It does mean the
artifacts now cover 216 of 586 queries, and the runner refuses to score an arm on
partial coverage rather than reranking 37% of the queries and leaving the rest in
first-stage order. So the live files currently measure 6 of 15 until one GPU run
regenerates the artifacts against the larger set.

The headline finding is in both, at two sample sizes:

| | n = 143 shared | n = 390 shared |
|---|---|---|
| original wording, significant improvements | 0 | 0 |
| paraphrased wording, significant improvements | 4 | 4 (diluted coverage) |

The n=390 paraphrased figures were produced before the coverage rule existed and
are not reproducible from the current code; they are described here rather than
committed as results, because a number nobody can regenerate is not evidence.

## `generation-n12-same-session-latency.json`

The generation run of 12 sampled queries in which **both arms made live calls in
the same session** — 11 retrieval and 10 long-context. It is kept because that is
the only condition under which the two arms' latencies are comparable, and the
current run cannot reproduce it: its answers come from cache, so re-measuring
latency would need a fresh long-context pass at roughly 1.4M prompt tokens, which
is more than a day's free-tier allowance.

Retrieval p95 4.542 s, long-context p95 16.908 s, ratio 3.7×. Cost figures here are
superseded by the current run, which measures the same thing on more queries; token
counts do not depend on when a call was made, so cost stays comparable across
sessions in a way latency does not.

## `generation-n66-live-retrieval-latency.json`

The 66-answer retrieval run, kept for its live latency measurement: 20 timed calls,
p95 15.489 s, made in the same session as the answers themselves. The long-context
arm was cache-only in that run and is recorded as not measured, so this is a
per-arm figure rather than a comparison.

Archived because the next run judges long-context faithfulness and answers nothing
new, so every answer comes from cache and no call is timed. `publish` refuses that
write on its own — latency is the one field a successful-looking cached re-run
destroys — and archiving is the path its message recommends: keep the measurement,
let the better run through.
