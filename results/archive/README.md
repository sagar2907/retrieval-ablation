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
