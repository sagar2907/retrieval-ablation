# Retrieval ablation

6 of 15 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `chunk-semantic95` | chunking | 0.2189 | [0.187, 0.252] | 0.6641 | 0.1828 | +0.0218 | 0.2622 | no | 390 |
| `chunk-struct512` | chunking | 0.2135 | [0.180, 0.249] | 0.5667 | 0.1905 | +0.0164 | 0.5629 | no | 390 |
| `retrieval-bm25-struct` | retrieval | 0.2135 | [0.180, 0.249] | 0.5667 | 0.1905 | +0.0164 | 0.5629 | no | 390 |
| `baseline-bm25-fixed512` | baseline | 0.1971 | [0.166, 0.230] | 0.5385 | 0.1769 | (baseline) | - | - | 390 |
| `tables-row-sentences` | table_rendering | 0.1644 | [0.135, 0.195] | 0.5324 | 0.1501 | -0.0327 | 0.1188 | no | 390 |
| `chunk-fixed256o32` | chunking | 0.1556 | [0.127, 0.185] | 0.4474 | 0.1376 | -0.0415 | 0.0285 | yes | 390 |

## Chunking reachability and reranking ceiling

`reachable` is the fraction of gold passages that any chunk of this
configuration covers. A configuration cannot score a query whose answer it
cannot represent, so a low value here means the headline number above rests
on fewer queries -- which is exactly why the shared subset is used.

`ceiling` is the first stage's recall at the reranking candidate depth: the
hard upper bound on what the cross-encoder could achieve. It separates a weak
reranker from one that never saw the answer.

| configuration | chunks | reachable | ceiling | nDCG low-overlap | nDCG high-overlap | seconds |
|---|---|---|---|---|---|---|
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.0823 | 0.2378 | 1 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0754 | 0.1840 | 2 |
| `chunk-semantic95` | 29,556 | 100.0% | - | 0.1376 | 0.2477 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.1016 | 0.2532 | 1 |
| `tables-row-sentences` | 40,155 | 95.9% | - | 0.0840 | 0.1929 | 1 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.1016 | 0.2532 | 1 |

## The lexical-overlap confound

Queries are generated from table row labels, so they reuse the document's
own wording and hand a lexical matcher an exact string. The two nDCG columns
above split the query set at 0.4 content-word overlap. A configuration whose
advantage exists only in the high-overlap column is winning at string
matching, not at retrieval.

## Not measured

These configurations did not run. No number is reported for them.

| configuration | reason |
|---|---|
| `retrieval-dense-bge` | query vectors for bge-m3 cover only 216 of 586 queries. Re-run the GPU notebook against the current eval set. |
| `retrieval-hybrid-rrf` | query vectors for bge-m3 cover only 216 of 586 queries. Re-run the GPU notebook against the current eval set. |
| `embed-e5-base` | query vectors for e5-base cover only 216 of 586 queries. Re-run the GPU notebook against the current eval set. |
| `embed-e5-base-v2` | no usable query vectors for e5-base-v2: queryvectors-e5-base-v2.npz is absent, or records different query text than the set being scored (check the loader warning above). A dense index needs both sides embedded by the same model on the same wording; reusing vectors from other text compares two things that were never asked and returns confident nonsense. Re-run the GPU notebook against this eval set to produce them. |
| `rerank-bm25-100` | cross-encoder scores cover only 216 of 582 queries. Reranking the covered fraction and leaving the rest in first-stage order measures neither. Re-run the GPU notebook against the current eval set. |
| `rerank-candidates-25` | cross-encoder scores cover only 216 of 582 queries. Reranking the covered fraction and leaving the rest in first-stage order measures neither. Re-run the GPU notebook against the current eval set. |
| `rerank-candidates-50` | cross-encoder scores cover only 216 of 582 queries. Reranking the covered fraction and leaving the rest in first-stage order measures neither. Re-run the GPU notebook against the current eval set. |
| `rerank-candidates-200` | cross-encoder scores cover only 216 of 582 queries. Reranking the covered fraction and leaving the rest in first-stage order measures neither. Re-run the GPU notebook against the current eval set. |
| `hybrid-plus-rerank` | query vectors for bge-m3 cover only 216 of 586 queries. Re-run the GPU notebook against the current eval set. |
