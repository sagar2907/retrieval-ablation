# Retrieval ablation

14 of 15 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `rerank-candidates-50` | candidates | 0.2198 | [0.186, 0.255] | 0.5667 | 0.1891 | +0.0227 | 1.0000 | no | 390 |
| `chunk-semantic95` | chunking | 0.2189 | [0.187, 0.252] | 0.6641 | 0.1828 | +0.0218 | 0.6991 | no | 390 |
| `chunk-struct512` | chunking | 0.2135 | [0.180, 0.249] | 0.5667 | 0.1905 | +0.0164 | 1.0000 | no | 390 |
| `retrieval-bm25-struct` | retrieval | 0.2135 | [0.180, 0.249] | 0.5667 | 0.1905 | +0.0164 | 1.0000 | no | 390 |
| `rerank-candidates-25` | candidates | 0.2116 | [0.179, 0.245] | 0.5667 | 0.1843 | +0.0145 | 1.0000 | no | 390 |
| `rerank-bm25-100` | reranking | 0.2068 | [0.174, 0.241] | 0.5487 | 0.1797 | +0.0097 | 1.0000 | no | 390 |
| `baseline-bm25-fixed512` | baseline | 0.1971 | [0.166, 0.230] | 0.5385 | 0.1769 | (baseline) | - | - | 390 |
| `rerank-candidates-200` | candidates | 0.1924 | [0.161, 0.226] | 0.5051 | 0.1674 | -0.0047 | 1.0000 | no | 390 |
| `retrieval-hybrid-rrf` | retrieval | 0.1817 | [0.150, 0.215] | 0.5205 | 0.1630 | -0.0154 | 1.0000 | no | 390 |
| `tables-row-sentences` | table_rendering | 0.1644 | [0.135, 0.195] | 0.5324 | 0.1501 | -0.0327 | 0.2673 | no | 390 |
| `chunk-fixed256o32` | chunking | 0.1556 | [0.127, 0.185] | 0.4474 | 0.1376 | -0.0415 | 0.0570 | no | 390 |
| `retrieval-dense-bge` | retrieval | 0.1044 | [0.079, 0.132] | 0.2718 | 0.0928 | -0.0927 | 0.0013 | yes | 390 |
| `embed-e5-base-v2` | embedding | 0.1007 | [0.077, 0.127] | 0.2949 | 0.0853 | -0.0965 | 0.0013 | yes | 390 |
| `embed-e5-base` | embedding | 0.0367 | [0.023, 0.052] | 0.1795 | 0.0312 | -0.1604 | 0.0013 | yes | 390 |

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
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.0676 | 0.1174 | 4 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.0921 | 0.2134 | 6 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.1016 | 0.2532 | 1 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0329 | 0.0380 | 3 |
| `embed-e5-base-v2` | 42,215 | 100.0% | - | 0.0898 | 0.1045 | 3 |
| `rerank-bm25-100` | 42,215 | 100.0% | 64.2% | 0.1682 | 0.2205 | 1 |
| `rerank-candidates-25` | 42,215 | 100.0% | 46.8% | 0.1664 | 0.2276 | 1 |
| `rerank-candidates-50` | 42,215 | 100.0% | 56.3% | 0.1680 | 0.2382 | 1 |
| `rerank-candidates-200` | 42,215 | 100.0% | 73.7% | 0.1564 | 0.2051 | 1 |

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
| `hybrid-plus-rerank` | no cross-encoder scores for candidates-hybrid-plus-rerank: the GPU run has not scored this configuration's shortlist. Falling back to a live reranker also failed (ModuleNotFoundError). |
