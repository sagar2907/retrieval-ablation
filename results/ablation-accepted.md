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
| `rerank-candidates-50` | candidates | 0.2299 | [0.195, 0.267] | 0.5854 | 0.1977 | +0.0250 | 1.0000 | no | 357 |
| `chunk-semantic95` | chunking | 0.2258 | [0.192, 0.261] | 0.6751 | 0.1888 | +0.0209 | 1.0000 | no | 357 |
| `chunk-struct512` | chunking | 0.2235 | [0.187, 0.261] | 0.5854 | 0.1997 | +0.0185 | 1.0000 | no | 357 |
| `retrieval-bm25-struct` | retrieval | 0.2235 | [0.187, 0.261] | 0.5854 | 0.1997 | +0.0185 | 1.0000 | no | 357 |
| `rerank-candidates-25` | candidates | 0.2199 | [0.185, 0.256] | 0.5854 | 0.1921 | +0.0150 | 1.0000 | no | 357 |
| `rerank-bm25-100` | reranking | 0.2157 | [0.181, 0.252] | 0.5658 | 0.1876 | +0.0107 | 1.0000 | no | 357 |
| `baseline-bm25-fixed512` | baseline | 0.2050 | [0.172, 0.239] | 0.5630 | 0.1845 | (baseline) | - | - | 357 |
| `rerank-candidates-200` | candidates | 0.2000 | [0.166, 0.236] | 0.5266 | 0.1745 | -0.0050 | 1.0000 | no | 357 |
| `retrieval-hybrid-rrf` | retrieval | 0.1895 | [0.155, 0.226] | 0.5434 | 0.1702 | -0.0154 | 1.0000 | no | 357 |
| `tables-row-sentences` | table_rendering | 0.1702 | [0.139, 0.202] | 0.5389 | 0.1558 | -0.0348 | 0.2943 | no | 357 |
| `chunk-fixed256o32` | chunking | 0.1607 | [0.131, 0.193] | 0.4678 | 0.1417 | -0.0443 | 0.0460 | yes | 357 |
| `retrieval-dense-bge` | retrieval | 0.1096 | [0.082, 0.139] | 0.2801 | 0.0977 | -0.0954 | 0.0013 | yes | 357 |
| `embed-e5-base-v2` | embedding | 0.1033 | [0.078, 0.131] | 0.2997 | 0.0874 | -0.1016 | 0.0013 | yes | 357 |
| `embed-e5-base` | embedding | 0.0352 | [0.021, 0.052] | 0.1709 | 0.0306 | -0.1698 | 0.0013 | yes | 357 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.0818 | 0.2440 | 1 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0704 | 0.1893 | 2 |
| `chunk-semantic95` | 29,556 | 100.0% | - | 0.1397 | 0.2532 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.1127 | 0.2586 | 1 |
| `tables-row-sentences` | 40,155 | 95.6% | - | 0.0866 | 0.1967 | 1 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.0764 | 0.1202 | 4 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.1012 | 0.2176 | 5 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.1127 | 0.2586 | 1 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0341 | 0.0356 | 3 |
| `embed-e5-base-v2` | 42,215 | 100.0% | - | 0.1020 | 0.1038 | 3 |
| `rerank-bm25-100` | 42,215 | 100.0% | 65.7% | 0.1847 | 0.2255 | 1 |
| `rerank-candidates-25` | 42,215 | 100.0% | 48.3% | 0.1789 | 0.2329 | 1 |
| `rerank-candidates-50` | 42,215 | 100.0% | 57.9% | 0.1844 | 0.2444 | 1 |
| `rerank-candidates-200` | 42,215 | 100.0% | 75.3% | 0.1707 | 0.2093 | 1 |

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
