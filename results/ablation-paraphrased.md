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
| `rerank-bm25-100` | reranking | 0.1149 | [0.088, 0.143] | 0.2538 | 0.1002 | +0.0621 | 0.0013 | yes | 390 |
| `rerank-candidates-200` | candidates | 0.1109 | [0.085, 0.139] | 0.2718 | 0.0971 | +0.0581 | 0.0013 | yes | 390 |
| `retrieval-hybrid-rrf` | retrieval | 0.1022 | [0.078, 0.129] | 0.2974 | 0.0891 | +0.0494 | 0.0013 | yes | 390 |
| `rerank-candidates-50` | candidates | 0.1004 | [0.075, 0.127] | 0.2385 | 0.0888 | +0.0476 | 0.0013 | yes | 390 |
| `rerank-candidates-25` | candidates | 0.0975 | [0.072, 0.125] | 0.2385 | 0.0906 | +0.0447 | 0.0016 | yes | 390 |
| `retrieval-dense-bge` | retrieval | 0.0839 | [0.061, 0.108] | 0.2564 | 0.0746 | +0.0311 | 0.0399 | yes | 390 |
| `chunk-semantic95` | chunking | 0.0663 | [0.047, 0.087] | 0.3154 | 0.0605 | +0.0134 | 0.6479 | no | 390 |
| `chunk-struct512` | chunking | 0.0643 | [0.045, 0.084] | 0.2385 | 0.0545 | +0.0114 | 0.6479 | no | 390 |
| `retrieval-bm25-struct` | retrieval | 0.0643 | [0.045, 0.084] | 0.2385 | 0.0545 | +0.0114 | 0.6479 | no | 390 |
| `tables-row-sentences` | table_rendering | 0.0597 | [0.042, 0.079] | 0.2264 | 0.0562 | +0.0069 | 0.8289 | no | 390 |
| `baseline-bm25-fixed512` | baseline | 0.0528 | [0.037, 0.071] | 0.2179 | 0.0493 | (baseline) | - | - | 390 |
| `embed-e5-base-v2` | embedding | 0.0506 | [0.035, 0.068] | 0.2282 | 0.0421 | -0.0022 | 0.8410 | no | 390 |
| `chunk-fixed256o32` | chunking | 0.0407 | [0.026, 0.057] | 0.1487 | 0.0329 | -0.0121 | 0.6479 | no | 390 |
| `embed-e5-base` | embedding | 0.0137 | [0.006, 0.023] | 0.0795 | 0.0125 | -0.0391 | 0.0013 | yes | 390 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.0414 | 0.1891 | 2 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0253 | 0.2263 | 3 |
| `chunk-semantic95` | 29,556 | 100.0% | - | 0.0600 | 0.1418 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0467 | 0.2752 | 2 |
| `tables-row-sentences` | 40,155 | 95.9% | - | 0.0451 | 0.2349 | 2 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.0767 | 0.1699 | 4 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.0806 | 0.3615 | 7 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0467 | 0.2752 | 2 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0090 | 0.0710 | 3 |
| `embed-e5-base-v2` | 42,215 | 100.0% | - | 0.0416 | 0.1587 | 3 |
| `rerank-bm25-100` | 42,215 | 100.0% | 33.3% | 0.1066 | 0.2151 | 2 |
| `rerank-candidates-25` | 42,215 | 100.0% | 18.9% | 0.0858 | 0.2387 | 2 |
| `rerank-candidates-50` | 42,215 | 100.0% | 25.6% | 0.0900 | 0.2248 | 2 |
| `rerank-candidates-200` | 42,215 | 100.0% | 43.0% | 0.1050 | 0.1815 | 2 |

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
