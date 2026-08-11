# Retrieval ablation

13 of 15 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `rerank-candidates-200` | candidates | 0.1210 | [0.077, 0.170] | 0.2867 | 0.1049 | +0.0735 | 0.0050 | yes | 143 |
| `hybrid-plus-rerank` | interaction | 0.1210 | [0.078, 0.169] | 0.2727 | 0.1016 | +0.0735 | 0.0044 | yes | 143 |
| `rerank-bm25-100` | reranking | 0.1178 | [0.074, 0.166] | 0.2657 | 0.1013 | +0.0703 | 0.0099 | yes | 143 |
| `retrieval-hybrid-rrf` | retrieval | 0.1088 | [0.070, 0.152] | 0.3077 | 0.0887 | +0.0613 | 0.0036 | yes | 143 |
| `rerank-candidates-50` | candidates | 0.1047 | [0.064, 0.151] | 0.2168 | 0.0897 | +0.0572 | 0.0160 | yes | 143 |
| `retrieval-dense-bge` | retrieval | 0.0991 | [0.059, 0.144] | 0.2797 | 0.0886 | +0.0516 | 0.0444 | yes | 143 |
| `rerank-candidates-25` | candidates | 0.0963 | [0.055, 0.143] | 0.2168 | 0.0881 | +0.0488 | 0.0392 | yes | 143 |
| `chunk-struct512` | chunking | 0.0482 | [0.024, 0.077] | 0.2168 | 0.0387 | +0.0008 | 1.0000 | no | 143 |
| `retrieval-bm25-struct` | retrieval | 0.0482 | [0.024, 0.077] | 0.2168 | 0.0387 | +0.0008 | 1.0000 | no | 143 |
| `tables-row-sentences` | table_rendering | 0.0475 | [0.026, 0.073] | 0.2224 | 0.0439 | +0.0000 | 1.0000 | no | 143 |
| `baseline-bm25-fixed512` | baseline | 0.0475 | [0.023, 0.077] | 0.2133 | 0.0403 | (baseline) | - | - | 143 |
| `chunk-fixed256o32` | chunking | 0.0311 | [0.012, 0.054] | 0.1643 | 0.0244 | -0.0164 | 1.0000 | no | 143 |
| `embed-e5-base` | embedding | 0.0223 | [0.006, 0.044] | 0.0909 | 0.0191 | -0.0252 | 0.4485 | no | 143 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.0406 | 0.1301 | 1 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0210 | 0.1520 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 1 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.0383 | 0.1584 | 1 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.0965 | 0.1301 | 2 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.0893 | 0.3422 | 2 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 1 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0165 | 0.0909 | 1 |
| `rerank-bm25-100` | 42,215 | 100.0% | 35.6% | 0.1069 | 0.2483 | 1 |
| `rerank-candidates-25` | 42,215 | 100.0% | 17.1% | 0.0886 | 0.1877 | 1 |
| `rerank-candidates-50` | 42,215 | 100.0% | 24.5% | 0.0938 | 0.2356 | 1 |
| `rerank-candidates-200` | 42,215 | 100.0% | 43.1% | 0.1135 | 0.2105 | 1 |
| `hybrid-plus-rerank` | 42,215 | 100.0% | 38.0% | 0.1084 | 0.2719 | 2 |

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
| `chunk-semantic95` | bge-m3 unavailable: ModuleNotFoundError: No module named 'sentence_transformers' |
| `embed-finance-e5` | finance-e5 unavailable: ModuleNotFoundError: No module named 'sentence_transformers' |
