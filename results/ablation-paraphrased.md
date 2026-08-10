# Retrieval ablation

8 of 15 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `retrieval-hybrid-rrf` | retrieval | 0.1088 | [0.070, 0.152] | 0.3077 | 0.0887 | +0.0613 | 0.0021 | yes | 143 |
| `retrieval-dense-bge` | retrieval | 0.0991 | [0.059, 0.144] | 0.2797 | 0.0886 | +0.0516 | 0.0444 | yes | 143 |
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
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0210 | 0.1520 | 2 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 1 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.0383 | 0.1584 | 1 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.0965 | 0.1301 | 3 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.0893 | 0.3422 | 4 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 1 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0165 | 0.0909 | 2 |

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
| `rerank-bm25-100` | rerank-scores-candidates-rerank-bm25-100.json.gz holds no cross-encoder scores for the wording being scored: it either predates query-text provenance or was computed against a different eval set. Reusing it would rerank these queries with scores derived from other questions. Re-run the GPU notebook against this eval set. |
| `rerank-candidates-25` | rerank-scores-candidates-rerank-bm25-100.json.gz holds no cross-encoder scores for the wording being scored: it either predates query-text provenance or was computed against a different eval set. Reusing it would rerank these queries with scores derived from other questions. Re-run the GPU notebook against this eval set. |
| `rerank-candidates-50` | rerank-scores-candidates-rerank-bm25-100.json.gz holds no cross-encoder scores for the wording being scored: it either predates query-text provenance or was computed against a different eval set. Reusing it would rerank these queries with scores derived from other questions. Re-run the GPU notebook against this eval set. |
| `rerank-candidates-200` | rerank-scores-candidates-rerank-bm25-100.json.gz holds no cross-encoder scores for the wording being scored: it either predates query-text provenance or was computed against a different eval set. Reusing it would rerank these queries with scores derived from other questions. Re-run the GPU notebook against this eval set. |
| `hybrid-plus-rerank` | rerank-scores-candidates-rerank-bm25-100.json.gz holds no cross-encoder scores for the wording being scored: it either predates query-text provenance or was computed against a different eval set. Reusing it would rerank these queries with scores derived from other questions. Re-run the GPU notebook against this eval set. |
