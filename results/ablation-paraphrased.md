# Retrieval ablation

9 of 15 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `rerank-bm25-100` | reranking | 0.1377 | [0.090, 0.189] | 0.2937 | 0.1232 | +0.0902 | 0.0016 | yes | 143 |
| `rerank-candidates-200` | candidates | 0.1342 | [0.088, 0.185] | 0.3287 | 0.1208 | +0.0867 | 0.0028 | yes | 143 |
| `rerank-candidates-50` | candidates | 0.1058 | [0.065, 0.150] | 0.2168 | 0.0912 | +0.0584 | 0.0036 | yes | 143 |
| `rerank-candidates-25` | candidates | 0.0919 | [0.053, 0.136] | 0.2168 | 0.0849 | +0.0444 | 0.0350 | yes | 143 |
| `chunk-struct512` | chunking | 0.0482 | [0.024, 0.077] | 0.2168 | 0.0387 | +0.0008 | 1.0000 | no | 143 |
| `retrieval-bm25-struct` | retrieval | 0.0482 | [0.024, 0.077] | 0.2168 | 0.0387 | +0.0008 | 1.0000 | no | 143 |
| `tables-row-sentences` | table_rendering | 0.0475 | [0.026, 0.073] | 0.2224 | 0.0439 | +0.0000 | 1.0000 | no | 143 |
| `baseline-bm25-fixed512` | baseline | 0.0475 | [0.023, 0.077] | 0.2133 | 0.0403 | (baseline) | - | - | 143 |
| `chunk-fixed256o32` | chunking | 0.0311 | [0.012, 0.054] | 0.1643 | 0.0244 | -0.0164 | 1.0000 | no | 143 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.0406 | 0.1301 | 4 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.0210 | 0.1520 | 6 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 4 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.0383 | 0.1584 | 3 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0381 | 0.1692 | 3 |
| `rerank-bm25-100` | 42,215 | 100.0% | 35.6% | 0.1273 | 0.2625 | 4 |
| `rerank-candidates-25` | 42,215 | 100.0% | 17.1% | 0.0887 | 0.1302 | 4 |
| `rerank-candidates-50` | 42,215 | 100.0% | 24.5% | 0.0981 | 0.1982 | 4 |
| `rerank-candidates-200` | 42,215 | 100.0% | 43.1% | 0.1308 | 0.1745 | 4 |

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
| `retrieval-dense-bge` | passage vectors for bge-m3 are present but queryvectors-bge-m3.npz is not. A dense index needs both sides embedded by the same model; embedding queries with a different model would compare vectors from two spaces and return confident nonsense. Re-run the GPU notebook to produce them. |
| `retrieval-hybrid-rrf` | passage vectors for bge-m3 are present but queryvectors-bge-m3.npz is not. A dense index needs both sides embedded by the same model; embedding queries with a different model would compare vectors from two spaces and return confident nonsense. Re-run the GPU notebook to produce them. |
| `embed-e5-base` | passage vectors for e5-base are present but queryvectors-e5-base.npz is not. A dense index needs both sides embedded by the same model; embedding queries with a different model would compare vectors from two spaces and return confident nonsense. Re-run the GPU notebook to produce them. |
| `embed-finance-e5` | finance-e5 unavailable: ModuleNotFoundError: No module named 'sentence_transformers' |
| `hybrid-plus-rerank` | passage vectors for bge-m3 are present but queryvectors-bge-m3.npz is not. A dense index needs both sides embedded by the same model; embedding queries with a different model would compare vectors from two spaces and return confident nonsense. Re-run the GPU notebook to produce them. |
