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
| `rerank-candidates-50` | candidates | 0.2456 | [0.181, 0.314] | 0.5727 | 0.2128 | +0.0254 | 1.0000 | no | 110 |
| `rerank-candidates-25` | candidates | 0.2368 | [0.173, 0.303] | 0.5727 | 0.2038 | +0.0166 | 1.0000 | no | 110 |
| `rerank-bm25-100` | reranking | 0.2341 | [0.172, 0.302] | 0.5909 | 0.2054 | +0.0140 | 1.0000 | no | 110 |
| `baseline-bm25-fixed512` | baseline | 0.2202 | [0.159, 0.287] | 0.5773 | 0.1979 | (baseline) | - | - | 110 |
| `chunk-struct512` | chunking | 0.2120 | [0.149, 0.280] | 0.5727 | 0.1966 | -0.0082 | 1.0000 | no | 110 |
| `retrieval-bm25-struct` | retrieval | 0.2120 | [0.149, 0.280] | 0.5727 | 0.1966 | -0.0082 | 1.0000 | no | 110 |
| `rerank-candidates-200` | candidates | 0.2080 | [0.147, 0.273] | 0.5727 | 0.1861 | -0.0121 | 1.0000 | no | 110 |
| `chunk-fixed256o32` | chunking | 0.1906 | [0.129, 0.256] | 0.4682 | 0.1805 | -0.0296 | 1.0000 | no | 110 |
| `tables-row-sentences` | table_rendering | 0.1889 | [0.132, 0.253] | 0.5030 | 0.1829 | -0.0313 | 1.0000 | no | 110 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.1276 | 0.2420 | 1 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.1038 | 0.2110 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.1157 | 0.2347 | 1 |
| `tables-row-sentences` | 40,155 | 94.2% | - | 0.1253 | 0.2039 | 1 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.1157 | 0.2347 | 1 |
| `rerank-bm25-100` | 42,215 | 100.0% | 65.7% | 0.3462 | 0.2077 | 1 |
| `rerank-candidates-25` | 42,215 | 100.0% | 48.8% | 0.2506 | 0.2336 | 1 |
| `rerank-candidates-50` | 42,215 | 100.0% | 57.6% | 0.2806 | 0.2373 | 1 |
| `rerank-candidates-200` | 42,215 | 100.0% | 78.5% | 0.3100 | 0.1840 | 1 |

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
