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
| `rerank-candidates-50` | candidates | 0.2145 | [0.160, 0.273] | 0.5245 | 0.1859 | +0.0192 | 1.0000 | no | 143 |
| `rerank-candidates-25` | candidates | 0.2103 | [0.157, 0.268] | 0.5245 | 0.1799 | +0.0150 | 1.0000 | no | 143 |
| `rerank-bm25-100` | reranking | 0.2057 | [0.153, 0.263] | 0.5385 | 0.1798 | +0.0104 | 1.0000 | no | 143 |
| `baseline-bm25-fixed512` | baseline | 0.1953 | [0.143, 0.251] | 0.5070 | 0.1741 | (baseline) | - | - | 143 |
| `chunk-struct512` | chunking | 0.1884 | [0.135, 0.246] | 0.5245 | 0.1734 | -0.0069 | 1.0000 | no | 143 |
| `retrieval-bm25-struct` | retrieval | 0.1884 | [0.135, 0.246] | 0.5245 | 0.1734 | -0.0069 | 1.0000 | no | 143 |
| `rerank-candidates-200` | candidates | 0.1854 | [0.134, 0.240] | 0.5035 | 0.1640 | -0.0099 | 1.0000 | no | 143 |
| `chunk-fixed256o32` | chunking | 0.1699 | [0.119, 0.226] | 0.4126 | 0.1604 | -0.0254 | 1.0000 | no | 143 |
| `tables-row-sentences` | table_rendering | 0.1688 | [0.122, 0.222] | 0.4935 | 0.1610 | -0.0265 | 1.0000 | no | 143 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.1091 | 0.2254 | 0 |
| `chunk-fixed256o32` | 75,083 | 100.0% | - | 0.1030 | 0.1932 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0837 | 0.2249 | 1 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.1014 | 0.1924 | 0 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0837 | 0.2249 | 0 |
| `rerank-bm25-100` | 42,215 | 100.0% | 61.6% | 0.2309 | 0.1969 | 1 |
| `rerank-candidates-25` | 42,215 | 100.0% | 44.4% | 0.1852 | 0.2190 | 1 |
| `rerank-candidates-50` | 42,215 | 100.0% | 53.2% | 0.1937 | 0.2218 | 1 |
| `rerank-candidates-200` | 42,215 | 100.0% | 73.6% | 0.2104 | 0.1767 | 1 |

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
