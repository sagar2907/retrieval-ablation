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
| `rerank-candidates-50` | candidates | 0.2145 | [0.160, 0.273] | 0.5245 | 0.1859 | +0.0192 | 1.0000 | no | 143 |
| `rerank-candidates-25` | candidates | 0.2103 | [0.157, 0.268] | 0.5245 | 0.1799 | +0.0150 | 1.0000 | no | 143 |
| `rerank-bm25-100` | reranking | 0.2057 | [0.153, 0.263] | 0.5385 | 0.1798 | +0.0104 | 1.0000 | no | 143 |
| `chunk-semantic95` | chunking | 0.2039 | [0.154, 0.258] | 0.6573 | 0.1697 | +0.0086 | 1.0000 | no | 143 |
| `retrieval-hybrid-rrf` | retrieval | 0.1991 | [0.146, 0.254] | 0.4965 | 0.1736 | +0.0039 | 1.0000 | no | 143 |
| `baseline-bm25-fixed512` | baseline | 0.1953 | [0.143, 0.251] | 0.5070 | 0.1741 | (baseline) | - | - | 143 |
| `chunk-struct512` | chunking | 0.1874 | [0.134, 0.245] | 0.5245 | 0.1722 | -0.0078 | 1.0000 | no | 143 |
| `retrieval-bm25-struct` | retrieval | 0.1874 | [0.134, 0.245] | 0.5245 | 0.1722 | -0.0078 | 1.0000 | no | 143 |
| `rerank-candidates-200` | candidates | 0.1854 | [0.134, 0.240] | 0.5035 | 0.1640 | -0.0099 | 1.0000 | no | 143 |
| `chunk-fixed256o32` | chunking | 0.1699 | [0.119, 0.226] | 0.4126 | 0.1604 | -0.0254 | 1.0000 | no | 143 |
| `tables-row-sentences` | table_rendering | 0.1688 | [0.122, 0.222] | 0.4935 | 0.1610 | -0.0265 | 1.0000 | no | 143 |
| `retrieval-dense-bge` | retrieval | 0.1196 | [0.077, 0.167] | 0.3077 | 0.1036 | -0.0757 | 0.1298 | no | 143 |
| `embed-e5-base` | embedding | 0.0413 | [0.019, 0.068] | 0.2168 | 0.0337 | -0.1540 | 0.0012 | yes | 143 |

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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.1091 | 0.2254 | 1 |
| `chunk-fixed256o32` | 75,084 | 100.0% | - | 0.1030 | 0.1932 | 2 |
| `chunk-semantic95` | 29,556 | 100.0% | - | 0.1254 | 0.2312 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0837 | 0.2236 | 1 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.1014 | 0.1924 | 1 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.1346 | 0.1143 | 2 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.1374 | 0.2207 | 2 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0837 | 0.2236 | 0 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0467 | 0.0394 | 1 |
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
| `embed-e5-base-v2` | no usable query vectors for e5-base-v2: queryvectors-e5-base-v2.npz is absent, or records different query text than the set being scored (check the loader warning above). A dense index needs both sides embedded by the same model on the same wording; reusing vectors from other text compares two things that were never asked and returns confident nonsense. Re-run the GPU notebook against this eval set to produce them. |
| `hybrid-plus-rerank` | no cross-encoder scores for candidates-hybrid-plus-rerank: the GPU run has not scored this configuration's shortlist. Falling back to a live reranker also failed (ModuleNotFoundError). |
