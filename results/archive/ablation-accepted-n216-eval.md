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
| `rerank-candidates-50` | candidates | 0.2456 | [0.181, 0.314] | 0.5727 | 0.2128 | +0.0254 | 1.0000 | no | 110 |
| `rerank-candidates-25` | candidates | 0.2368 | [0.173, 0.303] | 0.5727 | 0.2038 | +0.0166 | 1.0000 | no | 110 |
| `rerank-bm25-100` | reranking | 0.2341 | [0.172, 0.302] | 0.5909 | 0.2054 | +0.0140 | 1.0000 | no | 110 |
| `retrieval-hybrid-rrf` | retrieval | 0.2299 | [0.167, 0.298] | 0.5636 | 0.2001 | +0.0097 | 1.0000 | no | 110 |
| `hybrid-plus-rerank` | interaction | 0.2274 | [0.167, 0.293] | 0.6000 | 0.1966 | +0.0072 | 1.0000 | no | 110 |
| `baseline-bm25-fixed512` | baseline | 0.2202 | [0.159, 0.287] | 0.5773 | 0.1979 | (baseline) | - | - | 110 |
| `chunk-struct512` | chunking | 0.2120 | [0.149, 0.280] | 0.5727 | 0.1966 | -0.0082 | 1.0000 | no | 110 |
| `retrieval-bm25-struct` | retrieval | 0.2120 | [0.149, 0.280] | 0.5727 | 0.1966 | -0.0082 | 1.0000 | no | 110 |
| `rerank-candidates-200` | candidates | 0.2080 | [0.147, 0.273] | 0.5727 | 0.1861 | -0.0121 | 1.0000 | no | 110 |
| `chunk-fixed256o32` | chunking | 0.1906 | [0.129, 0.256] | 0.4682 | 0.1805 | -0.0296 | 1.0000 | no | 110 |
| `tables-row-sentences` | table_rendering | 0.1889 | [0.132, 0.253] | 0.5030 | 0.1829 | -0.0313 | 1.0000 | no | 110 |
| `retrieval-dense-bge` | retrieval | 0.1409 | [0.088, 0.199] | 0.3455 | 0.1227 | -0.0792 | 0.3982 | no | 110 |
| `embed-e5-base` | embedding | 0.0378 | [0.014, 0.067] | 0.2000 | 0.0325 | -0.1824 | 0.0012 | yes | 110 |

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
| `chunk-struct512` | 42,215 | 100.0% | - | 0.1157 | 0.2347 | 0 |
| `tables-row-sentences` | 40,155 | 94.2% | - | 0.1253 | 0.2039 | 0 |
| `retrieval-dense-bge` | 42,215 | 100.0% | - | 0.2212 | 0.1220 | 1 |
| `retrieval-hybrid-rrf` | 42,215 | 100.0% | - | 0.2093 | 0.2347 | 2 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.1157 | 0.2347 | 1 |
| `embed-e5-base` | 42,215 | 100.0% | - | 0.0618 | 0.0321 | 1 |
| `rerank-bm25-100` | 42,215 | 100.0% | 65.7% | 0.3462 | 0.2077 | 0 |
| `rerank-candidates-25` | 42,215 | 100.0% | 48.8% | 0.2506 | 0.2336 | 0 |
| `rerank-candidates-50` | 42,215 | 100.0% | 57.6% | 0.2806 | 0.2373 | 0 |
| `rerank-candidates-200` | 42,215 | 100.0% | 78.5% | 0.3100 | 0.1840 | 1 |
| `hybrid-plus-rerank` | 42,215 | 100.0% | 64.5% | 0.3399 | 0.2008 | 2 |

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
