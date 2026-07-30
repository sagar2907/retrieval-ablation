# Retrieval ablation

5 of 8 configurations measured.

`nDCG@10` and the confidence interval are computed on the **shared subset**
of queries every configuration can judge. Comparing configurations on their
own individually-judgeable subsets would let a configuration improve its
average by failing to represent hard queries at all.

`p (Holm)` is corrected across the whole family of comparisons against the
baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a
~51% chance of at least one false positive.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | delta vs base | p (Holm) | sig | n |
|---|---|---|---|---|---|---|---|---|---|
| `baseline-bm25-fixed512` | baseline | 0.1953 | [0.143, 0.251] | 0.5070 | 0.1741 | (baseline) | - | - | 143 |
| `chunk-struct512` | chunking | 0.1884 | [0.135, 0.246] | 0.5245 | 0.1734 | -0.0069 | 1.0000 | no | 143 |
| `retrieval-bm25-struct` | retrieval | 0.1884 | [0.135, 0.246] | 0.5245 | 0.1734 | -0.0069 | 1.0000 | no | 143 |
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
| `baseline-bm25-fixed512` | 37,498 | 100.0% | - | 0.1091 | 0.2254 | 1 |
| `chunk-fixed256o32` | 75,083 | 100.0% | - | 0.1030 | 0.1932 | 1 |
| `chunk-struct512` | 42,215 | 100.0% | - | 0.0837 | 0.2249 | 1 |
| `tables-row-sentences` | 40,155 | 95.4% | - | 0.1014 | 0.1924 | 1 |
| `retrieval-bm25-struct` | 42,215 | 100.0% | - | 0.0837 | 0.2249 | 1 |

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
| `retrieval-dense-bge` | bge-m3 unavailable: ModuleNotFoundError: No module named 'sentence_transformers' |
| `retrieval-hybrid-rrf` | bge-m3 unavailable: ModuleNotFoundError: No module named 'sentence_transformers' |
