# retrieval-ablation

A labelled retrieval benchmark over SEC filings, and a single-axis ablation study
measuring what each component of a retrieval pipeline is actually worth.

The point of this repository is not the demo. It is the evidence: retrieval
quality is measured independently of generation quality, against a set of queries
with known gold passages, with confidence intervals and significance testing on
every comparison — and with the confounds named and measured rather than hidden.

## Why measure retrieval separately

A question-answering system over documents does two things: it finds passages,
then it writes an answer from them. Most published RAG projects measure only the
second step. That conflates two failure modes with completely different fixes — if
retrieval returned the wrong passages, no amount of prompt engineering recovers
the answer.

Separating them requires a labelled set: queries paired with the passage that
actually contains the answer. With that, retrieval is scored with nDCG@10,
Recall@50 and MRR, and no language model is involved at all — so the retrieval
ablation is both free to run and fully deterministic.

## Corpus

**120 SEC 10-K filings: 30 companies × 4 consecutive fiscal years, 68.8M
characters (~17M tokens, ~27,500 pages), 14,537 extracted tables.**

Four consecutive years per company is deliberate and load-bearing. Annual reports
repeat their structure almost verbatim while the figures change, so every
year-specific question has three near-identical distractors. A single-year corpus
would make retrieval far too easy and flatter every configuration equally.

Filings are HTML, not PDF. This project parses the original HTML directly rather
than converting to PDF first — round-tripping through PDF layout analysis destroys
exactly the table structure being tested.

## Measured results

**13 of 15 configurations measured.** Full corpus, 42,215 chunks, 143 queries
judgeable by every configuration. Dense vectors and cross-encoder scores produced
on a Kaggle T4; reproduce with `python -m retrieval_ablation.ablation.runner`.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | Δ vs base | p (Holm) |
|---|---|---|---|---|---|---|---|
| `rerank-candidates-50` | candidates | **0.2145** | [0.160, 0.273] | 0.5245 | 0.1859 | +0.0192 | 1.000 |
| `rerank-candidates-25` | candidates | 0.2103 | [0.157, 0.268] | 0.5245 | 0.1799 | +0.0150 | 1.000 |
| `rerank-bm25-100` | reranking | 0.2057 | [0.153, 0.263] | 0.5385 | 0.1798 | +0.0104 | 1.000 |
| `hybrid-plus-rerank` | interaction | 0.2003 | [0.148, 0.256] | **0.5455** | 0.1729 | +0.0050 | 1.000 |
| `retrieval-hybrid-rrf` | retrieval | 0.1991 | [0.146, 0.254] | 0.4965 | 0.1736 | +0.0039 | 1.000 |
| `baseline-bm25-fixed512` | baseline | 0.1953 | [0.143, 0.251] | 0.5070 | 0.1741 | — | — |
| `chunk-struct512` | chunking | 0.1874 | [0.134, 0.245] | 0.5245 | 0.1722 | −0.0078 | 1.000 |
| `rerank-candidates-200` | candidates | 0.1854 | [0.134, 0.240] | 0.5035 | 0.1640 | −0.0099 | 1.000 |
| `chunk-fixed256o32` | chunking | 0.1699 | [0.119, 0.226] | 0.4126 | 0.1604 | −0.0254 | 1.000 |
| `tables-row-sentences` | table rendering | 0.1688 | [0.122, 0.222] | 0.4935 | 0.1610 | −0.0265 | 1.000 |
| `retrieval-dense-bge` | retrieval | 0.1196 | [0.077, 0.167] | 0.3077 | 0.1036 | −0.0757 | 0.130 |
| `embed-e5-base` | embedding | 0.0413 | [0.019, 0.068] | 0.2168 | 0.0337 | −0.1540 | **0.001** |

Full table with reachability and the overlap split:
[`results/ablation.md`](results/ablation.md).

### The headline: which retriever wins depends entirely on how the questions are worded

Rewriting the questions so they stop quoting the filing's own row labels — same
corpus, same gold spans, same query ids, same 143 shared queries, only the wording
different — reverses the ranking of the two retrieval families.

Both wordings now measure **13 of 15 configurations**.

| configuration | original | paraphrased | change | Δ vs base | p (Holm) |
|---|---|---|---|---|---|
| `rerank-candidates-200` | 0.1854 | **0.1210** | −35% | +0.0735 | **0.0050** ✓ |
| `hybrid-plus-rerank` | 0.2003 | **0.1210** | −40% | +0.0735 | **0.0044** ✓ |
| `rerank-bm25-100` | 0.2057 | **0.1178** | −43% | +0.0703 | **0.0099** ✓ |
| `retrieval-hybrid-rrf` | 0.1991 | **0.1088** | −45% | +0.0613 | **0.0036** ✓ |
| `rerank-candidates-50` | 0.2145 | **0.1047** | −51% | +0.0572 | **0.0160** ✓ |
| `retrieval-dense-bge` | 0.1196 | **0.0991** | **−17%** | +0.0516 | **0.0444** ✓ |
| `rerank-candidates-25` | 0.2103 | **0.0963** | −54% | +0.0488 | **0.0392** ✓ |
| `chunk-struct512` | 0.1874 | 0.0482 | −74% | +0.0008 | 1.000 |
| `baseline-bm25-fixed512` | 0.1953 | 0.0475 | **−76%** | — | — |
| `chunk-fixed256o32` | 0.1699 | 0.0311 | −82% | −0.0164 | 1.000 |
| `embed-e5-base` | 0.0413 | 0.0223 | −46% | −0.0252 | 0.449 |

On the original wording **nothing was a significant improvement**. On the
paraphrased wording **seven configurations are** — every reranking arm, both
hybrid arms, and dense retrieval.

The mechanism is the `change` column. BM25 loses 76% of its score once the
questions stop quoting their answers. Dense loses 17%, because it was never using
the overlap. Roughly three quarters of what the baseline scored was the benchmark
handing it back its own words, and every semantic method was being measured
against that inflated number.

Two conclusions from the original wording do not survive. `embed-e5-base` scoring
significantly *below* baseline (p = 0.001) was the single significant result
there; on paraphrased queries it is not significant (p = 0.449). And the
candidate-depth ordering **inverts** — depth 200 was the worst reranking
configuration on the original wording and is the best here, which is what a
reranker doing real work should do with a deeper shortlist.

What does *not* change is the chunking and table-rendering axes: null on both
wordings. The confound was specific rather than a haze over everything, which is
exactly why it survived so long — most of the table looked stable.

Mean lexical overlap falls 0.4613 → 0.1684; the high-overlap bucket goes from 158
queries to 17. Every artifact behind the paraphrased column records the query text
it was computed against, and the loaders verified all 216 matched before scoring.
An earlier version of this table reported reranking as significant while reusing
scores computed from the original wording; that result was retracted, and these
numbers come from a separate GPU run against the paraphrased shortlists.

**The reranking and dense arms are not measured on the paraphrased queries**, and
an earlier version of this section reported them as significant improvements. That
was wrong and is retracted. Both arms are served from precomputed GPU artifacts
keyed by query id, and query ids deliberately survive a rewrite of the query text —
that is what makes the two eval sets comparable. So the paraphrased run silently
reused cross-encoder scores and query vectors computed from the *original*
wording, which is precisely the lexical overlap the paraphrasing existed to
remove. The reranker was being handed the answer's own words while being credited
for finding them.

Both artifacts now record the text they were computed against, and the loaders
refuse anything they cannot tie to the queries being scored. Measuring those arms
needs one GPU run against `data/eval/queries-paraphrased.jsonl`.

The lesson stands independently of the numbers, and is sharper for having caught
this the hard way: a benchmark that systematically advantages one arm will report
that arm winning, with tight confidence intervals and a multiple-comparison
correction applied conscientiously to the wrong numbers. No statistical machinery
in this project detected either problem, because none of it was wrong.

### On the original queries, the only significant result was a configuration doing worse

Reranking takes the top four slots, and the brief predicted a "large jump" from
the cross-encoder. **No improvement over the baseline is statistically
significant.** The best configuration beats the baseline by +0.0192 with a
Holm-corrected p of 1.000.

Exactly one comparison in the grid survives correction, and it is
`embed-e5-base` scoring **0.1540 *below*** the baseline at p = 0.001. Nothing was
shown to help; one thing was shown to hurt.

With 143 queries the 95% CI on nDCG@10 spans roughly ±0.055, so a 0.019 gap is
well inside the noise floor while a 0.154 gap is not. Reporting "reranking lifted
nDCG@10 from 0.195 to 0.215" would be true arithmetic and a false finding. The
statistics module exists precisely to stop that, and here it earned its place by
refusing the result the project was set up to produce while passing the one
nobody wanted.

### Dense retrieval loses — and the overlap split says why

Both dense arms land far below every lexical configuration. Before reading that
as "embeddings are bad at finance", look at where each configuration's score
comes from:

| configuration | nDCG low-overlap | nDCG high-overlap | direction |
|---|---|---|---|
| `baseline-bm25-fixed512` | 0.1091 | 0.2254 | **+107%** high |
| `chunk-struct512` | 0.0837 | 0.2236 | +167% high |
| `retrieval-hybrid-rrf` | 0.1374 | 0.2207 | +61% high |
| `retrieval-dense-bge` | **0.1346** | **0.1143** | **−15%** — *inverted* |
| `embed-e5-base` | 0.0467 | 0.0394 | −16% — *inverted* |

`retrieval-dense-bge` is the **only** configuration whose low-overlap score beats
its high-overlap score, and `embed-e5-base` is the only other one close to flat.
Every lexical configuration roughly doubles when the query shares wording with the
answer, because that is what BM25 matches on. The dense arms do not, so they
neither gain from the overlap nor lose from its absence.

That matters because this benchmark's queries are generated from table rows and
reuse the document's own row labels, handing BM25 an exact string match on most of
them. The dense arms are being scored on a benchmark whose construction favours
their competitor. The defensible claim is **not** "BM25 beats dense on SEC filings"
but "BM25 beats dense on queries phrased in the filing's own words" — and on the
low-overlap subset, `retrieval-dense-bge` (0.1346) beats the baseline (0.1091) and
`retrieval-hybrid-rrf` (0.1374) beats everything except reranking.

Query paraphrasing is the fix, it is not done, and until it is the dense arms'
headline numbers are a lower bound. The next section shows the same split doing
the same work for reranking.

### The real finding, which the aggregate hides

Splitting queries at 0.4 content-word overlap with the gold passage changes the
picture completely:

| configuration | low-overlap nDCG | vs base | high-overlap nDCG | vs base |
|---|---|---|---|---|
| `baseline-bm25-fixed512` | 0.1091 | — | 0.2254 | — |
| `rerank-bm25-100` | **0.2309** | **+111.7%** | 0.1969 | −12.6% |
| `rerank-candidates-200` | 0.2104 | +92.9% | 0.1767 | −21.6% |
| `rerank-candidates-50` | 0.1937 | +77.6% | 0.2218 | −1.6% |
| `rerank-candidates-25` | 0.1852 | +69.8% | 0.2190 | −2.8% |

**The cross-encoder roughly doubles performance on queries that do not share
wording with their answer, and slightly hurts the ones that do.** That is exactly
the shape you would predict: where BM25 already has an exact string match there is
nothing to fix, and reranking can only shuffle a correct top hit downward. Where
the question is phrased differently from the filing — the case that actually needs
semantic retrieval — it is worth about 2×.

A single averaged number reports +0.019 and calls it noise. It is a large real
effect on half the queries, cancelled by a small negative effect on the other
half. This is why lexical overlap is recorded per query rather than left implicit.

### Candidate depth is non-monotonic, and more is worse

| depth | nDCG@10 | recall ceiling |
|---|---|---|
| 25 | 0.2103 | 44.4% |
| 50 | **0.2145** | 53.2% |
| 100 | 0.2057 | 61.6% |
| 200 | 0.1854 | **73.6%** |

Depth 200 has by far the best ceiling — the answer is in its shortlist 73.6% of
the time, against 44.4% at depth 25 — and the **worst** nDCG@10, below the
baseline that does no reranking at all. So the cross-encoder is not failing to
see the answer; given more candidates it actively promotes wrong ones past it.
The optimum is around 50, and buying a better ceiling past that point costs
accuracy as well as compute.

That is worth stating plainly because the intuitive tuning move — widen the
shortlist so the reranker has more to work with — measurably makes things worse
here.

### What these numbers mean, and what they do not

**The absolute values are low, and that is a real result rather than a bug.** The
project brief assumed a lexical baseline near 0.41. Measured here it is 0.195. Two
diagnosed causes, both verified by tracing individual queries:

1. **Year confusion accounts for 48% of top-10 failures.** Of 88 queries missed at
   top-10, 42 had a top hit that was the *same row from the same company in a
   different fiscal year*. This is the corpus doing what it was designed to do.

2. **Name-dense boilerplate outranks the answer.** Templated queries name the
   company, and exhibit indexes and legal-proceedings sections repeat the company
   name dozens of times in one chunk. BM25's term frequency rewards that, so an
   exhibit list beats the financial statement holding the figure. Classic
   term-frequency gaming, and precisely the failure a cross-encoder should fix.

Both are headroom the reranking and hybrid arms are meant to close. That makes the
ablation more informative than a high baseline would, but it also means **these
numbers are not comparable to published benchmarks** with easier corpora.

**Smaller chunks are clearly worse** (0.1699 vs 0.1953) on both nDCG and Recall.
**Structure-aware chunking improves Recall@50** (0.5245 vs 0.5070, the best
measured) while slightly lowering nDCG@10 — it finds the answer more often but not
higher up. **Row-sentence table rendering is worse than pipe tables** on this
corpus, which contradicts the intuition that repeating column headers next to
every value helps.

The GPU arms have since landed and the Holm-corrected tests run over the whole
grid. Every one of these chunking and rendering deltas comes back with a corrected
p of 1.000, so **none of them is a finding** — they are the shape of the noise at
n = 143. The one comparison that does survive correction is `embed-e5-base`
scoring below the baseline.

### Label quality, and whether the conclusions survive it

All 216 labels were audited by `gemini-3.5-flash-lite`, which was asked to reject
on specific checkable grounds rather than give a quality score. **44 of 216 were
rejected — a 20.4% rejection rate**, with 0 unparseable verdicts.

The rejections are concentrated in exactly the failure modes visible by hand
earlier: the subject of the question is a place or entity name rather than a line
item (*"united states"*, *"duke energy ohio"*), or the passage contains several
figures under the same label so the question is ambiguous (*"Expected life in
years" 4.6, 4.6, …*). Rejected labels are retained with their reason rather than
deleted, so the size of the discarded set stays visible.

**These are `MODEL_CHECKED`, not `HUMAN_VERIFIED`, and a test enforces that.** A
model auditing labels a program generated from table structure is not an
independent second opinion — both can be satisfied by a query that is grammatical
and meaningless — and it cannot see that a *different* passage would have been the
better gold. It is a bulk quality signal, not a substitute for a person.

Re-running the whole grid on the 172 accepted labels is the robustness check that
matters:

| configuration | all 216 | accepted 172 | Δ |
|---|---|---|---|
| `rerank-candidates-50` | 0.2145 | 0.2456 | +0.0311 |
| `rerank-bm25-100` | 0.2057 | 0.2341 | +0.0284 |
| `rerank-candidates-25` | 0.2103 | 0.2368 | +0.0265 |
| `baseline-bm25-fixed512` | 0.1953 | 0.2202 | +0.0249 |
| `chunk-struct512` | 0.1874 | 0.2120 | +0.0246 |
| `rerank-candidates-200` | 0.1854 | 0.2080 | +0.0226 |
| `tables-row-sentences` | 0.1688 | 0.1889 | +0.0201 |

Three things to take from it. **Twelve of thirteen configurations gain between
+0.020 and +0.031**, which is what you would expect if the rejected labels were
genuinely unanswerable — they penalised every system about equally. The exception
is `embed-e5-base`, which *loses* 0.0035: it was not being held back by bad
labels, and removing them does not rescue it.

**The ranking is stable but not identical.** The top three and the bottom four are
unchanged; `retrieval-hybrid-rrf` and `hybrid-plus-rerank` swap places at ranks 4
and 5, on a gap of 0.0025 — which is noise, and is exactly why neither is called a
finding. No conclusion here rests on the label defects.

And **the significance picture is unchanged**: no improvement survives Holm
correction on either label set, and `embed-e5-base` remains the single significant
comparison at p = 0.001 on both. Neither the null result nor the one real effect
is an artifact of noisy labels.

The overlap split gets *stronger* on the cleaner subset — `rerank-bm25-100` goes
from +112% to **+171%** on low-overlap queries while its high-overlap penalty
deepens from −12.6% to −14.2%. Removing ambiguous labels sharpens the effect rather
than dissolving it.

Full comparison: [`results/ablation-accepted.md`](results/ablation-accepted.md)
and [`data/eval/model_check.json`](data/eval/model_check.json).

## Retrieval versus long context — measured

`gemini-3.6-flash`, 12 of 216 queries (seeded stratified sample), full details in
[`results/generation.md`](results/generation.md).

| | retrieval (top-10) | long context (whole filing) | ratio |
|---|---|---|---|
| mean prompt tokens | 7,345 | 130,701 | 17.8× |
| cost per query | $0.011224 | $0.196322 | **17.5×** |
| p95 latency | 4.54 s | 8.54 s | 1.9× |
| value accuracy, answered | 0.600 | 0.556 | — |
| value accuracy, all queries | 0.300 | **0.556** | — |
| refusal rate | **5 of 10** | 0 of 9 | — |
| citation precision / recall | 0.567 / 0.800 | n/a by construction | — |

**The brief's "roughly 1,250× cheaper" is not reproducible. Measured: 17.5×.**
1,250× requires assuming a full 1M-token context, an ~800-token retrieval prompt,
and zero output cost. The brief's own draft résumé bullet says 1/40th, which is far
closer to what this measures.

**On this corpus, long context currently wins on accuracy** — 0.556 against 0.300
over all queries. The reason is visible in the refusal column: retrieval declined
to answer half the questions, because with nDCG@10 at 0.195 the answer often was
not in its top-10. When it *did* answer it was slightly more accurate (0.600 vs
0.556) and it cited its sources, which the long-context arm structurally cannot.

So the honest reading is: retrieval is 17.5× cheaper and 1.9× faster, and loses on
accuracy today because its first stage is weak — exactly the gap the hybrid and
reranking arms exist to close. That is a claim the completed grid can test, not one
to assert now.

Two caveats stated because they cut against the result:

- **Long context is handed the correct filing**; retrieval must find it among 120.
  The baseline is deliberately generous, so retrieval's cost and latency wins hold
  despite the comparison being stacked against it.
- **12 queries is a small sample** and the run was cut short by the free-tier daily
  quota (19 live calls, 15 rate-limited responses, 712 s spent waiting). Treat the
  accuracy figures as indicative; the cost and token ratios are solid because they
  come from the API's own reported token counts.

## Honest status

Nothing below is called verified unless it was run and its output inspected.

### Built and verified

| Component | Evidence |
|---|---|
| Corpus ingest, 120 filings | manifest with per-document SHA-256 of raw bytes and parsed text |
| Table-aware HTML parsing | all 4 Parts, 23 Items, 13 Notes recovered from a real filing; byte-identical reparse |
| Three chunkers | 66 tests; span-slices-to-text invariant asserted on real documents |
| Retrieval metrics (nDCG/Recall/MRR) | 29 tests against hand-computed values |
| Statistics (bootstrap CI, paired permutation, Holm) | 25 tests, determinism asserted under fixed seeds |
| BM25, dense, RRF fusion, reranking wiring | 66 tests, offline with fakes |
| Eval set, 216 queries | every gold passage verified to contain the value its query asks for |
| Ablation runner, 13 configurations | numbers above, on the full corpus |
| Cross-encoder reranking arms | Kaggle T4: 43,200 pairs scored, 46.6 pairs/s; scores committed |
| Dense and hybrid arms | Kaggle T4: 42,215 passage + 216 query vectors per model, aligned by id with 0 orphans |
| GPU embeddings (BGE-M3, E5-base) | Kaggle T4: 49.8 and 143.3 chunks/s; model commit hashes recorded |
| Model-assisted label audit | 216 labels rechecked, 44 rejected; conclusions unchanged on the accepted subset |
| Learning document + PDF renderer | every page rasterised and round-tripped through a text extractor |
| Gemini client: cached, quota-tolerant, token-accounted | live run; 19 calls, 15 rate-limited, resumed from cache |
| Generation + long-context comparison | numbers above, from the API's own reported token counts |
| FastAPI service, Docker, citation UI | live run: index 31.7 s, /search 1.9 ms, /answer 429 path verified |

**429 tests pass, offline, with no API key and no model download.** `ruff` clean.

### Not done

| Missing | Why | What unblocks it |
|---|---|---|
| `chunk-semantic95` | needs live embeddings — the sentences it embeds do not exist until it has already run, so there is no precomputed substitute; the GPU worker does not currently emit them | extend `notebooks/kaggle_gpu_arms.py` to run the semantic chunker on the GPU and ship its chunk boundaries |
| `embed-finance-e5` | the GPU run embedded BGE-M3 and E5-base only | add `finance-e5` to `EMBEDDING_JOBS` and re-run the notebook |
| Dense / hybrid arms **on the paraphrased queries** | the GPU run embedded the original wording. Query vectors are only valid for the text they were built from, and the loader now refuses to reuse them across a rewrite | one Kaggle session against `data/eval/queries-paraphrased.jsonl`; the notebook records `query_texts` so a mismatch can never pass silently again |
| Faithfulness judging | run was cut short by the daily quota before the judge pass | free-tier quota, or re-run tomorrow |
| Generation + long-context at full sample | 12 of 216 queries measured; quota-bound | free-tier quota, or re-run tomorrow |
| Human verification of eval labels | requires a person; the model-assisted pass is labelled `MODEL_CHECKED`, never `HUMAN_VERIFIED` | fill in `data/eval/verification_sample.md` |

### The parse was not reproducible across machines

Two of 120 documents disagreed between this machine and the GPU worker. Re-fetching
both from EDGAR showed their **raw bytes byte-identical** to the committed manifest,
last modified in 2023. The filings had not changed; the parser had.

| Filing | Symptom | Cause |
|---|---|---|
| `msft-10-k-2023-06-30` | same 357,277 chars, different digest | `&#149;` × 5. HTML5 reinterprets numeric references in `0x80–0x9F` through Windows-1252, so `&#149;` is `•`. Older libxml2 emits `U+0095` literally. Length-preserving, so no size check could see it. |
| `so-10-k-2022-12-31` | 1,216,695 → 1,217,055 chars | libxml2 applies an internal size ceiling to a 19.6 MB document. When it trips, libxml2 does not raise — it stops adding nodes. The filing silently lost 3 spans, 5 divs, 3 brs and its closing paragraph. |

Both are fixed in `corpus/html_parse.py`: the C1 range is mapped explicitly so the
result is identical whichever libxml2 is installed, and every parse now sets
`huge_tree`. The local corpus was rebuilt, and it now reproduces the GPU worker's
output exactly. **The vector artifacts align perfectly — 0 chunks without a vector,
0 vectors without a chunk**, where previously one chunk of 42,215 was orphaned.

Two things are worth recording. Deleting `data/interim/` is required for a parser
fix to take effect at all — the first rebuild reported "0 documents changed" purely
because it reused the cached parse. And the corpus was only 360 characters short in
68.8M, but that was luck: the same ceiling silently truncates *any* document past
it, and nothing in the run reports it.

The eval set survived: all 216 gold spans still resolve, Microsoft has none, and
Southern's single span sits at offset 287,704 — far below the tail that changed.

Configurations that could not run are recorded in `results/ablation.json` with
`"measured": false` and a stated reason. **No number is invented for them**, and
there is a test pinning that behaviour.

### The GPU blocker, in detail

Windows **Smart App Control is enforced** on the development machine
(`VerifiedAndReputablePolicyState = 1`). PyTorch ships unsigned native libraries,
so importing it from Windows Python fails with `WinError 4551`, and the Code
Integrity event log names `torch_cpu.dll` — meaning **CPU-only PyTorch is blocked
too**, not just the CUDA build.

Disabling Smart App Control is not an acceptable fix: it is a machine-wide
security control that cannot be re-enabled without reinstalling Windows.

WSL2 is the supported route and was set up ([`scripts/setup_wsl_gpu.sh`](scripts/setup_wsl_gpu.sh)).
Its Linux userspace is not governed by the Windows user-mode code-integrity
policy, and the NVIDIA WSL driver exposes the same RTX 4050 — verified working:
`nvidia-smi` reports the GPU with 6,141 MiB inside WSL. The remaining obstacle is
purely network: the Linux torch build resolves CUDA runtime libraries as separate
multi-gigabyte `nvidia-*` wheels from `pypi.nvidia.com`, and those downloads time
out repeatedly (6 retry rounds, failing on `nvidia-nccl-cu12`).

To finish the GPU arms:

```bash
wsl -d Ubuntu -- bash scripts/setup_wsl_gpu.sh
```

then re-run the ablation from inside WSL. Everything else — tests, lint, git, the
lexical arms — runs natively on Windows.

### Known limitations, stated up front

- **Eval labels are generated, not human-verified.** They are mechanically correct
  by construction — every gold passage provably contains the value asked for — but
  nobody has confirmed the queries read naturally or that the labelled span is what
  a person would cite. `data/eval/verification_sample.md` exists to change that.
  The schema tracks this per query and a test asserts labels are marked
  `GENERATED`.
- **Queries are templated, so they reuse the corpus's wording.** Median content-word
  overlap with the gold passage is 0.46 (range 0.22–0.88). This hands a lexical
  matcher an exact string match, so overlap is recorded per query and results are
  split at 0.4 into low- and high-overlap subsets. A configuration whose advantage
  exists only in the high-overlap column is winning at string matching.
- **Some row labels make awkward queries.** Segment tables yield geography labels
  ("united states") and actuarial tables yield fragments ("expected life in
  years"). Company-name rows and footnote digits are filtered; these are not yet.
- **Token counting is approximate** (characters ÷ 4) rather than a real tokenizer.
  Every configuration uses the same counter, so chunking comparisons are valid,
  but boundaries differ slightly from a model's own tokenisation.
- **The grid varies one axis at a time and cannot detect interactions.** One crossed
  cell (hybrid + reranking) is run explicitly because that interaction is the
  study's headline claim.

## Running the service

```bash
docker compose up --build      # then open http://localhost:8000
```

Or without Docker:

```bash
uv pip install -e ".[service]" && uv run python -m uvicorn retrieval_ablation.service.app:app
```

Verified end to end against the real corpus: the index builds in **31.7 s** over
**42,215 chunks from 120 filings**, and `/search` returns in **1.9 ms**.

Three routes. `/search` needs no API key and no quota, so the retrieval half is
always usable. `/answer` adds a generated answer with numbered citations, and
returns **HTTP 429 with an actionable message** when the free-tier quota is spent
rather than silently degrading — verified by exhausting the quota and observing it.
`/health` reports what is actually being served, including a note that the first
stage is lexical only, so a demo answer is not mistaken for the project''s best
configuration.

The UI exists for inspection rather than polish: every citation in the answer is a
button that scrolls to and highlights the passage it refers to, out-of-range
citations are flagged in a warning colour instead of dropped, and every passage
shows its rank, raw score, relative score bar, section path in the filing, and
character offsets. A wrong answer should be diagnosable from the page without
opening a terminal. It is a single self-contained document with no CDN asset, so
it works offline and under a strict content policy.

## Development

Python 3.12+. Default install is CPU-only with no API dependencies, so the full
test suite runs offline.

```bash
uv venv && uv pip install -e ".[dev]"
```

```bash
uv run python -m ruff check . && uv run python -m ruff format --check . && uv run python -m pytest
```

Rebuild everything from scratch:

```bash
uv run python -m retrieval_ablation.corpus.ingest && uv run python -m retrieval_ablation.evalset.build && uv run python -m retrieval_ablation.ablation.runner
```

## Reproducibility

- Every sampling decision derives from `GLOBAL_SEED` in `config.py`. Nothing that
  affects data reads the clock or the global RNG.
- The raw corpus is gitignored but reproducible from the checksummed manifest in
  `data/manifests/corpus.json`, which records the SHA-256 of both the raw bytes
  and the parsed text of all 120 documents.
- Query ids are content-addressed on (document, row label, period) — not on
  character offsets — so the same fact keeps its id when table rendering changes
  the document text.

## Licence

MIT. SEC filings are US public domain.
