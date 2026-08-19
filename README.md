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


**15 of 15 configurations measured** on both wordings, over an eval set of 586
queries with **390 judgeable by every configuration**. Dense vectors, query
vectors, cross-encoder scores and the semantic chunk boundaries were produced on a
Kaggle T4; reproduce with `python -m retrieval_ablation.ablation.runner`.

### The headline: which retriever wins is decided by how the questions are worded

Same corpus, same gold spans, same query ids, same 390 shared queries. Only the
wording of the questions differs.

<!-- generated:headline -->
| configuration | original | paraphrased | change | Δ vs base | p (Holm) |
|---|---|---|---|---|---|
| `hybrid-plus-rerank` | 0.1869 | **0.1208** | −35% | **+0.0680** | **0.0014** ✓ |
| `rerank-bm25-100` | 0.2068 | **0.1149** | −44% | **+0.0621** | **0.0014** ✓ |
| `rerank-candidates-200` | 0.1924 | **0.1109** | −42% | **+0.0581** | **0.0014** ✓ |
| `retrieval-hybrid-rrf` | 0.1817 | **0.1022** | −44% | **+0.0494** | **0.0014** ✓ |
| `rerank-candidates-50` | 0.2198 | **0.1004** | −54% | **+0.0476** | **0.0014** ✓ |
| `rerank-candidates-25` | 0.2116 | **0.0975** | −54% | **+0.0447** | **0.0016** ✓ |
| `retrieval-dense-bge` | 0.1044 | **0.0839** | −20% | **+0.0311** | **0.0399** ✓ |
| `chunk-semantic95` | 0.2189 | 0.0663 | −70% | +0.0134 | 0.648 |
| `chunk-struct512` | 0.2135 | 0.0643 | −70% | +0.0114 | 0.648 |
| `retrieval-bm25-struct` | 0.2135 | 0.0643 | −70% | +0.0114 | 0.648 |
| `tables-row-sentences` | 0.1644 | 0.0597 | −64% | +0.0069 | 0.829 |
| `embed-e5-base-v2` | 0.1007 | 0.0506 | −50% | −0.0022 | 0.841 |
| `chunk-fixed256o32` | 0.1556 | 0.0407 | −74% | −0.0121 | 0.648 |
| `embed-e5-base` | 0.0367 | 0.0137 | −63% | **−0.0391** | **0.0014** ✓ *(worse)* |
| `baseline-bm25-fixed512` | 0.1971 | 0.0528 | −73% | — | — |
<!-- /generated:headline -->

**On the original wording, not one configuration beats the baseline
significantly.** The only significant results are three doing significantly
*worse* — every dense arm, at p = 0.0013.

**On the paraphrased wording, seven beat it significantly**: both hybrid arms,
every reranking arm, and dense retrieval.

Two configurations reverse sign entirely:

| | original | paraphrased |
|---|---|---|
| `hybrid-plus-rerank` | 0.1869, **below** baseline | **0.1208, best in the study** (+0.0680, p = 0.0014) |
| `retrieval-dense-bge` | 0.1044, **−0.0927 significant** | 0.0839, **+0.0311 significant** |

`retrieval-dense-bge` is significantly *worse* than BM25 on the original questions
and significantly *better* on the paraphrased ones — same retriever, same corpus,
same labels, opposite conclusions at the same confidence, decided entirely by
whether the questions quote the filing's own row labels.

The mechanism is the `change` column. BM25 loses 73% of its score once the
questions stop repeating their answers; dense loses 20%, because it was never
using the overlap. Roughly three quarters of what the baseline scored was the
benchmark handing it back its own words, and every semantic method was being
compared against that inflated number.

Full tables, including reachability and the overlap split:
[`results/ablation.md`](results/ablation.md) and
[`results/ablation-paraphrased.md`](results/ablation-paraphrased.md).

**Semantic chunking has the best Recall@50 in the grid** — 0.6641 against 0.5667
for the next best — on 29,556 chunks rather than 42,215, so it is not winning by
cutting the corpus finer. Its nDCG@10 is unremarkable on both wordings. It finds
the answer far more often and does not put it near the top, which is exactly what
a first stage feeding a reranker should be judged on.

**One caveat, stated because it cuts against the results.** 8 queries on the
original wording (12 paraphrased) share their exact text with another query that
has different gold — a figure repeated across two consecutive filings produces the
same question twice. Every retriever sees one string and returns one ranking, so
at most one of each pair can score. They are kept because they penalise every
configuration identically, and the runner reports them on every run.

### What the earlier, smaller benchmark showed

The eval set was 216 queries (143 shared) until it was extended to 586. Those runs
are archived under [`results/archive/`](results/archive/) and told the same story
at lower power: nothing significant on the original wording, four reranking arms
significant on the paraphrased one. Two results then sat at p = 0.050 and 0.059 —
close enough that they moved across the threshold when two *unrelated*
configurations were added and Holm corrected over a larger family.

That is the correction working rather than failing, and it is why the set was
grown. At 390 shared queries the same comparisons come back at p = 0.0013, and
nothing is balanced on the third decimal place any more.

### The real finding, which the aggregate hides

Splitting queries at 0.4 content-word overlap with the gold passage changes the
picture completely:

<!-- generated:overlap-split -->
| configuration | low-overlap nDCG | vs base | high-overlap nDCG | vs base |
|---|---|---|---|---|
| `baseline-bm25-fixed512` | 0.0823 | — | 0.2378 | — |
| `rerank-bm25-100` | 0.1682 | +104.4% | 0.2205 | −7.3% |
| `rerank-candidates-50` | 0.1680 | +104.1% | 0.2382 | +0.2% |
| `rerank-candidates-25` | 0.1664 | +102.3% | 0.2276 | −4.3% |
| `rerank-candidates-200` | 0.1564 | +90.1% | 0.2051 | −13.7% |
| `hybrid-plus-rerank` | 0.1604 | +95.0% | 0.1963 | −17.5% |
| `retrieval-hybrid-rrf` | 0.0921 | +11.9% | 0.2134 | −10.3% |
| `retrieval-dense-bge` | 0.0676 | −17.8% | 0.1174 | −50.6% |
<!-- /generated:overlap-split -->

**The cross-encoder roughly doubles performance on queries that do not share
wording with their answer, and slightly hurts the ones that do.** That is exactly
the shape you would predict: where BM25 already has an exact string match there is
nothing to fix, and reranking can only shuffle a correct top hit downward. Where
the question is phrased differently from the filing — the case that actually needs
semantic retrieval — it is worth about 2×.

A single averaged number reports around +0.01 and calls it noise. It is a large
real effect on half the queries, cancelled by a small negative one on the other
half. This is why lexical overlap is recorded per query rather than left implicit.

The dense row is the same argument from the other side: it is the only
configuration that loses *more* on high-overlap queries (−50.6%) than on
low-overlap ones (−17.8%). It gains nothing from wording that repeats the answer,
which is exactly why paraphrasing reverses its verdict.

### Candidate depth is non-monotonic, and more is worse

<!-- generated:candidate-depth -->
| depth | nDCG@10 | recall ceiling |
|---|---|---|
| 25 | 0.2116 | 46.8% |
| 50 | **0.2198** | 56.3% |
| 100 | 0.2068 | 64.2% |
| 200 | 0.1924 | 73.7% |
<!-- /generated:candidate-depth -->

Depth 200 has by far the best ceiling — the answer is in its shortlist 73.7% of
the time, against 46.8% at depth 25 — and the **worst** nDCG@10, below the
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

Re-running the whole grid on the 542 accepted labels is the robustness check that
matters:

<!-- generated:accepted-subset -->
| configuration | all labels | accepted | Δ |
|---|---|---|---|
| `rerank-candidates-50` | 0.2198 | 0.2299 | +0.0101 |
| `chunk-semantic95` | 0.2189 | 0.2258 | +0.0069 |
| `chunk-struct512` | 0.2135 | 0.2235 | +0.0100 |
| `retrieval-bm25-struct` | 0.2135 | 0.2235 | +0.0100 |
| `rerank-candidates-25` | 0.2116 | 0.2199 | +0.0083 |
| `rerank-bm25-100` | 0.2068 | 0.2157 | +0.0089 |
| `baseline-bm25-fixed512` | 0.1971 | 0.2050 | +0.0078 |
| `rerank-candidates-200` | 0.1924 | 0.2000 | +0.0076 |
<!-- /generated:accepted-subset -->

Three things to take from it. **The differences are tiny** — every configuration
moves between −0.0015 and +0.0101, far smaller than the effects being measured.
**The ranking is identical**, so no conclusion here rests on the label defects.
And the significance picture is unchanged: the same configurations are significant
on both label sets.

The audit covered the original 216 queries; the 370 added later are unaudited and
carry `GENERATED`, so "accepted" here means "not rejected by the audit that ran",
not "checked". That distinction is why the verification status is stored per query
rather than as a single project-level claim.

### Built and verified

| Component | Evidence |
|---|---|
| Corpus ingest, 120 filings | manifest with per-document SHA-256 of raw bytes and parsed text |
| Table-aware HTML parsing | all 4 Parts, 23 Items, 13 Notes recovered from a real filing; byte-identical reparse |
| Three chunkers | 66 tests; span-slices-to-text invariant asserted on real documents |
| Retrieval metrics (nDCG/Recall/MRR) | 29 tests against hand-computed values |
| Statistics (bootstrap CI, paired permutation, Holm) | 25 tests, determinism asserted under fixed seeds |
| BM25, dense, RRF fusion, reranking wiring | 66 tests, offline with fakes |
| Eval set, 586 queries | every gold passage verified to contain the value its query asks for |
| Ablation runner, 15 configurations on both wordings | numbers above, on the full corpus |
| Cross-encoder reranking arms | Kaggle T4: 43,200 pairs scored, 46.6 pairs/s; scores committed |
| Dense and hybrid arms | Kaggle T4: 42,215 passage + 586 query vectors per model per wording, aligned by id with 0 orphans |
| GPU embeddings (BGE-M3, E5-base) | Kaggle T4: 49.8 and 143.3 chunks/s; model commit hashes recorded |
| Model-assisted label audit | 216 of the 586 labels rechecked, 44 rejected; conclusions unchanged on the accepted subset |
| Learning document + PDF renderer | every page rasterised and round-tripped through a text extractor |
| Gemini client: cached, quota-tolerant, token-accounted | live run; 19 calls, 15 rate-limited, resumed from cache |
| Generation + long-context comparison | numbers above, from the API's own reported token counts; cost only — latency is reported as not measured, because the two arms' timings came from different sessions |
| FastAPI service, Docker, citation UI | live run: index 31.7 s, /search 1.9 ms, /answer 429 path verified |

**543 tests pass, offline, with no API key and no model download.** `ruff` clean.

### Not done

| Missing | Why | What unblocks it |
|---|---|---|
| Faithfulness at a usable sample size | 23 judged answers, all faithful — enough to show the pass works, still too few to quote as a rate | free-tier quota for more answers; the judging itself is cheap |
| Faithfulness of the long-context arm | its context is a whole filing, ~130k tokens per judgement against ~7.5k for a retrieval answer | `--judge-long-context`, on a paid tier |
| Generation + long-context at full sample | 46 retrieval answers and 11 long-context ones, of 586 queries. The arms have separate budgets now, so the cheap one is no longer capped by the expensive one, but both are still quota-bound | free-tier quota, or re-run tomorrow; the run resumes from cache rather than restarting |
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

The results tables in this file and in `docs/learning.md` are generated from
`results/*.json`, not typed. After a re-run, regenerate them and re-read the prose
around each one — a regenerated table does not fix a sentence drawing the wrong
conclusion from it:

```bash
uv run python scripts/render_tables.py && uv run python scripts/render_pdf.py
```

The sentences *around* those tables are not generated, so a third check reads every
figure quoted in prose and confirms it appears somewhere in `results/`:

```bash
uv run python scripts/audit_figures.py
```

CI fails if any of them is stale: the suite runs `render_tables.py --check` and the
figure audit, and a workflow step re-renders the PDF and compares its extracted text
against the committed one.

## Reproducibility

- Every sampling decision derives from `GLOBAL_SEED` in `config.py`. Nothing that
  affects data reads the clock or the global RNG.
- The raw corpus is gitignored but reproducible from the checksummed manifest in
  `data/manifests/corpus.json`, which records the SHA-256 of both the raw bytes
  and the parsed text of all 120 documents.
- Query ids are content-addressed on (document, row label, period) — not on
  character offsets — so the same fact keeps its id when table rendering changes
  the document text.
- Two runs of the ablation produce **identical metrics, confidence intervals and
  significance results, and a different file**. The difference is the per-row
  `seconds` field, which is a wall-clock measurement of this machine rather than a
  property of the experiment. Verified by running the grid twice and diffing:
  zero substantive differences, `significance_vs_baseline` identical. Worth
  stating precisely, because "deterministic" and "byte-identical output" are not
  the same claim and only the first one is true here.

## Licence

MIT. SEC filings are US public domain.
