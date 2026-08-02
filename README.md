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

**9 of 15 configurations measured.** Full corpus, 42,215 chunks, 143 queries
judgeable by every configuration. Dense vectors and cross-encoder scores produced
on a Kaggle T4; reproduce with `python -m retrieval_ablation.ablation.runner`.

| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR | Δ vs base | p (Holm) |
|---|---|---|---|---|---|---|---|
| `rerank-candidates-50` | candidates | **0.2145** | [0.160, 0.273] | 0.5245 | 0.1859 | +0.0192 | 1.000 |
| `rerank-candidates-25` | candidates | 0.2103 | [0.157, 0.268] | 0.5245 | 0.1799 | +0.0150 | 1.000 |
| `rerank-bm25-100` | reranking | 0.2057 | [0.153, 0.263] | **0.5385** | 0.1798 | +0.0104 | 1.000 |
| `baseline-bm25-fixed512` | baseline | 0.1953 | [0.143, 0.251] | 0.5070 | 0.1741 | — | — |
| `chunk-struct512` | chunking | 0.1884 | [0.135, 0.246] | 0.5245 | 0.1734 | −0.0069 | 1.000 |
| `rerank-candidates-200` | candidates | 0.1854 | [0.134, 0.240] | 0.5035 | 0.1640 | −0.0099 | 1.000 |
| `chunk-fixed256o32` | chunking | 0.1699 | [0.119, 0.226] | 0.4126 | 0.1604 | −0.0254 | 1.000 |
| `tables-row-sentences` | table rendering | 0.1688 | [0.122, 0.222] | 0.4935 | 0.1610 | −0.0265 | 1.000 |

Full table with reachability and the overlap split:
[`results/ablation.md`](results/ablation.md).

### The headline: no difference here is statistically significant

Reranking takes the top three slots, and the brief predicted a "large jump" from
the cross-encoder. **Zero of eight comparisons survive Holm correction.** The
smallest *uncorrected* p-value in the entire grid is 0.256; the best configuration
beats the baseline by +0.0192 with a raw p of 0.538.

With 143 queries the 95% CI on nDCG@10 spans roughly ±0.055, so a 0.019 gap is
well inside the noise floor. Reporting "reranking lifted nDCG@10 from 0.195 to
0.215" would be true arithmetic and a false finding. The statistics module exists
precisely to stop that, and here it earned its place by refusing the result the
project was set up to produce.

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

None of these differences has been checked for significance yet — the
Holm-corrected paired tests run over the full grid, and the grid is incomplete.
**Do not quote any of these deltas as a finding until the GPU arms land.**

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
| Ablation runner, 5 lexical configurations | numbers above, on the full corpus |
| Gemini client: cached, quota-tolerant, token-accounted | live run; 19 calls, 15 rate-limited, resumed from cache |
| Generation + long-context comparison | numbers above, from the API's own reported token counts |
| FastAPI service, Docker, citation UI | live run: index 31.7 s, /search 1.9 ms, /answer 429 path verified |

**395 tests pass, offline, with no API key and no model download.** `ruff` clean.

### Not done

| Missing | Why | What unblocks it |
|---|---|---|
| Dense / hybrid / embedding-model arms | GPU stack could not be installed — see below | network access to `pypi.nvidia.com`, or a machine without Smart App Control |
| Cross-encoder reranking arms | same | same |
| Semantic chunking arm | needs an embedding model | same |
| Query paraphrasing | needs an LLM; would reduce the lexical-overlap confound | free-tier quota |
| Faithfulness judging | run was cut short by the daily quota before the judge pass | free-tier quota, or re-run tomorrow |
| Generation + long-context at full sample | 12 of 216 queries measured; quota-bound | free-tier quota, or re-run tomorrow |
| Human verification of eval labels | requires a person | fill in `data/eval/verification_sample.md` |
| Learning PDF | not started | nothing |

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
