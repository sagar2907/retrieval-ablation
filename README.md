# retrieval-ablation

A labelled retrieval benchmark over SEC filings, and a single-axis ablation study
measuring what each component of a retrieval pipeline is actually worth.

The point of this repository is not the demo. It is the evidence: retrieval
quality is measured independently of generation quality, against a hand-verified
set of queries with known gold passages, with confidence intervals and
significance testing on every comparison.

## Why measure retrieval separately

A question-answering system over documents does two things: it finds passages,
then it writes an answer from them. Almost every published RAG project measures
only the second step. That conflates two failure modes which have completely
different fixes — if retrieval returned the wrong passages, no amount of prompt
engineering recovers the answer.

Separating them requires a labelled set: queries paired with the passage IDs that
actually contain the answer. With that, retrieval is scored with nDCG@10,
Recall@50 and MRR, and no language model is involved at all. That makes the
retrieval ablation both free to run and fully deterministic.

## Corpus

SEC EDGAR 10-K and 10-Q filings. Chosen over the alternatives because:

- **Table-dense.** Financial statements are where naive chunking visibly fails —
  a table split across two chunks becomes unretrievable and unciteable.
- **Cross-referential.** "See Note 12 to the Consolidated Financial Statements"
  means the answer to a question is often not in the passage that mentions it.
- **Structured.** The mandated Item 1–15 hierarchy gives a real document tree to
  test structure-aware chunking against.
- **Public domain.** The eval set can be published without licensing questions,
  which matters because the labelled benchmark is the artifact most worth
  sharing.

Filings are HTML, not PDF. This project parses the original HTML directly rather
than converting to PDF first — round-tripping through PDF layout analysis
destroys exactly the table structure being tested.

## Honest status

Updated at the end of each phase. Nothing is listed as verified unless it has
been run and the output inspected.

### Built and verified

| Component | Status | Evidence |
|---|---|---|
| Repo scaffold, lint, offline CI | verified | `ruff check` clean, `pytest` green — see Phase 1 below |
| `metrics.retrieval` — nDCG@k, Recall@k, MRR | verified | 29 tests, hand-computed expected values |
| `metrics.stats` — bootstrap CI, paired permutation test, Holm-Bonferroni | verified | 25 tests, determinism asserted under fixed seeds |

### Not done yet

Everything else. In planned order: corpus ingest, chunking strategies, eval-set
construction, retrieval stack, the ablation run, generation eval, long-context
baseline, service and UI.

### Known limitations, stated up front

- **Free-tier request quotas bound the LLM-dependent arms.** Google no longer
  publishes per-model free-tier daily request limits; they are account-specific.
  The generation, judging and long-context measurements therefore run on a
  stratified query subsample with reported confidence intervals rather than the
  full set. This is disclosed in the results rather than hidden by reporting
  point estimates.
- **Unmeasured means unmeasured.** Any metric that requires data not yet
  collected returns `None` and renders as "not measured". This is enforced by
  test, not convention.

## Development

Requires Python 3.12+. The default install is CPU-only and has no API
dependencies, so the full test suite runs offline.

```bash
uv venv
uv pip install -e ".[dev]"
```

Run the checks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

GPU extras (BGE-M3 embeddings, cross-encoder reranking) are a separate optional
group so CI never downloads model weights:

```bash
uv pip install -e ".[gpu]"
```

## Reproducibility

- Every sampling decision derives from `GLOBAL_SEED` in `config.py`. No function
  that affects data reads the clock or the global RNG state.
- The raw corpus is gitignored but reproducible from checksummed manifests in
  `data/manifests/`.
- API responses are cached to `.cache/` keyed by content hash, so a re-run
  neither re-pays nor silently substitutes different output.

## Licence

Code under this repository is MIT. SEC filings are US public domain.
