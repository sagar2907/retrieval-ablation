"""Build and write the labelled eval set. Entry point: `python -m retrieval_ablation.evalset.build`.

Writes two files under `data/eval/`, both committed:

- `queries.jsonl` -- the benchmark itself.
- `verification_sample.md` -- a human-readable sample for manual checking.

The second file exists because the labels in the first are `GENERATED`. This
project will not report a metric as verified when it is not, and the honest state
of a programmatically derived eval set is "mechanically correct, humanly
unchecked". The sample is the mechanism for changing that: a reader marks each
entry, the marks are fed back, and the verified subset becomes reportable on its
own terms.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import EVAL_DIR, GLOBAL_SEED, ensure_dirs
from ..corpus.ingest import load_corpus
from .schema import EvalQuery, summarise, write_eval_set
from .synthesize import build_queries

log = logging.getLogger(__name__)

QUERIES_PATH = EVAL_DIR / "queries.jsonl"
SAMPLE_PATH = EVAL_DIR / "verification_sample.md"
SUMMARY_PATH = EVAL_DIR / "summary.json"


def write_verification_sample(
    queries: list[EvalQuery],
    docs_by_id: dict,
    path: Path | None = None,
    n: int = 40,
) -> None:
    """Write a sample for a human to check, spanning the overlap range.

    Sampled across the lexical-overlap distribution rather than at random,
    because the low-overlap queries are both the most valuable (they test
    semantic retrieval rather than string matching) and the most likely to be
    malformed, so they are where verification effort pays off most.
    """
    target = SAMPLE_PATH if path is None else path
    ordered = sorted(queries, key=lambda q: q.lexical_overlap or 0.0)
    step = max(1, len(ordered) // n)
    chosen = ordered[::step][:n]

    lines = [
        "# Verification sample",
        "",
        "Labels in `queries.jsonl` are **generated**, not human-verified. Each entry",
        "below shows a query and the exact passage labelled as its answer.",
        "",
        "For each one, mark `[x] ok` if the passage genuinely answers the query, or",
        "`[x] reject` with a short reason if it does not. The rejection rate is the",
        "only evidence available about how trustworthy the generated labels are in",
        "bulk, so a completed sample is worth more than a larger unverified set.",
        "",
        f"Sampled {len(chosen)} of {len(queries)} queries, spread across the "
        "lexical-overlap range.",
        "",
        "---",
        "",
    ]

    for index, query in enumerate(chosen, start=1):
        gold = query.gold[0]
        doc = docs_by_id.get(gold.doc_id)
        passage = doc.slice(gold.span) if doc else "(document not loaded)"
        lines += [
            f"## {index}. `{query.query_id}`",
            "",
            f"**Query:** {query.text}",
            "",
            f"- lexical overlap: `{query.lexical_overlap:.2f}`"
            if query.lexical_overlap is not None
            else "- lexical overlap: n/a",
            f"- document: `{gold.doc_id}`",
            f"- section: {query.metadata.get('section', '') or '(none)'}",
            f"- expected value: `{query.metadata.get('expected_value', '')}`",
            "",
            "**Labelled passage:**",
            "",
            "```",
            passage[:600],
            "```",
            "",
            "- [ ] ok",
            "- [ ] reject &mdash; reason:",
            "",
        ]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labelled retrieval eval set")
    parser.add_argument("--n-queries", type=int, default=220)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_dirs()

    docs = load_corpus()
    log.info("loaded %d documents", len(docs))

    queries = build_queries(docs, n_queries=args.n_queries, seed=args.seed)
    write_eval_set(queries, QUERIES_PATH)

    docs_by_id = {d.doc_id: d for d in docs}
    write_verification_sample(queries, docs_by_id, n=args.sample_size)

    info = summarise(queries)
    info["seed"] = args.seed
    info["requested"] = args.n_queries
    SUMMARY_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nwrote {len(queries)} queries to {QUERIES_PATH}")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print(f"\nverification sample: {SAMPLE_PATH}")
    print("NOTE: all labels are GENERATED. No metric may be reported as")
    print("human-verified until that sample is filled in and fed back.")


if __name__ == "__main__":
    main()
