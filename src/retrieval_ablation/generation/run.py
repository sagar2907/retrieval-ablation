"""Run both answer arms and write the generation + long-context comparison.

THE TWO ARMS

*retrieval* answers from the top-k chunks a real retriever returned. *long_context*
answers from the whole filing that contains the answer, stuffed into the prompt up
to a token budget. Same questions, same model, same prompt template, same scoring
code -- the only difference is how the context was assembled, which is the only way
the comparison means anything.

The long-context arm is deliberately given the *correct* document. That makes it a
generous baseline rather than a fair fight: retrieval has to find the right filing
among 120, while long-context is handed it. If retrieval still wins on cost and
latency at comparable accuracy, the finding is stronger for having stacked the
comparison against it. This is stated in the results because a reader would
otherwise reasonably assume both arms searched the same corpus.

SUBSAMPLING IS DISCLOSED, NOT HIDDEN

The free tier permits only a few requests per minute, so this runs on a seeded
stratified subsample rather than all 216 queries. The sample size is recorded in
the output and bootstrap confidence intervals are reported, because a point
estimate from 20 queries invites exactly the over-reading the rest of this project
avoids.

Everything is cached by request content, so an interrupted run resumes rather than
restarts, and a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Sequence

from ..ablation.runner import make_chunker
from ..chunking import approx_token_count
from ..config import GLOBAL_SEED, RESULTS_DIR, ensure_dirs
from ..corpus.ingest import load_corpus
from ..corpus.models import Document, Span
from ..evalset.build import QUERIES_PATH
from ..evalset.relevance import Chunk, build_qrels
from ..evalset.schema import EvalQuery, gold_by_query, read_eval_set
from ..index.bm25 import BM25Index
from ..llm.gemini import DEFAULT_MODEL, JUDGE_MODEL, GeminiClient, QuotaExhaustedError
from ..metrics.stats import bootstrap_ci
from .answer import GeneratedAnswer, generate_answer
from .score import (
    aggregate_by_arm,
    compare_arms,
    judge_faithfulness,
    latency_stats,
    score_answer,
    token_cost,
)

log = logging.getLogger(__name__)

RESULTS_PATH = RESULTS_DIR / "generation.json"
TABLE_PATH = RESULTS_DIR / "generation.md"

#: Published paid-tier prices for the default model, per million tokens. Used to
#: turn the API's reported token counts into a cost figure. The run itself is on
#: the free tier and costs nothing; these are what the same traffic *would* cost,
#: which is the only meaningful basis for a retrieval-versus-long-context
#: comparison.
INPUT_PRICE_PER_MILLION = 1.50
OUTPUT_PRICE_PER_MILLION = 7.50

#: Passages handed to the retrieval arm. The 2026 practitioner consensus is
#: 5-20 reranked chunks; 10 sits in the middle.
RETRIEVAL_TOP_K = 10

#: Character budget for the long-context arm, about 200k tokens at ~4 chars per
#: token. Chosen to match the brief's "stuff 100-200K tokens" specification.
LONG_CONTEXT_CHARS = 800_000


def stuff_document(doc: Document, budget_chars: int = LONG_CONTEXT_CHARS) -> list[Chunk]:
    """Return the filing as one oversized pseudo-chunk, truncated to budget.

    Presented as a single Chunk so both arms travel through identical code. Its
    id records the truncation, so a citation pointing at it is never mistaken for
    a citation of a real retrieved chunk when the two arms are compared.
    """
    text = doc.text[:budget_chars]
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#fulldoc",
            doc_id=doc.doc_id,
            span=Span(0, len(text)),
            text=text,
        )
    ]


def sample_queries(
    queries: Sequence[EvalQuery], n: int, seed: int = GLOBAL_SEED
) -> list[EvalQuery]:
    """Seeded stratified sample across the lexical-overlap range.

    Stratified rather than uniform so the sample spans easy string-match queries
    and genuinely hard ones in the same proportion as the full set; a uniform
    draw of 20 could easily land mostly in one regime and misrepresent both.
    """
    ordered = sorted(queries, key=lambda q: (q.lexical_overlap or 0.0, q.query_id))
    if n >= len(ordered):
        return list(ordered)
    step = len(ordered) / n
    picked = [ordered[min(len(ordered) - 1, int(i * step))] for i in range(n)]
    random.Random(seed).shuffle(picked)
    return picked


def main() -> None:  # noqa: PLR0915 - a linear pipeline; splitting it would hide the order
    parser = argparse.ArgumentParser(description="Run generation and long-context evaluation")
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--n-faithfulness", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--skip-long-context", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    docs = load_corpus()
    by_id = {d.doc_id: d for d in docs}
    queries = read_eval_set(QUERIES_PATH)
    chosen = sample_queries(queries, args.n_queries)
    log.info("sampled %d of %d queries", len(chosen), len(queries))

    # Best first stage currently measured: structure-aware chunking with BM25.
    chunker = make_chunker("struct512")
    chunks = chunker.chunk_corpus(docs)
    by_chunk = {c.chunk_id: c for c in chunks}
    index = BM25Index(chunks)
    qrels = build_qrels(gold_by_query(queries), chunks)
    log.info("indexed %d chunks", len(chunks))

    answers: list[GeneratedAnswer] = []
    scores = []
    incomplete: str | None = None

    with GeminiClient() as client:
        try:
            for i, query in enumerate(chosen, start=1):
                gold_ids = [c for c, g in qrels.get(query.query_id, {}).items() if g > 0]

                hits = index.search(query.text, top_k=args.top_k)
                retrieved = [by_chunk[h.chunk_id] for h in hits]
                if retrieved:
                    answer = generate_answer(
                        client,
                        query.query_id,
                        query.text,
                        retrieved,
                        model=args.model,
                        arm="retrieval",
                    )
                    answers.append(answer)
                    scores.append(score_answer(answer, query, gold_ids))

                if not args.skip_long_context:
                    gold_doc = by_id.get(query.gold[0].doc_id)
                    if gold_doc is not None:
                        stuffed = stuff_document(gold_doc)
                        answer = generate_answer(
                            client,
                            query.query_id,
                            query.text,
                            stuffed,
                            model=args.model,
                            arm="long_context",
                        )
                        answers.append(answer)
                        # None, not []. The long-context arm's context is one
                        # whole-document pseudo-chunk, so its citations can never
                        # match a gold chunk id however well it cites. An empty
                        # list would score precision as a real 0.0 and the report
                        # would show it citing badly; None records the metric as
                        # not applicable to this arm.
                        scores.append(score_answer(answer, query, None))

                log.info("%d/%d queries done (%d answers)", i, len(chosen), len(answers))

            # Faithfulness last: it is the only judged metric, so if quota runs
            # out here the deterministic metrics above are already complete.
            for score in scores[: args.n_faithfulness * 2]:
                answer = next(
                    a for a in answers if a.query_id == score.query_id and a.arm == score.arm
                )
                passages = [
                    by_chunk[c].text if c in by_chunk else "(full document)"
                    for c in answer.context_ids
                ]
                verdict = judge_faithfulness(client, answer, passages, model=JUDGE_MODEL)
                if verdict is not None:
                    object.__setattr__(score, "faithfulness", verdict)

        except QuotaExhaustedError as exc:
            # Not fatal. Everything already answered is cached and scored; the run
            # is reported as partial rather than discarded, and re-running later
            # resumes from the cache.
            incomplete = str(exc)
            log.warning("quota exhausted, reporting a partial run: %s", exc)

        usage = client.usage.to_json()

    by_arm = aggregate_by_arm(scores)
    cost = {
        arm: token_cost(
            [a for a in answers if a.arm == arm],
            INPUT_PRICE_PER_MILLION,
            OUTPUT_PRICE_PER_MILLION,
        )
        for arm in {a.arm for a in answers}
    }
    latency = {
        arm: latency_stats([a for a in answers if a.arm == arm]) for arm in {a.arm for a in answers}
    }
    comparison = compare_arms(cost, latency)

    accuracy_ci = {}
    for arm in by_arm:
        values = [
            float(s.value_correct) for s in scores if s.arm == arm and s.value_correct is not None
        ]
        ci = bootstrap_ci(values)
        accuracy_ci[arm] = (
            {"point": ci.point, "low": ci.low, "high": ci.high, "n": ci.n} if ci else None
        )

    payload = {
        "model": args.model,
        "n_queries_sampled": len(chosen),
        "n_queries_total": len(queries),
        "retrieval_top_k": args.top_k,
        "long_context_budget_chars": LONG_CONTEXT_CHARS,
        "long_context_budget_tokens_approx": approx_token_count("x" * LONG_CONTEXT_CHARS),
        "incomplete_reason": incomplete,
        "by_arm": by_arm,
        "value_accuracy_ci95": accuracy_ci,
        "cost": cost,
        "latency": latency,
        "comparison": comparison,
        "api_usage": usage,
        "answers": [a.to_json() for a in answers],
        "scores": [s.to_json() for s in scores],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    TABLE_PATH.write_text(render(payload), encoding="utf-8")

    print(f"\nwrote {RESULTS_PATH}\nwrote {TABLE_PATH}\n")
    print(json.dumps({"by_arm": by_arm, "comparison": comparison, "usage": usage}, indent=2))


def render(payload: dict) -> str:
    lines = [
        "# Generation and long-context evaluation",
        "",
        f"Model `{payload['model']}`, "
        f"{payload['n_queries_sampled']} of {payload['n_queries_total']} queries "
        f"(seeded stratified sample across the lexical-overlap range).",
        "",
        "The **long-context arm is handed the correct filing**, while the retrieval",
        "arm must find it among 120. That makes long-context a deliberately",
        "generous baseline: any retrieval win on cost or latency at comparable",
        "accuracy holds despite the comparison being stacked against it.",
        "",
    ]
    if payload.get("incomplete_reason"):
        lines += [
            f"> **Partial run.** {payload['incomplete_reason']}",
            "> Numbers below cover only the answers that completed.",
            "",
        ]

    lines += [
        "| arm | answered | refused | value acc (answered) | value acc (all) "
        "| citation prec | citation recall | faithfulness |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value) -> str:
        return "not measured" if value is None else f"{value:.3f}"

    for arm, summary in payload["by_arm"].items():
        lines.append(
            f"| `{arm}` | {summary['n_answered']} | {summary['n_refused']} | "
            f"{fmt(summary['value_accuracy_of_answered'])} | "
            f"{fmt(summary['value_accuracy_of_all'])} | "
            f"{fmt(summary['citation_precision'])} | "
            f"{fmt(summary['citation_recall'])} | "
            f"{fmt(summary['faithfulness'])} |"
        )

    comparison = payload["comparison"]
    lines += ["", "## Retrieval versus long context", ""]
    if not comparison.get("measured"):
        lines += [f"Not measured: {comparison.get('reason', 'unknown')}", ""]
    else:
        rag_tokens = comparison["retrieval_mean_prompt_tokens"]
        lc_tokens = comparison["long_context_mean_prompt_tokens"]
        token_ratio = lc_tokens / max(rag_tokens, 1)
        lines += [
            "| | retrieval | long context | ratio |",
            "|---|---|---|---|",
            f"| mean prompt tokens | {rag_tokens:,.0f} | {lc_tokens:,.0f} | {token_ratio:.1f}x |",
            f"| cost per query (USD) | {comparison['retrieval_cost_per_query_usd']:.6f} | "
            f"{comparison['long_context_cost_per_query_usd']:.6f} | "
            f"**{comparison['cost_ratio_long_context_over_retrieval']}x** |",
            f"| p95 latency (s) | {comparison['retrieval_p95_latency_s']} | "
            f"{comparison['long_context_p95_latency_s']} | "
            f"{comparison['latency_ratio']}x |",
            "",
            "### On the brief's 1,250x claim",
            "",
            "The project brief asserted retrieval is *roughly 1,250x cheaper per query*.",
            f"Measured here: **{comparison['cost_ratio_long_context_over_retrieval']}x**.",
            "",
            "1,250x is only reachable by assuming a full 1M-token context, an",
            "~800-token retrieval prompt, and zero output cost. The brief's own draft",
            "resume bullet says 1/40th, which is far closer to what this measures.",
            "",
        ]

    lines += [
        "## API usage for this run",
        "",
        "```json",
        json.dumps(payload["api_usage"], indent=2),
        "```",
        "",
        "Costs above are the published paid-tier prices applied to the API's own",
        "reported token counts. The run itself was on the free tier and cost $0.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
