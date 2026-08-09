"""Rewrite eval queries so they stop quoting the filing's own wording.

WHY THIS EXISTS

The queries in this benchmark are generated from table rows, and they reuse the
row's label verbatim: a row labelled "Research and development" becomes "What was
Apple's research and development expense in fiscal 2025?". Every content word in
the question is then present in the answer passage, which hands BM25 an exact
string match and makes the query a string-matching exercise rather than a
retrieval one.

That confound is measured -- `lexical_overlap` records it per query, and the
ablation reports low-overlap and high-overlap subsets separately. The measurement
showed it is not a small effect. Every lexical configuration roughly doubles its
nDCG on high-overlap queries, while the dense arm is the only configuration in
the grid whose low-overlap score *exceeds* its high-overlap score. In other words
the benchmark systematically favours one of the two families it is comparing.

Measuring a confound is not the same as removing it. This module removes it, by
asking a model to re-ask each question the way a person would -- without the row
label -- while keeping the question answerable by exactly the same passage.

WHAT IS DELIBERATELY NOT DONE HERE

The gold spans are never touched. A paraphrase changes only `EvalQuery.text`;
`gold`, `metadata` and `verification` carry over unchanged. If paraphrasing were
allowed to move the labels it would no longer be the same benchmark, and the
comparison against the original run would be meaningless.

The output is written to a separate file rather than over `queries.jsonl`. Both
sets are then runnable, which is the entire point: the interesting number is not
the paraphrased score but the *difference* between the two runs on the same
configurations.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass

from ..config import EVAL_DIR, ensure_dirs
from ..corpus.ingest import load_corpus
from ..llm.gemini import GeminiClient, QuotaExhaustedError
from .build import QUERIES_PATH
from .schema import EvalQuery, read_eval_set, write_eval_set
from .synthesize import lexical_overlap

log = logging.getLogger(__name__)

PARAPHRASE_MODEL = "gemini-3.5-flash-lite"

#: Written beside queries.jsonl, never over it. See the module docstring.
PARAPHRASED_PATH = EVAL_DIR / "queries-paraphrased.jsonl"
REPORT_PATH = EVAL_DIR / "paraphrase.json"

_PROMPT = """You are rewriting a question so that it tests retrieval rather than string matching.

The question below was generated from a row of a financial table, and it reuses \
the table's own row label word for word. Rewrite it the way a person who had not \
seen the table would ask it.

Rules:
- Keep it answerable by the SAME passage. Do not change what is being asked for.
- Keep the company name and the fiscal period. Without them the question is \
ambiguous, because this corpus holds four consecutive years for every company.
- Replace the accounting terminology with ordinary words wherever you can. \
"research and development expense" might become "how much it spent developing \
new products".
- Do not invent figures, and do not include the answer.
- One sentence. Output only the rewritten question, nothing else.

Company: {company}
Fiscal period: {period}
Question: {question}"""

#: Tokens whose loss makes a query ambiguous across a 4-year, 30-company corpus.
#: Checked mechanically after the rewrite rather than trusted to the instructions.
_YEAR_RE = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True, slots=True)
class ParaphraseResult:
    query_id: str
    original: str
    rewritten: str | None
    #: Why a rewrite was refused, or None if it was kept.
    rejected: str | None
    overlap_before: float
    #: None when the rewrite was refused -- there is nothing to measure.
    overlap_after: float | None

    @property
    def kept(self) -> bool:
        return self.rewritten is not None


def _company_tokens(company: str) -> set[str]:
    """Distinctive words of a company name, ignoring corporate suffixes.

    "The Southern Company" and "Southern Co" must count as the same name, or a
    rewrite would be rejected for dropping a word that carries no information.
    """
    noise = {"the", "inc", "co", "corp", "corporation", "company", "group", "ltd", "plc"}
    return {w for w in re.findall(r"[a-z]+", company.lower()) if w not in noise and len(w) > 2}


def check_rewrite(query: EvalQuery, rewritten: str, company: str) -> str | None:
    """Return a rejection reason, or None if the rewrite is usable.

    A model asked to remove wording will sometimes remove too much. Dropping the
    company or the year turns a question with one correct answer into one with
    thirty or four, and the eval set cannot represent that -- the gold span still
    points at a single row, so the query would simply be scored as wrong no matter
    what a retriever returned. Rejecting is honest; silently keeping it would
    depress every configuration's score for a reason unrelated to retrieval.

    `company` is passed in rather than read from `query.metadata`, which does not
    carry it -- it holds `ticker`. The first version of this function looked the
    key up anyway, got "", and skipped the check for every single query while
    appearing to perform it. A guard that cannot fail is not a guard.
    """
    if not rewritten or len(rewritten) < 15:
        return "empty or too short"
    if len(rewritten) > 400:
        return "not a single question"

    wanted = _company_tokens(company)
    if not wanted:
        return None  # nothing distinctive to check against; do not invent a failure
    if not (wanted & _company_tokens(rewritten)):
        return f"dropped the company name ({company})"

    period = query.metadata.get("period", "") or query.metadata.get("report_date", "")
    years_wanted = {m.group(0) for m in _YEAR_RE.finditer(period)}
    if years_wanted and not (years_wanted & {m.group(0) for m in _YEAR_RE.finditer(rewritten)}):
        return f"dropped the fiscal period ({period})"

    # A rewrite that still quotes the row label has not done its job. This is not
    # a correctness failure, so it is reported rather than rejected -- see main().
    return None


def paraphrase_query(
    client: GeminiClient,
    query: EvalQuery,
    passage: str,
    company: str,
    model: str = PARAPHRASE_MODEL,
) -> ParaphraseResult:
    """Rewrite one query, or record why the rewrite was refused."""
    before = lexical_overlap(query.text, passage)
    completion = client.generate(
        _PROMPT.format(
            company=company or "(unknown)",
            period=query.metadata.get("period", "(unknown)") or "(unknown)",
            question=query.text,
        ),
        model=model,
        max_output_tokens=1024,
        temperature=0.0,
    )
    text = completion.text.strip().strip('"').split("\n")[0].strip()

    reason = check_rewrite(query, text, company)
    if reason:
        return ParaphraseResult(query.query_id, query.text, None, reason, before, None)
    return ParaphraseResult(
        query.query_id,
        query.text,
        text,
        None,
        before,
        lexical_overlap(text, passage),
    )


def apply_results(
    queries: list[EvalQuery], results: dict[str, ParaphraseResult], passages: dict[str, str]
) -> list[EvalQuery]:
    """Return a new query list carrying the accepted rewrites.

    Queries with no rewrite keep their original text rather than being dropped.
    Removing them would change which queries the two runs share, and the whole
    comparison rests on the two sets being the same queries.
    """
    out: list[EvalQuery] = []
    for query in queries:
        result = results.get(query.query_id)
        if result is None or not result.kept:
            out.append(query)
            continue
        passage = passages.get(query.query_id, "")
        out.append(
            EvalQuery(
                query_id=query.query_id,
                text=result.rewritten or query.text,
                gold=query.gold,
                kind=query.kind,
                verification=query.verification,
                lexical_overlap=lexical_overlap(result.rewritten or query.text, passage),
                metadata=query.metadata,
                paraphrase_source=PARAPHRASE_MODEL,
                checked_by=query.checked_by,
                check_reason=query.check_reason,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase eval queries to cut lexical overlap")
    parser.add_argument("--limit", type=int, default=None, help="rewrite only the first N")
    parser.add_argument("--model", default=PARAPHRASE_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    queries = read_eval_set(QUERIES_PATH)
    docs = {d.doc_id: d for d in load_corpus()}
    passages = {
        q.query_id: docs[q.gold[0].doc_id].slice(q.gold[0].span)
        for q in queries
        if q.gold and q.gold[0].doc_id in docs
    }
    companies = {
        q.query_id: docs[q.gold[0].doc_id].metadata.get("company", "")
        for q in queries
        if q.gold and q.gold[0].doc_id in docs
    }
    todo = queries[: args.limit] if args.limit else queries
    log.info("paraphrasing %d of %d queries with %s", len(todo), len(queries), args.model)

    results: dict[str, ParaphraseResult] = {}
    stopped: str | None = None

    with GeminiClient() as client:
        for i, query in enumerate(todo, start=1):
            passage = passages.get(query.query_id)
            if passage is None:
                continue
            try:
                result = paraphrase_query(
                    client, query, passage, companies.get(query.query_id, ""), model=args.model
                )
            except QuotaExhaustedError as exc:
                # Partial output is still useful: the queries that were rewritten
                # are rewritten correctly, and the shortfall is stated in the
                # report rather than left for a reader to notice.
                stopped = str(exc)
                log.warning("stopping at %d/%d: %s", i, len(todo), exc)
                break
            results[query.query_id] = result
            if i % 20 == 0:
                kept = sum(1 for r in results.values() if r.kept)
                log.info("%d/%d rewritten, %d kept", i, len(todo), kept)
        usage = client.usage.to_json()

    kept = [r for r in results.values() if r.kept]
    refused = [r for r in results.values() if not r.kept]
    before = [r.overlap_before for r in kept]
    after = [r.overlap_after for r in kept if r.overlap_after is not None]

    written = apply_results(queries, results, passages)
    write_eval_set(written, PARAPHRASED_PATH)

    report = {
        "model": args.model,
        "_note": (
            "Paraphrases rewrite EvalQuery.text only. Gold spans, grades and "
            "verification status are carried over untouched, so this file indexes "
            "the same passages as queries.jsonl and the two runs are comparable. "
            "Queries whose rewrite was refused keep their original text rather "
            "than being dropped, so both files contain the same query ids."
        ),
        "n_queries_total": len(queries),
        "n_attempted": len(results),
        "n_kept": len(kept),
        "n_refused": len(refused),
        "mean_overlap_before": round(sum(before) / len(before), 4) if before else None,
        "mean_overlap_after": round(sum(after) / len(after), 4) if after else None,
        "refusal_reasons": sorted({r.rejected for r in refused if r.rejected}),
        "stopped_early": stopped,
        "usage": usage,
        "examples": [
            {
                "query_id": r.query_id,
                "before": r.original,
                "after": r.rewritten,
                "overlap": [r.overlap_before, r.overlap_after],
            }
            for r in kept[:10]
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    log.info("kept %d, refused %d", len(kept), len(refused))
    if before and after:
        log.info(
            "mean lexical overlap %.4f -> %.4f", sum(before) / len(before), sum(after) / len(after)
        )
    log.info("wrote %s and %s", PARAPHRASED_PATH.name, REPORT_PATH.name)
    if stopped:
        log.warning("INCOMPLETE: %s", stopped)


if __name__ == "__main__":
    main()
