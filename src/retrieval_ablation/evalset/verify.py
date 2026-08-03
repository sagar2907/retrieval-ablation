"""Model-assisted adjudication of the generated eval labels.

WHAT THIS IS, AND WHAT IT IS EXPLICITLY NOT

It marks labels `MODEL_CHECKED`. It never marks them `HUMAN_VERIFIED`, and there
is a test asserting that. The distinction is not bureaucratic: the eval set is the
artifact this project publishes, and its verification field is the only thing
telling a reader how much to trust it. Recording a model's opinion under a human's
label would misrepresent the one number that matters most.

Why a model is a genuinely weaker check here, stated so the result is read
correctly:

  - **It is not independent.** The labels were produced by a program from table
    structure, and a model reading the same table can be fooled by the same
    defects. "How much did American Express Co report for american express
    company in 2022?" is grammatical, has a passage containing the figure, and is
    meaningless -- a judge with no other context can reasonably approve it.
  - **It cannot see the counterfactual.** It is shown one gold passage and asked
    whether that passage answers the query. It has no way to notice that a
    *different* passage in the same filing would have been the better label.
  - **It shares the generator's blind spots** about what makes a question natural,
    because both were built around the same table-derived facts.

What it is good for is real: catching malformed queries at a scale nobody will sit
through by hand, and producing a rejection rate that says something concrete about
label quality in bulk. A 5% rejection rate and a 40% rejection rate imply very
different things about every metric computed from this set, and until now the
project had no evidence either way.

The judge is asked to reject on specific, checkable grounds rather than to give a
vague quality score, so a rejection can be argued with.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config import EVAL_DIR, ensure_dirs
from ..corpus.ingest import load_corpus
from ..llm.gemini import JUDGE_MODEL, GeminiClient, QuotaExhaustedError
from .build import QUERIES_PATH
from .schema import EvalQuery, Verification, read_eval_set, write_eval_set

log = logging.getLogger(__name__)

REPORT_PATH = EVAL_DIR / "model_check.json"

_PROMPT = """\
You are auditing a retrieval benchmark built from SEC filings.

A program extracted a fact from a table and generated a question from it. Your job
is to decide whether the question and the passage form a sound benchmark entry.

Question: {question}
Expected answer: {expected}

Labelled passage (from {document}, section: {section}):
---
{passage}
---

Reply with exactly one line in this form:

VERDICT: OK|REJECT — <short reason>

Reject only for one of these specific, checkable reasons:
- The passage does not contain the expected answer.
- The question is not answerable as written (it is nonsense, or the subject of the
  question is a company or place name rather than a reportable line item).
- The question is ambiguous enough that several unrelated figures would answer it
  equally well.

Do not reject merely because the question is terse, templated, or awkwardly worded.
Benchmark questions are allowed to be blunt.
"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(OK|REJECT)\b[^\S\n]*[-–—:]?\s*(.*)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CheckResult:
    query_id: str
    accepted: bool | None
    reason: str
    model: str

    @property
    def parsed(self) -> bool:
        """Whether a verdict could be read at all.

        An unparseable reply is `accepted=None` rather than a rejection. Treating
        a failed read as "REJECT" would silently inflate the rejection rate with
        the judge's formatting mistakes, and that rate is the whole output here.
        """
        return self.accepted is not None


def parse_verdict(text: str) -> tuple[bool | None, str]:
    match = _VERDICT_RE.search(text)
    if not match:
        return None, text.strip()[:200]
    return match.group(1).upper() == "OK", match.group(2).strip()[:300]


def check_query(
    client: GeminiClient,
    query: EvalQuery,
    passage: str,
    model: str = JUDGE_MODEL,
) -> CheckResult:
    """Ask the judge whether one label is sound."""
    gold = query.gold[0]
    completion = client.generate(
        _PROMPT.format(
            question=query.text,
            expected=query.metadata.get("expected_value", "(none recorded)"),
            document=gold.doc_id,
            section=query.metadata.get("section", "(unknown)") or "(unknown)",
            passage=passage[:2000],
        ),
        model=model,
        max_output_tokens=1024,
        temperature=0.0,
    )
    accepted, reason = parse_verdict(completion.text)
    return CheckResult(query.query_id, accepted, reason, model)


def apply_results(
    queries: Sequence[EvalQuery],
    results: Mapping[str, CheckResult],
) -> list[EvalQuery]:
    """Return queries with verification updated from the check results.

    A rejected label becomes REJECTED rather than being deleted, so the rejection
    rate stays measurable and the ablation can report on the accepted subset while
    still disclosing how large the discarded one was.

    A human verdict is never overwritten: if someone has already looked at a
    label, a model does not get to overrule them.
    """
    out: list[EvalQuery] = []
    for query in queries:
        result = results.get(query.query_id)
        if result is None or not result.parsed:
            out.append(query)
            continue
        if query.verification in {Verification.HUMAN_VERIFIED, Verification.REJECTED}:
            out.append(query)
            continue
        out.append(
            EvalQuery(
                query_id=query.query_id,
                text=query.text,
                gold=query.gold,
                kind=query.kind,
                verification=(
                    Verification.MODEL_CHECKED if result.accepted else Verification.REJECTED
                ),
                lexical_overlap=query.lexical_overlap,
                metadata=query.metadata,
                paraphrase_source=query.paraphrase_source,
                checked_by=result.model,
                check_reason=result.reason,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-check the generated eval labels")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument(
        "--write",
        action="store_true",
        help="update queries.jsonl in place (otherwise only the report is written)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()

    queries = read_eval_set(QUERIES_PATH)
    docs = {d.doc_id: d for d in load_corpus()}
    todo = queries[: args.limit] if args.limit else queries
    log.info("checking %d of %d labels with %s", len(todo), len(queries), args.model)

    results: dict[str, CheckResult] = {}
    stopped: str | None = None

    with GeminiClient() as client:
        for i, query in enumerate(todo, start=1):
            gold = query.gold[0]
            doc = docs.get(gold.doc_id)
            if doc is None:
                continue
            try:
                result = check_query(client, query, doc.slice(gold.span), model=args.model)
            except QuotaExhaustedError as exc:
                # Partial results are still worth reporting; the rate is computed
                # over what completed and the shortfall is stated.
                stopped = str(exc)
                log.warning("stopping at %d/%d: %s", i, len(todo), exc)
                break
            results[query.query_id] = result
            if i % 10 == 0:
                accepted = sum(1 for r in results.values() if r.accepted)
                log.info("%d/%d checked, %d accepted so far", i, len(todo), accepted)
        usage = client.usage.to_json()

    accepted = [r for r in results.values() if r.accepted is True]
    rejected = [r for r in results.values() if r.accepted is False]
    unparsed = [r for r in results.values() if r.accepted is None]

    report = {
        "model": args.model,
        "_note": (
            "MODEL_CHECKED, not human-verified. The labels audited here were "
            "generated by a program from table structure, so a model reading the "
            "same table is not an independent second opinion and can be fooled by "
            "the same defects. Useful as a bulk quality signal; not a substitute "
            "for a person."
        ),
        "n_queries_total": len(queries),
        "n_checked": len(results),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_unparseable": len(unparsed),
        "acceptance_rate": len(accepted) / len(results) if results else None,
        "stopped_early": stopped,
        "api_usage": usage,
        "rejections": [{"query_id": r.query_id, "reason": r.reason} for r in rejected[:60]],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.write and results:
        write_eval_set(apply_results(queries, results), QUERIES_PATH)
        log.info("updated %s", QUERIES_PATH)

    print(json.dumps({k: v for k, v in report.items() if k != "rejections"}, indent=2))
    if rejected:
        print(f"\nfirst {min(10, len(rejected))} rejections:")
        for r in rejected[:10]:
            print(f"  {r.query_id}: {r.reason[:110]}")


if __name__ == "__main__":
    main()
