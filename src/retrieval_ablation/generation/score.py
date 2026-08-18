"""Score generated answers. Two metrics need no model at all; one does.

WHAT IS COMPUTED WITHOUT AN LLM, AND WHY THAT MATTERS
-----------------------------------------------------
**Value accuracy.** Every query in this eval set asks for a specific figure that
the gold passage provably contains, and the expected value is recorded in the
query metadata. So correctness is a string match against a normalised number, not
a judgement. An LLM judge asked "is 34,550 the right answer?" would agree
~always, cost a call, and occasionally be wrong -- strictly worse than comparing
the digits.

**Citation accuracy.** The prompt requires citations by passage number, and the
gold chunk ids are known, so this is set arithmetic. Precision is the fraction of
cited passages that are genuinely relevant; recall is whether any gold passage was
cited at all.

That leaves only **faithfulness** -- is every claim supported by the supplied
context -- as a genuine judgement, and it is the one thing an LLM is called for.
This ordering is deliberate: on a free tier where a request takes ~20 seconds, the
difference between three judged metrics and one is the difference between a run
that finishes and one that does not.

REFUSALS ARE SCORED SEPARATELY, NOT AS WRONG ANSWERS
-----------------------------------------------------
A model that says "NOT IN CONTEXT" when the passages genuinely lack the answer has
behaved correctly, and folding that into an accuracy figure would reward
hallucination. Refusal rate is reported beside accuracy, and accuracy is reported
both over all queries and over answered queries only, because those two numbers
say different things and quoting one alone is misleading.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..evalset.schema import EvalQuery
from ..llm.gemini import JUDGE_MODEL, GeminiClient
from .answer import GeneratedAnswer

#: Strip everything that varies between two spellings of the same figure:
#: currency, thousands separators, whitespace, trailing percent, and the
#: parentheses filings use for negatives.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalise_number(text: str) -> str | None:
    """Reduce a reported figure to a comparable canonical form.

    "$ 34,550", "34550" and "(34,550)" all reduce to a comparable string, with the
    parenthesised form marked negative because in a filing that is what the
    parentheses mean. Without this, a correct answer written in a different but
    equally valid style would score as wrong.
    """
    if not text:
        return None
    negative = "(" in text and ")" in text
    match = _NUMBER_RE.search(text.replace(" ", ""))
    if match is None:
        return None
    value = match.group().replace(",", "")
    if negative and not value.startswith("-"):
        value = f"-{value}"
    # Trailing ".0" is a formatting choice, not a different number.
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def contains_expected_value(answer: str, expected: str) -> bool:
    """Whether the answer states the expected figure anywhere in it.

    Scanned across every number in the answer rather than only the first, because
    a correct reply often restates the question ("R&D was 34,550 in 2025") and the
    year would otherwise be compared against the figure.
    """
    target = normalise_number(expected)
    if target is None:
        return False
    cleaned = answer.replace(" ", "")
    negative_context = "(" in cleaned and ")" in cleaned
    for match in _NUMBER_RE.finditer(cleaned):
        candidate = match.group().replace(",", "")
        if "." in candidate:
            candidate = candidate.rstrip("0").rstrip(".")
        if candidate == target:
            return True
        if negative_context and f"-{candidate}" == target:
            return True
    return False


@dataclass(frozen=True, slots=True)
class AnswerScore:
    """Per-answer scores. `faithfulness` is None until a judge has run."""

    query_id: str
    arm: str
    refused: bool
    #: None when the query carries no expected value, so it cannot be scored.
    value_correct: bool | None
    #: Fraction of cited passages that are gold. None when nothing was cited.
    citation_precision: float | None
    #: Whether any gold passage was cited.
    citation_recall: bool | None
    cited_a_hallucinated_passage: bool
    faithfulness: float | None = None

    def to_json(self) -> dict:
        return {
            "query_id": self.query_id,
            "arm": self.arm,
            "refused": self.refused,
            "value_correct": self.value_correct,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "cited_a_hallucinated_passage": self.cited_a_hallucinated_passage,
            "faithfulness": self.faithfulness,
        }


def score_answer(
    answer: GeneratedAnswer,
    query: EvalQuery,
    gold_chunk_ids: Sequence[str] | None,
) -> AnswerScore:
    """Score one answer on everything that needs no model.

    `gold_chunk_ids=None` means citation metrics are **not applicable to this
    arm**, and they come back as None rather than 0.0.

    That distinction is not pedantic; getting it wrong produced a wrong result.
    The long-context arm receives the whole filing as one pseudo-chunk, so its
    citations can never match a gold chunk id no matter how well it cites.
    Passing an empty gold list made precision compute as 0.0, and the report then
    showed long-context with citation precision 0.000 beside retrieval's 0.567 --
    which reads as "long context cites badly" when the truth is that the metric is
    undefined for it. An incomparable metric rendered as a real zero is exactly
    the fabricated measurement this project is built to avoid.
    """
    expected = query.metadata.get("expected_value", "")
    value_correct: bool | None = None
    if expected and not answer.refused:
        value_correct = contains_expected_value(answer.answer, expected)
    elif expected and answer.refused:
        # A refusal is not a wrong value; it is a declined answer. Scoring it
        # False here would make abstention indistinguishable from error.
        value_correct = None

    precision: float | None = None
    recall: bool | None = None
    if gold_chunk_ids is not None:
        gold = set(gold_chunk_ids)
        cited = set(answer.cited_ids)
        if cited:
            precision = len(cited & gold) / len(cited)
            recall = bool(cited & gold)
        elif gold and not answer.refused:
            # Answered without citing anything, while a gold passage existed.
            precision = 0.0
            recall = False

    return AnswerScore(
        query_id=answer.query_id,
        arm=answer.arm,
        refused=answer.refused,
        value_correct=value_correct,
        citation_precision=precision,
        citation_recall=recall,
        cited_a_hallucinated_passage=bool(answer.invalid_citations),
    )


_FAITHFULNESS_PROMPT = """\
You are checking whether an answer is supported by its source passages.

Passages:
{passages}

Answer:
{answer}

Is every factual claim in the answer directly supported by the passages above?
Reply with one word only: SUPPORTED, PARTIAL, or UNSUPPORTED.
"""

_FAITHFULNESS_SCALE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


def judge_faithfulness(
    client: GeminiClient,
    answer: GeneratedAnswer,
    passages: Sequence[str],
    model: str = JUDGE_MODEL,
) -> float | None:
    """Ask a judge whether the answer is grounded. None when it cannot be scored.

    A refusal has no claims to check, so it returns None rather than a score --
    scoring an abstention as perfectly faithful would let a model that refuses
    everything top the faithfulness column.

    Three coarse buckets rather than a 1-10 scale, because a judge's fine-grained
    numbers are not reliable enough to justify the precision they imply, and a
    single word costs fewer output tokens on a rate-limited tier.
    """
    if answer.refused or not answer.answer.strip():
        return None

    numbered = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, start=1))
    completion = client.generate(
        _FAITHFULNESS_PROMPT.format(passages=numbered, answer=answer.answer),
        model=model,
        max_output_tokens=512,
        temperature=0.0,
    )
    verdict = completion.text.strip().lower().split()
    for word in verdict:
        cleaned = word.strip(".,:;!*")
        if cleaned in _FAITHFULNESS_SCALE:
            return _FAITHFULNESS_SCALE[cleaned]
    # An unparseable verdict is not a zero. Returning 0.0 would silently record
    # "unfaithful" for what is actually a failed measurement.
    return None


def aggregate(scores: Sequence[AnswerScore]) -> dict:
    """Summarise a set of scores, keeping unmeasurable things unmeasured."""

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    answered = [s for s in scores if not s.refused]
    value_scored = [s for s in scores if s.value_correct is not None]
    precision = [s.citation_precision for s in scores if s.citation_precision is not None]
    recall = [s.citation_recall for s in scores if s.citation_recall is not None]
    faithful = [s.faithfulness for s in scores if s.faithfulness is not None]

    return {
        "n_answers": len(scores),
        "n_answered": len(answered),
        "n_refused": len(scores) - len(answered),
        "refusal_rate": (len(scores) - len(answered)) / len(scores) if scores else None,
        # Reported over answered queries only, and the denominator is stated, so
        # this cannot be read as accuracy over the whole set.
        "value_accuracy_of_answered": mean([float(s.value_correct) for s in value_scored]),
        "n_value_scored": len(value_scored),
        # Over every query, counting a refusal as not-correct. This is the
        # pessimistic reading, and both are reported because neither alone is
        # honest: one ignores refusals, the other punishes correct abstention.
        "value_accuracy_of_all": (
            mean([float(bool(s.value_correct)) for s in scores]) if scores else None
        ),
        "citation_precision": mean(precision),
        "citation_recall": mean([float(r) for r in recall]),
        "hallucinated_citation_rate": (
            mean([float(s.cited_a_hallucinated_passage) for s in scores]) if scores else None
        ),
        "faithfulness": mean(faithful),
        "n_faithfulness_judged": len(faithful),
    }


def aggregate_by_arm(scores: Sequence[AnswerScore]) -> dict[str, dict]:
    by_arm: dict[str, list[AnswerScore]] = {}
    for score in scores:
        by_arm.setdefault(score.arm, []).append(score)
    return {arm: aggregate(group) for arm, group in sorted(by_arm.items())}


def token_cost(
    answers: Sequence[GeneratedAnswer],
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict:
    """Cost per query from the API's own reported token counts.

    Reported counts, not estimates. The long-context comparison is a cost claim,
    and a cost claim computed from a characters-divided-by-four approximation
    would not be evidence of anything.
    """
    if not answers:
        return {"n": 0}
    prompt = sum(a.prompt_tokens for a in answers)
    output = sum(a.output_tokens for a in answers)
    total = (prompt * input_price_per_million + output * output_price_per_million) / 1e6
    return {
        "n": len(answers),
        "prompt_tokens": prompt,
        "output_tokens": output,
        "mean_prompt_tokens": round(prompt / len(answers), 1),
        "total_cost_usd": round(total, 6),
        "cost_per_query_usd": round(total / len(answers), 8),
    }


def latency_stats(answers: Sequence[GeneratedAnswer]) -> dict:
    """Latency over answers that actually hit the API in this run.

    Filtered on `from_cache`, not on whether a latency value exists. The client
    stores the measured latency inside the cached response body, so a cache hit
    carries the timing of whenever that call was first made -- which meant this
    function's own docstring described an exclusion it did not perform.

    It mattered. A run whose long-context answers all came from cache and whose
    retrieval answers were made live during a throttled window reported the
    long-context arm as 2.5x *faster*, comparing a quiet earlier session against a
    congested current one. The number was real and the comparison was meaningless.
    """
    live = sorted(
        a.latency_seconds for a in answers if a.latency_seconds is not None and not a.from_cache
    )
    if not live:
        return {
            "n_live": 0,
            "p50": None,
            "p95": None,
            "note": "no answers were generated live in this run; cached timings "
            "describe an earlier session and are not comparable",
        }
    return {
        "n_live": len(live),
        "p50": round(live[len(live) // 2], 3),
        "p95": round(live[min(len(live) - 1, int(0.95 * len(live)))], 3),
        "max": round(live[-1], 3),
    }


def compare_arms(
    per_arm_cost: Mapping[str, dict],
    per_arm_latency: Mapping[str, dict],
    retrieval_arm: str = "retrieval",
    long_context_arm: str = "long_context",
) -> dict:
    """The headline retrieval-versus-long-context comparison.

    The project brief asserted retrieval is "roughly 1,250x cheaper per query".
    That figure is not reproducible at realistic settings: it requires assuming a
    1M-token context, an ~800-token retrieval prompt, and zero output cost. This
    function reports the measured ratio instead, whatever it turns out to be.
    """
    rag = per_arm_cost.get(retrieval_arm, {})
    lc = per_arm_cost.get(long_context_arm, {})
    if not rag.get("cost_per_query_usd") or not lc.get("cost_per_query_usd"):
        return {"measured": False, "reason": "one or both arms have no cost data"}

    rag_latency = per_arm_latency.get(retrieval_arm, {}).get("p95")
    lc_latency = per_arm_latency.get(long_context_arm, {}).get("p95")
    return {
        "measured": True,
        "retrieval_cost_per_query_usd": rag["cost_per_query_usd"],
        "long_context_cost_per_query_usd": lc["cost_per_query_usd"],
        "cost_ratio_long_context_over_retrieval": round(
            lc["cost_per_query_usd"] / rag["cost_per_query_usd"], 1
        ),
        "retrieval_mean_prompt_tokens": rag.get("mean_prompt_tokens"),
        "long_context_mean_prompt_tokens": lc.get("mean_prompt_tokens"),
        "retrieval_p95_latency_s": rag_latency,
        "long_context_p95_latency_s": lc_latency,
        "latency_ratio": (
            round(lc_latency / rag_latency, 1) if rag_latency and lc_latency else None
        ),
        # A null ratio needs a reason, or a reader cannot tell "not measured" from
        # "measured as nothing". Cost is comparable across sessions because token
        # counts do not depend on when the call was made; latency is not.
        "latency_note": (
            None
            if rag_latency and lc_latency
            else "not comparable: an arm produced no live call in this run, so its "
            "timings would come from a different session under different load"
        ),
    }
