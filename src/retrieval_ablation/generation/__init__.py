"""Answer generation and its evaluation, for both the retrieval and long-context arms."""

from .answer import NOT_IN_CONTEXT, GeneratedAnswer, build_prompt, generate_answer, parse_citations
from .score import (
    AnswerScore,
    aggregate,
    aggregate_by_arm,
    compare_arms,
    contains_expected_value,
    judge_faithfulness,
    latency_stats,
    normalise_number,
    score_answer,
    token_cost,
)

__all__ = [
    "NOT_IN_CONTEXT",
    "AnswerScore",
    "GeneratedAnswer",
    "aggregate",
    "aggregate_by_arm",
    "build_prompt",
    "compare_arms",
    "contains_expected_value",
    "generate_answer",
    "judge_faithfulness",
    "latency_stats",
    "normalise_number",
    "parse_citations",
    "score_answer",
    "token_cost",
]
