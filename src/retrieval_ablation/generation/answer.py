"""Answer generation, for both the retrieval pipeline and the long-context baseline.

Both arms answer the same questions and are scored by the same code. That is the
whole point: a comparison between retrieval and long-context is only meaningful if
the only thing that differs is how the model got its context.

CITATIONS ARE REQUESTED IN A MACHINE-CHECKABLE FORM
---------------------------------------------------
The prompt numbers each passage and requires the answer to cite by number. That
turns citation accuracy into set arithmetic against the gold passage ids -- no
judge model, no rubric, no cost. Asking for prose citations and then having an LLM
grade them would be slower, more expensive, and less trustworthy than a string
comparison, and it would make citation accuracy depend on the judge's mood.

Numbering also lets a wrong citation be distinguished from a missing one, which
matters: a model that answers correctly while citing nothing has a different
failure than one that cites a passage it did not use.

REFUSAL IS AN ALLOWED ANSWER
----------------------------
The prompt explicitly permits "NOT IN CONTEXT". Without that, a model handed
passages that do not contain the answer will invent one, and the faithfulness
metric would then be measuring the prompt's pressure to answer rather than the
retrieval quality underneath it. Since roughly half the queries here have their
gold passage outside the top-10, refusal has to be available or the generation
numbers say nothing about retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..evalset.relevance import Chunk
from ..llm.gemini import DEFAULT_MODEL, Completion, GeminiClient

#: Marker the model is told to emit when the context does not contain the answer.
NOT_IN_CONTEXT = "NOT IN CONTEXT"

_CITATION_RE = re.compile(r"\[(\d+)\]")

_SYSTEM = (
    "You answer questions about SEC filings using only the numbered passages "
    "provided. You never use outside knowledge and never guess."
)

_TEMPLATE = """\
Answer the question using ONLY the numbered passages below.

Rules:
- Cite every passage you used, by number, in square brackets like [3].
- If the passages do not contain the answer, reply with exactly: {refusal}
- Give the figure and its units exactly as the filing states them.
- Be brief. One or two sentences.

Passages:
{passages}

Question: {question}

Answer:"""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """One answer, with everything needed to score it without re-calling the model."""

    query_id: str
    question: str
    answer: str
    #: Chunk ids supplied as context, in the order they were numbered.
    context_ids: tuple[str, ...]
    #: Chunk ids the answer actually cited, resolved from the bracket numbers.
    cited_ids: tuple[str, ...]
    #: Bracket numbers that pointed outside the supplied range. A model citing
    #: [12] when given 10 passages has hallucinated a source, which is a distinct
    #: and more serious failure than citing the wrong real passage.
    invalid_citations: tuple[int, ...]
    refused: bool
    prompt_tokens: int
    output_tokens: int
    latency_seconds: float | None
    from_cache: bool
    model: str
    arm: str

    def to_json(self) -> dict:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "answer": self.answer,
            "context_ids": list(self.context_ids),
            "cited_ids": list(self.cited_ids),
            "invalid_citations": list(self.invalid_citations),
            "refused": self.refused,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "from_cache": self.from_cache,
            "model": self.model,
            "arm": self.arm,
        }


def build_prompt(question: str, passages: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    return _TEMPLATE.format(passages=numbered, question=question, refusal=NOT_IN_CONTEXT)


def parse_citations(answer: str, n_passages: int) -> tuple[list[int], list[int]]:
    """Split bracket citations into valid 1-based indices and invalid ones.

    Duplicates are collapsed and order is preserved, so citing [3] twice counts
    once -- otherwise a repeated citation would inflate any count computed from
    this list.
    """
    valid: list[int] = []
    invalid: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if number in seen:
            continue
        seen.add(number)
        if 1 <= number <= n_passages:
            valid.append(number)
        else:
            invalid.append(number)
    return valid, invalid


def _is_refusal(answer: str) -> bool:
    # Substring rather than equality: models reliably emit the marker but often
    # wrap it in a sentence, and treating "The passages do not contain the answer,
    # so: NOT IN CONTEXT" as a non-refusal would count a correct abstention as a
    # wrong answer.
    return NOT_IN_CONTEXT.lower() in answer.lower()


def generate_answer(
    client: GeminiClient,
    query_id: str,
    question: str,
    chunks: Sequence[Chunk],
    *,
    model: str = DEFAULT_MODEL,
    arm: str = "retrieval",
    max_output_tokens: int = 1024,
) -> GeneratedAnswer:
    """Answer one question from the supplied passages."""
    passages = [c.text for c in chunks]
    completion: Completion = client.generate(
        build_prompt(question, passages),
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        system=_SYSTEM,
    )

    valid, invalid = parse_citations(completion.text, len(passages))
    return GeneratedAnswer(
        query_id=query_id,
        question=question,
        answer=completion.text,
        context_ids=tuple(c.chunk_id for c in chunks),
        cited_ids=tuple(chunks[i - 1].chunk_id for i in valid),
        invalid_citations=tuple(invalid),
        refused=_is_refusal(completion.text),
        prompt_tokens=completion.prompt_tokens,
        output_tokens=completion.output_tokens,
        latency_seconds=completion.latency_seconds,
        from_cache=completion.from_cache,
        model=model,
        arm=arm,
    )
