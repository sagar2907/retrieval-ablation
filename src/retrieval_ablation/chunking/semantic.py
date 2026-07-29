"""Semantic chunking: break where consecutive sentence meanings diverge.

The method, as popularised by Greg Kamradt and adopted by LangChain and
LlamaIndex: embed each sentence, take the cosine distance between consecutive
sentences, and start a new chunk wherever that distance exceeds a percentile
threshold of all distances in the document. The intuition is that a topic shift
shows up as a spike in distance.

Two things about it are worth stating plainly, because this project is supposed to
measure claims rather than repeat them:

**The threshold is relative, not absolute.** Using a percentile of the document's
own distance distribution means a fixed fraction of positions always become
breakpoints, whatever the document looks like. A perfectly uniform document still
gets broken at the 95th percentile of a nearly flat distribution. That makes the
method robust to embedding-scale differences but also means it cannot decline to
split.

**It is expensive and structure-blind.** It requires one embedding per sentence of
the corpus -- far more than one per chunk -- and it knows nothing about tables. A
financial table linearised into rows produces a run of very similar "sentences", so
distances stay low and the table is likely to be swallowed into a neighbouring
chunk or split at an arbitrary row. The published evidence for semantic chunking
beating fixed-size is mixed; on a table-heavy corpus there is a reasonable prior
that it does not. Measuring that is the point of including it.

The embedding function is injected, so tests run offline against a deterministic
fake and the real runs use the same model as the retrieval arm.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from ..corpus.models import BlockKind, Document, Span
from ..evalset.relevance import Chunk
from .base import Chunker, TokenCounter, approx_token_count

#: Embeds a batch of strings, returning one row per input.
EmbedFn = Callable[[Sequence[str]], np.ndarray]

# Sentence boundary: terminal punctuation followed by whitespace and a capital or
# digit. Deliberately conservative -- filings are full of abbreviations ("U.S.",
# "Inc.", "No. 12") and an aggressive splitter would shatter them into fragments
# whose embeddings carry no meaning. Missing a boundary merely produces a longer
# sentence; inventing one produces noise.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str, offset: int = 0) -> list[Span]:
    """Sentence spans within `text`, shifted by `offset`."""
    spans: list[Span] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if text[cursor:end].strip():
            spans.append(Span(offset + cursor, offset + end))
        cursor = match.end()
    if text[cursor:].strip():
        spans.append(Span(offset + cursor, offset + len(text)))
    return spans


class SemanticChunker(Chunker):
    """Breaks at percentile-threshold spikes in consecutive sentence distance."""

    def __init__(  # noqa: PLR0917 - configuration surface, all with defaults
        self,
        embed: EmbedFn,
        breakpoint_percentile: float = 95.0,
        target_tokens: int = 512,
        max_tokens: int = 1024,
        count_tokens: TokenCounter = approx_token_count,
        name: str | None = None,
    ) -> None:
        if not 50.0 <= breakpoint_percentile < 100.0:
            raise ValueError(
                f"breakpoint_percentile must be in [50, 100), got {breakpoint_percentile}"
            )
        self._embed = embed
        self.breakpoint_percentile = breakpoint_percentile
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self._count = count_tokens
        self.name = name or f"semantic{int(breakpoint_percentile)}"

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = self._sentence_spans(doc)
        if not sentences:
            return []
        if len(sentences) == 1:
            built = self._build(doc, sentences[0].start, sentences[0].end)
            return [built] if built else []

        vectors = self._embed([doc.text[s.start : s.end] for s in sentences])
        breakpoints = self._breakpoints(vectors)

        chunks: list[Chunk] = []
        group_start = 0
        group_tokens = 0

        for index, span in enumerate(sentences):
            group_tokens += self._count(doc.text[span.start : span.end])
            is_break = index in breakpoints
            # The size cap overrides the semantic signal. Without it a document
            # with flat distances yields one chunk spanning the whole filing,
            # which no embedding model could encode.
            too_big = group_tokens >= self.max_tokens
            is_last = index == len(sentences) - 1

            if is_break or too_big or is_last:
                built = self._build(doc, sentences[group_start].start, span.end)
                if built is not None:
                    chunks.append(built)
                group_start = index + 1
                group_tokens = 0

        return chunks

    def _sentence_spans(self, doc: Document) -> list[Span]:
        """Sentence spans, with tables kept intact as single units.

        A table is emitted as one "sentence" rather than being sentence-split.
        Splitting a linearised table on punctuation would produce fragments of
        rows whose embeddings are meaningless, and would let a breakpoint fall
        inside the table -- reintroducing exactly the failure the structure-aware
        chunker exists to avoid, in the chunker that has no other reason to hit it.
        """
        spans: list[Span] = []
        for block in doc.blocks:
            if block.kind is BlockKind.BOILERPLATE:
                continue
            if block.is_table:
                spans.append(block.span)
                continue
            spans.extend(split_sentences(doc.slice(block.span), block.span.start))
        return spans

    def _breakpoints(self, vectors: np.ndarray) -> set[int]:
        """Indices after which a new chunk starts."""
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Guard against a zero vector, which an embedding model can return for
        # whitespace-only input; dividing by it would produce NaN distances that
        # silently propagate into the percentile and disable all breaking.
        norms[norms == 0] = 1.0
        unit = matrix / norms

        similarities = np.sum(unit[:-1] * unit[1:], axis=1)
        distances = 1.0 - similarities
        if distances.size == 0:
            return set()

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        # Strictly greater: with a degenerate distribution where many distances
        # equal the threshold, >= would break at nearly every position.
        return {int(i) for i in np.flatnonzero(distances > threshold)}
