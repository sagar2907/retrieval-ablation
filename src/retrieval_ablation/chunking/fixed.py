"""Fixed-size chunking: the baseline the other strategies are measured against.

Splits on a token budget with a sliding overlap and no regard for document
structure. This is the "naive chunk-and-retrieve" approach, and it is here as the
control: the value of structure-aware chunking is only meaningful as a difference
against something.

Its expected failure mode on this corpus is specific and worth predicting in
advance so the measurement can confirm or refute it: a financial table longer than
the token budget is guaranteed to be split, which severs values from the column
headers that give them meaning and can leave a long table unreachable by any
single chunk. `evalset.relevance.reachability` is what surfaces that.
"""

from __future__ import annotations

from ..corpus.models import Document
from ..evalset.relevance import Chunk
from .base import Chunker, TokenCounter, approx_token_count, atomize


class FixedSizeChunker(Chunker):
    """Packs word atoms up to a token budget, with a sliding overlap."""

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        count_tokens: TokenCounter = approx_token_count,
        name: str | None = None,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not 0 <= overlap_tokens < target_tokens:
            # Overlap equal to or larger than the target would advance zero or
            # negative tokens per step and never terminate.
            raise ValueError(
                f"overlap_tokens must be in [0, target_tokens), "
                f"got {overlap_tokens} with target {target_tokens}"
            )
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._count = count_tokens
        self.name = name or f"fixed{target_tokens}o{overlap_tokens}"

    def chunk(self, doc: Document) -> list[Chunk]:
        atoms = atomize(doc.text, self._count)
        if not atoms:
            return []

        chunks: list[Chunk] = []
        start_index = 0

        while start_index < len(atoms):
            budget = 0
            end_index = start_index
            while end_index < len(atoms) and budget + atoms[end_index].tokens <= self.target_tokens:
                budget += atoms[end_index].tokens
                end_index += 1

            # A single atom larger than the whole budget would otherwise make no
            # progress. Emit it alone: truncating it would break the span-to-text
            # correspondence the eval set depends on.
            if end_index == start_index:
                end_index = start_index + 1

            chunk = self._build(doc, atoms[start_index].span.start, atoms[end_index - 1].span.end)
            if chunk is not None:
                chunks.append(chunk)

            if end_index >= len(atoms):
                break

            # Step back by the overlap, measured in tokens rather than atoms so
            # the overlap is comparable across configurations with different
            # tokenizers.
            step_back = 0
            new_start = end_index
            while new_start > start_index + 1 and step_back < self.overlap_tokens:
                new_start -= 1
                step_back += atoms[new_start].tokens
            start_index = new_start

        return chunks
