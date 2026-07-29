"""The chunker interface, and the atom packing every chunker builds on.

Every chunker returns `Chunk` objects whose spans index into the document's
canonical text. Chunkers may only *slice* that text: never rewrite, re-normalise
or reflow it. Any transformation would break the correspondence with gold labels,
which are spans into the same string (see `corpus.models`).

Token counting is injected rather than imported. Two reasons:

1.  The offline test suite must not download a tokenizer. Tests pass a
    deterministic fake and assert exact boundaries.
2.  Which tokenizer is used changes where boundaries fall, so it is an
    experimental parameter that belongs in the recorded configuration rather
    than hidden inside a module.

Chunkers pack *atoms* -- maximal runs of non-whitespace with known character
offsets -- rather than splitting on token indices. A token-index split would need
a tokenizer that reports character offsets, which not every tokenizer does, and
would let a chunk boundary fall inside a word. Packing atoms keeps boundaries on
word edges and keeps character spans exact, which is what the eval set needs.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..corpus.models import Block, Document, Span
from ..evalset.relevance import Chunk

#: Counts tokens in a string. Injected so tests are offline and deterministic.
TokenCounter = Callable[[str], int]

_ATOM_RE = re.compile(r"\S+")

#: Characters per token for the approximate counter. 4.0 is the usual rule of
#: thumb for English prose; financial filings run slightly denser because of
#: numerals and punctuation, but the approximate counter exists for tests and
#: fast iteration, not for the recorded runs, so precision here is not the point.
_CHARS_PER_TOKEN = 4.0


def approx_token_count(text: str) -> int:
    """Deterministic token estimate requiring no model download.

    Used by tests and by the fast development loop. Real runs pass a genuine
    tokenizer; the ablation records which counter produced each configuration so
    the two are never silently compared.
    """
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True, slots=True)
class Atom:
    """A word-sized unit with its character span and token cost."""

    span: Span
    tokens: int


def atomize(text: str, count_tokens: TokenCounter) -> list[Atom]:
    """Split text into non-whitespace runs with spans and token costs."""
    return [
        Atom(span=Span(m.start(), m.end()), tokens=count_tokens(m.group()))
        for m in _ATOM_RE.finditer(text)
    ]


def chunk_id_for(doc_id: str, span: Span) -> str:
    """Stable, sortable, self-describing chunk identifier.

    Encoding the span in the id means a retrieval result can be traced back to
    its exact source location without a lookup table, which matters for the
    citation-highlighting UI and for debugging a bad result by hand. Zero padding
    keeps lexicographic and numeric order the same.
    """
    return f"{doc_id}#{span.start:08d}-{span.end:08d}"


class Chunker(ABC):
    """Splits a document into retrievable units.

    Implementations must be pure functions of the document plus their own
    configuration: no clock reads, no unseeded randomness, no network. Running a
    chunker twice on the same document must produce identical spans, because the
    ablation depends on chunk ids being reproducible across runs.
    """

    #: Short identifier used in configuration names, index names, and results
    #: tables. Must be filesystem- and collection-name safe.
    name: str

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Return this document's chunks, ordered by start offset."""

    def chunk_corpus(self, docs: Sequence[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk(doc))
        return out

    # -- shared helpers -------------------------------------------------------

    @staticmethod
    def _build(doc: Document, start: int, end: int) -> Chunk | None:
        """Materialise a chunk from a character range, or None if it is blank.

        Trims to the exact non-whitespace extent so a chunk's span, its text, and
        the substring of the document at that span are always the same thing.
        Without trimming, a chunk boundary landing on the block separator would
        give a span whose slice starts with a newline, and `chunk.text` would then
        disagree with `document.slice(chunk.span)` after any later strip.
        """
        raw = doc.text[start:end]
        stripped = raw.strip()
        if not stripped:
            return None
        lead = len(raw) - len(raw.lstrip())
        true_start = start + lead
        true_end = true_start + len(stripped)
        span = Span(true_start, true_end)

        blocks = doc.blocks_overlapping(span)
        return Chunk(
            chunk_id=chunk_id_for(doc.doc_id, span),
            doc_id=doc.doc_id,
            span=span,
            text=stripped,
            # The section path of the chunk's first overlapping block. A chunk
            # spanning a section boundary is attributed to where it begins, which
            # matches how a reader would cite it.
            section_path=blocks[0].section_path if blocks else (),
            contains_table=any(b.is_table for b in blocks),
        )

    @staticmethod
    def _block_tokens(doc: Document, block: Block, count_tokens: TokenCounter) -> int:
        return count_tokens(doc.slice(block.span))
