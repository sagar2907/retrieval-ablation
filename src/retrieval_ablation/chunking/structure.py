"""Structure-aware chunking: respects document hierarchy and keeps tables whole.

Two rules, both of which fixed-size chunking violates:

1.  **A table is never split.** A financial table split across two chunks leaves
    numbers in one chunk and their column headers in the other. The half with the
    numbers is unretrievable (nothing lexical to match) and unciteable (nothing to
    tell a reader what the figures mean). Keeping the table whole is the single
    largest expected difference between this chunker and the baseline on a
    filing corpus.

2.  **Chunks do not straddle a section boundary.** A chunk half in "Note 7 --
    Income Taxes" and half in "Note 8 -- Leases" has no single correct
    attribution, which makes both its citation and any metadata filter on it
    wrong.

The cost is variable chunk sizes, including chunks well over target when a single
table exceeds the budget. That is an accepted, measured trade-off rather than a
bug: oversized chunks are counted and reported, because they raise embedding cost
and can exceed an embedding model's context window, and pretending otherwise
would hide a real limitation.

A heading is prepended to each chunk's *retrieval* text in the caller when that
variant is enabled; this module does not alter document text, because chunk spans
must keep indexing the canonical string.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..corpus.models import Block, BlockKind, Document
from ..evalset.relevance import Chunk
from .base import Chunker, TokenCounter, approx_token_count
from .fixed import FixedSizeChunker


@dataclass(frozen=True, slots=True)
class OversizeReport:
    """Chunks that exceeded the target because an atomic block could not be split."""

    n_chunks: int
    n_oversize: int
    largest_tokens: int

    @property
    def fraction_oversize(self) -> float:
        return self.n_oversize / self.n_chunks if self.n_chunks else 0.0


class StructureAwareChunker(Chunker):
    """Packs whole blocks into chunks, never crossing sections or splitting tables."""

    def __init__(
        self,
        target_tokens: int = 512,
        max_tokens: int = 2048,
        count_tokens: TokenCounter = approx_token_count,
        split_long_prose: bool = True,
        name: str | None = None,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be at least target_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self._count = count_tokens
        #: Long *prose* blocks may be split on word boundaries; tables never are.
        #: Without this a single 8,000-token risk-factor paragraph would produce
        #: one chunk far past any embedding model's window.
        self.split_long_prose = split_long_prose
        self.name = name or f"struct{target_tokens}"
        self._oversize: list[int] = []

    def chunk(self, doc: Document) -> list[Chunk]:
        self._oversize = []
        chunks: list[Chunk] = []

        # Boilerplate is excluded from chunking but its offsets remain valid; it
        # is page furniture and contents listings, which add nothing retrievable
        # and dilute every index they enter.
        blocks = [b for b in doc.blocks if b.kind is not BlockKind.BOILERPLATE]

        group: list[Block] = []
        group_tokens = 0
        group_section: tuple[str, ...] | None = None

        def flush() -> None:
            nonlocal group, group_tokens, group_section
            if group:
                chunk = self._build(doc, group[0].span.start, group[-1].span.end)
                if chunk is not None:
                    chunks.append(chunk)
                    if group_tokens > self.target_tokens:
                        self._oversize.append(group_tokens)
            group = []
            group_tokens = 0
            group_section = None

        for block in blocks:
            tokens = self._block_tokens(doc, block, self._count)

            # Section change closes the current chunk before anything else, so a
            # chunk never spans two sections.
            if group_section is not None and block.section_path != group_section:
                flush()

            if tokens > self.max_tokens and not block.is_table and self.split_long_prose:
                flush()
                chunks.extend(self._split_prose_block(doc, block))
                continue

            if group and group_tokens + tokens > self.target_tokens:
                flush()

            group.append(block)
            group_tokens += tokens
            group_section = block.section_path

        flush()
        return chunks

    def _split_prose_block(self, doc: Document, block: Block) -> list[Chunk]:
        """Split one over-long prose block on word boundaries.

        Reuses the fixed-size packer rather than reimplementing it, but confined
        to a single block so section attribution is preserved.
        """
        inner = FixedSizeChunker(
            target_tokens=self.target_tokens,
            overlap_tokens=0,
            count_tokens=self._count,
        )
        # A one-block document sharing the parent's text, so offsets computed
        # inside it are already absolute.
        window = Document(doc_id=doc.doc_id, text=doc.text[: block.span.end], blocks=(block,))
        pieces = []
        for piece in inner.chunk(window):
            if piece.span.start >= block.span.start:
                pieces.append(piece)
        return pieces

    def oversize_report(self, n_chunks: int) -> OversizeReport:
        """Chunks that could not be kept under target, for reporting alongside metrics."""
        return OversizeReport(
            n_chunks=n_chunks,
            n_oversize=len(self._oversize),
            largest_tokens=max(self._oversize, default=0),
        )
