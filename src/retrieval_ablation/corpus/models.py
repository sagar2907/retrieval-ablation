"""The parsed document representation, and the span algebra the eval set rests on.

THE CENTRAL DESIGN DECISION OF THIS PROJECT LIVES HERE.

A retrieval ablation varies the chunker. So gold relevance labels must not refer
to chunks. If a gold label says "chunk 47 of document X is the answer", then
switching from fixed-size to structure-aware chunking renumbers every chunk and
silently invalidates the entire eval set -- and the resulting ablation table
would be comparing systems against different ground truth while appearing to
compare them against the same. That failure is invisible in the output: every
number still looks plausible.

So labels anchor to **character spans in the document's canonical text**, which
no chunker may alter. A chunk is then judged relevant to a gold span by span
overlap, computed at scoring time. Consequences:

  - The canonical `Document.text` is immutable ground truth. Chunkers may only
    slice it; they may never rewrite, re-normalise, or reflow it. Any
    normalisation must happen once, during parsing, before offsets are assigned.
  - The same eval set scores every chunking configuration without modification,
    which is what makes the chunking axis of the ablation meaningful at all.
  - Adding a new chunker later requires no relabelling.

`Block` records where structural elements (paragraphs, headings, tables) sit
within that text, so a structure-aware chunker can respect boundaries and a
table can be kept whole -- but blocks are metadata *about* offsets, never a
replacement for them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Self


class BlockKind(enum.StrEnum):
    """What a block of document text structurally is."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    #: Page furniture, running heads, and similar boilerplate. Retained with
    #: offsets rather than deleted, because deleting it would shift every
    #: downstream character offset and break span stability.
    BOILERPLATE = "boilerplate"


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A half-open character interval [start, end) into a document's text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap_length(self, other: Self | Span) -> int:
        """Number of characters shared with `other`."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def coverage_of(self, other: Span) -> float:
        """Fraction of `other` that this span contains, in [0, 1].

        Deliberately asymmetric. Relevance asks "does this chunk contain the
        gold passage?", not "do these two spans resemble each other". A Jaccard
        or symmetric measure would penalise a large chunk that fully contains a
        short gold span, which is a correct retrieval, and would reward a chunk
        that happens to be the same size as the gold span while overlapping it
        only halfway.
        """
        if other.length == 0:
            return 0.0
        return self.overlap_length(other) / other.length

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True, slots=True)
class Table:
    """A parsed table: a rectangular grid plus how many leading rows are headers.

    Stored as a grid rather than pre-rendered text so that different
    linearisation strategies can be compared without re-parsing the filing.
    Financial tables are the case naive pipelines lose, and how a table is
    turned into retrievable text is itself an experimental variable.
    """

    rows: tuple[tuple[str, ...], ...]
    n_header_rows: int = 1
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_markdown(self) -> str:
        """Render as a pipe table.

        Header rows are emitted as-is and separated by a delimiter row so the
        text remains legible to both a lexical index and a language model.
        """
        if not self.rows:
            return ""
        width = self.n_cols
        lines: list[str] = []
        if self.caption:
            lines.append(self.caption)
        for i, row in enumerate(self.rows):
            padded = list(row) + [""] * (width - len(row))
            lines.append("| " + " | ".join(c.strip() for c in padded) + " |")
            if i == self.n_header_rows - 1:
                lines.append("|" + "|".join([" --- "] * width) + "|")
        return "\n".join(lines)

    def to_row_sentences(self) -> str:
        """Render each data row as a self-contained line carrying its headers.

        A pipe table is compact but positional: the cell "12,345" three columns
        into row nine means nothing once retrieval returns it out of context, and
        a lexical index cannot match "fiscal 2024 research and development
        expense" against a bare number. Repeating the column header alongside
        every value trades size for retrievability.

        Which of these two renderings actually retrieves better is measured, not
        assumed -- it is one of the table-handling variants in the ablation.
        """
        if self.n_rows <= self.n_header_rows:
            return self.to_markdown()

        header = self.rows[self.n_header_rows - 1] if self.n_header_rows else ()
        out: list[str] = []
        if self.caption:
            out.append(self.caption)
        for row in self.rows[self.n_header_rows :]:
            if not row:
                continue
            label = row[0].strip()
            pairs = [
                f"{header[i].strip()}: {cell.strip()}"
                for i, cell in enumerate(row)
                if i > 0 and cell.strip() and i < len(header) and header[i].strip()
            ]
            out.append(f"{label} -- " + "; ".join(pairs) if pairs else label)
        return "\n".join(out)


@dataclass(frozen=True, slots=True)
class Block:
    """A structural element located by character span in the document text."""

    block_id: str
    kind: BlockKind
    span: Span
    #: Hierarchical location, outermost first, e.g.
    #: ("Part II", "Item 8. Financial Statements", "Note 12 -- Income Taxes").
    #: Used by the structure-aware chunker and carried into chunk metadata so
    #: retrieval results can be filtered and displayed with provenance.
    section_path: tuple[str, ...] = ()
    table: Table | None = None

    def __post_init__(self) -> None:
        if (self.kind is BlockKind.TABLE) != (self.table is not None):
            raise ValueError(
                f"block {self.block_id}: kind={self.kind} and table="
                f"{'set' if self.table else 'None'} disagree"
            )

    @property
    def is_table(self) -> bool:
        return self.kind is BlockKind.TABLE


@dataclass(frozen=True, slots=True)
class Document:
    """A parsed filing.

    `text` is canonical and immutable: every span in `blocks`, every gold label,
    and every chunk refers to offsets into this exact string. Two parses of the
    same filing must produce byte-identical text or the eval set no longer
    applies, which is why the parser is covered by golden-file tests.
    """

    doc_id: str
    text: str
    blocks: tuple[Block, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = len(self.text)
        for block in self.blocks:
            if block.span.end > length:
                raise ValueError(
                    f"block {block.block_id} ends at {block.span.end} but "
                    f"document {self.doc_id} has only {length} characters"
                )
        # Blocks must be ordered and non-overlapping. Overlapping blocks would
        # make "the section this offset belongs to" ambiguous, and the
        # structure-aware chunker would emit duplicated text.
        for prev, nxt in zip(self.blocks, self.blocks[1:], strict=False):
            if nxt.span.start < prev.span.end:
                raise ValueError(
                    f"blocks {prev.block_id} and {nxt.block_id} overlap in document {self.doc_id}"
                )

    def slice(self, span: Span) -> str:
        return self.text[span.start : span.end]

    def blocks_overlapping(self, span: Span) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.span.overlap_length(span) > 0)

    def section_path_at(self, offset: int) -> tuple[str, ...]:
        """Section path of the block containing `offset`, or () if outside any."""
        for block in self.blocks:
            if block.span.start <= offset < block.span.end:
                return block.section_path
        return ()


@dataclass(frozen=True, slots=True)
class GoldPassage:
    """A labelled answer location: a document and a character span within it.

    Chunker-independent by construction. `passage_id` is a stable handle for
    reporting and for the published dataset; scoring uses the span.
    """

    passage_id: str
    doc_id: str
    span: Span
    #: Graded relevance. 2 = passage directly answers the query, 1 = partially
    #: relevant or supporting. Kept graded rather than binary because nDCG can
    #: use the distinction, and collapsing to binary later is lossless whereas
    #: inventing grades later is not.
    gain: int = 2
