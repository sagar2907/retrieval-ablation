"""Parse EDGAR filing HTML into a canonical `Document` with tables preserved.

Why parse HTML directly instead of converting to PDF and using a document AI
layout model (Docling, Unstructured): EDGAR filings are *born* as HTML. Their
tables carry explicit `<tr>`/`<td>` structure with colspan and rowspan. Rendering
to PDF discards that and forces a layout model to re-infer cell boundaries from
pixel positions -- reconstructing, imperfectly, information that was never lost.
The PDF route is the right choice for a scanned or PDF-native corpus; here it
would inject avoidable error into exactly the axis this project measures.

The two genuinely hard problems in filing HTML:

**Layout tables.** Filings use `<table>` for visual arrangement as much as for
data. A cover page, a signature block, and a page header are all frequently
tables. Treating them as data tables floods the corpus with junk grids; treating
real financial statements as prose destroys the numbers. `is_data_table`
separates them on structural evidence rather than on tag name alone.

**The table of contents.** A 10-K lists "Item 1.", "Item 1A." ... in a contents
block near the front, then repeats every heading in the body. Naive heading
detection therefore finds each section twice and assigns the first, empty copy a
section path that captures nothing. Detected and suppressed by
`_suppress_contents_headings`, which uses the one reliable signal: a contents
entry is followed by almost no text before the next heading.

Determinism: parsing is a pure function of the input bytes. Two parses of the
same filing produce byte-identical `Document.text`, which the golden-file tests
enforce -- required because gold labels are character offsets into that text.
"""

from __future__ import annotations

import re
import unicodedata

import lxml.html

from .models import Block, BlockKind, Document, Span, Table

#: Elements whose end implies a paragraph boundary in the rendered document.
_BLOCK_LEVEL = frozenset(
    {
        "p",
        "div",
        "tr",
        "li",
        "br",
        "hr",
        "table",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "td",
        "th",
    }
)

_DROP = frozenset({"script", "style", "head", "noscript", "meta", "link"})

#: Separator written between blocks. Part of the canonical text and therefore
#: part of the offset space, so changing it invalidates existing gold labels.
_BLOCK_SEP = "\n\n"

#: Punctuation that can separate a heading label from its title. Written as
#: escapes because em dash and en dash are visually indistinguishable from a
#: hyphen in source, and filings use all three interchangeably.
_HEADING_SEP = ".:—–-"  # period, colon, em dash, en dash, hyphen

_PART_RE = re.compile(r"^part\s+([ivx]+)\b", re.IGNORECASE)
_ITEM_RE = re.compile(
    rf"^item\s+(\d{{1,2}}[a-z]?)\s*[{_HEADING_SEP}]?\s*(.{{0,120}})$", re.IGNORECASE
)
_NOTE_RE = re.compile(rf"^note\s+(\d{{1,2}})\s*[{_HEADING_SEP}]?\s*(.{{0,120}})$", re.IGNORECASE)

#: A heading candidate longer than this is prose that merely starts with the
#: word "Item", e.g. a cross-reference sentence.
_MAX_HEADING_CHARS = 200

#: Body text below this length between two headings makes them "tightly packed".
_MIN_SECTION_CHARS = 600

#: How many tightly-packed headings in a row constitute a table of contents. A
#: 10-K contents block lists 20+ entries; no genuine sequence of real sections is
#: this short-bodied for this long.
_MIN_CONTENTS_RUN = 5

_NUMERIC_CELL_RE = re.compile(r"\d")


def normalize_text(raw: str) -> str:
    """Collapse whitespace and canonicalise Unicode, once and irreversibly.

    Runs before any offset is assigned, which is the whole point: normalising
    later would shift every character position and silently invalidate the eval
    set. NFKC folds the typographic variants filings are full of, including the
    non-breaking spaces used for numeric alignment and full-width digits, so a
    lexical index does not treat two spellings of one figure as unrelated. NFKC
    also folds NBSP to an ordinary space, so no explicit NBSP handling is needed
    here; an earlier version replaced it separately, which was dead code.

    Zero-width and soft-hyphen characters are stripped separately because NFKC
    preserves them: they are not compatibility variants of anything, yet they
    split a word invisibly and so break token matching.
    """
    text = unicodedata.normalize("NFKC", raw)
    # U+200B zero-width space, U+200C/U+200D zero-width joiners, U+FEFF byte
    # order mark, U+00AD soft hyphen. All survive NFKC and all break tokens.
    for invisible in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"):
        text = text.replace(invisible, "")
    # Collapse runs of whitespace to a single space. Filing HTML is generated and
    # contains large amounts of indentation that carries no meaning.
    return re.sub(r"\s+", " ", text).strip()


def _cell_text(cell: lxml.html.HtmlElement) -> str:
    return normalize_text(cell.text_content())


def _expand_grid(
    element: lxml.html.HtmlElement,
) -> tuple[list[list[str]], list[list[int]]]:
    """Expand a `<table>` into a dense grid, plus the source cell of each column.

    Returns `(values, origins)`. `origins[r][c]` identifies the single `<td>` or
    `<th>` that produced that physical column, which is what makes the later
    collapse exact rather than heuristic: a run of identical adjacent values is an
    artefact of one spanning cell exactly when those columns share an origin, and
    is real data when they do not. Without this, a balance sheet row where two
    fiscal years happen to hold the same figure is indistinguishable from a row
    label smeared across three columns by colspan=3.

    Values are repeated across a span rather than placed once, because repetition
    is what preserves column alignment. A filing renders "2025" as one cell with
    colspan=3 above three physical columns holding "$", "34,550" and "". Placing
    the value once would leave "34,550" under a blank header; repeating it puts
    every physical column under the header that visually governs it.

    The cost of repetition is a very wide, very redundant grid -- filings pad
    tables with spacer columns of width 0.1% -- which `_collapse_grid` removes.
    """
    values: list[list[str]] = []
    origins: list[list[int]] = []
    # Pending rowspans: column index -> (remaining rows, value, origin)
    carry: dict[int, tuple[int, str, int]] = {}
    next_origin = 0

    for row_el in element.iter("tr"):
        row: list[str] = []
        row_origin: list[int] = []
        col = 0

        def drain_carry(col: int, row: list[str] = row, ro: list[int] = row_origin) -> int:
            """Re-emit cells still spanning down into this row, preserving columns."""
            while col in carry:
                remaining, value, origin = carry[col]
                row.append(value)
                ro.append(origin)
                if remaining - 1 <= 0:
                    del carry[col]
                else:
                    carry[col] = (remaining - 1, value, origin)
                col += 1
            return col

        col = drain_carry(col)

        for cell in row_el.iter("td", "th"):
            value = _cell_text(cell)
            next_origin += 1
            origin = next_origin
            try:
                colspan = max(1, int(cell.get("colspan", "1")))
                rowspan = max(1, int(cell.get("rowspan", "1")))
            except ValueError:
                colspan = rowspan = 1
            # Malformed filings occasionally declare enormous spans; cap them so
            # one bad attribute cannot allocate a million-column row.
            colspan = min(colspan, 64)
            rowspan = min(rowspan, 64)

            for _ in range(colspan):
                row.append(value)
                row_origin.append(origin)
                if rowspan > 1:
                    carry[col] = (rowspan - 1, value, origin)
                col += 1
                col = drain_carry(col)

        if any(c for c in row):
            values.append(row)
            origins.append(row_origin)

    width = max((len(r) for r in values), default=0)
    # Pad with a sentinel origin of 0, which no real cell uses, so padding can
    # never be mistaken for a shared origin and merged with a neighbour.
    return (
        [r + [""] * (width - len(r)) for r in values],
        [r + [0] * (width - len(r)) for r in origins],
    )


def _column_groups(values: list[list[str]], n_header_rows: int) -> list[list[int]]:
    """Partition columns into runs sharing the same header label.

    A run of columns under one repeated header label is one original colspan
    group, so grouping this way reassembles the logical cell the filing displays.
    Blank labels are never merged: those are the stub column and genuinely
    separate unlabelled columns.
    """
    width = len(values[0]) if values else 0
    if n_header_rows <= 0 or len(values) <= n_header_rows:
        return [[i] for i in range(width)]

    labels = [c.strip() for c in values[n_header_rows - 1]]
    groups: list[list[int]] = []
    for idx, label in enumerate(labels):
        if groups and label and label == labels[groups[-1][0]]:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _merge_row(row: list[str], origin_row: list[int], groups: list[list[int]]) -> list[str]:
    """Collapse one row's columns, using provenance to tell artefact from data."""
    cells: list[str] = []
    cell_origins: list[int] = []

    for group in groups:
        # Within a group: drop values repeated from a single source cell, but keep
        # genuinely distinct pieces such as "$" followed by "34,550".
        parts: list[str] = []
        seen_origin: int | None = None
        for index in group:
            value = row[index].strip()
            if value and origin_row[index] != seen_origin:
                parts.append(value)
                seen_origin = origin_row[index]
        merged = " ".join(parts)

        # Across groups: fold into the previous cell when both came from the same
        # source cell, which is how a colspan row label smeared over several
        # header groups gets reassembled. Origin 0 is padding and never folds.
        #
        # The fold decision depends on origin ALONE, never on whether this cell
        # has text. Requiring non-empty text was a real bug: a header row whose
        # blank stub spans three columns kept all three (each merging to ""),
        # while the data row below folded its colspan=3 label into one -- so the
        # header row was three cells wider than its data rows and every column
        # label lined up against the wrong value.
        first_origin = origin_row[group[0]]
        if cells and first_origin != 0 and cell_origins[-1] == first_origin:
            continue
        cells.append(merged)
        cell_origins.append(first_origin)

    return cells


def _collapse_grid(
    grid: list[list[str]],
    origins: list[list[int]],
    n_header_rows: int,
) -> list[list[str]]:
    """Remove spacer columns and undo colspan smearing, using cell provenance.

    Filing tables are laid out, not tabulated. A three-column financial table
    arrives as thirty physical columns: currency symbols, percent signs, and
    sub-1%-width spacers each occupy their own column. Rendered naively this
    produces rows like `| R&D | $ | 34,550 |  |  |  | 10 | % |` with a header row
    repeating "2025" fifteen times -- unreadable, and actively harmful to lexical
    matching because the repetition inflates term frequencies.

    Three passes:

    1.  Drop columns empty across all *data* rows. Judging emptiness on all rows
        keeps every spacer alive, because `_expand_grid` repeats a spanning
        header across the spacers it covers, so a 0.1%-wide spacer under a "2025"
        header is non-empty in the header row despite never holding a value.

    2.  Merge adjacent columns sharing the same header label. This reassembles
        "$" and "34,550" into the single logical cell the filing displays.

    3.  Collapse adjacent cells that share a *source cell*. This is the pass that
        needs `origins`. A row label under colspan=3 lands in three columns that
        may fall into three different header groups, so pass 2 alone leaves
        "Cash and cash equivalents | Cash and cash equivalents | Cash and cash
        equivalents".

        Collapsing on equal *text* would fix that but would also silently merge a
        balance sheet row where two fiscal years genuinely hold the same figure,
        destroying real data. Collapsing on equal *origin* distinguishes the two
        exactly: same origin means one cell was stretched, different origins mean
        two cells coincidentally agree.
    """
    if not grid:
        return grid

    width = len(grid[0])

    data_rows = range(n_header_rows, len(grid)) if n_header_rows < len(grid) else range(len(grid))
    keep = [c for c in range(width) if any(grid[r][c].strip() for r in data_rows)]
    if not keep:
        keep = [c for c in range(width) if any(row[c].strip() for row in grid)]
    if not keep:
        return []

    values = [[row[c] for c in keep] for row in grid]
    provenance = [[row[c] for c in keep] for row in origins]
    groups = _column_groups(values, n_header_rows)

    out = [_merge_row(row, provenance[index], groups) for index, row in enumerate(values)]

    # Rows collapse to different widths -- a full-width "ASSETS:" heading row
    # legitimately becomes a single cell. Pad to a common width so the rendering
    # stays a rectangle, which is exactly how such a row displays in the filing.
    final_width = max((len(r) for r in out), default=0)
    return [r + [""] * (final_width - len(r)) for r in out]


def parse_table(element: lxml.html.HtmlElement) -> Table:
    """Convert a `<table>` into a compact logical grid."""
    expanded, origins = _expand_grid(element)
    n_header = _guess_header_rows(expanded)
    collapsed = _collapse_grid(expanded, origins, n_header)
    if collapsed and n_header >= len(collapsed):
        n_header = max(0, len(collapsed) - 1)
    return Table(rows=tuple(tuple(r) for r in collapsed), n_header_rows=n_header)


def _guess_header_rows(grid: list[list[str]]) -> int:
    """Leading rows that label columns rather than carry data.

    Primary rule: **a header row has an empty stub cell.** Financial tables put
    the row label in column zero and leave it blank while the column headers are
    being declared, so the first row whose column zero is non-empty is the first
    data row. This is close to universal in filings and survives multi-level
    headers ("Years ended September 27," stacked over "2025 2024 2023").

    An earlier version instead assumed header rows contain no digits. That is
    wrong on exactly the tables that matter: the headers of a financial statement
    are years. It classified every income statement as having zero header rows,
    which stripped the column labels the row-sentence rendering depends on.
    """
    count = 0
    for row in grid[:4]:
        if row and not row[0].strip():
            count += 1
        else:
            break
    if count:
        # Never treat the entire table as header; something must remain as data.
        return min(count, max(0, len(grid) - 1))

    # No blank stub anywhere: fall back to treating a first row free of
    # quantity-shaped cells as a single header row.
    if grid:
        tail = grid[0][1:]
        if tail and not any(_NUMERIC_CELL_RE.search(c) for c in tail):
            return 1
    return 0


def is_data_table(element: lxml.html.HtmlElement) -> bool:
    """Whether a `<table>` holds tabular data rather than page layout.

    Structural evidence only, in deliberate order:

    - A table containing another table is a layout container. Financial data
      tables in filings do not nest.
    - Fewer than two rows or two columns cannot express a relation.
    - A data table has numbers in it. Filing layout tables (cover page blocks,
      signature grids, page headers) are almost entirely prose.

    Imperfect by nature. `parse_filing` therefore keeps rejected tables as
    ordinary text rather than discarding them, so a misclassification costs
    table structure but never content.
    """
    if any(True for _ in element.iterdescendants("table")):
        return False

    rows = list(element.iter("tr"))
    if len(rows) < 2:
        return False

    widths = [len(list(r.iter("td", "th"))) for r in rows]
    if max(widths, default=0) < 2:
        return False

    cells = [_cell_text(c) for r in rows for c in r.iter("td", "th")]
    non_empty = [c for c in cells if c]
    if len(non_empty) < 4:
        return False

    numeric = sum(1 for c in non_empty if _NUMERIC_CELL_RE.search(c))
    return numeric / len(non_empty) >= 0.2


class _Builder:
    """Accumulates canonical text and the blocks that index into it."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self._pending: list[str] = []
        self.blocks: list[Block] = []
        self._counter = 0
        self._part: str | None = None
        self._item: str | None = None
        self._note: str | None = None

    # -- text -----------------------------------------------------------------

    def add(self, text: str) -> None:
        if text:
            self._pending.append(text)

    def _append_raw(self, text: str) -> Span:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        self.parts.append(_BLOCK_SEP)
        self.length += len(_BLOCK_SEP)
        # The span covers the block's own text only. Excluding the separator
        # keeps `document.slice(block.span)` free of trailing whitespace, which
        # matters because those slices become chunk text and prompt content.
        return Span(start, start + len(text))

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:05d}"

    # -- blocks ---------------------------------------------------------------

    def flush_text(self) -> None:
        """Finalise accumulated inline text as a paragraph or heading block."""
        text = normalize_text("".join(self._pending))
        self._pending.clear()
        if not text:
            return

        kind = BlockKind.PARAGRAPH
        if len(text) <= _MAX_HEADING_CHARS:
            kind = self._classify_heading(text)

        span = self._append_raw(text)
        self.blocks.append(
            Block(
                block_id=self._next_id("b"),
                kind=kind,
                span=span,
                section_path=self.section_path,
            )
        )

    def flush_table(self, table: Table, rendered: str) -> None:
        self.flush_text()
        if not rendered:
            return
        span = self._append_raw(rendered)
        self.blocks.append(
            Block(
                block_id=self._next_id("t"),
                kind=BlockKind.TABLE,
                span=span,
                section_path=self.section_path,
                table=table,
            )
        )

    # -- section tracking -----------------------------------------------------

    @property
    def section_path(self) -> tuple[str, ...]:
        return tuple(p for p in (self._part, self._item, self._note) if p)

    def _classify_heading(self, text: str) -> BlockKind:
        """Update the section cursor if `text` is a heading, and report the kind.

        Order matters. A Part heading resets Item and Note; an Item heading
        resets Note. Without the resets, "Note 12" from Item 8 would remain
        attached to passages in Item 9, mislabelling their provenance and
        corrupting any metadata-filtered retrieval configuration.
        """
        if match := _PART_RE.match(text):
            self._part = f"Part {match.group(1).upper()}"
            self._item = None
            self._note = None
            return BlockKind.HEADING

        if match := _ITEM_RE.match(text):
            title = match.group(2).strip(" .:—–-")
            label = f"Item {match.group(1).upper()}"
            self._item = f"{label}. {title}" if title else label
            self._note = None
            return BlockKind.HEADING

        if match := _NOTE_RE.match(text):
            title = match.group(2).strip(" .:—–-")
            label = f"Note {match.group(1)}"
            self._note = f"{label} - {title}" if title else label
            return BlockKind.HEADING

        return BlockKind.PARAGRAPH

    def text(self) -> str:
        return "".join(self.parts)


def _walk(element: lxml.html.HtmlElement, builder: _Builder, render_tables: str) -> None:
    tag = element.tag
    if not isinstance(tag, str) or tag in _DROP:
        # Comments and processing instructions have a callable .tag in lxml.
        return

    if tag == "table" and is_data_table(element):
        table = parse_table(element)
        rendered = (
            table.to_row_sentences() if render_tables == "row_sentences" else table.to_markdown()
        )
        # Deliberately NOT passed through normalize_text: that collapses all
        # whitespace including newlines, which would flatten the rendered table
        # into a single unreadable line and destroy row boundaries.
        builder.flush_table(table, rendered)
        # Not recursed into: its content is fully represented by the grid.
        return

    if element.text:
        builder.add(element.text)

    for child in element:
        _walk(child, builder, render_tables)
        if child.tail:
            builder.add(child.tail)

    if tag in _BLOCK_LEVEL:
        builder.flush_text()


def _suppress_contents_headings(blocks: list[Block], text: str) -> list[Block]:
    """Reclassify table-of-contents entries from HEADING to BOILERPLATE.

    A contents entry and a real section heading are textually identical, so the
    signal has to come from context. The reliable one is that a contents block is
    a long *run* of headings with almost nothing between them.

    An earlier version suppressed any heading followed by less than
    `_MIN_SECTION_CHARS` of text, judging each heading in isolation. That was
    wrong and measurably so: it deleted "PART I" (always immediately followed by
    "Item 1", so the gap is tiny by construction), "Item 4. Mine Safety
    Disclosures" and "Item 6. [Reserved]" (genuinely one-line sections), and
    "Item 2. Properties" (short but real). Losing the Part headings meant no
    passage in the corpus carried a Part in its section path at all.

    Requiring a run of at least `_MIN_CONTENTS_RUN` tightly-packed headings was
    the second attempt, and still not enough on its own. The last contents entry
    sits immediately before the first body heading, so the run does not stop at
    the end of the contents block -- it continues into the body and swallows the
    first real "PART I" and "Item 1" with it.

    So a second condition is required, and it is the decisive one: **a contents
    entry's text appears again later in the document.** That is what a table of
    contents *is*. Requiring both conditions means a repeated heading inside a
    dense run is contents, while the final occurrence of that text -- the body
    heading -- is always kept, and a genuinely short section that never repeats is
    never touched.

    Reclassified rather than removed. Deleting them would be the obvious fix and
    is wrong -- every block after the deletion would keep its original offsets
    while the caller assumed a compacted document, and the mismatch would surface
    only as subtly wrong gold labels much later.
    """
    positions = [i for i, b in enumerate(blocks) if b.kind is BlockKind.HEADING]
    if len(positions) < _MIN_CONTENTS_RUN:
        return blocks

    def heading_text(block_index: int) -> str:
        block = blocks[block_index]
        return text[block.span.start : block.span.end].strip().casefold()

    # Last heading position at which each heading text occurs. Anything earlier
    # than this is a duplicate of something that appears again later.
    last_occurrence: dict[str, int] = {}
    for block_index in positions:
        last_occurrence[heading_text(block_index)] = block_index

    def gap_after(idx: int) -> int:
        """Characters of body text between heading `idx` and the next heading."""
        here = positions[idx]
        nxt = positions[idx + 1] if idx + 1 < len(positions) else None
        end = blocks[nxt].span.start if nxt is not None else len(text)
        return end - blocks[here].span.end

    # Maximal runs of consecutive headings separated by almost no text.
    runs: list[list[int]] = []
    current: list[int] = [0]
    for idx in range(len(positions) - 1):
        if gap_after(idx) < _MIN_SECTION_CHARS:
            current.append(idx + 1)
        else:
            runs.append(current)
            current = [idx + 1]
    runs.append(current)

    suppress = {
        positions[i]
        for run in runs
        if len(run) >= _MIN_CONTENTS_RUN
        for i in run
        # Keep the final occurrence: that is the body heading, not the listing.
        if last_occurrence[heading_text(positions[i])] != positions[i]
    }
    if not suppress:
        return blocks

    return [
        Block(
            block_id=b.block_id,
            kind=BlockKind.BOILERPLATE if i in suppress else b.kind,
            span=b.span,
            section_path=b.section_path,
            table=b.table,
        )
        for i, b in enumerate(blocks)
    ]


def _reassign_section_paths(blocks: list[Block]) -> list[Block]:
    """Recompute section paths after contents suppression.

    Must run after `_suppress_contents_headings`, because the first pass assigned
    paths while walking and treated contents entries as real headings. Every
    block between the contents block and the body therefore carries a section
    path derived from the contents, which would be wrong.
    """
    part: str | None = None
    item: str | None = None
    note: str | None = None
    out: list[Block] = []

    for block in blocks:
        if block.kind is BlockKind.HEADING:
            head = block.section_path
            # The walk stored the cumulative path on each heading; the deepest
            # component is what that heading introduced.
            if head:
                deepest = head[-1]
                if deepest.startswith("Part "):
                    part, item, note = deepest, None, None
                elif deepest.startswith("Item "):
                    item, note = deepest, None
                elif deepest.startswith("Note "):
                    note = deepest

        path = tuple(p for p in (part, item, note) if p)
        out.append(
            Block(
                block_id=block.block_id,
                kind=block.kind,
                span=block.span,
                section_path=path,
                table=block.table,
            )
        )
    return out


_XML_DECL_RE = re.compile(rb"^\s*<\?xml[^>]*\?>")
_XML_ENCODING_RE = re.compile(rb"<\?xml[^>]*encoding\s*=", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"<meta[^>]*charset", re.IGNORECASE)


def _declares_encoding(payload: bytes) -> bool:
    """Whether the document states its own encoding.

    Only the head of the document is examined: a declaration has to appear before
    content to be usable, and scanning a 10 MB filing for it is wasteful.
    """
    head = payload[:4096]
    return bool(_XML_ENCODING_RE.search(head) or _META_CHARSET_RE.search(head))


def _to_tree(html: str | bytes) -> lxml.html.HtmlElement:
    """Build a tree, letting the document's own encoding declaration win.

    Bytes are the primary contract because filings are inline XBRL: XHTML with
    an `<?xml version="1.0" encoding="..."?>` prologue. lxml refuses a `str`
    carrying such a declaration, and rightly so -- the string has already been
    decoded by somebody's guess, so the declaration can no longer be honoured.

    A `str` input is still accepted for tests and fixtures. There the
    declaration is stripped rather than trusted, because by that point the
    caller has already committed to a decoding and pretending otherwise would
    hide a real inconsistency.
    """
    if isinstance(html, str):
        payload = _XML_DECL_RE.sub(b"", html.encode("utf-8"))
        declared = True  # we just encoded it ourselves, so the encoding is known
    else:
        payload = html
        declared = _declares_encoding(payload)

    if not payload.strip():
        raise ValueError("cannot parse empty document")

    if declared:
        return lxml.html.fromstring(payload)

    # No declaration anywhere. lxml then falls back to a legacy single-byte
    # encoding, which turns "Café" into "CafÃ©" -- mis-decoded text would become
    # the canonical text and every gold-label offset would be built on top of it.
    # UTF-8 is the correct assumption for undeclared EDGAR documents; if it does
    # not decode, fall back rather than crash, since a slightly wrong character
    # is better than losing the filing entirely.
    parser = lxml.html.HTMLParser(encoding="utf-8")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        parser = lxml.html.HTMLParser(encoding="cp1252")
    return lxml.html.fromstring(payload, parser=parser)


def parse_filing(
    doc_id: str,
    html: str | bytes,
    metadata: dict[str, str] | None = None,
    render_tables: str = "markdown",
) -> Document:
    """Parse filing HTML into a `Document`.

    `render_tables` selects how a table becomes retrievable text -- "markdown"
    for a compact pipe table, "row_sentences" to repeat column headers next to
    every value. Which retrieves better is an ablation variable, not a default
    to be assumed, so it is a parameter rather than a hardcoded choice.
    """
    if render_tables not in {"markdown", "row_sentences"}:
        raise ValueError(f"unknown render_tables={render_tables!r}")

    tree = _to_tree(html)
    for element in tree.xpath("//script | //style | //noscript"):
        element.drop_tree()

    builder = _Builder()
    _walk(tree, builder, render_tables)
    builder.flush_text()

    text = builder.text()
    blocks = _suppress_contents_headings(builder.blocks, text)
    blocks = _reassign_section_paths(blocks)

    return Document(
        doc_id=doc_id,
        text=text,
        blocks=tuple(blocks),
        metadata=dict(metadata or {}),
    )
