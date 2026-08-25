"""Render docs/learning.md to PDF, then verify the result by rasterising it.

WHY THERE IS A VERIFICATION STEP AT ALL

A PDF that is missing a font renders the affected characters as blank boxes, or as
nothing. The file is still valid, the byte count still looks right, and the
generator reports success. Checking the exit code proves nothing. The only honest
check is to rasterise the pages and look at what actually landed on them, which is
what `verify()` does: it renders every page to an image, confirms each carries a
plausible amount of ink, and extracts the text back out to confirm known strings
survived the round trip.

WHY fpdf2 AND NOT A HTML-BASED PIPELINE

WeasyPrint and wkhtmltopdf produce prettier output and both need native libraries
that Windows Smart App Control blocks on this machine, the same policy that stops
PyTorch loading. fpdf2 is pure Python, so it runs where the rest of the project
runs. The markdown handling here is deliberately small -- headings, paragraphs,
lists, tables, code blocks, bold and inline code -- because that is all the
document uses.

    python scripts/render_pdf.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "learning.md"
OUTPUT = ROOT / "docs" / "retrieval-ablation-learning.pdf"
PREVIEW_DIR = ROOT / "docs" / "_preview"

#: Core-14 fonts are metric-standard and embedded in every PDF reader, so no font
#: file has to be found, licensed or shipped. The trade-off is Latin-1 only, which
#: `sanitise` handles explicitly rather than letting it fail at write time.
BODY = "Helvetica"
MONO = "Courier"

#: Characters the document uses that Latin-1 cannot encode. Mapped deliberately
#: rather than dropped: silently deleting a minus sign from "-0.0069" would change
#: a reported number, which is the one thing this project must never do.
_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
    "×": "x",
    "≈": "~",
    "≥": ">=",
    "≤": "<=",
    "→": "->",
    "•": "-",
    "·": "-",
    " ": " ",
    "±": "+/-",
    "²": "^2",
    "α": "alpha",
    "Δ": "delta",
    "√": "sqrt",
    "₂": "2",
    "©": "(c)",
    # Marks "significant at 0.05" in the results tables. Unmapped, it became a bare
    # '?' sitting next to a p-value, which reads as doubt about the number rather
    # than as the flag it is. '*' is the conventional notation and needs no glyph.
    "✓": "*",
    "✗": "x",
}

#: Characters `sanitise` had to destroy because nothing maps them. Collected rather
#: than ignored: the old code turned them into '?' and only a *run* of two or more
#: was reported, so a single lost glyph shipped -- which is exactly what happened to
#: the significance tick. One lost character is already a corrupted document.
UNMAPPED: set[str] = set()

#: Table cells that did not fit and were shortened. A shortened number or metric
#: name is a changed fact, so this is reported rather than accepted.
TRUNCATED: list[tuple[str, str]] = []


#: A whole-line HTML comment, which markdown viewers hide and this renderer used
#: to print. Anchored so a comment sharing a line with prose is left alone rather
#: than half-removed.
HTML_COMMENT = re.compile(r"\s*<!--.*?-->\s*")


def sanitise(text: str) -> str:
    """Make text Latin-1 safe, mapping known characters and flagging the rest."""
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    # Anything still unencodable becomes '?' rather than raising deep inside the
    # writer, but the character is recorded first so the run can fail naming it.
    UNMAPPED.update(c for c in text if c.encode("latin-1", "ignore") == b"")
    return text.encode("latin-1", "replace").decode("latin-1")


@dataclass
class Style:
    size: float
    style: str = ""
    space_before: float = 0.0
    space_after: float = 1.5
    font: str = BODY


HEADINGS = {
    1: Style(20, "B", 6, 4),
    2: Style(15, "B", 6, 3),
    3: Style(12, "B", 4, 2),
    4: Style(11, "B", 3, 1.5),
}
BODY_STYLE = Style(10)


class Renderer(FPDF):
    def header(self) -> None:  # noqa: D102 - FPDF hook
        if self.page_no() == 1:
            return
        self.set_font(BODY, "", 8)
        self.set_text_color(130)
        self.cell(0, 6, "retrieval-ablation - a first-principles walkthrough", align="L")
        self.ln(8)
        self.set_text_color(0)

    def footer(self) -> None:  # noqa: D102 - FPDF hook
        self.set_y(-14)
        self.set_font(BODY, "", 8)
        self.set_text_color(130)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(0)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")

#: Splits a paragraph into styled and unstyled runs. Bold must precede italic in
#: the alternation so `**x**` is not consumed as two italic markers.
_SPAN_RE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*\n]+\*(?!\*)|`[^`]+`)")


def write_rich(pdf: Renderer, text: str, size: float) -> None:
    """Write a paragraph, honouring **bold**, *italic* and `code` spans.

    Takes a whole paragraph, not a source line. Markdown wraps paragraphs across
    lines, so an emphasis span frequently straddles a line break; rendering line
    by line meant those spans never matched their closing marker and the raw
    asterisks were printed to the page. Verified by looking at a rasterised page,
    which is the only way that class of defect shows up -- the text extracts
    cleanly and every automated check passes while the reader sees `**cosine
    similarity**`.

    Runs are emitted with `write()` rather than `multi_cell()` so several styles
    share a line; `multi_cell` breaks at every style change and would turn an
    emphasised phrase into its own paragraph.
    """
    pdf.set_font(BODY, "", size)
    for part in _SPAN_RE.split(text):
        if not part:
            continue
        if bold := _BOLD_RE.fullmatch(part):
            _emit_styled(pdf, bold.group(1), "B", size)
        elif italic := _ITALIC_RE.fullmatch(part):
            _emit_styled(pdf, italic.group(1), "I", size)
        elif code := _CODE_RE.fullmatch(part):
            _emit(pdf, sanitise(code.group(1)), MONO, "", size - 0.5)
        else:
            _emit(pdf, sanitise(part), BODY, "", size)
    pdf.ln(6)


def write_indented(pdf: Renderer, text: str, offset: float) -> None:
    """Write a block indented by `offset`, keeping wrapped lines aligned with it.

    Setting only the x position indents the first line: `write()` wraps against the
    page's left margin, so a bullet long enough to wrap put its continuation flush
    left and the list visibly came apart. Moving the margin for the duration makes
    the whole item hang together, and it is restored immediately because the page
    header and footer are drawn with whatever margin is current.
    """
    original = pdf.l_margin
    pdf.set_left_margin(original + offset)
    pdf.set_x(original + offset)
    try:
        write_rich(pdf, text, BODY_STYLE.size)
    finally:
        pdf.set_left_margin(original)
        pdf.set_x(original)


def _emit_styled(pdf: Renderer, inner: str, style: str, size: float) -> None:
    """Emit a bold or italic run that may itself contain `code` spans.

    `_SPAN_RE` finds top-level spans only, so a bold run wrapping inline code was
    emitted whole and its backticks printed as literal characters -- 22 of them in
    the document, mostly in the module-by-module list where every entry is a bold
    filename in backticks. Splitting the inner text one level deeper renders the
    code in Courier while keeping the outer weight, which is what the markdown
    means. Courier-Bold and Courier-Oblique are both core-14, so no font is needed.
    """
    for index, piece in enumerate(_CODE_RE.split(inner)):
        if not piece:
            continue
        # re.split with one capture group alternates plain, captured, plain, ...
        if index % 2:
            _emit(pdf, sanitise(piece), MONO, style, size - 0.5)
        else:
            _emit(pdf, sanitise(piece), BODY, style, size)


def _emit(pdf: Renderer, text: str, font: str, style: str, size: float) -> None:
    """Write one styled run, wrapping to a new line rather than splitting a word.

    `write()` continues from the current x position, so a styled run beginning
    near the right margin can have its very first token broken mid-word. That is
    how "526,296" came out as "526" at the end of one line and ",296" at the
    start of the next: fpdf treats the comma as a break opportunity when the
    token does not fit the space remaining.

    The bug only appears when a style change lands near the margin, which is why
    it survived the automated checks -- the text extracts with a newline inside
    the number, so a naive substring search for the figure fails, and every
    ink-coverage check passes. On a document whose entire point is reported
    figures, a number split across lines is not cosmetic.

    Moving to the next line when the first token cannot fit preserves the token.
    """
    if not text:
        return
    pdf.set_font(font, style, size)
    # c_margin is the padding fpdf reserves inside every cell, on both sides.
    # Omitting it made the usable width look wider than it is, so a word that
    # "fitted" by this check still overflowed fpdf's own limit and was split --
    # which is how "project" rendered as "proje" / "ct" after the first fix.
    right = pdf.w - pdf.r_margin - 2 * pdf.c_margin

    # Emitted word by word, with the wrap decided here rather than by fpdf.
    # Guarding only the first token of a run was not enough -- any token can
    # arrive at the margin, and "together" came out as "togethe" / "r". Deciding
    # every break explicitly removes the whole class rather than the instance.
    for word in re.split(r"( )", text):
        if not word:
            continue
        if word == " ":
            # A space at the very start of a line is dropped, otherwise every
            # wrapped line would begin with a visible indent.
            if pdf.get_x() > pdf.l_margin:
                pdf.write(5, " ")
            continue
        width = pdf.get_string_width(word)
        if pdf.get_x() + width > right and pdf.get_x() > pdf.l_margin:
            pdf.ln(5)
        # A single word wider than the whole column cannot be preserved; fpdf
        # splits it, which is correct here and only affects pathological input.
        pdf.write(5, word)


def strip_markup(text: str) -> str:
    """Remove emphasis markers, for contexts that cannot render styled runs."""
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    return _CODE_RE.sub(r"\1", text)


def write_table(pdf: Renderer, rows: list[list[str]]) -> None:
    """Render a pipe table, sized to the page.

    Column widths are proportional to the longest cell in each column rather than
    equal, because the tables here pair one long configuration name with several
    short numeric columns and equal widths would wrap the names to three lines.
    """
    if not rows:
        return
    width = pdf.w - pdf.l_margin - pdf.r_margin
    columns = max(len(r) for r in rows)
    rows = [r + [""] * (columns - len(r)) for r in rows]

    widths = _column_widths(pdf, rows, columns, width)

    pdf.ln(1)
    for index, row in enumerate(rows):
        header = index == 0
        pdf.set_font(BODY, "B" if header else "", 8)
        if header:
            pdf.set_fill_color(232, 236, 242)
        height = 5.5
        if pdf.get_y() + height > pdf.h - 20:
            pdf.add_page()
        for cell, cell_width in zip(row, widths, strict=True):
            clean = sanitise(strip_markup(cell))
            intended = clean
            # Truncate rather than wrap: a wrapped cell would desynchronise this
            # row's height from its neighbours' and stagger the grid.
            while pdf.get_string_width(clean) > cell_width - 2 and len(clean) > 1:
                clean = clean[:-1]
            if clean != intended:
                TRUNCATED.append((intended, clean))
            pdf.cell(cell_width, height, clean, border=1, fill=header)
        pdf.ln(height)
    pdf.ln(2)


def _column_widths(pdf: Renderer, rows: list[list[str]], columns: int, width: float) -> list[float]:
    """Column widths measured in the font each row is actually drawn in.

    The previous version sized columns from `len()`, in characters. The header row
    is drawn in bold, and bold Helvetica is wider per character than regular, so a
    header could overflow a column sized for the same number of regular-weight
    characters and be silently shaved. That is how "nDCG@10" printed as "nDCG@1" --
    not a clipped label but the name of a different metric, in a document whose
    whole subject is which metric said what.

    Measuring with `get_string_width` under the correct font removes the
    approximation rather than padding around it.
    """
    pad = 2 * pdf.c_margin + 0.8
    need = [1.0] * columns
    for index, row in enumerate(rows):
        pdf.set_font(BODY, "B" if index == 0 else "", 8)
        for column, cell in enumerate(row):
            text = sanitise(strip_markup(cell))
            need[column] = max(need[column], pdf.get_string_width(text) + pad)
    total = sum(need)
    # Scaled to fill the text block exactly. If the content genuinely cannot fit,
    # this shrinks columns and truncation happens -- which TRUNCATED then reports
    # instead of letting it pass as it did before.
    return [w * width / total for w in need]


def render(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    """Convert the markdown source to a PDF."""
    # Reset the two collectors so a second render in the same process reports its
    # own problems rather than the previous run's -- otherwise a passing render
    # could inherit a failure and a failing one could be masked by ordering.
    UNMAPPED.clear()
    TRUNCATED.clear()
    lines = source.read_text(encoding="utf-8").splitlines()

    pdf = Renderer(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    in_code = False
    code: list[str] = []
    table: list[list[str]] = []
    paragraph: list[str] = []
    quote: list[str] = []
    item: list[str] = []
    item_offset = 4.0

    def flush_table() -> None:
        nonlocal table
        if table:
            write_table(pdf, table)
            table = []

    def flush_paragraph() -> None:
        """Emit accumulated body lines as one paragraph.

        Markdown wraps a paragraph across several source lines; joining them
        before rendering is what lets an emphasis span that straddles a line
        break be recognised, and it also lets fpdf fill lines to the margin
        instead of inheriting the source file's arbitrary wrap points.
        """
        nonlocal paragraph
        if paragraph:
            write_rich(pdf, " ".join(paragraph), BODY_STYLE.size)
            paragraph = []

    def flush_quote() -> None:
        """Emit accumulated blockquote lines as one run, for the same reason.

        Blockquotes were rendered line by line, so the joining fix above never
        applied to them: the project's central design decision is stated as a
        two-line blockquote, and its `**` markers printed as literal asterisks
        because the opening marker's line never contained the closing one.
        """
        nonlocal quote
        if quote:
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_text_color(70)
            write_rich(pdf, " ".join(quote), BODY_STYLE.size)
            pdf.set_text_color(0)
            pdf.set_x(pdf.l_margin)
            quote = []

    def flush_item() -> None:
        """Emit an accumulated list item, continuation lines included.

        Markdown continues a list item on an indented following line. Those lines
        used to fall through to the paragraph accumulator and be emitted as a
        separate, unindented paragraph, so a two-line bullet visibly broke out of
        its list -- and any emphasis straddling the break lost its markers too. The
        third bug of this shape: blockquotes, then paragraphs, now list items.
        """
        nonlocal item
        if item:
            write_indented(pdf, " ".join(item), item_offset)
            item = []

    def flush_text() -> None:
        """Close whichever text block is open; only one ever is."""
        flush_item()
        flush_quote()
        flush_paragraph()

    for raw in lines:
        line = raw.rstrip()

        # An HTML comment is invisible in every markdown viewer, so a reader who
        # checked the rendered document would never see one -- but this renderer
        # printed them verbatim, and the generated-table markers put six of them
        # into the PDF while its own verification still reported success. The gate
        # only looks for expected strings and ink, so it cannot see text that should
        # not be there at all. Stripped only outside code fences: inside one, the
        # comment is being quoted deliberately.
        if not in_code and HTML_COMMENT.fullmatch(line):
            continue

        if line.startswith("```"):
            flush_text()
            if in_code:
                pdf.set_font(MONO, "", 8)
                pdf.set_fill_color(244, 245, 247)
                for entry in code:
                    pdf.cell(0, 4.4, sanitise("  " + entry), fill=True)
                    pdf.ln(4.4)
                pdf.ln(2)
                code = []
            else:
                flush_table()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue

        # Pipe tables: a row of dashes is the header delimiter and is not content.
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            flush_text()
            if not all(set(c) <= set("-: ") for c in cells):
                table.append(cells)
            continue
        flush_table()

        if not line:
            flush_text()
            pdf.ln(2)
            continue
        if line.startswith("---") and set(line) <= {"-"}:
            flush_text()
            pdf.ln(1)
            pdf.set_draw_color(200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            continue

        if heading := re.match(r"^(#{1,4})\s+(.*)", line):
            flush_text()
            level = len(heading.group(1))
            style = HEADINGS[level]
            # Top-level sections start a page so the document has a spine a
            # reader can navigate by.
            if level == 1 and pdf.page_no() > 0 and pdf.get_y() > 40:
                pdf.add_page()
            pdf.ln(style.space_before)
            pdf.set_font(BODY, style.style, style.size)
            pdf.multi_cell(0, style.size * 0.5, sanitise(strip_markup(heading.group(2))))
            pdf.ln(style.space_after)
            continue

        if bullet := re.match(r"^(\s*)[-*]\s+(.*)", line):
            flush_text()
            item_offset = 4.0 + (len(bullet.group(1)) // 2) * 4
            item = ["- " + bullet.group(2)]
            continue

        if numbered := re.match(r"^(\s*)(\d+)\.\s+(.*)", line):
            flush_text()
            item_offset = 4.0
            item = [f"{numbered.group(2)}. {numbered.group(3)}"]
            continue

        # An indented line while a list item is open continues that item.
        if item and raw[:1].isspace():
            item.append(line.strip())
            continue

        if line.startswith("> "):
            # Only the paragraph is closed here: the quote being accumulated is
            # this one, and flushing it per line is the defect.
            flush_paragraph()
            quote.append(line[2:])
            continue

        flush_quote()
        paragraph.append(line)

    flush_text()
    flush_table()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return output


def stripped_comments(source: Path) -> list[str]:
    """The whole-line HTML comments `render` is expected to remove.

    Checking for the *pattern* `<!--.*?-->` in the rendered text was the obvious
    approach and it was wrong: the document quotes the marker syntax in prose to
    explain the mechanism, and a pattern cannot tell a quotation from a leak. The
    exact strings that were stripped can be read off the source, so the check
    becomes "did any of these specific lines reach the page" -- which is the
    question that was actually being asked.
    """
    out: list[str] = []
    in_code = False
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and HTML_COMMENT.fullmatch(line):
            out.append(line.strip())
    return out


def literal_markup_budget(source: Path) -> dict[str, int]:
    """How many inline markdown markers the document legitimately prints.

    Two sources, and missing the second cost a false alarm on a correct document
    for the second time in one sitting. Code fences are rendered verbatim, so
    everything inside one is meant to reach the page. Inline code spans are too:
    this document explains the renderer, so it quotes markers deliberately, and a
    span written as a pair of backticks around two asterisks *should* print two
    asterisks in Courier. Only a marker outside both means a span was not
    recognised.

    Counting gives an exact budget rather than a blanket "there must be none".
    """
    budget = {"`": 0, "**": 0}
    in_code = False
    for raw in source.read_text(encoding="utf-8").splitlines():
        if raw.rstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            for marker in budget:
                budget[marker] += raw.count(marker)
            continue
        # Outside a fence, only the contents of inline code spans are printed
        # verbatim; the delimiting backticks themselves are consumed.
        for span in _CODE_RE.findall(raw):
            for marker in budget:
                budget[marker] += span.count(marker)
    return budget


def verify(
    pdf_path: Path,
    expect: list[str],
    preview: bool = True,
    forbidden: list[str] | None = None,
    markup_budget: dict[str, int] | None = None,
) -> bool:
    """Rasterise every page and confirm real content landed on it.

    Four checks, because each catches something the others miss:

    1.  **Ink coverage per page.** A page whose glyphs failed to render is valid
        but blank. Only rasterising detects that; text extraction would still
        report the characters as present.
    2.  **Text round-trip.** Known strings -- measured numbers, section titles --
        are extracted back out. This catches content that was silently dropped
        during layout.
    3.  **Replacement characters.** `sanitise` maps unencodable characters to '?',
        so any surviving '?' clusters mean a character was lost.
    4.  **Forbidden markup.** The three checks above all ask whether something
        expected is present, so none of them could see six lines of raw
        `<!-- generated:... -->` printed into the document -- this function reported
        "verified" on that PDF. Asking what must be *absent* is a different question
        and needs its own check.
    """
    import pymupdf

    forbidden = list(forbidden or [])
    doc = pymupdf.open(pdf_path)
    print(f"  pages: {doc.page_count}")

    if preview:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for stale in PREVIEW_DIR.glob("page-*.png"):
            stale.unlink()

    blank: list[int] = []
    for number, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=80)
        # Fraction of non-white bytes. A rendered text page sits well above 1%;
        # a page whose font failed sits near zero.
        ink = sum(1 for b in pix.samples if b < 250) / max(len(pix.samples), 1)
        if ink < 0.004:
            blank.append(number)
        if preview and number <= 4:
            pix.save(PREVIEW_DIR / f"page-{number:02d}.png")

    text = "".join(page.get_text() for page in doc)
    doc.close()

    print(f"  extracted characters: {len(text):,}")
    ok = True

    if blank:
        print(f"  FAIL near-blank pages: {blank}")
        ok = False
    else:
        print("  every page carries ink")

    missing = [needle for needle in expect if needle not in text]
    if missing:
        print(f"  FAIL strings absent from the PDF: {missing}")
        ok = False
    else:
        print(f"  all {len(expect)} expected strings survived the round trip")

    if UNMAPPED:
        print(f"  FAIL characters with no Latin-1 mapping: {sorted(UNMAPPED)}")
        print("       add each to _REPLACEMENTS -- it printed as '?' in the document")
        ok = False
    else:
        print("  every character had a mapping")

    if TRUNCATED:
        shown = [f"{a!r} -> {b!r}" for a, b in TRUNCATED[:3]]
        print(f"  FAIL {len(TRUNCATED)} table cell(s) shortened to fit: {shown}")
        ok = False
    else:
        print("  no table cell was shortened")

    leaked = sorted({c for c in forbidden if c in text})
    if leaked:
        print(f"  FAIL raw markup printed into the PDF: {leaked[:3]}")
        ok = False
    elif forbidden:
        print(f"  none of the {len(forbidden)} stripped comment(s) reached the page")
    else:
        print("  no comments to strip")

    if markup_budget is not None:
        over = {
            marker: (text.count(marker), allowed)
            for marker, allowed in markup_budget.items()
            if text.count(marker) > allowed
        }
        if over:
            print(f"  FAIL unrendered markdown markers, found vs allowed: {over}")
            print("       a span was not recognised, so its markers printed as text")
            ok = False
        else:
            print("  no unrendered markdown markers")

    # Isolated '?' is ordinary punctuation; a run of them is a lost glyph.
    runs = re.findall(r"\?{2,}", text)
    if runs:
        print(f"  FAIL {len(runs)} replacement-character run(s), e.g. {runs[:3]}")
        ok = False
    else:
        print("  no replacement-character runs")

    if preview:
        print(f"  wrote page images to {PREVIEW_DIR}")
    return ok


#: Strings that must survive rendering. Deliberately the *measured numbers* and
#: the findings, not decorative text: if a table cell is dropped during layout the
#: document would still look complete while having lost a result.
#: Strings that must survive the round trip out of the rendered PDF.
#:
#: Two kinds, deliberately. Structural landmarks catch a truncated or reordered
#: render. Load-bearing *figures* catch something subtler: this list is pinned to
#: the study's headline numbers, so when the results change and the prose is not
#: updated, rendering fails rather than quietly producing a document that
#: contradicts results/. That has already happened -- "0.1953" was the baseline
#: at 216 queries and is 0.1971 at 586, and this check is what noticed.
#:
#: Update it when a headline number legitimately changes, and treat a failure as a
#: question about the document rather than about the list.
EXPECTED = [
    "0.1208",  # hybrid-plus-rerank, best configuration on the paraphrased wording
    "0.1971",  # baseline nDCG@10, original wording
    "+0.0680",  # its delta over that baseline
    "0.6641",  # best Recall@50 in the grid, semantic chunking
    "18.0x",  # retrieval vs long context, cost ratio
    "20.4%",  # label audit rejection rate
    "526,296",  # inline-XBRL characters in one filing
    "42,215",  # chunks under struct512
    "Glossary",
    "Decisions that turned out wrong",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and verify the learning PDF")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    print(f"rendering {SOURCE.relative_to(ROOT)}")
    path = render()
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(ROOT)} ({size_kb:.0f} KB)\n")

    print("verifying by rasterising")
    if not verify(
        path,
        EXPECTED,
        preview=not args.no_preview,
        forbidden=stripped_comments(SOURCE),
        markup_budget=literal_markup_budget(SOURCE),
    ):
        print("\nVERIFICATION FAILED")
        sys.exit(1)
    print("\nverified")


if __name__ == "__main__":
    main()
