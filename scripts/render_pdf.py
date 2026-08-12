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
}


def sanitise(text: str) -> str:
    """Make text Latin-1 safe, mapping known characters and flagging the rest."""
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    # Anything still unencodable becomes '?', which is visible in the output and
    # caught by verify(), rather than raising deep inside the writer.
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
            _emit(pdf, sanitise(bold.group(1)), BODY, "B", size)
        elif italic := _ITALIC_RE.fullmatch(part):
            _emit(pdf, sanitise(italic.group(1)), BODY, "I", size)
        elif code := _CODE_RE.fullmatch(part):
            _emit(pdf, sanitise(code.group(1)), MONO, "", size - 0.5)
        else:
            _emit(pdf, sanitise(part), BODY, "", size)
    pdf.ln(6)


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

    longest = [max(len(row[c]) for row in rows) or 1 for c in range(columns)]
    total = sum(longest)
    widths = [max(14.0, width * (n / total)) for n in longest]
    scale = width / sum(widths)
    widths = [w * scale for w in widths]

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
            # Truncate rather than wrap: a wrapped cell would desynchronise this
            # row's height from its neighbours' and stagger the grid.
            while pdf.get_string_width(clean) > cell_width - 2 and len(clean) > 1:
                clean = clean[:-1]
            pdf.cell(cell_width, height, clean, border=1, fill=header)
        pdf.ln(height)
    pdf.ln(2)


def render(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    """Convert the markdown source to a PDF."""
    lines = source.read_text(encoding="utf-8").splitlines()

    pdf = Renderer(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    in_code = False
    code: list[str] = []
    table: list[list[str]] = []
    paragraph: list[str] = []

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

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            flush_paragraph()
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
            flush_paragraph()
            if not all(set(c) <= set("-: ") for c in cells):
                table.append(cells)
            continue
        flush_table()

        if not line:
            flush_paragraph()
            pdf.ln(2)
            continue
        if line.startswith("---") and set(line) <= {"-"}:
            flush_paragraph()
            pdf.ln(1)
            pdf.set_draw_color(200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            continue

        if heading := re.match(r"^(#{1,4})\s+(.*)", line):
            flush_paragraph()
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
            flush_paragraph()
            indent = len(bullet.group(1)) // 2
            pdf.set_x(pdf.l_margin + 4 + indent * 4)
            write_rich(pdf, "- " + bullet.group(2), 10)
            pdf.set_x(pdf.l_margin)
            continue

        if numbered := re.match(r"^(\s*)(\d+)\.\s+(.*)", line):
            flush_paragraph()
            pdf.set_x(pdf.l_margin + 4)
            write_rich(pdf, f"{numbered.group(2)}. {numbered.group(3)}", 10)
            pdf.set_x(pdf.l_margin)
            continue

        if line.startswith("> "):
            flush_paragraph()
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_text_color(70)
            write_rich(pdf, line[2:], 10)
            pdf.set_text_color(0)
            pdf.set_x(pdf.l_margin)
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_table()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    return output


def verify(pdf_path: Path, expect: list[str], preview: bool = True) -> bool:
    """Rasterise every page and confirm real content landed on it.

    Three checks, because each catches something the others miss:

    1.  **Ink coverage per page.** A page whose glyphs failed to render is valid
        but blank. Only rasterising detects that; text extraction would still
        report the characters as present.
    2.  **Text round-trip.** Known strings -- measured numbers, section titles --
        are extracted back out. This catches content that was silently dropped
        during layout.
    3.  **Replacement characters.** `sanitise` maps unencodable characters to '?',
        so any surviving '?' clusters mean a character was lost.
    """
    import fitz

    doc = fitz.open(pdf_path)
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
    "17.5x",  # retrieval vs long context, cost ratio
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
    if not verify(path, EXPECTED, preview=not args.no_preview):
        print("\nVERIFICATION FAILED")
        sys.exit(1)
    print("\nverified")


if __name__ == "__main__":
    main()
