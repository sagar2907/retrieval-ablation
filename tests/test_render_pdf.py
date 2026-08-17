"""Tests for the PDF renderer, which had no tests and six defects.

Every failure pinned here reached a published PDF and survived that script's own
verification step. The pattern in all six is the same: the gate asked whether
something expected was present, and none of these defects removed anything
expected -- they added markup, shortened a label, or swapped one character.

Offline: the pure functions need nothing, and the one test that renders writes to
tmp_path. No network, no fonts to install -- the renderer uses the core-14 fonts
precisely so it runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

render_pdf = pytest.importorskip("render_pdf")

#: The real header of the results tables. Its width is the thing under test, so it
#: is written out rather than imported -- a change to the generator should make
#: these assertions stale and visible, not silently retarget them.
HEADER = ["configuration", "nDCG@10", "95% CI", "Recall@50", "MRR", "delta vs base", "p (Holm)"]
ROW = ["baseline-bm25-fixed512", "0.1971", "[0.166, 0.230]", "0.5385", "0.1769", "—", "—"]


@pytest.fixture(autouse=True)
def clean_collectors():
    """The collectors are module state, so a leaked entry would fail the next test."""
    render_pdf.UNMAPPED.clear()
    render_pdf.TRUNCATED.clear()
    yield
    render_pdf.UNMAPPED.clear()
    render_pdf.TRUNCATED.clear()


def renderer() -> render_pdf.Renderer:
    """A page set up exactly as `render` sets it up, since widths depend on margins."""
    pdf = render_pdf.Renderer(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    return pdf


class TestSanitise:
    def test_the_significance_tick_becomes_a_conventional_marker(self):
        """Regression: an unmapped tick printed as '?' beside a p-value.

        The cell read "0.0014 ?", which in a column headed "p (Holm)" reads as
        doubt about the number rather than as the significance flag it is. Three
        such cells shipped, and the run-of-two check could not see a single '?'.
        """
        assert render_pdf.sanitise("0.0014 ✓") == "0.0014 *"
        assert "?" not in render_pdf.sanitise("0.0014 ✓")

    def test_an_unmapped_character_is_recorded_not_swallowed(self):
        """One destroyed character is already a corrupted document.

        `sanitise` still degrades to '?' rather than raising mid-write, but the
        character is collected so the run can fail naming it. Previously the only
        signal was a run of two or more '?' in the extracted text.
        """
        out = render_pdf.sanitise("temperature 20℃")

        assert "?" in out
        assert "℃" in render_pdf.UNMAPPED

    def test_mapped_characters_leave_nothing_behind(self):
        # The ambiguous-character rule is suppressed because the input must be the
        # real minus sign the documents use -- a hyphen would test nothing.
        assert render_pdf.sanitise("−0.0391 → 0.6641") == "-0.0391 -> 0.6641"  # noqa: RUF001
        assert not render_pdf.UNMAPPED

    def test_ascii_is_untouched(self):
        assert render_pdf.sanitise("nDCG@10 = 0.1971") == "nDCG@10 = 0.1971"


class TestHtmlComments:
    def test_a_whole_line_comment_is_recognised(self):
        assert render_pdf.HTML_COMMENT.fullmatch("<!-- generated:headline -->")
        assert render_pdf.HTML_COMMENT.fullmatch("  <!-- /generated:headline -->  ")

    def test_prose_containing_a_comment_is_not_recognised(self):
        """Half-removing a line would be worse than printing the comment."""
        assert not render_pdf.HTML_COMMENT.fullmatch("the marker <!-- x --> is visible")

    def test_comments_are_stripped_from_the_rendered_text(self, tmp_path):
        """Regression: six marker lines printed into the published PDF.

        Markdown viewers hide HTML comments, so reading the source or the rendered
        markdown showed nothing wrong, and the verification step reported success
        because every expected string was still present. Only extracting the text
        and asking what should *not* be there found it.
        """
        fitz = pytest.importorskip("fitz")
        source = tmp_path / "doc.md"
        source.write_text(
            "# Title\n\n<!-- generated:headline -->\nA sentence of body text.\n"
            "<!-- /generated:headline -->\n",
            encoding="utf-8",
        )

        out = render_pdf.render(source, tmp_path / "doc.pdf")
        with fitz.open(out) as doc:
            text = "".join(page.get_text() for page in doc)

        assert "generated:headline" not in text
        assert "<!--" not in text
        # The content between the markers must survive; stripping is not deleting.
        assert "A sentence of body text." in text

    def test_a_comment_inside_a_code_fence_is_kept(self, tmp_path):
        """The document quotes the marker syntax to explain it.

        Stripping inside a fence would delete the example that documents the
        mechanism, which is a different kind of wrong document.
        """
        fitz = pytest.importorskip("fitz")
        source = tmp_path / "doc.md"
        source.write_text(
            "# Title\n\n```\n<!-- generated:headline -->\n```\n\nBody.\n", encoding="utf-8"
        )

        out = render_pdf.render(source, tmp_path / "doc.pdf")
        with fitz.open(out) as doc:
            text = "".join(page.get_text() for page in doc)

        assert "generated:headline" in text


def rendered_text(tmp_path: Path, markdown: str) -> str:
    """Render a fragment and read the text back out, which is the only real check."""
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "doc.md"
    source.write_text(markdown, encoding="utf-8")
    out = render_pdf.render(source, tmp_path / "doc.pdf")
    with fitz.open(out) as doc:
        return "".join(page.get_text() for page in doc)


class TestNestedSpans:
    def test_inline_code_inside_bold_loses_its_backticks(self, tmp_path):
        """Regression: 22 literal backticks printed in the published document.

        `_SPAN_RE` matches top-level spans only, so `**`corpus/models.py`**` was
        emitted as one bold run with the backticks intact. The module-by-module
        section is built entirely from bold filenames in backticks, which is where
        most of the 22 were.
        """
        text = rendered_text(tmp_path, "Body **`corpus/models.py`** holds the model.\n")

        assert "`" not in text
        assert "corpus/models.py" in text

    def test_the_bold_text_around_the_code_survives(self, tmp_path):
        """Splitting a run must not drop the halves either side of the code."""
        text = rendered_text(tmp_path, "Body **before `code` after** end.\n")

        assert "before" in text
        assert "after" in text
        assert "`" not in text


class TestBlockquotes:
    def test_emphasis_straddling_two_quoted_lines_is_rendered(self, tmp_path):
        """Regression: the project's central design decision printed its asterisks.

        Blockquotes were written line by line, so the paragraph-joining fix never
        applied to them. The statement in section 7 spans two source lines, its
        `**` never closed on either one, and the markers printed as text -- in the
        one sentence the document calls load-bearing.
        """
        quote = "> **Gold labels anchor to spans,\n> never to chunk ids.**\n"
        text = rendered_text(tmp_path, quote)

        assert "**" not in text
        assert "Gold labels anchor to spans" in text
        assert "never to chunk ids." in text

    def test_a_quote_and_the_paragraph_after_it_stay_separate(self, tmp_path):
        """Accumulating the quote must not swallow the prose that follows it."""
        text = rendered_text(tmp_path, "> quoted line\n\nA following paragraph.\n")

        assert "quoted line" in text
        assert "A following paragraph." in text


class TestListItems:
    def test_a_two_line_item_is_rendered_as_one_item(self, tmp_path):
        """Regression: a continuation line became its own unindented paragraph.

        Markdown continues a list item on an indented following line. Those lines
        fell through to the paragraph accumulator, so a two-line bullet broke out of
        its list and sat flush left -- and emphasis straddling the break lost its
        markers, because the closing one was on the other line. Third defect of this
        shape after paragraphs and blockquotes.
        """
        text = rendered_text(tmp_path, "- First part continues\n  onto the next *with emphasis*.\n")

        assert "*" not in text
        assert "with emphasis" in text

    def test_an_unindented_line_after_an_item_starts_a_paragraph(self, tmp_path):
        """Accumulating the item must not swallow the prose after the list."""
        text = rendered_text(tmp_path, "- An item.\nA new paragraph.\n")

        assert "An item." in text
        assert "A new paragraph." in text


class TestLiteralMarkupBudget:
    def test_markers_inside_a_fence_are_budgeted(self, tmp_path):
        """A fence is rendered verbatim, so its markers are meant to print."""
        source = tmp_path / "doc.md"
        source.write_text("```\na `b` c **d**\n```\n", encoding="utf-8")

        assert render_pdf.literal_markup_budget(source) == {"`": 2, "**": 2}

    def test_markers_outside_a_fence_are_not_budgeted(self, tmp_path):
        """Those are spans that must be consumed by the renderer, not printed."""
        source = tmp_path / "doc.md"
        source.write_text("Prose with `code` and **bold**.\n", encoding="utf-8")

        assert render_pdf.literal_markup_budget(source) == {"`": 0, "**": 0}

    def test_markers_inside_an_inline_code_span_are_budgeted(self, tmp_path):
        """Regression: a correct document failed the check for explaining itself.

        This document quotes markers in prose -- a pair of backticks around two
        asterisks renders two asterisks in Courier, which is right. Counting only
        fences reported that as an unrendered span, which is the second false alarm
        from the same instinct within an hour. A false alarm spends exactly the
        credibility the next real failure needs.
        """
        source = tmp_path / "doc.md"
        source.write_text("It printed its `**` markers as text.\n", encoding="utf-8")

        assert render_pdf.literal_markup_budget(source)["**"] == 1


class TestStrippedComments:
    """What the forbidden-strings check is built from."""

    def test_it_lists_the_whole_line_comments(self, tmp_path):
        source = tmp_path / "doc.md"
        source.write_text(
            "# T\n\n<!-- generated:headline -->\nbody\n<!-- /generated:headline -->\n",
            encoding="utf-8",
        )

        assert render_pdf.stripped_comments(source) == [
            "<!-- generated:headline -->",
            "<!-- /generated:headline -->",
        ]

    def test_a_comment_quoted_in_prose_is_not_listed(self, tmp_path):
        """Regression: the check flagged the document for explaining itself.

        The first version searched the rendered text for the pattern `<!--.*?-->`,
        and section 14d quotes the marker syntax to describe the mechanism. The
        renderer failed on a correct document -- a false alarm, which costs the same
        credibility as a missed one because the next failure gets ignored. Listing
        the exact lines that were stripped asks the question that was meant.
        """
        source = tmp_path / "doc.md"
        source.write_text("Six lines of `<!-- generated:... -->` were printed.\n", encoding="utf-8")

        assert render_pdf.stripped_comments(source) == []

    def test_a_comment_inside_a_code_fence_is_not_listed(self, tmp_path):
        """It is not stripped, so it must not be forbidden either."""
        source = tmp_path / "doc.md"
        source.write_text("```\n<!-- generated:headline -->\n```\n", encoding="utf-8")

        assert render_pdf.stripped_comments(source) == []


class TestTableWidths:
    def test_the_bold_header_is_not_shortened(self):
        """Regression: "nDCG@10" printed as "nDCG@1".

        Widths were allocated from character counts, but the header row is drawn in
        bold and bold Helvetica is wider per character, so the header overflowed a
        column sized for the same number of regular-weight characters and was
        shaved by one character. "nDCG@1" is not a clipped label, it is the name of
        a different metric -- in a document about which metric said what. Two
        headers were affected and the PDF gate passed.
        """
        render_pdf.write_table(renderer(), [HEADER, ROW])

        assert render_pdf.TRUNCATED == []

    def test_truncation_is_reported_when_content_cannot_fit(self):
        """The check must be able to fail, or it is decoration.

        Truncating is still the behaviour -- wrapping one cell would stagger the
        grid -- but it is now reported rather than silently accepted.
        """
        wide = [["x" * 90 for _ in range(7)] for _ in range(2)]

        render_pdf.write_table(renderer(), wide)

        assert render_pdf.TRUNCATED
        intended, shown = render_pdf.TRUNCATED[0]
        assert len(shown) < len(intended)

    def test_widths_fill_the_text_block_exactly(self):
        """A table narrower or wider than the margins would look like a layout bug."""
        pdf = renderer()
        width = pdf.w - pdf.l_margin - pdf.r_margin

        widths = render_pdf._column_widths(pdf, [HEADER, ROW], len(HEADER), width)

        assert sum(widths) == pytest.approx(width)

    def test_an_empty_table_is_not_an_error(self):
        render_pdf.write_table(renderer(), [])
