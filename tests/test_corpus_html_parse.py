"""Tests for filing HTML parsing.

All fixtures are small synthetic documents built to mirror the structures real
filings use, so the suite runs offline and deterministically. Each regression
test names the failure it prevents in its docstring.
"""

from __future__ import annotations

import lxml.html
import pytest

from retrieval_ablation.corpus.html_parse import (
    _guess_header_rows,
    is_data_table,
    normalize_text,
    parse_filing,
    parse_table,
)
from retrieval_ablation.corpus.models import BlockKind


def table_of(html: str) -> lxml.html.HtmlElement:
    return lxml.html.fromstring(html).xpath("//table")[0]


class TestNormalizeText:
    def test_collapses_whitespace_runs(self):
        assert normalize_text("a  \n\t b") == "a b"

    def test_strips_ends(self):
        assert normalize_text("  x  ") == "x"

    def test_nfkc_folds_non_breaking_space(self):
        assert normalize_text("12\u00a0345") == "12 345"

    def test_nfkc_folds_full_width_digits(self):
        assert normalize_text("\uff12\uff10\uff12\uff15") == "2025"

    def test_removes_zero_width_space(self):
        """Regression: NFKC preserves U+200B, which splits a token invisibly.

        "Rev<ZWSP>enue" renders as "Revenue" but tokenises as two fragments, so
        lexical index would never match a query for "revenue".
        """
        assert normalize_text("Rev\u200benue") == "Revenue"

    def test_removes_soft_hyphen_and_bom(self):
        assert normalize_text("\ufeffco\u00adoperate") == "cooperate"

    def test_empty_input(self):
        assert normalize_text("   ") == ""


class TestIsDataTable:
    def test_financial_table_is_data(self):
        html = """<table>
          <tr><td></td><td>2025</td><td>2024</td></tr>
          <tr><td>Revenue</td><td>416,161</td><td>391,035</td></tr>
        </table>"""
        assert is_data_table(table_of(html))

    def test_nested_table_is_layout(self):
        """A table containing a table is a layout container.

        Financial data tables in filings never nest, so nesting is a reliable
        signal that the outer element is arranging the page.
        """
        html = """<table><tr><td>
          <table><tr><td></td><td>2025</td></tr><tr><td>Rev</td><td>1,000</td></tr></table>
        </td></tr></table>"""
        outer = lxml.html.fromstring(html).xpath("//table")[0]
        assert not is_data_table(outer)

    def test_single_row_is_not_data(self):
        assert not is_data_table(table_of("<table><tr><td>a</td><td>b</td></tr></table>"))

    def test_single_column_is_not_data(self):
        html = "<table><tr><td>a</td></tr><tr><td>b</td></tr></table>"
        assert not is_data_table(table_of(html))

    def test_prose_only_table_is_layout(self):
        # A signature block: structurally a table, no numbers at all.
        html = """<table>
          <tr><td>By:</td><td>Chief Executive Officer</td></tr>
          <tr><td>By:</td><td>Chief Financial Officer</td></tr>
        </table>"""
        assert not is_data_table(table_of(html))

    def test_too_few_non_empty_cells_is_layout(self):
        html = "<table><tr><td>1</td><td></td></tr><tr><td></td><td></td></tr></table>"
        assert not is_data_table(table_of(html))


class TestGuessHeaderRows:
    def test_blank_stub_marks_a_header_row(self):
        grid = [["", "2025", "2024"], ["Revenue", "416,161", "391,035"]]
        assert _guess_header_rows(grid) == 1

    def test_two_stacked_header_rows(self):
        grid = [
            ["", "Years ended", "Years ended"],
            ["", "2025", "2024"],
            ["Revenue", "416,161", "391,035"],
        ]
        assert _guess_header_rows(grid) == 2

    def test_year_headers_are_headers_despite_containing_digits(self):
        """Regression: an earlier rule assumed header rows contain no digits.

        That is wrong on exactly the tables that matter -- the headers of a
        financial statement are years. The rule reported zero header rows for
        every income statement and balance sheet, which stripped the column
        labels that the row-sentence rendering depends on, leaving bare numbers
        with nothing to match a query against.
        """
        grid = [["", "2025", "2024", "2023"], ["Revenue", "416,161", "391,035", "383,285"]]
        assert _guess_header_rows(grid) == 1

    def test_all_rows_blank_stub_still_leaves_data(self):
        grid = [["", "a"], ["", "b"]]
        assert _guess_header_rows(grid) < len(grid)

    def test_no_blank_stub_falls_back_to_text_first_row(self):
        grid = [["Metric", "Amount"], ["Revenue", "100"]]
        assert _guess_header_rows(grid) == 1

    def test_empty_grid(self):
        assert _guess_header_rows([]) == 0


class TestParseTable:
    def test_simple_grid(self):
        html = """<table>
          <tr><td></td><td>2025</td><td>2024</td></tr>
          <tr><td>Revenue</td><td>416,161</td><td>391,035</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[0] == ("", "2025", "2024")
        assert table.rows[1] == ("Revenue", "416,161", "391,035")
        assert table.n_header_rows == 1

    def test_spacer_columns_are_dropped(self):
        """Filings pad tables with sub-1%-width columns that never hold data."""
        html = """<table>
          <tr><td></td><td></td><td>2025</td><td></td></tr>
          <tr><td>Revenue</td><td></td><td>416,161</td><td></td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1] == ("Revenue", "416,161")

    def test_currency_symbol_is_joined_to_its_value(self):
        """ "$" and "34,550" arrive as separate columns under one header."""
        html = """<table>
          <tr><td></td><td colspan="2">2025</td></tr>
          <tr><td>Research and development</td><td>$</td><td>34,550</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1] == ("Research and development", "$ 34,550")

    def test_colspan_row_label_is_not_repeated(self):
        """Regression: a colspan row label was smeared across three columns.

        The stub cell carries colspan=3, so `_expand_grid` repeats its text into
        three physical columns which fall into three different header groups.
        Merging by header label alone left
        "Cash and cash equivalents | Cash and cash equivalents | Cash and cash
        equivalents". Collapsing on shared cell origin fixes it exactly.
        """
        html = """<table>
          <tr><td colspan="3"></td><td>2025</td><td>2024</td></tr>
          <tr><td colspan="3">Cash and cash equivalents</td>
          <td>35,934</td><td>29,943</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1] == ("Cash and cash equivalents", "35,934", "29,943")

    def test_two_years_holding_the_same_value_are_both_kept(self):
        """The reason collapsing uses provenance rather than matching text.

        Collapsing adjacent cells whose *text* is equal would silently delete one
        of these columns, corrupting the data. They come from two distinct source
        cells, so origin-based collapsing leaves both.
        """
        html = """<table>
          <tr><td></td><td>2025</td><td>2024</td></tr>
          <tr><td>Effective tax rate</td><td>16.1</td><td>16.1</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1] == ("Effective tax rate", "16.1", "16.1")

    def test_full_width_spanning_row_collapses_to_one_cell(self):
        html = """<table>
          <tr><td></td><td>2025</td><td>2024</td></tr>
          <tr><td colspan="3">ASSETS:</td></tr>
          <tr><td>Cash</td><td>35,934</td><td>29,943</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1][0] == "ASSETS:"
        assert all(c == "" for c in table.rows[1][1:])

    def test_rowspan_carries_the_label_downward(self):
        html = """<table>
          <tr><td></td><td>2025</td></tr>
          <tr><td rowspan="2">Segment</td><td>100</td></tr>
          <tr><td>200</td></tr>
        </table>"""
        table = parse_table(table_of(html))
        assert table.rows[1][0] == "Segment"
        assert table.rows[2][0] == "Segment"

    def test_absurd_span_is_capped(self):
        html = '<table><tr><td colspan="99999">x</td></tr><tr><td>1</td><td>2</td></tr></table>'
        table = parse_table(table_of(html))
        assert table.n_cols <= 64

    def test_non_integer_span_does_not_crash(self):
        html = (
            '<table><tr><td colspan="two">a</td><td>2025</td></tr>'
            "<tr><td>b</td><td>1</td></tr></table>"
        )
        assert parse_table(table_of(html)).n_rows == 2

    def test_fully_empty_table_yields_no_rows(self):
        html = "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
        assert parse_table(table_of(html)).rows == ()


def _filler(sentence: str, times: int = 25) -> str:
    return sentence * times


# Mirrors the real structure of a 10-K: a contents block listing every heading,
# then a body that repeats all of them with substantive text underneath. The
# repetition is the point -- it is what makes contents entries identifiable, and
# an earlier version of this fixture omitted several body items, which made the
# suppression logic look broken when it was the fixture that was unrealistic.
_ITEMS = [
    ("Item 1. Business", "The Company designs and markets smartphones. "),
    ("Item 1A. Risk Factors", "The Company's business is subject to risks. "),
    ("Item 1B. Unresolved Staff Comments", "None applicable to this filing. "),
    ("Item 2. Properties", "The headquarters are located in California. "),
    ("Item 3. Legal Proceedings", "Various claims are pending against the Company. "),
    ("Item 4. Mine Safety Disclosures", "Not applicable to the Company. "),
]

_CONTENTS = (
    ["<div>PART I</div>"]
    + [f"<div>{label}</div>" for label, _ in _ITEMS]
    + ["<div>PART II</div>", "<div>Item 8. Financial Statements</div>"]
)

_BODY = ["<div>PART I</div>"]
for _label, _sentence in _ITEMS:
    _BODY.append(f"<div>{_label}</div>")
    _BODY.append(f"<p>{_filler(_sentence)}</p>")

_BODY += [
    "<div>PART II</div>",
    "<div>Item 8. Financial Statements</div>",
    f"<p>{_filler('See the accompanying notes to the consolidated statements. ')}</p>",
    "<div>Note 7 - Income Taxes</div>",
    f"<p>{_filler('The provision for income taxes consisted of the following. ')}</p>",
    """<table>
        <tr><td></td><td>2025</td><td>2024</td></tr>
        <tr><td>Federal current</td><td>11,140</td><td>9,388</td></tr>
        <tr><td>State current</td><td>1,562</td><td>1,183</td></tr>
       </table>""",
]

FILING = (
    "<html><body><div>ANNUAL REPORT PURSUANT TO SECTION 13</div>"
    + "".join(_CONTENTS)
    + "".join(_BODY)
    + "</body></html>"
)


class TestParseFiling:
    @pytest.fixture
    def doc(self):
        return parse_filing("test-10k-2025", FILING)

    def test_produces_text_and_blocks(self, doc):
        assert doc.text
        assert doc.blocks

    def test_every_block_span_slices_to_its_own_text(self, doc):
        # The invariant the whole eval set depends on.
        for block in doc.blocks:
            assert doc.slice(block.span).strip() == doc.slice(block.span)
            assert doc.slice(block.span)

    def test_reparse_is_byte_identical(self, doc):
        again = parse_filing("test-10k-2025", FILING)
        assert again.text == doc.text
        assert [b.span for b in again.blocks] == [b.span for b in doc.blocks]

    def test_section_hierarchy_is_nested(self, doc):
        paths = {b.section_path for b in doc.blocks if b.section_path}
        assert ("Part I", "Item 1. Business") in paths
        assert any(p[:2] == ("Part II", "Item 8. Financial Statements") for p in paths)
        assert any(len(p) == 3 and p[2].startswith("Note 7") for p in paths)

    def test_note_resets_when_a_new_item_starts(self, doc):
        """A Note must not leak into the next Item's section path.

        Without resetting, "Note 7" from Item 8 stays attached to every later
        passage, mislabelling provenance and corrupting metadata-filtered
        retrieval.
        """
        for block in doc.blocks:
            if len(block.section_path) == 3:
                assert block.section_path[1].startswith("Item 8")

    def test_contents_run_is_suppressed(self, doc):
        boilerplate = [doc.slice(b.span) for b in doc.blocks if b.kind is BlockKind.BOILERPLATE]
        # The contents block lists nine entries back to back.
        assert len(boilerplate) >= 8
        assert "PART I" in boilerplate

    def test_real_short_sections_survive_suppression(self, doc):
        """Regression: judging each heading alone deleted real short sections.

        The first version suppressed any heading followed by little text. That
        removed "PART I" (always immediately followed by "Item 1", so the gap is
        tiny by construction) and one-line sections such as "Item 4. Mine Safety
        Disclosures". Losing the Part headings meant no passage in the corpus
        carried a Part in its section path at all.
        """
        headings = [doc.slice(b.span) for b in doc.blocks if b.kind is BlockKind.HEADING]
        assert "PART I" in headings
        assert "Item 2. Properties" in headings
        parts = {b.section_path[0] for b in doc.blocks if b.section_path}
        assert parts == {"Part I", "Part II"}

    def test_table_is_kept_whole_as_one_block(self, doc):
        tables = [b for b in doc.blocks if b.is_table]
        assert len(tables) == 1
        rendered = doc.slice(tables[0].span)
        assert "Federal current" in rendered
        assert "11,140" in rendered

    def test_table_block_carries_its_section_path(self, doc):
        table_block = next(b for b in doc.blocks if b.is_table)
        assert table_block.section_path[-1].startswith("Note 7")

    def test_row_sentence_rendering_attaches_headers(self):
        doc = parse_filing("t", FILING, render_tables="row_sentences")
        table_block = next(b for b in doc.blocks if b.is_table)
        assert "Federal current -- 2025: 11,140" in doc.slice(table_block.span)

    def test_unknown_render_mode_is_rejected(self):
        with pytest.raises(ValueError, match="render_tables"):
            parse_filing("t", FILING, render_tables="latex")

    def test_metadata_is_carried_through(self):
        doc = parse_filing("t", FILING, metadata={"ticker": "AAPL"})
        assert doc.metadata["ticker"] == "AAPL"


class TestEncodingHandling:
    def test_bytes_with_xml_declaration_parse(self):
        """Inline-XBRL filings are XHTML carrying an encoding declaration.

        Regression: the parser originally took `str`, and lxml refuses a string
        with an encoding declaration -- the real corpus failed on the first
        document. Bytes let lxml honour the declaration itself.
        """
        payload = (
            b'<?xml version="1.0" encoding="UTF-8"?><html><body><p>Revenue rose.</p></body></html>'
        )
        assert "Revenue rose." in parse_filing("t", payload).text

    def test_str_with_xml_declaration_is_accepted(self):
        payload = '<?xml version="1.0" encoding="UTF-8"?><html><body><p>Hello.</p></body></html>'
        assert "Hello." in parse_filing("t", payload).text

    def test_utf8_bytes_decode_correctly(self):
        payload = "<html><body><p>Café revenue</p></body></html>".encode()
        assert "Café revenue" in parse_filing("t", payload).text

    def test_empty_document_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_filing("t", b"   ")
