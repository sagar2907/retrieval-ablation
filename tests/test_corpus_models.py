"""Tests for the canonical document representation and span algebra."""

from __future__ import annotations

import pytest

from retrieval_ablation.corpus.models import (
    Block,
    BlockKind,
    Document,
    GoldPassage,
    Span,
    Table,
)


class TestSpan:
    def test_length(self):
        assert Span(10, 25).length == 15

    def test_empty_span_is_allowed(self):
        assert Span(5, 5).length == 0

    def test_negative_start_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Span(-1, 5)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="precedes start"):
            Span(10, 4)

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (Span(0, 10), Span(5, 15), 5),
            (Span(0, 10), Span(10, 20), 0),  # half-open: touching is not overlapping
            (Span(0, 10), Span(20, 30), 0),
            (Span(0, 10), Span(2, 8), 6),
            (Span(2, 8), Span(0, 10), 6),  # symmetric
        ],
    )
    def test_overlap_length(self, a: Span, b: Span, expected: int):
        assert a.overlap_length(b) == expected

    def test_coverage_is_asymmetric_by_design(self):
        """A big chunk fully containing a short gold span is a success.

        coverage_of measures "how much of the argument do I contain", so a large
        chunk covering a small gold span scores 1.0 while the reverse scores a
        small fraction. A symmetric measure such as Jaccard would score both
        directions low and wrongly mark the retrieval a miss.
        """
        chunk = Span(0, 1000)
        gold = Span(400, 500)
        assert chunk.coverage_of(gold) == pytest.approx(1.0)
        assert gold.coverage_of(chunk) == pytest.approx(0.1)

    def test_partial_coverage(self):
        assert Span(0, 100).coverage_of(Span(50, 150)) == pytest.approx(0.5)

    def test_coverage_of_empty_span_is_zero_not_a_crash(self):
        assert Span(0, 10).coverage_of(Span(5, 5)) == 0.0

    def test_contains(self):
        assert Span(0, 100).contains(Span(10, 20))
        assert not Span(0, 100).contains(Span(90, 110))

    def test_ordering_enables_sorting_blocks(self):
        assert sorted([Span(5, 6), Span(1, 2), Span(3, 4)])[0] == Span(1, 2)


class TestTable:
    @pytest.fixture
    def income_table(self) -> Table:
        return Table(
            rows=(
                ("", "2025", "2024"),
                ("Research and development", "31,370", "29,915"),
                ("Selling, general and administrative", "26,097", "24,932"),
            ),
            n_header_rows=1,
            caption="Operating expenses (in millions)",
        )

    def test_dimensions(self, income_table: Table):
        assert income_table.n_rows == 3
        assert income_table.n_cols == 3

    def test_ragged_rows_report_widest(self):
        assert Table(rows=(("a",), ("b", "c", "d"))).n_cols == 3

    def test_markdown_includes_delimiter_after_header(self, income_table: Table):
        md = income_table.to_markdown()
        lines = md.splitlines()
        assert lines[0] == "Operating expenses (in millions)"
        assert lines[1] == "|  | 2025 | 2024 |"
        assert set(lines[2]) <= set("|- ")
        assert "| Research and development | 31,370 | 29,915 |" in md

    def test_markdown_pads_ragged_rows(self):
        md = Table(rows=(("a", "b"), ("c",))).to_markdown()
        assert md.splitlines()[-1] == "| c |  |"

    def test_empty_table_renders_empty(self):
        assert Table(rows=()).to_markdown() == ""

    def test_row_sentences_attach_column_headers_to_values(self, income_table: Table):
        """The rendering that makes a number findable without its position.

        "31,370" alone is unretrievable; "Research and development -- 2025:
        31,370" can be matched lexically and read by a model out of context.
        """
        text = income_table.to_row_sentences()
        assert "Research and development -- 2025: 31,370; 2024: 29,915" in text
        assert "Operating expenses (in millions)" in text

    def test_row_sentences_skips_blank_cells(self):
        table = Table(rows=(("", "2025", "2024"), ("Revenue", "100", "")))
        assert table.to_row_sentences().endswith("Revenue -- 2025: 100")

    def test_row_sentences_falls_back_when_no_data_rows(self):
        header_only = Table(rows=(("", "2025"),), n_header_rows=1)
        assert header_only.to_row_sentences() == header_only.to_markdown()


class TestBlock:
    def test_table_kind_requires_a_table(self):
        with pytest.raises(ValueError, match="disagree"):
            Block("b1", BlockKind.TABLE, Span(0, 10))

    def test_non_table_kind_must_not_carry_a_table(self):
        with pytest.raises(ValueError, match="disagree"):
            Block("b1", BlockKind.PARAGRAPH, Span(0, 10), table=Table(rows=(("a",),)))

    def test_valid_table_block(self):
        block = Block("b1", BlockKind.TABLE, Span(0, 10), table=Table(rows=(("a",),)))
        assert block.is_table

    def test_paragraph_is_not_a_table(self):
        assert not Block("b1", BlockKind.PARAGRAPH, Span(0, 10)).is_table


class TestDocument:
    @pytest.fixture
    def doc(self) -> Document:
        text = "Heading here.\nFirst paragraph.\nSecond paragraph about taxes."
        return Document(
            doc_id="d1",
            text=text,
            blocks=(
                Block("b0", BlockKind.HEADING, Span(0, 13), ("Item 1",)),
                Block("b1", BlockKind.PARAGRAPH, Span(14, 30), ("Item 1",)),
                Block("b2", BlockKind.PARAGRAPH, Span(31, len(text)), ("Item 1", "Taxes")),
            ),
        )

    def test_slice_returns_exact_text(self, doc: Document):
        assert doc.slice(Span(0, 13)) == "Heading here."

    def test_block_beyond_text_is_rejected(self):
        with pytest.raises(ValueError, match="only 5 characters"):
            Document(
                doc_id="d",
                text="short",
                blocks=(Block("b", BlockKind.PARAGRAPH, Span(0, 99)),),
            )

    def test_overlapping_blocks_are_rejected(self):
        """Overlap makes "which section is this offset in" ambiguous."""
        with pytest.raises(ValueError, match="overlap"):
            Document(
                doc_id="d",
                text="x" * 100,
                blocks=(
                    Block("b1", BlockKind.PARAGRAPH, Span(0, 50)),
                    Block("b2", BlockKind.PARAGRAPH, Span(40, 90)),
                ),
            )

    def test_adjacent_blocks_are_fine(self):
        Document(
            doc_id="d",
            text="x" * 100,
            blocks=(
                Block("b1", BlockKind.PARAGRAPH, Span(0, 50)),
                Block("b2", BlockKind.PARAGRAPH, Span(50, 100)),
            ),
        )

    def test_blocks_overlapping_a_span(self, doc: Document):
        found = doc.blocks_overlapping(Span(10, 20))
        assert [b.block_id for b in found] == ["b0", "b1"]

    def test_section_path_at_offset(self, doc: Document):
        assert doc.section_path_at(35) == ("Item 1", "Taxes")
        assert doc.section_path_at(5) == ("Item 1",)

    def test_section_path_in_a_gap_is_empty(self, doc: Document):
        # Offset 13 is the newline between blocks and belongs to no block.
        assert doc.section_path_at(13) == ()

    def test_document_with_no_blocks_is_valid(self):
        assert Document(doc_id="d", text="text").section_path_at(0) == ()


class TestGoldPassage:
    def test_defaults_to_directly_relevant(self):
        gold = GoldPassage("p1", "d1", Span(0, 100))
        assert gold.gain == 2

    def test_graded_gain_is_preserved(self):
        assert GoldPassage("p1", "d1", Span(0, 100), gain=1).gain == 1
