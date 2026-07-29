"""Tests for corpus serialisation and integrity checking. No network, no keys."""

from __future__ import annotations

import json

import pytest

from retrieval_ablation.corpus.companies import (
    CORPUS_FORM,
    CORPUS_TICKERS,
    FILINGS_PER_COMPANY,
)
from retrieval_ablation.corpus.ingest import (
    document_from_json,
    document_to_json,
    sha256_text,
)
from retrieval_ablation.corpus.models import Block, BlockKind, Document, Span, Table


@pytest.fixture
def doc() -> Document:
    text = "Heading\n\nSome prose about taxes.\n\n| a | b |\n| 1 | 2 |\n\n"
    return Document(
        doc_id="aapl-10-k-2025-09-27",
        text=text,
        blocks=(
            Block("b00001", BlockKind.HEADING, Span(0, 7), ("Part I",)),
            Block("b00002", BlockKind.PARAGRAPH, Span(9, 32), ("Part I", "Item 1. Business")),
            Block(
                "t00003",
                BlockKind.TABLE,
                Span(34, 52),
                ("Part I", "Item 1. Business"),
                table=Table(rows=(("a", "b"), ("1", "2")), n_header_rows=1, caption="Cap"),
            ),
        ),
        metadata={"ticker": "AAPL", "sector": "technology"},
    )


class TestRoundTrip:
    def test_json_round_trip_is_lossless(self, doc: Document):
        assert document_from_json(document_to_json(doc)) == doc

    def test_survives_a_real_json_encode_decode(self, doc: Document):
        """Guards against a value that is fine in Python but not JSON-encodable."""
        payload = json.loads(json.dumps(document_to_json(doc), ensure_ascii=False))
        assert document_from_json(payload) == doc

    def test_spans_survive_exactly(self, doc: Document):
        restored = document_from_json(document_to_json(doc))
        assert [b.span for b in restored.blocks] == [b.span for b in doc.blocks]

    def test_table_grid_survives(self, doc: Document):
        restored = document_from_json(document_to_json(doc))
        table = next(b.table for b in restored.blocks if b.table)
        assert table.rows == (("a", "b"), ("1", "2"))
        assert table.caption == "Cap"
        assert table.n_header_rows == 1

    def test_section_paths_become_tuples_again(self, doc: Document):
        # JSON has no tuples; a list would break hashing and set membership.
        restored = document_from_json(document_to_json(doc))
        assert all(isinstance(b.section_path, tuple) for b in restored.blocks)

    def test_non_ascii_text_survives(self):
        original = Document(doc_id="d", text="Café — naïve ₹100")
        payload = json.loads(json.dumps(document_to_json(original), ensure_ascii=False))
        assert document_from_json(payload).text == original.text


class TestDigest:
    def test_stable_for_identical_text(self):
        assert sha256_text("abc") == sha256_text("abc")

    def test_differs_on_a_single_character(self):
        """The integrity check has to be sensitive enough to catch an edit.

        A one-character change shifts every subsequent offset, which would
        silently misalign every gold label in that document.
        """
        assert sha256_text("Revenue was 416,161") != sha256_text("Revenue was 416,162")

    def test_whitespace_is_significant(self):
        assert sha256_text("a b") != sha256_text("a  b")


class TestCorpusDefinition:
    def test_thirty_companies(self):
        assert len(CORPUS_TICKERS) == 30

    def test_tickers_are_uppercase_and_unique(self):
        assert all(t == t.upper() for t in CORPUS_TICKERS)
        assert len(set(CORPUS_TICKERS)) == len(CORPUS_TICKERS)

    def test_multiple_sectors_are_represented(self):
        """A single-sector corpus would measure one house style of filing."""
        assert len(set(CORPUS_TICKERS.values())) >= 12

    def test_no_sector_dominates(self):
        counts: dict[str, int] = {}
        for sector in CORPUS_TICKERS.values():
            counts[sector] = counts.get(sector, 0) + 1
        assert max(counts.values()) <= 6

    def test_annual_reports_not_quarterly(self):
        assert CORPUS_FORM == "10-K"

    def test_multiple_years_per_company(self):
        """Several years per company is what makes the corpus adversarial.

        Consecutive annual reports repeat their structure almost verbatim while
        the figures change, so a query about one fiscal year has near-identical
        distractor passages. With one year per company, retrieval would be far
        too easy and every configuration would score alike.
        """
        assert FILINGS_PER_COMPANY >= 3
