"""Tests for deterministic eval-set construction. Offline, no model, no key."""

from __future__ import annotations

import pytest

from retrieval_ablation.corpus.models import Block, BlockKind, Document, Span, Table
from retrieval_ablation.evalset.schema import (
    EvalQuery,
    QueryKind,
    Verification,
    gold_by_query,
    read_eval_set,
    summarise,
    write_eval_set,
)
from retrieval_ablation.evalset.synthesize import (
    build_queries,
    extract_table_facts,
    lexical_overlap,
)


def table_doc(
    rows: tuple[tuple[str, ...], ...],
    *,
    n_header_rows: int = 1,
    company: str = "Test Corp",
    ticker: str = "TST",
    sector: str = "technology",
    caption: str | None = None,
) -> Document:
    """A document whose entire content is one rendered table."""
    table = Table(rows=rows, n_header_rows=n_header_rows, caption=caption)
    rendered = table.to_markdown()
    return Document(
        doc_id="tst-10-k-2025-12-31",
        text=rendered + "\n\n",
        blocks=(
            Block(
                block_id="t00001",
                kind=BlockKind.TABLE,
                span=Span(0, len(rendered)),
                section_path=("Part II", "Item 8. Financial Statements"),
                table=table,
            ),
        ),
        metadata={
            "company": company,
            "ticker": ticker,
            "sector": sector,
            "report_date": "2025-12-31",
        },
    )


class TestLexicalOverlap:
    def test_full_overlap(self):
        text = "research development expense"
        assert lexical_overlap(text, text) == 1.0

    def test_no_overlap(self):
        assert lexical_overlap("alpha beta", "gamma delta") == 0.0

    def test_partial(self):
        assert lexical_overlap("research development", "research only") == pytest.approx(0.5)

    def test_stopwords_are_ignored(self):
        # "what was the" contributes nothing, so this is full overlap on content.
        assert lexical_overlap("what was the revenue", "revenue") == pytest.approx(1.0)

    def test_empty_query_is_zero_not_a_crash(self):
        assert lexical_overlap("the of and", "anything") == 0.0

    def test_case_insensitive(self):
        assert lexical_overlap("REVENUE", "revenue") == pytest.approx(1.0)


class TestExtractTableFacts:
    def test_extracts_a_labelled_value_for_a_period(self):
        doc = table_doc(
            (
                ("", "2025", "2024"),
                ("Research and development", "34,550", "31,370"),
            )
        )
        facts = extract_table_facts(doc)
        assert len(facts) == 1
        assert facts[0].row_label == "Research and development"
        assert facts[0].period == "2025"
        assert facts[0].value == "34,550"

    def test_gold_span_is_the_row_not_the_whole_table(self):
        """A tight span is what makes reachability meaningful.

        If the gold span were the whole table, a chunker that split the table
        would still "cover" it partially and the failure would be invisible.
        """
        doc = table_doc(
            (
                ("", "2025", "2024"),
                ("Research and development", "34,550", "31,370"),
                ("Selling and administrative", "26,097", "24,932"),
            )
        )
        facts = extract_table_facts(doc)
        for fact in facts:
            passage = doc.slice(fact.span)
            assert fact.row_label in passage
            assert passage.startswith("|")
            # The span covers one line, not the entire table.
            assert "\n" not in passage

    def test_span_slices_to_text_containing_the_value(self):
        doc = table_doc((("", "2025"), ("Total net sales", "416,161")))
        fact = extract_table_facts(doc)[0]
        assert fact.value in doc.slice(fact.span)

    def test_table_without_a_period_header_is_skipped(self):
        """Without a fiscal period the query has no unique answer.

        Four consecutive annual reports each contain "revenue", so a query that
        does not name a year is ambiguous across the corpus by construction.
        """
        doc = table_doc((("", "Segment", "Region"), ("Revenue", "100", "200")))
        assert extract_table_facts(doc) == []

    def test_non_quantity_cells_are_skipped(self):
        doc = table_doc((("", "2025"), ("Auditor name", "Ernst & Young LLP")))
        assert extract_table_facts(doc) == []

    def test_short_labels_are_skipped(self):
        doc = table_doc((("", "2025"), ("Cash", "100")))
        assert extract_table_facts(doc) == []

    def test_statement_section_headings_are_skipped(self):
        doc = table_doc((("", "2025"), ("Total assets", "364,980")))
        assert extract_table_facts(doc) == []

    def test_entity_name_rows_are_skipped(self):
        """Regression: segment tables list company names down the stub column.

        Those produced queries like "How much did American Express Co report for
        american express company in 2022?" -- grammatical and meaningless.
        """
        doc = table_doc(
            (("", "2025"), ("American Express Company", "10.3")),
            company="AMERICAN EXPRESS CO",
        )
        assert extract_table_facts(doc) == []

    def test_footnote_digit_is_stripped_from_the_label(self):
        """Regression: "...net of current maturities2" leaked into query text.

        Filings attach footnote markers as bare digits. Left in, the query reads
        like a typo and gains a token that matches the gold passage for no
        semantic reason, inflating lexical overlap.
        """
        doc = table_doc((("", "2025"), ("Long-term debt, net of current maturities2", "38,038")))
        facts = extract_table_facts(doc)
        assert len(facts) == 1
        assert facts[0].row_label == "Long-term debt, net of current maturities"

    def test_one_fact_per_row(self):
        # Two period columns must not yield two near-duplicate facts.
        doc = table_doc((("", "2025", "2024"), ("Research and development", "34,550", "31,370")))
        assert len(extract_table_facts(doc)) == 1

    def test_parenthesised_negative_is_a_quantity(self):
        doc = table_doc((("", "2025"), ("Net cash used in financing activities", "(9,415)")))
        facts = extract_table_facts(doc)
        assert len(facts) == 1
        assert facts[0].value == "(9,415)"

    def test_percentage_is_a_quantity(self):
        doc = table_doc((("", "2025"), ("Effective income tax rate", "16.1 %")))
        assert len(extract_table_facts(doc)) == 1

    def test_caption_offsets_the_row_index(self):
        doc = table_doc(
            (("", "2025"), ("Research and development", "34,550")),
            caption="Operating expenses (in millions)",
        )
        fact = extract_table_facts(doc)[0]
        assert "Research and development" in doc.slice(fact.span)

    def test_document_without_tables_yields_nothing(self):
        doc = Document(doc_id="d", text="Just prose here.", blocks=())
        assert extract_table_facts(doc) == []


class TestBuildQueries:
    @pytest.fixture
    def docs(self) -> list[Document]:
        out = []
        for index, sector in enumerate(["technology", "banking", "retail"]):
            rows = [("", "2025", "2024")]
            rows += [
                (f"Operating expense line {i} detail", f"{i * 1000}", f"{i * 900}")
                for i in range(1, 12)
            ]
            doc = table_doc(tuple(rows), company=f"Company {index}", sector=sector)
            out.append(
                Document(
                    doc_id=f"c{index}-10-k-2025-12-31",
                    text=doc.text,
                    blocks=doc.blocks,
                    metadata=dict(doc.metadata, sector=sector, ticker=f"C{index}"),
                )
            )
        return out

    def test_produces_queries(self, docs):
        queries = build_queries(docs, n_queries=12)
        assert len(queries) == 12
        assert all(q.kind is QueryKind.TABLE_LOOKUP for q in queries)

    def test_labels_are_marked_generated_not_verified(self, docs):
        """The honesty requirement, enforced by test rather than convention."""
        assert all(q.verification is Verification.GENERATED for q in build_queries(docs, 9))

    def test_deterministic_for_a_fixed_seed(self, docs):
        first = build_queries(docs, n_queries=9, seed=123)
        second = build_queries(docs, n_queries=9, seed=123)
        assert [q.query_id for q in first] == [q.query_id for q in second]
        assert [q.text for q in first] == [q.text for q in second]

    def test_different_seed_changes_the_sample(self, docs):
        a = build_queries(docs, n_queries=9, seed=1)
        b = build_queries(docs, n_queries=9, seed=2)
        assert [q.query_id for q in a] != [q.query_id for q in b]

    def test_query_ids_are_content_addressed_not_positional(self, docs):
        """A positional id would renumber the set whenever sampling changed,
        invalidating verification work already done against it."""
        small = build_queries(docs, n_queries=3, seed=5)
        large = build_queries(docs, n_queries=9, seed=5)
        shared = {q.query_id for q in small} & {q.query_id for q in large}
        assert shared, "ids should be stable across different sample sizes"

    def test_ids_are_unique(self, docs):
        queries = build_queries(docs, n_queries=20)
        assert len({q.query_id for q in queries}) == len(queries)

    def test_stratifies_across_sectors(self, docs):
        queries = build_queries(docs, n_queries=9)
        sectors = {q.metadata["sector"] for q in queries}
        assert len(sectors) == 3

    def test_lexical_overlap_is_recorded(self, docs):
        """The confound must be measured, not left implicit."""
        queries = build_queries(docs, n_queries=6)
        assert all(q.lexical_overlap is not None for q in queries)
        assert all(0.0 <= q.lexical_overlap <= 1.0 for q in queries)

    def test_gold_span_contains_the_expected_value(self, docs):
        by_id = {d.doc_id: d for d in docs}
        for query in build_queries(docs, n_queries=9):
            gold = query.gold[0]
            assert query.metadata["expected_value"] in by_id[gold.doc_id].slice(gold.span)

    def test_several_templates_are_used(self, docs):
        # A single phrasing would make the benchmark a test of one sentence form.
        queries = build_queries(docs, n_queries=15)
        starts = {q.text.split()[0] for q in queries}
        assert len(starts) > 1

    def test_empty_corpus_yields_nothing(self):
        assert build_queries([], n_queries=10) == []

    def test_asking_for_more_than_available_returns_what_exists(self, docs):
        queries = build_queries(docs, n_queries=10_000)
        assert 0 < len(queries) < 10_000


class TestSchemaRoundTrip:
    @pytest.fixture
    def queries(self):
        docs = [table_doc((("", "2025"), ("Research and development", "34,550")))]
        return build_queries(docs, n_queries=1)

    def test_json_round_trip(self, queries):
        restored = [EvalQuery.from_json(q.to_json()) for q in queries]
        assert restored == queries

    def test_file_round_trip(self, queries, tmp_path):
        path = tmp_path / "queries.jsonl"
        write_eval_set(queries, path)
        assert read_eval_set(path) == queries

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build the eval set"):
            read_eval_set(tmp_path / "absent.jsonl")

    def test_gold_by_query_shape(self, queries):
        mapping = gold_by_query(queries)
        assert set(mapping) == {q.query_id for q in queries}
        assert all(isinstance(v, list) for v in mapping.values())

    def test_summary_reports_verification_state(self, queries):
        info = summarise(queries)
        assert info["n_queries"] == len(queries)
        assert info["by_verification"]["generated"] == len(queries)
        assert info["mean_lexical_overlap"] is not None

    def test_summary_of_empty_set_does_not_crash(self):
        info = summarise([])
        assert info["n_queries"] == 0
        assert info["mean_lexical_overlap"] is None
