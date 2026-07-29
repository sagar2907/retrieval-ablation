"""Tests for the three chunking strategies.

The invariant asserted everywhere here is that a chunk's span slices back to
exactly its own text. Gold labels are spans into the same string, so any chunker
that drifted from that would silently invalidate the eval set rather than fail.

Embedding is faked deterministically, so the semantic chunker is tested offline
with exactly known breakpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_ablation.chunking import (
    FixedSizeChunker,
    SemanticChunker,
    StructureAwareChunker,
    approx_token_count,
    atomize,
    chunk_id_for,
    split_sentences,
)
from retrieval_ablation.corpus.models import Block, BlockKind, Document, Span, Table


def word_tokens(text: str) -> int:
    """One token per whitespace-delimited word: exact, so boundaries are known."""
    return max(1, len(text.split()))


def make_doc(paragraphs: list[str], sections: list[tuple[str, ...]] | None = None) -> Document:
    """Build a document whose blocks are the given paragraphs, separated by blank lines."""
    parts: list[str] = []
    blocks: list[Block] = []
    cursor = 0
    for index, para in enumerate(paragraphs):
        start = cursor
        parts.append(para)
        cursor += len(para)
        blocks.append(
            Block(
                block_id=f"b{index:03d}",
                kind=BlockKind.PARAGRAPH,
                span=Span(start, cursor),
                section_path=sections[index] if sections else (),
            )
        )
        parts.append("\n\n")
        cursor += 2
    return Document(doc_id="d1", text="".join(parts), blocks=tuple(blocks))


class TestAtomize:
    def test_spans_point_at_the_words(self):
        atoms = atomize("alpha beta", word_tokens)
        assert [(a.span.start, a.span.end) for a in atoms] == [(0, 5), (6, 10)]

    def test_ignores_whitespace_runs(self):
        assert len(atomize("  a \n\n b  ", word_tokens)) == 2

    def test_empty_text_has_no_atoms(self):
        assert atomize("   ", word_tokens) == []


class TestApproxTokenCount:
    def test_scales_with_length(self):
        assert approx_token_count("x" * 400) == 100

    def test_never_returns_zero(self):
        # A zero-cost atom would let the packer loop without making progress.
        assert approx_token_count("") >= 1
        assert approx_token_count("a") >= 1


class TestChunkId:
    def test_encodes_the_span(self):
        assert chunk_id_for("doc", Span(5, 120)) == "doc#00000005-00000120"

    def test_zero_padding_makes_string_order_match_numeric_order(self):
        ids = [chunk_id_for("d", Span(s, s + 10)) for s in (9, 100, 1000)]
        assert ids == sorted(ids)


class TestFixedSizeChunker:
    def test_respects_the_token_budget(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(100))])
        chunks = FixedSizeChunker(10, 0, word_tokens).chunk(doc)
        assert all(word_tokens(c.text) <= 10 for c in chunks)
        assert len(chunks) == 10

    def test_covers_the_whole_document_with_no_overlap(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(30))])
        chunks = FixedSizeChunker(10, 0, word_tokens).chunk(doc)
        recovered = " ".join(c.text for c in chunks).split()
        assert recovered == [f"w{i}" for i in range(30)]

    def test_overlap_repeats_trailing_words(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(20))])
        chunks = FixedSizeChunker(10, 3, word_tokens).chunk(doc)
        first = chunks[0].text.split()
        second = chunks[1].text.split()
        assert first[-3:] == second[:3]

    def test_spans_slice_back_to_the_chunk_text(self):
        doc = make_doc([" ".join(f"word{i}" for i in range(60))])
        for chunk in FixedSizeChunker(7, 2, word_tokens).chunk(doc):
            assert doc.slice(chunk.span) == chunk.text

    def test_deterministic_across_runs(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(80))])
        chunker = FixedSizeChunker(9, 2, word_tokens)
        assert [c.chunk_id for c in chunker.chunk(doc)] == [c.chunk_id for c in chunker.chunk(doc)]

    def test_atom_larger_than_the_budget_is_emitted_alone(self):
        """Regression: an over-budget atom would otherwise stall the packer.

        A single token costing more than the whole budget can never fit, so the
        inner loop admits nothing and start and end stay equal -- an infinite
        loop. It is emitted as its own chunk instead, because truncating it would
        break the span-to-text correspondence the eval set relies on.
        """
        doc = make_doc(["short " + "x" * 500])
        chunks = FixedSizeChunker(2, 0, word_tokens).chunk(doc)
        assert any("x" * 500 in c.text for c in chunks)

    def test_empty_document_yields_nothing(self):
        assert FixedSizeChunker(10, 0, word_tokens).chunk(Document("d", "   ")) == []

    @pytest.mark.parametrize(("target", "overlap"), [(0, 0), (-1, 0), (10, 10), (10, 11)])
    def test_invalid_configuration_is_rejected(self, target: int, overlap: int):
        with pytest.raises(ValueError):
            FixedSizeChunker(target, overlap, word_tokens)

    def test_overlap_equal_to_target_would_never_terminate(self):
        """Pinned as a constructor error rather than discovered as a hang."""
        with pytest.raises(ValueError, match="overlap_tokens"):
            FixedSizeChunker(64, 64, word_tokens)


class TestStructureAwareChunker:
    def test_does_not_cross_a_section_boundary(self):
        doc = make_doc(
            ["alpha beta gamma", "delta epsilon zeta"],
            sections=[("Item 1",), ("Item 2",)],
        )
        chunks = StructureAwareChunker(100, 200, word_tokens).chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].section_path == ("Item 1",)
        assert chunks[1].section_path == ("Item 2",)

    def test_packs_several_blocks_of_one_section_together(self):
        doc = make_doc([f"para {i} text" for i in range(4)], sections=[("Item 1",)] * 4)
        chunks = StructureAwareChunker(100, 200, word_tokens).chunk(doc)
        assert len(chunks) == 1

    def test_never_splits_a_table_even_over_budget(self):
        """The rule that distinguishes this chunker from the baseline.

        Splitting a financial table leaves the numbers in one chunk and their
        column headers in the other: the half with the numbers has nothing
        lexical to match a query and nothing to tell a reader what the figures
        mean.
        """
        rows = "\n".join(f"| row{i} | {i * 1000} |" for i in range(200))
        text = "intro\n\n" + rows + "\n\n"
        doc = Document(
            doc_id="d1",
            text=text,
            blocks=(
                Block("b000", BlockKind.PARAGRAPH, Span(0, 5), ("Note 7",)),
                Block(
                    "t001",
                    BlockKind.TABLE,
                    Span(7, 7 + len(rows)),
                    ("Note 7",),
                    table=Table(rows=(("a", "b"),)),
                ),
            ),
        )
        chunks = StructureAwareChunker(10, 20, word_tokens).chunk(doc)
        table_chunks = [c for c in chunks if c.contains_table]
        assert len(table_chunks) == 1
        assert "row0" in table_chunks[0].text
        assert "row199" in table_chunks[0].text

    def test_oversize_chunks_are_counted_not_hidden(self):
        rows = "\n".join(f"| row{i} | {i} |" for i in range(100))
        doc = Document(
            doc_id="d1",
            text=rows,
            blocks=(
                Block(
                    "t001",
                    BlockKind.TABLE,
                    Span(0, len(rows)),
                    (),
                    table=Table(rows=(("a", "b"),)),
                ),
            ),
        )
        chunker = StructureAwareChunker(10, 20, word_tokens)
        chunks = chunker.chunk(doc)
        report = chunker.oversize_report(len(chunks))
        assert report.n_oversize == 1
        assert report.largest_tokens > 10
        assert report.fraction_oversize == pytest.approx(1.0)

    def test_boilerplate_is_excluded_from_chunks(self):
        doc = Document(
            doc_id="d1",
            text="TABLE OF CONTENTS\n\nreal content here\n\n",
            blocks=(
                Block("b000", BlockKind.BOILERPLATE, Span(0, 17), ()),
                Block("b001", BlockKind.PARAGRAPH, Span(19, 36), ("Item 1",)),
            ),
        )
        chunks = StructureAwareChunker(100, 200, word_tokens).chunk(doc)
        assert len(chunks) == 1
        assert "CONTENTS" not in chunks[0].text

    def test_long_prose_block_is_split(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(500))], sections=[("Item 1A",)])
        chunks = StructureAwareChunker(20, 40, word_tokens).chunk(doc)
        assert len(chunks) > 1
        assert all(word_tokens(c.text) <= 40 for c in chunks)

    def test_long_prose_split_can_be_disabled(self):
        doc = make_doc([" ".join(f"w{i}" for i in range(200))], sections=[("Item 1A",)])
        chunker = StructureAwareChunker(20, 40, word_tokens, split_long_prose=False)
        assert len(chunker.chunk(doc)) == 1

    def test_spans_slice_back_to_the_chunk_text(self):
        doc = make_doc([f"paragraph {i} with words" for i in range(10)])
        for chunk in StructureAwareChunker(12, 40, word_tokens).chunk(doc):
            assert doc.slice(chunk.span) == chunk.text

    def test_max_below_target_is_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            StructureAwareChunker(target_tokens=100, max_tokens=50)


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        spans = split_sentences("One thing. Two things. Three.")
        assert len(spans) == 3

    def test_offset_is_applied(self):
        spans = split_sentences("A cat. A dog.", offset=100)
        assert spans[0].start == 100

    def test_does_not_split_common_filing_abbreviations(self):
        """Filings are dense with "U.S.", "Inc." and "No. 12".

        An aggressive splitter shatters these into fragments whose embeddings
        carry no meaning. Missing a boundary only makes a sentence longer, so the
        conservative direction is the correct one.
        """
        text = "The U.S. segment grew. Apple Inc. reported more."
        assert len(split_sentences(text)) == 2

    def test_single_sentence_without_punctuation(self):
        spans = split_sentences("no terminal punctuation here")
        assert len(spans) == 1

    def test_whitespace_only_yields_nothing(self):
        assert split_sentences("   \n  ") == []


class TestSemanticChunker:
    @staticmethod
    def embed_by_marker(texts):
        """Deterministic fake: sentences containing "SHIFT" point the other way.

        Gives exactly known cosine distances, so breakpoints are predictable and
        the test needs no model and no network.
        """
        vectors = []
        for text in texts:
            vectors.append([0.0, 1.0] if "SHIFT" in text else [1.0, 0.0])
        return np.array(vectors, dtype=np.float64)

    def test_breaks_where_meaning_diverges(self):
        doc = make_doc(["Alpha one. Alpha two. SHIFT topic. SHIFT again."])
        chunker = SemanticChunker(self.embed_by_marker, 50.0, count_tokens=word_tokens)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        assert "Alpha one." in chunks[0].text
        assert "SHIFT" in chunks[-1].text

    def test_uniform_document_is_not_broken_below_the_cap(self):
        doc = make_doc(["Same one. Same two. Same three. Same four."])
        chunker = SemanticChunker(
            lambda texts: np.tile([1.0, 0.0], (len(texts), 1)),
            95.0,
            max_tokens=1000,
            count_tokens=word_tokens,
        )
        assert len(chunker.chunk(doc)) == 1

    def test_size_cap_overrides_the_semantic_signal(self):
        """A flat distance distribution must not yield one filing-sized chunk."""
        doc = make_doc([" ".join(f"Sentence {i} here." for i in range(40))])
        chunker = SemanticChunker(
            lambda texts: np.tile([1.0, 0.0], (len(texts), 1)),
            99.0,
            max_tokens=10,
            count_tokens=word_tokens,
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_table_is_one_unit_and_is_never_split(self):
        rows = "\n".join(f"| row{i} | {i} |" for i in range(30))
        text = "Intro sentence. Another one.\n\n" + rows
        doc = Document(
            doc_id="d1",
            text=text,
            blocks=(
                Block("b000", BlockKind.PARAGRAPH, Span(0, 27), ("Note 7",)),
                Block(
                    "t001",
                    BlockKind.TABLE,
                    Span(29, len(text)),
                    ("Note 7",),
                    table=Table(rows=(("a", "b"),)),
                ),
            ),
        )
        chunker = SemanticChunker(
            self.embed_by_marker, 50.0, max_tokens=10_000, count_tokens=word_tokens
        )
        table_chunks = [c for c in chunker.chunk(doc) if c.contains_table]
        assert len(table_chunks) == 1
        assert "row0" in table_chunks[0].text and "row29" in table_chunks[0].text

    def test_zero_vector_does_not_produce_nan_distances(self):
        """Regression: a zero embedding makes cosine distance NaN.

        NaN propagates into the percentile, the threshold becomes NaN, every
        comparison against it is False, and the chunker silently stops breaking
        anywhere -- producing one enormous chunk with no error.
        """
        doc = make_doc(["One. Two. Three. Four."])
        chunker = SemanticChunker(
            lambda texts: np.zeros((len(texts), 2)),
            90.0,
            max_tokens=10_000,
            count_tokens=word_tokens,
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        assert all(c.text for c in chunks)

    def test_spans_slice_back_to_the_chunk_text(self):
        doc = make_doc(["Alpha one. Alpha two. SHIFT here. More SHIFT text."])
        for chunk in SemanticChunker(self.embed_by_marker, 50.0, count_tokens=word_tokens).chunk(
            doc
        ):
            assert doc.slice(chunk.span) == chunk.text

    def test_single_sentence_document(self):
        doc = make_doc(["Only one sentence"])
        chunks = SemanticChunker(self.embed_by_marker, 90.0, count_tokens=word_tokens).chunk(doc)
        assert len(chunks) == 1

    def test_empty_document_yields_nothing(self):
        chunker = SemanticChunker(self.embed_by_marker, 90.0, count_tokens=word_tokens)
        assert chunker.chunk(Document("d", "   ")) == []

    @pytest.mark.parametrize("bad", [10.0, 49.9, 100.0, 150.0])
    def test_invalid_percentile_is_rejected(self, bad: float):
        with pytest.raises(ValueError, match="breakpoint_percentile"):
            SemanticChunker(self.embed_by_marker, bad)


class TestChunkersAgreeOnInvariants:
    """Properties every chunker must satisfy, checked against all three."""

    @pytest.fixture
    def doc(self) -> Document:
        return make_doc(
            [f"Section {i} sentence one. Section {i} sentence two." for i in range(6)],
            sections=[("Item 1",)] * 3 + [("Item 2",)] * 3,
        )

    @pytest.fixture(params=["fixed", "structure", "semantic"])
    def chunker(self, request):
        if request.param == "fixed":
            return FixedSizeChunker(8, 2, word_tokens)
        if request.param == "structure":
            return StructureAwareChunker(8, 40, word_tokens)
        return SemanticChunker(
            lambda texts: np.tile([1.0, 0.0], (len(texts), 1)),
            95.0,
            max_tokens=12,
            count_tokens=word_tokens,
        )

    def test_span_slices_to_text(self, chunker, doc):
        for chunk in chunker.chunk(doc):
            assert doc.slice(chunk.span) == chunk.text

    def test_chunks_are_ordered_by_start(self, chunker, doc):
        starts = [c.span.start for c in chunker.chunk(doc)]
        assert starts == sorted(starts)

    def test_no_empty_chunks(self, chunker, doc):
        assert all(c.text.strip() for c in chunker.chunk(doc))

    def test_chunk_ids_are_unique(self, chunker, doc):
        ids = [c.chunk_id for c in chunker.chunk(doc)]
        assert len(ids) == len(set(ids))

    def test_doc_id_is_carried(self, chunker, doc):
        assert all(c.doc_id == "d1" for c in chunker.chunk(doc))

    def test_running_twice_gives_identical_results(self, chunker, doc):
        first = chunker.chunk(doc)
        second = chunker.chunk(doc)
        assert [(c.chunk_id, c.text) for c in first] == [(c.chunk_id, c.text) for c in second]

    def test_chunk_corpus_concatenates(self, chunker, doc):
        assert len(chunker.chunk_corpus([doc, doc])) == 2 * len(chunker.chunk(doc))
