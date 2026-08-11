"""Tests for replaying a chunking recorded on another machine.

Offline: boundaries are written and read from tmp_path, and the chunking being
replayed is produced by a deterministic chunker so no GPU or model is involved.
"""

from __future__ import annotations

import pytest

from retrieval_ablation.chunking import (
    BoundaryMismatchError,
    FixedSizeChunker,
    approx_token_count,
    corpus_digest,
    load_boundaries,
)
from retrieval_ablation.chunking.replay import write_boundaries
from retrieval_ablation.corpus.models import Block, BlockKind, Document, Span

TEXT = " ".join(f"Sentence number {i} about operating revenue and expenses." for i in range(60))


def document(doc_id: str = "d1", text: str = TEXT) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        blocks=(
            Block(
                block_id="b1",
                kind=BlockKind.PARAGRAPH,
                span=Span(0, len(text)),
                section_path=("Part I",),
            ),
        ),
        metadata={},
    )


def record(tmp_path, docs, chunks, digest=None):
    path = tmp_path / "chunks-semantic95.json.gz"
    write_boundaries(path, "semantic95", "bge-m3", digest or corpus_digest(docs), chunks)
    return path


class TestReplayChunker:
    def test_replay_reproduces_the_original_chunks_exactly(self, tmp_path):
        """A chunk is its document and its span, so boundaries are sufficient.

        This is what makes chunk-semantic95 measurable at all: its breakpoints
        need a GPU to place, but nothing about the resulting chunks needs one to
        rebuild. If replay did not reproduce ids and text exactly, the gold labels
        -- which are character spans -- would not line up.
        """
        docs = [document()]
        original = FixedSizeChunker(40, 8, approx_token_count, name="fixed40o8").chunk_corpus(docs)
        assert len(original) > 1

        replayed = load_boundaries(record(tmp_path, docs, original), docs).chunk_corpus(docs)

        assert [c.chunk_id for c in replayed] == [c.chunk_id for c in original]
        assert [c.text for c in replayed] == [c.text for c in original]
        assert [(c.span.start, c.span.end) for c in replayed] == [
            (c.span.start, c.span.end) for c in original
        ]

    def test_boundaries_from_a_different_corpus_are_refused(self, tmp_path):
        """Spans are offsets, and offsets into changed text point anywhere.

        The same class of failure as a query vector reused after a rewrite: the
        artifact still loads, still covers every document, and quietly indexes the
        wrong characters. A digest of document ids and lengths is enough to catch
        it, because a span cannot become wrong while every document keeps its
        length.
        """
        docs = [document()]
        chunks = FixedSizeChunker(40, 8, approx_token_count, name="f").chunk_corpus(docs)
        path = record(tmp_path, docs, chunks)

        moved = [document(text=TEXT + " One more sentence appended later.")]
        with pytest.raises(BoundaryMismatchError, match="different corpus"):
            load_boundaries(path, moved)

    def test_a_document_with_no_recorded_boundaries_yields_nothing(self, tmp_path):
        """Better an empty document than a silently mixed chunking.

        Falling back to another chunker for the missing document would produce a
        corpus that is part semantic and part fixed-size -- a configuration nobody
        specified and no reader could interpret from the results table.
        """
        docs = [document("d1"), document("d2")]
        only_d1 = FixedSizeChunker(40, 8, approx_token_count, name="f").chunk(docs[0])
        # Digest must still describe both documents, or the load is refused first.
        chunker = load_boundaries(record(tmp_path, docs, only_d1), docs)

        assert chunker.chunk(docs[0])
        assert chunker.chunk(docs[1]) == []

    def test_digest_ignores_document_order(self):
        """Corpus order is an artefact of iteration, not a property of the corpus."""
        a, b = document("d1"), document("d2", TEXT[:500])

        assert corpus_digest([a, b]) == corpus_digest([b, a])
