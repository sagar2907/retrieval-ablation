"""Rebuild a chunking from committed boundaries instead of recomputing it.

WHY THIS EXISTS

Every other chunker in this project is a pure function of the document, so the
ablation can rebuild its chunks anywhere. Semantic chunking is not: it embeds
each sentence and breaks where consecutive sentences are least similar, so it
needs a GPU that this machine cannot provide. Without a way to carry its
boundaries across, `chunk-semantic95` is permanently unmeasurable -- one of
fifteen configurations reported as "not measured" forever, for an infrastructure
reason rather than a finding.

A chunk is fully determined by its document and its character span: `Chunker._build`
derives the id, the text, the section path and the table flag from those two
things alone. So the boundaries are the whole artifact, and they are small --
tens of thousands of integer pairs rather than tens of megabytes of text.

WHAT IS CHECKED, AND WHY

Boundaries computed against one corpus are meaningless against another, in exactly
the way a query vector is meaningless for text it was not built from. Both failure
modes have already happened here. So the file records the corpus digest and the
embedder that produced it, and loading refuses on a mismatch rather than replaying
spans into a document that has since changed length.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Chunker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..corpus.models import Document
    from ..evalset.relevance import Chunk

log = logging.getLogger(__name__)


class BoundaryMismatchError(RuntimeError):
    """Raised when committed boundaries do not belong to this corpus."""


class ReplayChunker(Chunker):
    """Replays a chunking recorded elsewhere, span for span."""

    def __init__(self, boundaries: dict[str, list[tuple[int, int]]], name: str) -> None:
        self._boundaries = boundaries
        self.name = name

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = self._boundaries.get(doc.doc_id)
        if spans is None:
            # A document with no recorded boundaries contributes nothing rather
            # than falling back to some other chunking. A silently mixed corpus --
            # part semantic, part fixed-size -- would be a configuration nobody
            # described and nobody could interpret.
            log.warning("%s: no recorded boundaries for %s", self.name, doc.doc_id)
            return []
        out: list[Chunk] = []
        for start, end in spans:
            chunk = self._build(doc, start, end)
            if chunk is not None:
                out.append(chunk)
        return out


def write_boundaries(
    path: Path,
    chunker_name: str,
    embedder: str,
    corpus_digest: str,
    chunks: list[Chunk],
) -> None:
    """Record a chunking as document ids and spans."""
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append((chunk.span.start, chunk.span.end))
    payload = {
        "chunker": chunker_name,
        "embedder": embedder,
        "corpus_digest": corpus_digest,
        "n_chunks": len(chunks),
        "boundaries": {doc_id: sorted(spans) for doc_id, spans in by_doc.items()},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def corpus_digest(docs: list[Document]) -> str:
    """A digest of the corpus a chunking was computed against.

    Built from document ids and lengths rather than full text: it has to change
    whenever a span could point somewhere different, and a document cannot change
    length without that being true. Cheap enough to compute on both sides.
    """
    parts = "".join(f"{d.doc_id}:{len(d.text)};" for d in sorted(docs, key=lambda d: d.doc_id))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def load_boundaries(path: Path, docs: list[Document]) -> ReplayChunker:
    """Load recorded boundaries, refusing them if the corpus has moved."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected = payload.get("corpus_digest")
    actual = corpus_digest(docs)
    if expected != actual:
        raise BoundaryMismatchError(
            f"{path.name} was computed against a different corpus "
            f"({expected} vs {actual}). Replaying its spans would index into text "
            f"that has since changed. Re-run the GPU notebook."
        )
    boundaries = {
        doc_id: [(int(a), int(b)) for a, b in spans]
        for doc_id, spans in payload["boundaries"].items()
    }
    log.info(
        "%s: replaying %d chunks from %s",
        payload.get("chunker", path.stem),
        payload.get("n_chunks", 0),
        path.name,
    )
    return ReplayChunker(boundaries, name=payload.get("chunker", path.stem))
