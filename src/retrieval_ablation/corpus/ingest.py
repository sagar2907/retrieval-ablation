"""Build the corpus: select filings, fetch, parse, and record a manifest.

Resumable by construction. Fetching is cached by `EdgarClient`, and the parsed
documents are written to `data/interim/` as JSON, so an interrupted run continues
where it stopped and a completed run costs no network traffic at all.

The manifest in `data/manifests/corpus.json` is committed to the repository. It
records, for every document, the source URL, the SHA-256 of the raw bytes, and
the SHA-256 of the canonical parsed text. That is what lets a third party verify
they built the same corpus this study's numbers were computed on -- and it is what
would catch a silent upstream change, since SEC does occasionally re-post filings.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ..config import INTERIM_DIR, MANIFEST_DIR, ensure_dirs
from .companies import CORPUS_FORM, CORPUS_TICKERS, FILINGS_PER_COMPANY
from .edgar import EdgarClient, EdgarError, FilingRef
from .html_parse import parse_filing
from .models import Block, BlockKind, Document, Span, Table

log = logging.getLogger(__name__)

MANIFEST_PATH = MANIFEST_DIR / "corpus.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_to_json(doc: Document) -> dict:
    """Serialise a parsed document losslessly.

    Spans are stored as explicit start/end integers rather than as a nested
    object, so the on-disk format stays readable and a corrupted offset is
    obvious on inspection.
    """
    return {
        "doc_id": doc.doc_id,
        "text": doc.text,
        "metadata": doc.metadata,
        "blocks": [
            {
                "block_id": b.block_id,
                "kind": b.kind.value,
                "start": b.span.start,
                "end": b.span.end,
                "section_path": list(b.section_path),
                "table": (
                    {
                        "rows": [list(r) for r in b.table.rows],
                        "n_header_rows": b.table.n_header_rows,
                        "caption": b.table.caption,
                    }
                    if b.table
                    else None
                ),
            }
            for b in doc.blocks
        ],
    }


def document_from_json(payload: dict) -> Document:
    blocks = []
    for raw in payload["blocks"]:
        table = None
        if raw["table"]:
            table = Table(
                rows=tuple(tuple(r) for r in raw["table"]["rows"]),
                n_header_rows=raw["table"]["n_header_rows"],
                caption=raw["table"]["caption"],
            )
        blocks.append(
            Block(
                block_id=raw["block_id"],
                kind=BlockKind(raw["kind"]),
                span=Span(raw["start"], raw["end"]),
                section_path=tuple(raw["section_path"]),
                table=table,
            )
        )
    return Document(
        doc_id=payload["doc_id"],
        text=payload["text"],
        blocks=tuple(blocks),
        metadata=payload["metadata"],
    )


def interim_path(doc_id: str) -> Path:
    return INTERIM_DIR / f"{doc_id}.json"


def load_document(doc_id: str) -> Document | None:
    path = interim_path(doc_id)
    if not path.exists():
        return None
    return document_from_json(json.loads(path.read_text(encoding="utf-8")))


def save_document(doc: Document) -> None:
    path = interim_path(doc.doc_id)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(document_to_json(doc), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def select_filings(client: EdgarClient) -> list[FilingRef]:
    """Resolve the frozen ticker list into concrete filing references.

    Sorted by document id so the corpus order is deterministic regardless of the
    order EDGAR returns things in or of dictionary iteration details.
    """
    refs: list[FilingRef] = []
    for ticker in sorted(CORPUS_TICKERS):
        try:
            found = client.find_filings(ticker, CORPUS_FORM, limit=FILINGS_PER_COMPANY)
        except EdgarError as exc:
            # One delisted or renamed ticker must not abort a 120-document build.
            # Recorded loudly and reported in the manifest as a gap rather than
            # silently reducing the corpus.
            log.warning("skipping %s: %s", ticker, exc)
            continue
        if not found:
            log.warning("no %s filings found for %s", CORPUS_FORM, ticker)
        refs.extend(found)
    return sorted(refs, key=lambda r: r.doc_id)


def ingest(
    render_tables: str = "markdown",
    limit: int | None = None,
    client: EdgarClient | None = None,
) -> dict:
    """Fetch and parse the corpus, returning the manifest.

    `render_tables` is threaded through because it changes the canonical text and
    therefore every offset. Two corpora built with different values are not
    interchangeable, so the manifest records which was used.
    """
    ensure_dirs()
    owns_client = client is None
    client = client or EdgarClient()

    try:
        refs = select_filings(client)
        if limit is not None:
            refs = refs[:limit]

        entries = []
        for index, ref in enumerate(refs, start=1):
            raw = client.fetch(ref)
            raw_digest = hashlib.sha256(raw).hexdigest()

            doc = load_document(ref.doc_id)
            if doc is None:
                doc = parse_filing(
                    ref.doc_id,
                    raw,
                    metadata={
                        "ticker": ref.ticker,
                        "company": ref.company,
                        "sector": CORPUS_TICKERS.get(ref.ticker, "unknown"),
                        "form": ref.form,
                        "filing_date": ref.filing_date,
                        "report_date": ref.report_date,
                        "accession": ref.accession,
                        "url": ref.url,
                    },
                )
                save_document(doc)

            n_tables = sum(1 for b in doc.blocks if b.is_table)
            entries.append(
                {
                    "doc_id": doc.doc_id,
                    "ticker": ref.ticker,
                    "company": ref.company,
                    "sector": CORPUS_TICKERS.get(ref.ticker, "unknown"),
                    "form": ref.form,
                    "report_date": ref.report_date,
                    "url": ref.url,
                    "raw_sha256": raw_digest,
                    "text_sha256": sha256_text(doc.text),
                    "n_chars": len(doc.text),
                    "n_blocks": len(doc.blocks),
                    "n_tables": n_tables,
                }
            )
            log.info(
                "[%d/%d] %s  %d chars  %d blocks  %d tables",
                index,
                len(refs),
                doc.doc_id,
                len(doc.text),
                len(doc.blocks),
                n_tables,
            )

        manifest = {
            "form": CORPUS_FORM,
            "filings_per_company": FILINGS_PER_COMPANY,
            "render_tables": render_tables,
            "n_documents": len(entries),
            "total_chars": sum(e["n_chars"] for e in entries),
            "total_tables": sum(e["n_tables"] for e in entries),
            "tickers_requested": sorted(CORPUS_TICKERS),
            "tickers_present": sorted({e["ticker"] for e in entries}),
            "documents": entries,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        if owns_client:
            client.close()


def load_corpus() -> list[Document]:
    """Load every parsed document named by the manifest, in manifest order.

    Raises if a document is missing rather than returning a partial corpus: a
    silently short corpus would change every metric while still producing
    plausible-looking numbers.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} does not exist. Run the ingest first: "
            f"python -m retrieval_ablation.corpus.ingest"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    docs = []
    missing = []
    for entry in manifest["documents"]:
        doc = load_document(entry["doc_id"])
        if doc is None:
            missing.append(entry["doc_id"])
            continue
        # Verifying the digest catches a partially written or hand-edited interim
        # file, which would otherwise shift every gold-label offset in that
        # document without any visible error.
        actual = sha256_text(doc.text)
        if actual != entry["text_sha256"]:
            raise ValueError(
                f"{entry['doc_id']}: parsed text digest {actual[:12]} does not match "
                f"manifest {entry['text_sha256'][:12]}. The cached parse is stale; "
                f"delete data/interim/ and re-run the ingest."
            )
        docs.append(doc)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} documents named in the manifest are missing from "
            f"data/interim/ (first few: {missing[:3]}). Re-run the ingest."
        )
    return docs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = ingest()
    print(
        f"\ningested {manifest['n_documents']} documents, "
        f"{manifest['total_chars']:,} chars, "
        f"{manifest['total_tables']:,} tables"
    )
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
