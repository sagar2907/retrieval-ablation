"""Export first-stage candidate lists for the GPU worker to rerank.

Reranking is split across two machines: the first stage runs here (BM25 is
CPU-only and fast), the cross-encoder runs on a Kaggle T4 because no free
reranking API exists and PyTorch cannot load locally. This script writes the
handoff.

Only candidate *ids* are written, not their text, and the file is gzipped so it can
be committed and travel with a `git clone`.

Shipping the text was the first design, on the grounds that any drift between the
local corpus and the remote rebuild would silently pair a query with the wrong
passage and the cross-encoder would score it confidently. That produced an 86 MB
file, too large to commit, which would have forced a manual dataset upload into
every run.

It was also unnecessary. The remote side calls `load_corpus()`, which verifies the
SHA-256 of every document's parsed text against the committed manifest and raises
on any mismatch. Given an identical corpus, chunking is a deterministic pure
function, so chunk ids and chunk texts are guaranteed identical on both sides. The
invariant that makes ids sufficient was already enforced; the safeguard had to be
noticed rather than duplicated. The remote side additionally asserts that every
exported id is present in its rebuilt corpus, so a divergence fails loudly instead
of producing scores against the wrong passages.

Run: python -m retrieval_ablation.ablation.export_candidates
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging

from ..config import RESULTS_DIR, ensure_dirs
from ..corpus.ingest import load_corpus
from ..evalset.build import QUERIES_PATH
from ..evalset.paraphrase import PARAPHRASED_PATH
from ..evalset.schema import read_eval_set
from ..index.artifacts import dense_index_from_artifact, load_query_vectors
from ..index.base import Retriever
from ..index.bm25 import BM25Index
from ..index.fusion import HybridRetriever
from .configs import build_grid
from .runner import make_chunker

log = logging.getLogger(__name__)

#: Deepest shortlist any reranking configuration in the grid asks for. Exporting
#: the deepest once lets every shallower configuration be evaluated from the same
#: scores by truncation, instead of re-running the cross-encoder per depth.
MAX_CANDIDATES = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paraphrased",
        action="store_true",
        help=(
            "build shortlists from data/eval/queries-paraphrased.jsonl and write to "
            "candidates-<config>-paraphrased.json.gz"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_dirs()

    # The shortlist is produced by running the first stage over the queries, so it
    # belongs to a wording as much as the cross-encoder scores computed from it do.
    # Reranking the paraphrased queries against shortlists BM25 retrieved for the
    # original ones would measure a pipeline nobody would ever build: a first stage
    # answering one question and a second stage re-ranking it for another.
    queries_path = PARAPHRASED_PATH if args.paraphrased else QUERIES_PATH
    if not queries_path.exists():
        raise SystemExit(
            f"{queries_path} does not exist. Run "
            f"`python -m retrieval_ablation.evalset.paraphrase` first."
        )
    suffix = "-paraphrased" if args.paraphrased else ""

    docs = load_corpus()
    queries = read_eval_set(queries_path)
    log.info("loaded %d documents, %d queries from %s", len(docs), len(queries), queries_path.name)

    reranking_configs = [c for c in build_grid() if c.reranker is not None]
    # Group by what actually determines the candidate list: the chunker and the
    # first stage. Depth is handled by exporting the maximum.
    seen: set[tuple[str, str]] = set()

    for config in reranking_configs:
        signature = (config.chunker, config.retrieval)
        if signature in seen:
            continue
        seen.add(signature)

        chunker = make_chunker(config.chunker)
        chunks = chunker.chunk_corpus(docs)
        bm25 = BM25Index(chunks)
        log.info("%s: %d chunks indexed", config.name, len(chunks))

        # A shortlist is whatever the configuration's *own* first stage returned.
        # Exporting a BM25 shortlist for a hybrid configuration would score a
        # pipeline whose two stages disagree about what was retrieved, and the
        # results table would still call it hybrid-plus-rerank.
        #
        # This branch used to skip hybrid outright, on the grounds that dense
        # vectors "do not exist yet" -- true when it was written, and quietly
        # false ever since the GPU run produced them. The consequence was not a
        # missing row but a mislabelled one: hybrid-plus-rerank was measured, on a
        # lexical shortlist. Presence is now checked rather than assumed.
        index: Retriever = bm25
        if config.retrieval != "bm25":
            vectors = RESULTS_DIR / f"vectors-{config.embedding}-{config.chunker}.npz"
            query_vectors = (
                load_query_vectors(config.embedding, {q.query_id: q.text for q in queries})
                if config.embedding
                else None
            )
            # Coverage, not mere presence. PrecomputedEmbedder raises KeyError for a
            # query it has no vector for, and that aborts the export -- the same
            # crash the runner hit when the eval set grew from 216 to 586, in a
            # second place that made the same "are there vectors?" check instead of
            # "are these queries covered?". Partial coverage would also produce a
            # shortlist for a subset while the filename claims the whole set.
            covered = len(query_vectors or {})
            if not vectors.exists() or covered < len(queries):
                log.warning(
                    "skipping %s: a %s first stage needs vectors for %s covering all "
                    "%d queries; %d are covered",
                    config.name,
                    config.retrieval,
                    config.embedding,
                    len(queries),
                    covered,
                )
                continue
            dense = dense_index_from_artifact(vectors, chunks, query_vectors)
            index = (
                dense
                if config.retrieval == "dense"
                else HybridRetriever([bm25, dense], k=config.rrf_k)
            )
            log.info("%s: first stage is %s", config.name, index.name)

        payload: dict[str, dict] = {}
        for query in queries:
            hits = index.search(query.text, top_k=MAX_CANDIDATES)
            if not hits:
                continue
            payload[query.query_id] = {
                "query": query.text,
                "chunker": config.chunker,
                "candidate_ids": [h.chunk_id for h in hits],
                "first_stage_scores": [round(h.score, 4) for h in hits],
            }

        out = RESULTS_DIR / f"candidates-{config.name}{suffix}.json.gz"
        with gzip.open(out, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)

        pairs = sum(len(v["candidate_ids"]) for v in payload.values())
        size_mb = out.stat().st_size / 1e6
        log.info("wrote %s: %d queries, %d pairs, %.2f MB", out.name, len(payload), pairs, size_mb)
        print(
            f"\n{out}\n  {len(payload)} queries, {pairs:,} query-passage pairs "
            f"to score, {size_mb:.2f} MB gzipped"
        )
        print("  candidate texts are re-derived remotely from the checksum-verified corpus")


if __name__ == "__main__":
    main()
