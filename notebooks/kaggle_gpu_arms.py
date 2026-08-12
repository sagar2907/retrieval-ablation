"""Kaggle GPU worker: produce the embedding and reranking artifacts the ablation needs.

WHY THIS RUNS ON KAGGLE RATHER THAN LOCALLY OR THROUGH AN API
------------------------------------------------------------
Two independent constraints, both measured rather than assumed.

Locally, Windows Smart App Control is enforced on the development machine, so
PyTorch cannot load at all -- the Code Integrity log names torch_cpu.dll, which
rules out the CPU build too. Disabling that control is machine-wide and
irreversible without reinstalling Windows, so it is not an acceptable fix.

Through the Gemini API, embedding is possible but impractical at this scale. The
free tier caps batchEmbedContents at 32 texts per request and permits roughly
three requests per minute per model (quota
EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier), which measures
out at about 96 texts/minute -- roughly 7.3 hours for one pass over a
42,215-chunk corpus, per embedding model. More decisively, there is no free
cross-encoder reranking endpoint at all, and reranking is the component the study
most needs to measure.

A Kaggle T4 session has none of those limits: torch is preinstalled, the GPU is
free for ~30 hours a week, and internet access is available.

WHAT THIS PRODUCES
------------------
Four artifacts, written to /kaggle/working and downloaded back into data/indexes/:

  vectors-<model>-<chunker>.npz   dense vectors keyed by chunk id
  rerank-scores-<config>.json     cross-encoder scores per (query, candidate)
  gpu_report.json                 the device and throughput actually observed
  MANIFEST.json                   checksums, so the local side can verify these
                                  artifacts match the corpus they were built from

HOW TO RUN
----------
1. New Kaggle notebook, Settings -> Accelerator -> GPU T4 x2, Internet -> On.
2. Paste this file into one cell and run it. Nothing needs editing.
3. When it finishes, download from /kaggle/working -- the .npz files, the
   rerank-scores-*.json files, chunks-*.json.gz and MANIFEST.json. Ignore the
   retrieval-ablation/ folder, which is this repository's own clone.
4. Copy them into results/ locally and re-run the ablation.

One session covers both wordings and every model. Everything it writes is keyed
to the eval set committed in the repository at clone time, so the local side
refuses any artifact that does not match the queries it is scoring rather than
quietly using it.

The corpus is rebuilt from EDGAR inside the notebook rather than uploaded, so the
notebook is self-contained and no 60 MB dataset upload is needed. Rebuilding is
deterministic: the committed manifest's checksums are verified against it, and the
run aborts if they disagree, because embedding a different corpus than the one the
gold labels index into would produce confidently meaningless vectors.
"""

# ---------------------------------------------------------------------------
# Configuration. EDGAR requires a User-Agent naming a real contact.
# ---------------------------------------------------------------------------
EDGAR_USER_AGENT = "sagar2907 sagarsahu2907@gmail.com"
REPO = "https://github.com/sagar2907/retrieval-ablation.git"

#: (embedding model, chunker) pairs to build vectors for. Each pair is one full
#: pass over the corpus.
EMBEDDING_JOBS = [
    ("bge-m3", "struct512"),
    ("e5-base", "struct512"),
    ("e5-base-v2", "struct512"),
]

#: Chunkings that cannot be recomputed locally and must be recorded here.
#: Semantic chunking places its breakpoints by embedding every sentence, so it
#: needs this GPU. The boundaries it produces are replayed locally, which is what
#: makes chunk-semantic95 measurable instead of permanently "not measured".
BOUNDARY_JOBS = [("semantic95", "bge-m3")]

#: Reranking runs over candidate lists the local side commits as
#: results/candidates-<config>.json.gz. Those files hold chunk *ids* only; the
#: texts are rebuilt here from the checksum-verified corpus, which is what keeps
#: them small enough to travel with a git clone.
#:
#: If a file is absent, reranking for it is skipped with a recorded reason rather
#: than silently producing nothing.
RERANK_CANDIDATE_FILES = [
    "candidates-rerank-bm25-100.json.gz",
    "candidates-hybrid-plus-rerank.json.gz",
]

RERANKER = "BAAI/bge-reranker-v2-m3"
BATCH_EMBED = 64
BATCH_RERANK = 64

#: Which eval sets to embed queries from, and rerank against.
#:
#: "original"    -> data/eval/queries.jsonl
#: "paraphrased" -> data/eval/queries-paraphrased.jsonl
#:
#: Both files hold the same query ids and the same gold spans and differ only in
#: the wording of the questions, which is what makes the two ablation runs
#: comparable -- and also why a query vector is only valid for the text it was
#: built from. The local loader refuses vectors whose recorded text does not match
#: the queries being scored, so embedding the wrong file yields an artifact that
#: is silently useless for the run it was meant to unblock. Output filenames carry
#: the choice so both sets live side by side.
#:
#: Both are produced in one session because the two expensive steps -- rebuilding
#: the corpus from EDGAR and embedding 42,215 passages per model -- do not depend
#: on the wording at all. Doing one set per session paid for them twice.
QUERY_SETS = ["original", "paraphrased"]

# ---------------------------------------------------------------------------
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = Path("/kaggle/working/retrieval-ablation")


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:  # noqa: PLR0912,PLR0915 - a linear script; splitting it would obscure the order
    report: dict[str, object] = {}

    # -- environment ------------------------------------------------------
    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["device"] = props.name
        report["vram_mib"] = props.total_memory // (1024 * 1024)
    print(json.dumps(report, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Set Accelerator to GPU T4 in notebook settings.")

    # -- code and dependencies -------------------------------------------
    #
    # INSTALL STRATEGY, AND WHY IT IS NOT JUST `pip install -e`
    #
    # Kaggle images ship a large, mutually-consistent set of preinstalled
    # packages, and torch is compiled against the exact numpy that shipped with
    # the image. A plain editable install lets pip resolve this project's
    # dependency floors against that environment and upgrade numpy or scipy to
    # satisfy them -- which breaks torch's ABI, and the failure appears later as
    # an unrelated import error deep inside a model load.
    #
    # So: --no-deps for the project, then only the handful of pure-Python
    # packages that are genuinely imported and are unlikely to be present. Every
    # numeric and ML package (numpy, scipy, torch, transformers) is left exactly
    # as the image provides it.
    #
    # sentence-transformers is requested unpinned on purpose. Kaggle pins
    # transformers>=5.0.0; sentence-transformers 5.x declares
    # transformers<6.0.0,>=4.41.0 and so resolves cleanly, whereas pinning an
    # older sentence-transformers could drag transformers back to 4.x and
    # disturb the rest of the image.
    if not REPO_DIR.exists():
        sh(f"git clone --depth 1 {REPO} {REPO_DIR}")

    sh(f"{sys.executable} -m pip install -q --no-deps -e {REPO_DIR}")
    sh(f"{sys.executable} -m pip install -q lxml pydantic pydantic-settings python-dotenv tenacity")
    sh(f"{sys.executable} -m pip install -q sentence-transformers")
    sys.path.insert(0, str(REPO_DIR / "src"))

    # Confirm the image's numeric stack is intact before spending an hour on it.
    # A broken numpy/torch pairing is far cheaper to discover here than after the
    # corpus rebuild.
    import numpy

    print(f"numpy {numpy.__version__}, torch {torch.__version__}", flush=True)
    _ = torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")
    print("cuda matmul ok", flush=True)

    os.environ["EDGAR_USER_AGENT"] = EDGAR_USER_AGENT
    os.chdir(REPO_DIR)

    from retrieval_ablation.corpus.ingest import MANIFEST_PATH, ingest, load_corpus
    from retrieval_ablation.evalset.build import QUERIES_PATH
    from retrieval_ablation.evalset.schema import read_eval_set

    # -- corpus -----------------------------------------------------------
    # The committed manifest records the SHA-256 of every document's parsed text,
    # and load_corpus() checks each document against it. That check alone is
    # worthless here: ingest() REWRITES the manifest from whatever it just
    # downloaded, so load_corpus() would be verifying the new corpus against a
    # description of itself and could never disagree. An earlier run printed
    # "corpus verified" while holding a copy of the Southern Company 2022 10-K
    # that was 360 characters longer than the committed one, and the resulting
    # vector file contains a chunk id that does not exist in this repository.
    #
    # So snapshot the committed digests BEFORE ingest() can overwrite them, and
    # compare against the snapshot afterwards. The committed corpus is the one the
    # gold spans were labelled against, and a document that differs invalidates
    # every offset in it. When this check first fired it named two documents whose
    # raw bytes were byte-identical to the manifest: the disagreement was in the
    # parser, not the source. That is exactly the case a digest comparison exists
    # to catch, because nothing else about the run looks wrong.
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {d["doc_id"]: d["text_sha256"] for d in committed["documents"]}
    print(f"committed manifest: {committed['n_documents']} documents", flush=True)
    print("rebuilding corpus from EDGAR (this takes roughly 20-40 minutes)", flush=True)
    ingest()
    docs = load_corpus()

    # Three distinct problems, reported separately. An earlier version folded the
    # first two together, because `expected.get(doc_id)` returns None for a
    # document the manifest has never heard of and None never equals a digest. So
    # a brand-new filing was announced as "1 document with a different text
    # digest", which describes a corrupted document rather than an extra one and
    # sends the reader looking in entirely the wrong place.
    committed_ids = set(expected)
    present_ids = {d.doc_id for d in docs}
    unexpected = sorted(present_ids - committed_ids)
    absent = sorted(committed_ids - present_ids)
    drifted = sorted(
        d.doc_id
        for d in docs
        if d.doc_id in committed_ids
        and expected[d.doc_id] != hashlib.sha256(d.text.encode("utf-8")).hexdigest()
    )
    if drifted or absent or unexpected:
        # Fail loudly rather than embed it. Vectors are keyed by chunk id, and a
        # chunk id encodes character offsets, so a drifted document yields ids
        # that silently fail to match on the local side -- a missing arm reported
        # as a successful run.
        raise SystemExit(
            "corpus diverged from the committed manifest: "
            f"{len(drifted)} changed {drifted[:5]}, "
            f"{len(absent)} missing {absent[:5]}, "
            f"{len(unexpected)} not in the manifest {unexpected[:5]}. Refusing to embed. "
            "If filings appeared or disappeared, the corpus was re-selected rather "
            "than pinned -- ingest(pinned=True) rebuilds exactly the committed set."
        )
    print(f"corpus verified against committed digests: {len(docs)} documents", flush=True)

    # Both wordings in one session. The corpus rebuild and the passage embedding
    # passes are the expensive parts and neither depends on the query wording, so
    # doing one set per session pays for them twice. The local grid needs both
    # anyway: the original and paraphrased runs are only comparable when every arm
    # is measured on each.
    query_sets: dict[str, list] = {}
    for name in QUERY_SETS:
        if name not in ("original", "paraphrased"):
            raise SystemExit(
                f"QUERY_SETS entries must be 'original' or 'paraphrased', got {name!r}"
            )
        path = (
            QUERIES_PATH
            if name == "original"
            else QUERIES_PATH.with_name("queries-paraphrased.jsonl")
        )
        if not path.exists():
            raise SystemExit(
                f"{path} is not in the repository. Run "
                f"`python -m retrieval_ablation.evalset.paraphrase` locally and commit it first."
            )
        query_sets[name] = read_eval_set(path)
        print(f"query set {name}: {len(query_sets[name])} queries ({path.name})", flush=True)

    # Every arm is scored on the shared subset, so the two files must describe the
    # same benchmark. They differ only in wording, and a mismatch means one was
    # regenerated without the other.
    sizes = {name: len(qs) for name, qs in query_sets.items()}
    if len(set(sizes.values())) > 1:
        raise SystemExit(
            f"query sets disagree on size {sizes}. They must hold the same query ids; "
            f"re-run the paraphraser locally so both describe one benchmark."
        )
    queries = next(iter(query_sets.values()))

    from retrieval_ablation.ablation.runner import make_chunker
    from retrieval_ablation.embed.local import SentenceTransformerEmbedder

    # -- embedding passes -------------------------------------------------
    import numpy as np

    manifest: dict[str, object] = {
        "gpu": report,
        "n_documents": len(docs),
        "n_queries": len(queries),
        "artifacts": [],
    }

    for model_key, chunker_name in EMBEDDING_JOBS:
        out = WORK / f"vectors-{model_key}-{chunker_name}.npz"
        # The query set is part of the identity of a query-vector file. Writing
        # both to one name would mean a paraphrased run silently replacing the
        # original run's artifact, and the local loader would then refuse the
        # file for the original eval set -- trading one measured arm for another.
        qouts = {
            name: WORK / f"queryvectors-{model_key}{'' if name == 'original' else '-' + name}.npz"
            for name in query_sets
        }
        embedder = None

        # Skip work already on disk. A Kaggle session survives a failed cell, and
        # the embedding passes take about nineteen minutes combined -- repeating
        # them because a later stage crashed wastes GPU quota for nothing.
        #
        # This branch used to `continue`, which also skipped the query-vector
        # write below it. That is how a session whose passage vectors survived a
        # crash produced, on the retry, exactly the same unusable output as the
        # run before it: passage vectors present, query vectors absent, dense arm
        # still unmeasurable. The two artifacts are now written independently, so
        # reusing one never suppresses the other.
        if out.exists():
            existing = np.load(out, allow_pickle=True)
            print(
                f"\n=== {model_key} / {chunker_name}: reusing {out.name} "
                f"({existing['vectors'].shape[0]:,} vectors) ===",
                flush=True,
            )
            manifest["artifacts"].append(
                {
                    "file": out.name,
                    "embedder": model_key,
                    "chunker": chunker_name,
                    "n_chunks": int(existing["vectors"].shape[0]),
                    "dimension": int(existing["vectors"].shape[1]),
                    "reused": True,
                }
            )
        else:
            chunker = make_chunker(chunker_name)
            chunks = chunker.chunk_corpus(docs)
            print(f"\n=== {model_key} / {chunker_name}: {len(chunks):,} chunks ===", flush=True)

            embedder = SentenceTransformerEmbedder(model_key, batch_size=BATCH_EMBED)
            started = time.monotonic()
            vectors = embedder.encode_passages([c.text for c in chunks])
            elapsed = time.monotonic() - started
            rate = len(chunks) / elapsed if elapsed else 0.0
            print(f"  {elapsed / 60:.1f} min  ({rate:.0f} chunks/sec)", flush=True)

            np.savez_compressed(
                out,
                vectors=vectors,
                chunk_ids=np.array([c.chunk_id for c in chunks], dtype=object),
                embedder=model_key,
            )
            print(f"  wrote {out.name}", flush=True)
            manifest["artifacts"].append(
                {
                    "file": out.name,
                    "embedder": model_key,
                    "chunker": chunker_name,
                    "n_chunks": len(chunks),
                    "dimension": int(vectors.shape[1]),
                    "seconds": round(elapsed, 1),
                    "chunks_per_sec": round(rate, 1),
                }
            )

        # Query vectors, from the same model in the same session.
        #
        # Easy to forget and fatal to omit: a dense index needs BOTH sides
        # embedded by the same model. Passage vectors alone are unusable, because
        # the query cannot be embedded anywhere else -- a different model produces
        # a different vector space, and cosine similarity across two spaces is
        # meaningless while still returning confident-looking numbers. The first
        # version of this notebook shipped only passage vectors and the dense arm
        # could not run at all.
        #
        # Cheap: a few hundred queries, so this adds seconds per wording -- which
        # is why it is worth loading the model for even when the passage vectors
        # were reused, and why both wordings are done while it is loaded.
        for set_name, set_queries in query_sets.items():
            qout = qouts[set_name]
            if qout.exists():
                print(f"  reusing {qout.name}", flush=True)
                continue
            if embedder is None:
                embedder = SentenceTransformerEmbedder(model_key, batch_size=BATCH_EMBED)
            query_vectors = embedder.encode_queries([q.text for q in set_queries])
            np.savez_compressed(
                qout,
                vectors=query_vectors,
                query_ids=np.array([q.query_id for q in set_queries], dtype=object),
                # The text each vector was actually built from. Ids alone are not
                # enough: they survive a rewrite of the query text, which is what
                # makes the paraphrased eval set comparable to the original, and
                # therefore also what let the local side serve original-wording
                # vectors for paraphrased queries without noticing. The loader
                # refuses any artifact lacking this field.
                query_texts=np.array([q.text for q in set_queries], dtype=object),
                embedder=model_key,
            )
            manifest["artifacts"].append(
                {
                    "file": qout.name,
                    "embedder": model_key,
                    "query_set": set_name,
                    "n_queries": len(set_queries),
                    "dimension": int(query_vectors.shape[1]),
                }
            )
            print(f"  wrote {qout.name} ({len(set_queries)} query vectors)", flush=True)

        # Released before the next model loads. Two of these resident at once
        # exceeds a T4's memory once activations are counted.
        if embedder is not None:
            embedder.release()

    # -- chunkings that only a GPU can produce -----------------------------
    from retrieval_ablation.chunking import corpus_digest
    from retrieval_ablation.chunking.replay import write_boundaries

    digest = corpus_digest(docs)
    for chunker_name, model_key in BOUNDARY_JOBS:
        bout = WORK / f"chunks-{chunker_name}.json.gz"
        if bout.exists():
            print(f"  reusing {bout.name}", flush=True)
            continue
        boundary_embedder = SentenceTransformerEmbedder(model_key, batch_size=BATCH_EMBED)
        started = time.monotonic()
        chunks = make_chunker(chunker_name, boundary_embedder).chunk_corpus(docs)
        elapsed = time.monotonic() - started
        boundary_embedder.release()
        write_boundaries(bout, chunker_name, model_key, digest, chunks)
        manifest["artifacts"].append(
            {
                "file": bout.name,
                "chunker": chunker_name,
                "embedder": model_key,
                "n_chunks": len(chunks),
                "corpus_digest": digest,
                "seconds": round(elapsed, 1),
            }
        )
        mins = elapsed / 60
        print(f"=== {chunker_name}: {len(chunks):,} chunks in {mins:.1f} min ===", flush=True)

    # -- reranking --------------------------------------------------------
    from retrieval_ablation.index.rerank import CrossEncoderReranker

    reranker = None
    # The shortlist belongs to a wording: it is what the first stage returned for
    # those questions. Reranking paraphrased queries against shortlists retrieved
    # for the original ones would score a pipeline whose two stages are answering
    # different questions.
    rerank_jobs = [
        (
            set_name,
            base_file
            if set_name == "original"
            else base_file.replace(".json.gz", f"-{set_name}.json.gz"),
        )
        for set_name in query_sets
        for base_file in RERANK_CANDIDATE_FILES
    ]
    for set_name, candidate_file in rerank_jobs:
        path = REPO_DIR / "results" / candidate_file
        if not path.exists():
            print(f"\nskipping {candidate_file}: not committed by the local side", flush=True)
            manifest["artifacts"].append(
                {"file": candidate_file, "skipped": "candidate list not present in repo"}
            )
            continue

        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        # Skip a shortlist whose scores already exist, whether from this session or
        # from a previous run committed to the repository.
        #
        # Reranking is by far the most expensive step here: 117,200 pairs per file
        # at roughly 47 pairs a second is about forty minutes each. The embedding,
        # query-vector and boundary passes all skip completed work and this one did
        # not, so a fresh session re-scored every shortlist -- including the BM25
        # ones already committed and still valid, which is over an hour of GPU time
        # spent reproducing files byte for byte.
        #
        # The check is on the query *text* rather than the file's existence, for the
        # same reason everything else here is: a scores file is only valid for the
        # wording it was computed against, and ids survive a rewrite.
        stem = candidate_file.removesuffix(".json.gz")
        out = WORK / f"rerank-scores-{stem}.json"
        wanted = {qid: entry["query"] for qid, entry in payload.items()}
        existing = None
        for candidate in (out, REPO_DIR / "results" / f"rerank-scores-{stem}.json.gz"):
            if not candidate.exists():
                continue
            opener = gzip.open if candidate.suffix == ".gz" else open
            with opener(candidate, "rt", encoding="utf-8") as handle:
                blob = json.load(handle)
            texts = blob.get("query_texts") if isinstance(blob, dict) else None
            if texts and all(texts.get(q) == t for q, t in wanted.items()):
                existing = candidate
                break
        if existing is not None:
            print(f"\nskipping {candidate_file}: {existing.name} already covers it", flush=True)
            manifest["artifacts"].append(
                {"file": out.name, "query_set": set_name, "reused": existing.name}
            )
            continue

        # Candidate texts are rebuilt here rather than shipped. Safe because the
        # corpus was compared against the SNAPSHOTTED committed digests above,
        # and chunking is a deterministic pure function of the document, so these
        # chunk ids and texts are identical to the local ones. The check below is
        # kept even so: it is independent of the manifest comparison, and back
        # when the manifest comparison was vacuous it was the only thing standing
        # between a drifted corpus and a confidently wrong set of scores.
        chunker_name = next(iter(payload.values()))["chunker"]
        rebuilt = {c.chunk_id: c.text for c in make_chunker(chunker_name).chunk_corpus(docs)}
        missing = {
            cid
            for entry in payload.values()
            for cid in entry["candidate_ids"]
            if cid not in rebuilt
        }
        if missing:
            # Fail loudly. Scoring a query against a passage it never retrieved
            # would produce confident numbers for an experiment that did not run.
            raise SystemExit(
                f"{len(missing):,} candidate ids are absent from the rebuilt corpus. "
                f"The rebuild diverged from the committed manifest; refusing to score."
            )
        for entry in payload.values():
            entry["candidate_texts"] = [rebuilt[c] for c in entry["candidate_ids"]]

        if reranker is None:
            reranker = CrossEncoderReranker(RERANKER, batch_size=BATCH_RERANK)

        print(f"\n=== reranking {candidate_file}: {len(payload)} queries ===", flush=True)
        scores: dict[str, dict[str, float]] = {}
        started = time.monotonic()
        for i, (query_id, entry) in enumerate(payload.items(), 1):
            values = reranker.score(entry["query"], entry["candidate_texts"])
            scores[query_id] = {
                cid: float(s) for cid, s in zip(entry["candidate_ids"], values, strict=True)
            }
            if i % 25 == 0:
                print(f"  {i}/{len(payload)}", flush=True)
        elapsed = time.monotonic() - started

        # `stem` and `out` were computed above, when checking whether this
        # shortlist had already been scored.
        # Wrapped with the text each score was computed against. A cross-encoder
        # score is a function of the query wording as much as the passage, and
        # query ids survive a rewrite of that wording, so ids alone let scores
        # from one eval set be served for another -- which is exactly what
        # happened, and produced four significant improvements that were not real.
        payload_out = {
            "scores": scores,
            "query_texts": {qid: payload[qid]["query"] for qid in scores},
        }
        out.write_text(json.dumps(payload_out), encoding="utf-8")
        pairs = sum(len(v) for v in scores.values())
        manifest["artifacts"].append(
            {
                "file": out.name,
                "reranker": RERANKER,
                "query_set": set_name,
                "n_queries": len(scores),
                "n_pairs": pairs,
                "seconds": round(elapsed, 1),
                "pairs_per_sec": round(pairs / elapsed, 1) if elapsed else None,
            }
        )
        print(f"  {elapsed / 60:.1f} min for {pairs:,} pairs -> {out.name}", flush=True)

    (WORK / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n=== done ===", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    print("\nDownload every vectors-*.npz, rerank-scores-*.json and MANIFEST.json")
    print("from the Output tab, put them in data/indexes/, then re-run:")
    print("  python -m retrieval_ablation.ablation.runner")


if __name__ == "__main__":
    main()
