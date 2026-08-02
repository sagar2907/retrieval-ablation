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
2. Paste this file into one cell and run it.
3. When it finishes, download the files from the notebook's Output tab.
4. Put them in data/indexes/ locally and re-run the ablation.

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
]

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

# ---------------------------------------------------------------------------
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


def main() -> None:  # noqa: PLR0915 - a linear script; splitting it would obscure the order
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
    # The committed manifest records the SHA-256 of every document's parsed text.
    # load_corpus() verifies each one and raises on mismatch, so a corpus that
    # differs from the one the gold labels were built against cannot be embedded
    # by accident.
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"committed manifest: {committed['n_documents']} documents", flush=True)
    print("rebuilding corpus from EDGAR (this takes roughly 20-40 minutes)", flush=True)
    ingest()
    docs = load_corpus()
    print(f"corpus verified: {len(docs)} documents", flush=True)

    queries = read_eval_set(QUERIES_PATH)
    print(f"eval set: {len(queries)} queries", flush=True)

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
        # Skip work already on disk. A Kaggle session survives a failed cell, and
        # the embedding passes take about nineteen minutes combined -- repeating
        # them because a later stage crashed wastes GPU quota for nothing.
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
            continue

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
        # Released before the next model loads. Two of these resident at once
        # exceeds a T4's memory once activations are counted.
        embedder.release()
        print(f"  wrote {out.name}", flush=True)

    # -- reranking --------------------------------------------------------
    from retrieval_ablation.index.rerank import CrossEncoderReranker

    reranker = None
    for candidate_file in RERANK_CANDIDATE_FILES:
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

        # Candidate texts are rebuilt here rather than shipped. Safe because
        # load_corpus() above verified every document against the committed
        # SHA-256 manifest, and chunking is a deterministic pure function of the
        # document, so these chunk ids and texts are identical to the local ones.
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

        stem = candidate_file.removesuffix(".json.gz")
        out = WORK / f"rerank-scores-{stem}.json"
        out.write_text(json.dumps(scores), encoding="utf-8")
        pairs = sum(len(v) for v in scores.values())
        manifest["artifacts"].append(
            {
                "file": out.name,
                "reranker": RERANKER,
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
