"""Add the missing `query_texts` field to query-vector artifacts.

WHY THIS EXISTS

`load_query_vectors` refuses an artifact that cannot prove which text each vector
was built from, because query ids survive a rewrite of the query text and are
therefore not evidence of a match. The artifacts produced before that field
existed are not wrong -- they simply cannot say what they embedded.

Discarding them would mean re-running a GPU session to recover a measurement that
is already correct. Backfilling them is only defensible if the text they embedded
can be established rather than assumed, so this script establishes it and refuses
when it cannot:

  * The notebook embeds `read_eval_set(QUERIES_PATH)`, i.e. data/eval/queries.jsonl.
  * That file was last modified on 2026-08-03 and committed; the vectors were
    produced on 2026-08-09 against a clean working tree. So the text the GPU run
    read is the text in that file today.
  * The ids in the artifact must match the file exactly, in the same order. A
    reordered or partial artifact would pair vectors with the wrong text, which is
    the precise failure the field was added to prevent, so any mismatch aborts.

This is a one-off. Artifacts written by the current notebook carry the field.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from retrieval_ablation.config import RESULTS_DIR
from retrieval_ablation.evalset.build import QUERIES_PATH
from retrieval_ablation.evalset.schema import read_eval_set
from retrieval_ablation.index.artifacts import load_query_vectors


def backfill(path: Path, text_by_id: dict[str, str]) -> str:
    payload = dict(np.load(path, allow_pickle=True))
    if "query_texts" in payload:
        return "already carries query_texts"

    ids = [str(q) for q in payload["query_ids"]]
    missing = [q for q in ids if q not in text_by_id]
    if missing:
        raise SystemExit(
            f"{path.name}: {len(missing)} query ids are absent from {QUERIES_PATH.name} "
            f"(first few: {missing[:3]}). The artifact was not built from this eval "
            f"set, so its text cannot be established. Refusing to backfill."
        )
    if len(ids) != len(text_by_id):
        raise SystemExit(
            f"{path.name}: holds {len(ids)} vectors but {QUERIES_PATH.name} has "
            f"{len(text_by_id)} queries. Refusing to backfill a partial artifact."
        )

    shutil.copy2(path, path.with_suffix(".npz.bak"))
    payload["query_texts"] = np.array([text_by_id[q] for q in ids], dtype=object)
    np.savez_compressed(path, **payload)
    return f"backfilled {len(ids)} texts"


def backfill_rerank_scores(path: Path, candidates_path: Path) -> str:
    """Attach the query text each cross-encoder score was computed against.

    The authority here is the candidates file, not the eval set. The notebook
    scores `entry["query"]` taken straight from that file, so it records what was
    actually sent to the model rather than what we believe was sent. Any query id
    the candidates file does not cover is left out instead of guessed.
    """
    payload = json.loads(_read_text(path))
    if isinstance(payload, dict) and "scores" in payload and "query_texts" in payload:
        return "already carries query_texts"

    candidates = json.loads(_read_text(candidates_path))
    texts = {qid: entry["query"] for qid, entry in candidates.items() if "query" in entry}
    missing = [q for q in payload if q not in texts]
    if missing:
        raise SystemExit(
            f"{path.name}: {len(missing)} scored queries are absent from "
            f"{candidates_path.name}, so the text they were scored against cannot "
            f"be established (first few: {missing[:3]}). Refusing to backfill."
        )

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    wrapped = {"scores": payload, "query_texts": {q: texts[q] for q in payload}}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(wrapped, handle)
    return f"backfilled {len(payload)} query texts"


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def main() -> None:
    queries = read_eval_set(QUERIES_PATH)
    text_by_id = {q.query_id: q.text for q in queries}
    print(f"{QUERIES_PATH.name}: {len(text_by_id)} queries")

    for scores_path in sorted(RESULTS_DIR.glob("rerank-scores-*.json*")):
        if scores_path.name.endswith(".bak"):
            continue
        stem = scores_path.name.removeprefix("rerank-scores-").removesuffix(".json.gz")
        candidates_path = RESULTS_DIR / f"{stem}.json.gz"
        if not candidates_path.exists():
            print(f"  {scores_path.name}: no {candidates_path.name}; skipped")
            continue
        print(f"  {scores_path.name}: {backfill_rerank_scores(scores_path, candidates_path)}")

    paths = sorted(RESULTS_DIR.glob("queryvectors-*.npz"))
    if not paths:
        print("no query-vector artifacts found")
        return
    for path in paths:
        print(f"  {path.name}: {backfill(path, text_by_id)}")

    # Verify by reloading through the real loader rather than trusting the write.
    for path in paths:
        name = path.stem.replace("queryvectors-", "")
        vectors = load_query_vectors(name, text_by_id)
        if vectors is None or len(vectors) != len(text_by_id):
            print(f"  VERIFY FAILED for {name}: {vectors and len(vectors)}", file=sys.stderr)
            raise SystemExit(1)
        print(f"  verified {name}: loader returns {len(vectors)} vectors")


if __name__ == "__main__":
    main()
