"""Run the ablation grid and write the results table.

Two properties matter more than speed here.

**Nothing is invented.** A configuration that cannot run -- because its embedding
model is unavailable, or the GPU is not reachable -- is recorded with metrics of
`None` and a stated reason. It is never silently dropped and never given a
plausible-looking number. `Result.measured` is the flag, and the markdown table
prints "not measured" for those rows.

**Comparisons are fair.** Every configuration is scored twice: once on all
queries it can judge, and once on the subset every configuration can judge. The
second is the headline, because a configuration can otherwise raise its average
by failing to represent hard queries at all, which looks exactly like an
improvement. See `evalset.relevance` for why that happens.

Expensive artifacts are shared. Chunking the corpus and embedding it are keyed on
`Config.index_key`, so 15 rows over 4 distinct (chunker, table rendering,
embedding) triples pay for 4 chunkings and 4 embedding passes rather than 15.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking import (
    Chunker,
    FixedSizeChunker,
    SemanticChunker,
    StructureAwareChunker,
    approx_token_count,
)
from ..config import RAW_DIR, RESULTS_DIR, ensure_dirs
from ..corpus.ingest import load_corpus
from ..corpus.models import Document, GoldPassage
from ..embed.base import Embedder
from ..evalset.build import QUERIES_PATH
from ..evalset.paraphrase import PARAPHRASED_PATH
from ..evalset.relevance import (
    Chunk,
    build_qrels,
    common_judgeable_queries,
    reachability,
)
from ..evalset.schema import EvalQuery, Verification, gold_by_query, read_eval_set
from ..evalset.synthesize import _query_id, extract_table_facts
from ..index.artifacts import (
    PrecomputedReranker,
    dense_index_from_artifact,
    load_query_vectors,
    load_rerank_scores,
    load_vectors,
)
from ..index.bm25 import BM25Index
from ..index.dense import DenseIndex
from ..index.fusion import HybridRetriever
from ..index.rerank import CrossEncoderReranker, RerankingRetriever, recall_ceiling
from ..metrics.retrieval import per_query_scores, score_run
from ..metrics.stats import (
    bootstrap_ci,
    holm_bonferroni,
    paired_randomization_test,
)
from .configs import BASELINE, Config, build_grid, group_by_index_key

log = logging.getLogger(__name__)

RESULTS_PATH = RESULTS_DIR / "ablation.json"
TABLE_PATH = RESULTS_DIR / "ablation.md"


@dataclass
class Result:
    """One configuration's outcome, or the reason there isn't one."""

    config: Config
    #: None when the configuration could not be run at all.
    skipped_reason: str | None = None

    n_chunks: int = 0
    #: Metrics over every query this configuration can judge.
    metrics_all: dict[str, float | None] = field(default_factory=dict)
    #: Metrics over the subset every configuration can judge. The headline.
    metrics_common: dict[str, float | None] = field(default_factory=dict)
    n_scored_all: int = 0
    n_scored_common: int = 0
    #: Fraction of gold passages any chunk of this configuration can cover.
    fraction_reachable: float | None = None
    #: Upper bound on what reranking at this candidate depth could achieve.
    first_stage_recall_ceiling: float | None = None
    ndcg_ci: tuple[float, float] | None = None
    #: Metrics split by query lexical overlap, to expose the template confound.
    ndcg_low_overlap: float | None = None
    ndcg_high_overlap: float | None = None
    seconds: float = 0.0
    #: Per-query nDCG on the common subset, kept for the paired tests.
    per_query_ndcg: dict[str, float] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.skipped_reason is None and self.metrics_common.get("ndcg@10") is not None

    def to_json(self) -> dict:
        return {
            "name": self.config.name,
            "axis": self.config.axis,
            "chunker": self.config.chunker,
            "retrieval": self.config.retrieval,
            "embedding": self.config.embedding,
            "reranker": self.config.reranker,
            "candidate_k": self.config.candidate_k,
            "table_rendering": self.config.table_rendering,
            "notes": self.config.notes,
            "measured": self.measured,
            "skipped_reason": self.skipped_reason,
            "n_chunks": self.n_chunks,
            "metrics_all": self.metrics_all,
            "metrics_common_subset": self.metrics_common,
            "n_scored_all": self.n_scored_all,
            "n_scored_common": self.n_scored_common,
            "fraction_reachable": self.fraction_reachable,
            "first_stage_recall_ceiling": self.first_stage_recall_ceiling,
            "ndcg10_ci95": list(self.ndcg_ci) if self.ndcg_ci else None,
            "ndcg10_low_overlap_queries": self.ndcg_low_overlap,
            "ndcg10_high_overlap_queries": self.ndcg_high_overlap,
            "seconds": round(self.seconds, 1),
        }


def make_chunker(name: str, embedder: Embedder | None = None) -> Chunker:
    """Construct a chunker from its grid name.

    Token counting uses the approximate counter throughout. That is a deliberate
    simplification and a limitation worth naming: a real tokenizer would place
    boundaries slightly differently. What matters for the ablation is that every
    configuration uses the *same* counter, so a chunking difference is a
    difference in strategy rather than in tokenisation.
    """
    if name.startswith("fixed"):
        # e.g. "fixed512o64"
        body = name.removeprefix("fixed")
        target_str, _, overlap_str = body.partition("o")
        return FixedSizeChunker(
            int(target_str), int(overlap_str or 0), approx_token_count, name=name
        )
    if name.startswith("struct"):
        target = int(name.removeprefix("struct"))
        return StructureAwareChunker(target, target * 4, approx_token_count, name=name)
    if name.startswith("semantic"):
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        percentile = float(name.removeprefix("semantic"))
        return SemanticChunker(
            embedder.encode_passages,
            percentile,
            max_tokens=1024,
            count_tokens=approx_token_count,
            name=name,
        )
    raise ValueError(f"unknown chunker {name!r}")


def _artifact_vectors(embedding_key: str, chunker: str) -> Path | None:
    """Path to GPU-produced vectors for this (embedding, chunker) pair, if present."""
    candidate = RESULTS_DIR / f"vectors-{embedding_key}-{chunker}.npz"
    return candidate if candidate.exists() else None


def _artifact_rerank_scores() -> Path | None:
    """Path to GPU-produced cross-encoder scores, if present."""
    for name in (
        "rerank-scores-candidates-rerank-bm25-100.json",
        "rerank-scores-candidates-rerank-bm25-100.json.gz",
    ):
        path = RESULTS_DIR / name
        if path.exists():
            return path
    return None


def _load_embedder(key: str):
    """Load a real embedding model, or return the reason it is unavailable.

    Returns `(embedder, None)` or `(None, reason)`. Failure is a recorded reason
    rather than an exception so one unavailable model does not abort a 15-row run,
    and so the results table can say exactly why a row is missing.
    """
    try:
        # Imported here, not at module scope: sentence-transformers and torch are
        # optional dependencies, and the ablation must still run its lexical
        # configurations on a machine where neither is installed.
        from ..embed.local import SentenceTransformerEmbedder  # noqa: PLC0415
    except ImportError as exc:
        return None, f"sentence-transformers unavailable: {exc}"
    try:
        embedder = SentenceTransformerEmbedder(key)
        # Force the load now: a lazy failure would otherwise surface halfway
        # through a long run rather than before any work is done.
        embedder.encode_passages(["warmup"])
        return embedder, None
    except Exception as exc:
        return None, f"{key} unavailable: {type(exc).__name__}: {exc}"


@dataclass
class _Prepared:
    """Artifacts shared by every configuration with the same index key."""

    chunks: list[Chunk]
    qrels: dict[str, dict[str, int]]
    bm25: BM25Index
    dense: DenseIndex | None
    skipped_reason: str | None = None


def _corpus_for_rendering(
    docs: Sequence[Document],
    queries: Sequence[EvalQuery],
    rendering: str,
) -> tuple[list[Document], dict[str, list]]:
    """Return documents and gold labels for a given table rendering.

    THE COLLISION THIS FUNCTION RESOLVES

    Table rendering is an ablation axis, but it is applied during *parsing*, and
    parsing produces the canonical text that every gold span indexes into.
    Changing the rendering therefore changes every character offset in the
    document, so the committed eval set -- built against the default rendering --
    does not apply.

    The first version of this runner missed that entirely and simply passed the
    rendering through the index key. The axis was inert: the row-sentence
    configuration scored identically to its markdown twin, to four decimal places,
    with the same chunk count, because it was silently reusing the
    markdown-rendered corpus. A plausible number for an experiment that never ran.

    The fix re-parses from the cached raw bytes and re-derives gold spans for the
    new text. That is sound because query ids are content-addressed on (document,
    row label, period) rather than on offsets, so the same fact keeps its id under
    both renderings and the two configurations are compared on a shared query set
    with each side carrying its own correct spans.
    """
    if rendering == "markdown":
        return list(docs), gold_by_query(list(queries))

    from ..corpus.html_parse import parse_filing  # noqa: PLC0415 - avoids an import cycle

    wanted = {q.query_id for q in queries}
    reparsed: list[Document] = []
    gold: dict[str, list] = {}

    for doc in docs:
        raw_path = RAW_DIR / f"{doc.doc_id}.htm"
        if not raw_path.exists():
            # Without the raw bytes this document cannot be re-rendered. Skipping
            # it would silently shrink the corpus for one configuration only, so
            # the original is kept and the discrepancy is visible in chunk counts.
            reparsed.append(doc)
            continue
        fresh = parse_filing(
            doc.doc_id,
            raw_path.read_bytes(),
            metadata=dict(doc.metadata),
            render_tables=rendering,
        )
        reparsed.append(fresh)
        for fact in extract_table_facts(fresh):
            query_id = _query_id(fact)
            if query_id in wanted:
                gold.setdefault(query_id, []).append(
                    GoldPassage(
                        passage_id=f"{fact.doc_id}:{fact.span.start}-{fact.span.end}",
                        doc_id=fact.doc_id,
                        span=fact.span,
                        gain=2,
                    )
                )

    return reparsed, gold


def _prepare(
    key: str,
    configs: Sequence[Config],
    docs: Sequence[Document],
    queries: Sequence[EvalQuery],
) -> _Prepared | None:
    """Chunk, index and build qrels once for a group of configurations."""
    example = configs[0]
    embedding_key = example.embedding
    embedder = None
    skip: str | None = None
    artifact = _artifact_vectors(embedding_key, example.chunker) if embedding_key else None

    # Precomputed vectors are preferred over loading a model. They are what the
    # GPU worker produced, torch cannot load on this machine at all, and using
    # them keeps the dense arm reproducible from committed artifacts rather than
    # from whatever model version happens to be installed.
    if embedding_key and artifact is None:
        embedder, skip = _load_embedder(embedding_key)
        if skip:
            log.warning("group %s: %s", key, skip)
    elif example.chunker.startswith("semantic"):
        embedder, skip = _load_embedder("bge-m3")
        if skip:
            log.warning("group %s: %s", key, skip)
            # Semantic chunking needs live embeddings for its breakpoints; there
            # is no precomputed substitute, because the sentences it embeds do
            # not exist until it has already run.
            return _Prepared([], {}, BM25Index([]), None, skipped_reason=skip)

    # Re-parse when this group uses a non-default table rendering, because the
    # rendering changes the canonical text and therefore every gold offset.
    group_docs, gold = _corpus_for_rendering(docs, queries, example.table_rendering)
    if example.table_rendering != "markdown":
        log.info(
            "group %s: re-parsed %d documents with %s rendering, %d gold facts relocated",
            key,
            len(group_docs),
            example.table_rendering,
            len(gold),
        )

    chunker = make_chunker(example.chunker, embedder)
    log.info("group %s: chunking %d documents with %s", key, len(group_docs), chunker.name)
    chunks = chunker.chunk_corpus(group_docs)
    log.info("group %s: %d chunks", key, len(chunks))

    qrels = build_qrels(gold, chunks)

    bm25 = BM25Index(chunks)
    dense = None
    if embedding_key and artifact is not None:
        loaded = load_vectors(artifact, chunks)
        log.info(
            "group %s: loaded %s (%d-dim), %d/%d chunks covered",
            key,
            artifact.name,
            loaded.dimension,
            len(loaded.chunk_ids),
            len(chunks),
        )
        if loaded.n_missing_locally or loaded.n_unmatched_remotely:
            log.warning(
                "group %s: %d chunks without vectors, %d vectors without chunks",
                key,
                loaded.n_missing_locally,
                loaded.n_unmatched_remotely,
            )
        query_vectors = load_query_vectors(embedding_key, {q.query_id: q.text for q in queries})
        if query_vectors is None:
            log.warning(
                "group %s: no queryvectors-%s.npz. Passage vectors alone cannot "
                "serve a dense arm -- both sides must come from the same model.",
                key,
                embedding_key,
            )
        dense = dense_index_from_artifact(artifact, chunks, query_vectors)
    elif embedding_key and embedder is not None:
        log.info("group %s: embedding %d chunks with %s", key, len(chunks), embedding_key)
        dense = DenseIndex.build(chunks, embedder)

    return _Prepared(chunks, qrels, bm25, dense, skipped_reason=skip if embedding_key else None)


def _build_retriever(  # noqa: PLR0911 - each return is a distinct, named unavailability
    config: Config, prepared: _Prepared, query_ids: dict[str, str] | None = None
):
    """Assemble the retriever this configuration describes, or return a reason."""
    if config.retrieval == "bm25":
        first = prepared.bm25
    elif config.retrieval == "dense":
        if prepared.dense is None:
            return None, prepared.skipped_reason or "dense index unavailable"
        if not getattr(prepared.dense.embedder, "can_embed_queries", True):
            return None, _NO_QUERY_VECTORS.format(model=config.embedding)
        first = prepared.dense
    else:
        if prepared.dense is None:
            return None, prepared.skipped_reason or "dense index unavailable for hybrid"
        if not getattr(prepared.dense.embedder, "can_embed_queries", True):
            return None, _NO_QUERY_VECTORS.format(model=config.embedding)
        first = HybridRetriever([prepared.bm25, prepared.dense], k=config.rrf_k)

    if config.reranker is None:
        return first, None

    # Precomputed cross-encoder scores are preferred, and on this machine they are
    # the only option: torch cannot load under the enforced code-integrity policy.
    # Using them also keeps the reranking arms reproducible from a committed
    # artifact rather than from whatever model version happens to download.
    scores_path = _artifact_rerank_scores()
    if scores_path is not None and query_ids:
        # query_ids maps text -> id; the loader needs the inverse to check that
        # each score was computed against the wording being asked now.
        by_id = {query_id: text for text, query_id in query_ids.items()}
        scores = load_rerank_scores(scores_path, by_id)
        if not scores:
            return None, _STALE_RERANK_SCORES.format(file=scores_path.name)
        return (
            PrecomputedReranker(
                first,
                scores,
                query_ids,
                candidate_k=config.candidate_k,
                name=config.name,
            ),
            None,
        )

    texts = {c.chunk_id: c.text for c in prepared.chunks}
    try:
        reranker = CrossEncoderReranker(f"BAAI/{config.reranker}")
        reranker.score("warmup", ["warmup passage"])
    except Exception as exc:
        return None, f"reranker unavailable: {type(exc).__name__}: {exc}"

    return RerankingRetriever(first, reranker, texts, candidate_k=config.candidate_k), None


#: Reported verbatim in the results table when a dense arm cannot run. Stated as
#: a missing artifact rather than a code failure, because that is what it is.
#:
#: Deliberately does not claim the file is absent. It may well be present and
#: still unusable: the loader refuses vectors whose recorded text differs from the
#: query being scored, which is exactly what happens on the paraphrased eval set.
#: An earlier version of this message said the file "is not" there, which would
#: send a reader looking for something that is sitting in results/.
#: A cross-encoder score depends on the query wording as much as the passage, so
#: scores computed against other text are not a weaker measurement of this one --
#: they are a measurement of a different experiment.
_STALE_RERANK_SCORES = (
    "{file} holds no cross-encoder scores for the wording being scored: it either "
    "predates query-text provenance or was computed against a different eval set. "
    "Reusing it would rerank these queries with scores derived from other "
    "questions. Re-run the GPU notebook against this eval set."
)

_NO_QUERY_VECTORS = (
    "no usable query vectors for {model}: queryvectors-{model}.npz is absent, or "
    "records different query text than the set being scored (check the loader "
    "warning above). A dense index needs both sides embedded by the same model on "
    "the same wording; reusing vectors from other text compares two things that "
    "were never asked and returns confident nonsense. Re-run the GPU notebook "
    "against this eval set to produce them."
)

#: Queries whose content words overlap the gold passage below this fraction are'
#: the ones that genuinely test retrieval rather than string matching.
LOW_OVERLAP_THRESHOLD = 0.4


def run_ablation(  # noqa: PLR0915 - a linear pipeline; splitting it would hide the order
    docs: Sequence[Document],
    queries: Sequence[EvalQuery],
    grid: Sequence[Config] | None = None,
    top_k: int = 50,
) -> list[Result]:
    """Execute the grid and return one Result per configuration."""
    grid = list(grid or build_grid())
    groups = group_by_index_key(grid)
    gold = gold_by_query(list(queries))

    # First pass: prepare each group and collect qrels, so the shared judgeable
    # subset can be computed before any configuration is scored. Scoring first
    # and intersecting later would mean re-running everything.
    prepared: dict[str, _Prepared] = {}
    for key, configs in groups.items():
        prepared[key] = _prepare(key, configs, docs, queries)

    qrels_by_config = {
        config.name: prepared[config.index_key].qrels
        for config in grid
        if prepared[config.index_key].qrels
    }
    common = common_judgeable_queries(qrels_by_config) if qrels_by_config else set()
    log.info("queries judgeable by every configuration: %d of %d", len(common), len(queries))

    overlap = {q.query_id: (q.lexical_overlap or 0.0) for q in queries}
    query_text = {q.query_id: q.text for q in queries}
    # Reverse map: the Retriever interface takes query text, but precomputed
    # rerank scores are keyed by query id.
    id_by_text = {q.text: q.query_id for q in queries}

    results: list[Result] = []
    for config in grid:
        group = prepared[config.index_key]
        result = Result(config=config, n_chunks=len(group.chunks))

        if group.skipped_reason and not group.chunks:
            result.skipped_reason = group.skipped_reason
            results.append(result)
            log.warning("%s: SKIPPED (%s)", config.name, result.skipped_reason)
            continue

        retriever, reason = _build_retriever(config, group, id_by_text)
        if retriever is None:
            result.skipped_reason = reason
            results.append(result)
            log.warning("%s: SKIPPED (%s)", config.name, reason)
            continue

        started = time.monotonic()
        run = retriever.run(query_text, top_k=top_k)
        result.seconds = time.monotonic() - started

        summaries = score_run(run, group.qrels)
        result.metrics_all = {k: v.value for k, v in summaries.items()}
        result.n_scored_all = summaries["ndcg@10"].n_scored

        common_run = {qid: r for qid, r in run.items() if qid in common}
        common_summaries = score_run(common_run, group.qrels)
        result.metrics_common = {k: v.value for k, v in common_summaries.items()}
        result.n_scored_common = common_summaries["ndcg@10"].n_scored

        per_query = per_query_scores(common_run, group.qrels, "ndcg", 10)
        result.per_query_ndcg = per_query
        ci = bootstrap_ci(list(per_query.values()))
        result.ndcg_ci = (ci.low, ci.high) if ci else None

        low = [v for q, v in per_query.items() if overlap.get(q, 0.0) < LOW_OVERLAP_THRESHOLD]
        high = [v for q, v in per_query.items() if overlap.get(q, 0.0) >= LOW_OVERLAP_THRESHOLD]
        result.ndcg_low_overlap = sum(low) / len(low) if low else None
        result.ndcg_high_overlap = sum(high) / len(high) if high else None

        report = reachability(config.name, gold, group.chunks)
        result.fraction_reachable = report.fraction_reachable

        if config.reranker is not None:
            first_only, _ = _build_retriever(
                Config(
                    name=f"{config.name}-firststage",
                    axis=config.axis,
                    chunker=config.chunker,
                    retrieval=config.retrieval,
                    embedding=config.embedding,
                    table_rendering=config.table_rendering,
                ),
                group,
                id_by_text,
            )
            if first_only is not None:
                first_run = first_only.run(query_text, top_k=max(config.candidate_k, top_k))
                result.first_stage_recall_ceiling = recall_ceiling(
                    first_run, group.qrels, config.candidate_k
                )

        results.append(result)
        log.info(
            "%s: nDCG@10=%s recall@50=%s mrr=%s (%.0fs)",
            config.name,
            _fmt(result.metrics_common.get("ndcg@10")),
            _fmt(result.metrics_common.get("recall@50")),
            _fmt(result.metrics_common.get("mrr")),
            result.seconds,
        )

    return results


def _fmt(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.4f}"


def significance(results: Sequence[Result]) -> dict[str, dict]:
    """Paired tests against the baseline, corrected across the whole family."""
    baseline = next((r for r in results if r.config.name == BASELINE.name), None)
    if baseline is None or not baseline.per_query_ndcg:
        return {}

    tests = {}
    for result in results:
        if result.config.name == BASELINE.name or not result.per_query_ndcg:
            continue
        test = paired_randomization_test(baseline.per_query_ndcg, result.per_query_ndcg)
        if test is not None:
            tests[result.config.name] = test

    corrected = holm_bonferroni(tests)
    return {
        name: {
            "baseline_ndcg10": test.baseline_mean,
            "config_ndcg10": test.system_mean,
            "delta": test.delta,
            "p_raw": test.p_value,
            "p_holm": test.p_adjusted,
            "significant_at_05": test.significant(0.05),
            "n_pairs": test.n_pairs,
        }
        for name, test in corrected.items()
    }


def write_results(results: Sequence[Result], tests: dict[str, dict], suffix: str = "") -> None:
    ensure_dirs()
    json_path = RESULTS_PATH.with_name(f"ablation{suffix}.json")
    table_path = TABLE_PATH.with_name(f"ablation{suffix}.md")
    json_path.write_text(
        json.dumps(
            {
                "n_configurations": len(results),
                "n_measured": sum(1 for r in results if r.measured),
                "results": [r.to_json() for r in results],
                "significance_vs_baseline": tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    table_path.write_text(render_table(results, tests), encoding="utf-8")


def render_table(results: Sequence[Result], tests: dict[str, dict]) -> str:
    """Render the ablation as markdown.

    Confidence intervals and corrected p-values are in the table itself, not in a
    footnote, because a bare column of point estimates invites exactly the
    over-reading this project exists to avoid.
    """
    measured = [r for r in results if r.measured]
    skipped = [r for r in results if not r.measured]

    lines = [
        "# Retrieval ablation",
        "",
        f"{len(measured)} of {len(results)} configurations measured.",
        "",
        "`nDCG@10` and the confidence interval are computed on the **shared subset**",
        "of queries every configuration can judge. Comparing configurations on their",
        "own individually-judgeable subsets would let a configuration improve its",
        "average by failing to represent hard queries at all.",
        "",
        "`p (Holm)` is corrected across the whole family of comparisons against the",
        "baseline. Uncorrected, comparing 14 configurations at alpha=0.05 carries a",
        "~51% chance of at least one false positive.",
        "",
        "| configuration | axis | nDCG@10 | 95% CI | Recall@50 | MRR |"
        " delta vs base | p (Holm) | sig | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for result in sorted(measured, key=lambda r: -(r.metrics_common.get("ndcg@10") or 0.0)):
        test = tests.get(result.config.name)
        ci = f"[{result.ndcg_ci[0]:.3f}, {result.ndcg_ci[1]:.3f}]" if result.ndcg_ci else "-"
        lines.append(
            (
                "| `{name}` | {axis} | {ndcg} | {ci} | {recall} | {mrr} "
                "| {delta} | {p} | {sig} | {n} |"
            ).format(
                name=result.config.name,
                axis=result.config.axis,
                ndcg=_fmt(result.metrics_common.get("ndcg@10")),
                ci=ci,
                recall=_fmt(result.metrics_common.get("recall@50")),
                mrr=_fmt(result.metrics_common.get("mrr")),
                delta=f"{test['delta']:+.4f}" if test else "(baseline)",
                p=f"{test['p_holm']:.4f}" if test and test["p_holm"] is not None else "-",
                sig="yes" if test and test["significant_at_05"] else ("no" if test else "-"),
                n=result.n_scored_common,
            )
        )

    lines += [
        "",
        "## Chunking reachability and reranking ceiling",
        "",
        "`reachable` is the fraction of gold passages that any chunk of this",
        "configuration covers. A configuration cannot score a query whose answer it",
        "cannot represent, so a low value here means the headline number above rests",
        "on fewer queries -- which is exactly why the shared subset is used.",
        "",
        "`ceiling` is the first stage's recall at the reranking candidate depth: the",
        "hard upper bound on what the cross-encoder could achieve. It separates a weak",
        "reranker from one that never saw the answer.",
        "",
        "| configuration | chunks | reachable | ceiling |"
        " nDCG low-overlap | nDCG high-overlap | seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in measured:
        lines.append(
            "| `{name}` | {chunks:,} | {reach} | {ceil} | {low} | {high} | {secs:.0f} |".format(
                name=result.config.name,
                chunks=result.n_chunks,
                reach=f"{result.fraction_reachable:.1%}"
                if result.fraction_reachable is not None
                else "-",
                ceil=f"{result.first_stage_recall_ceiling:.1%}"
                if result.first_stage_recall_ceiling is not None
                else "-",
                low=_fmt(result.ndcg_low_overlap),
                high=_fmt(result.ndcg_high_overlap),
                secs=result.seconds,
            )
        )

    lines += [
        "",
        "## The lexical-overlap confound",
        "",
        "Queries are generated from table row labels, so they reuse the document's",
        "own wording and hand a lexical matcher an exact string. The two nDCG columns",
        "above split the query set at 0.4 content-word overlap. A configuration whose",
        "advantage exists only in the high-overlap column is winning at string",
        "matching, not at retrieval.",
        "",
    ]

    if skipped:
        lines += [
            "## Not measured",
            "",
            "These configurations did not run. No number is reported for them.",
            "",
            "| configuration | reason |",
            "|---|---|",
        ]
        for result in skipped:
            lines.append(f"| `{result.config.name}` | {result.skipped_reason} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval ablation")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated axis names to run (default: all)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="limit the corpus, for a fast smoke run",
    )
    parser.add_argument(
        "--exclude-rejected",
        action="store_true",
        help="drop labels a checker rejected, and write to results/ablation-accepted.*",
    )
    parser.add_argument(
        "--paraphrased",
        action="store_true",
        help=(
            "score the paraphrased queries, and write to results/ablation-paraphrased.*. "
            "Same query ids and same gold spans, so the two runs are directly comparable"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    docs = load_corpus()
    if args.max_docs:
        docs = docs[: args.max_docs]

    # The paraphrased set rewrites question text only; gold spans, ids and grades
    # are carried over untouched, which is what makes a difference between the two
    # runs attributable to the wording rather than to a different benchmark.
    queries_path = PARAPHRASED_PATH if args.paraphrased else QUERIES_PATH
    if args.paraphrased and not queries_path.exists():
        raise SystemExit(
            f"{queries_path} does not exist. Run "
            f"`python -m retrieval_ablation.evalset.paraphrase` first."
        )
    queries = read_eval_set(queries_path)
    if args.exclude_rejected:
        # Robustness check, not the headline. Roughly one label in five was
        # rejected by the checker, so the question is whether the ordering of the
        # ablation survives dropping them. If the conclusions change, the label
        # set is doing more work than the retrievers are.
        before = len(queries)
        queries = [q for q in queries if q.verification is not Verification.REJECTED]
        log.info("excluding rejected labels: %d of %d retained", len(queries), before)

    grid = build_grid()
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        grid = [c for c in grid if c.axis in wanted or c.name == BASELINE.name]

    log.info("running %d configurations over %d documents", len(grid), len(docs))
    results = run_ablation(docs, queries, grid, top_k=args.top_k)
    tests = significance(results)
    suffix = ""
    if args.paraphrased:
        suffix += "-paraphrased"
    if args.exclude_rejected:
        suffix += "-accepted"
    write_results(results, tests, suffix)

    print(f"\nwrote {RESULTS_PATH}")
    print(f"wrote {TABLE_PATH}")
    print(f"\nmeasured {sum(1 for r in results if r.measured)} of {len(results)} configurations")
    for result in results:
        if result.measured:
            print(f"  {result.config.name:32s} nDCG@10={_fmt(result.metrics_common['ndcg@10'])}")
        else:
            print(f"  {result.config.name:32s} NOT MEASURED: {result.skipped_reason}")


if __name__ == "__main__":
    main()
