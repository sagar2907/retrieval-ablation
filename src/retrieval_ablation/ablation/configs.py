"""The ablation grid: one axis varied at a time, from a single named baseline.

A full factorial over chunking x embedding x retrieval x reranking x candidate
size would be hundreds of runs and would answer a question nobody asked. The
useful question is "what is each component worth?", and that is a one-axis-at-a-
time design: hold everything at the baseline, change one thing, measure the
difference, attribute it.

The consequence, stated because it is a real limitation rather than an oversight:
this design cannot detect interactions. If reranking only helps when the first
stage is hybrid, a one-axis grid measures reranking against a lexical baseline and
under-reports it. Two crossed cells are therefore included deliberately -- hybrid
with and without reranking -- because that specific interaction is the study's
headline claim and cannot be assumed away.

Every configuration is a pure description. Nothing here builds an index or touches
a model, so the grid can be inspected, counted and tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Which axis a configuration varies relative to the baseline. Recorded so the
#: results table can group rows by the question each one answers.
AXES = (
    "baseline",
    "chunking",
    "embedding",
    "retrieval",
    "reranking",
    "candidates",
    "table_rendering",
    "interaction",
)


@dataclass(frozen=True, slots=True)
class Config:
    """One retrieval pipeline, fully specified."""

    name: str
    axis: str

    # -- chunking --
    chunker: str = "fixed512o64"

    # -- first stage --
    #: "bm25", "dense", or "hybrid".
    retrieval: str = "bm25"
    embedding: str | None = None
    rrf_k: int = 60

    # -- reranking --
    reranker: str | None = None
    candidate_k: int = 100

    # -- how tables become retrievable text --
    table_rendering: str = "markdown"

    #: Free-form notes carried into the results table.
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown axis {self.axis!r}; known: {AXES}")
        if self.retrieval not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"unknown retrieval {self.retrieval!r}")
        if self.retrieval in {"dense", "hybrid"} and not self.embedding:
            raise ValueError(f"{self.name}: {self.retrieval} retrieval needs an embedding model")
        if self.retrieval == "bm25" and self.embedding:
            raise ValueError(f"{self.name}: bm25 retrieval must not name an embedding model")

    @property
    def uses_gpu(self) -> bool:
        return self.retrieval in {"dense", "hybrid"} or self.reranker is not None

    @property
    def index_key(self) -> str:
        """Identifies the reusable artifacts this configuration needs.

        Configurations differing only in reranking or candidate size share the
        same chunking and the same vectors. Keying on that means the expensive
        work -- chunking the corpus and embedding it -- happens once per distinct
        (chunker, table rendering, embedding) triple rather than once per row.
        With 15 rows over 4 distinct triples that is most of the runtime.
        """
        return f"{self.chunker}|{self.table_rendering}|{self.embedding or 'none'}"


#: The reference configuration. Naive chunk-and-retrieve: fixed-size chunks,
#: lexical search, no reranking. Every other row is a difference against this.
BASELINE = Config(
    name="baseline-bm25-fixed512",
    axis="baseline",
    chunker="fixed512o64",
    retrieval="bm25",
    notes="naive chunk-and-retrieve control",
)


def build_grid() -> list[Config]:
    """The full ablation grid."""
    grid: list[Config] = [BASELINE]

    # -- Chunking: vary the chunker, hold lexical retrieval fixed. -------------
    # Lexical rather than dense as the held-constant retriever, so a chunking
    # difference cannot be confounded by an embedding model's sequence-length
    # truncation: BM25 sees every token of a chunk however long it is.
    for chunker in ("fixed256o32", "semantic95", "struct512"):
        grid.append(
            Config(
                name=f"chunk-{chunker}",
                axis="chunking",
                chunker=chunker,
                retrieval="bm25",
                notes=f"chunking axis: {chunker} vs baseline fixed512o64",
            )
        )

    # -- Table rendering: same chunker, different linearisation. ---------------
    grid.append(
        Config(
            name="tables-row-sentences",
            axis="table_rendering",
            chunker="struct512",
            retrieval="bm25",
            table_rendering="row_sentences",
            notes="repeat column headers beside every value instead of a pipe table",
        )
    )

    # -- Retrieval: lexical vs dense vs hybrid, on the best-structured chunks. -
    grid.append(
        Config(
            name="retrieval-dense-bge",
            axis="retrieval",
            chunker="struct512",
            retrieval="dense",
            embedding="bge-m3",
            notes="dense only",
        )
    )
    grid.append(
        Config(
            name="retrieval-hybrid-rrf",
            axis="retrieval",
            chunker="struct512",
            retrieval="hybrid",
            embedding="bge-m3",
            notes="BM25 + dense fused with RRF",
        )
    )
    grid.append(
        Config(
            name="retrieval-bm25-struct",
            axis="retrieval",
            chunker="struct512",
            retrieval="bm25",
            notes="lexical control at the same chunking as the dense arms",
        )
    )

    # -- Embedding model, dense-only so the model is the only difference. ------
    for model in ("e5-base", "finance-e5"):
        grid.append(
            Config(
                name=f"embed-{model}",
                axis="embedding",
                chunker="struct512",
                retrieval="dense",
                embedding=model,
                notes=f"embedding axis: {model} vs bge-m3",
            )
        )

    # -- Reranking on the lexical baseline. -----------------------------------
    grid.append(
        Config(
            name="rerank-bm25-100",
            axis="reranking",
            chunker="struct512",
            retrieval="bm25",
            reranker="bge-reranker-v2-m3",
            candidate_k=100,
            notes="cross-encoder on a lexical first stage",
        )
    )

    # -- Candidate-set size, the reranker's cost/benefit knob. -----------------
    for candidates in (25, 50, 200):
        grid.append(
            Config(
                name=f"rerank-candidates-{candidates}",
                axis="candidates",
                chunker="struct512",
                retrieval="bm25",
                reranker="bge-reranker-v2-m3",
                candidate_k=candidates,
                notes=f"shortlist depth {candidates}",
            )
        )

    # -- The one deliberate interaction cell. ---------------------------------
    # A one-axis grid would measure reranking only against a lexical first stage.
    # The headline claim is about hybrid *plus* reranking, so that cell is run
    # explicitly rather than inferred by adding two independent deltas.
    grid.append(
        Config(
            name="hybrid-plus-rerank",
            axis="interaction",
            chunker="struct512",
            retrieval="hybrid",
            embedding="bge-m3",
            reranker="bge-reranker-v2-m3",
            candidate_k=100,
            notes="the production configuration: hybrid retrieval then cross-encoder",
            tags=("headline",),
        )
    )

    _assert_single_axis(grid)
    return grid


def _assert_single_axis(grid: list[Config]) -> None:
    """Check that each non-interaction row differs from the baseline on one axis.

    A design error here is silent and fatal to interpretation: a row that changes
    two things at once still produces a number, and that number gets attributed to
    whichever axis the row is filed under. Checked in code so the grid cannot drift
    as configurations are added.
    """
    tracked = ("chunker", "retrieval", "embedding", "reranker", "candidate_k", "table_rendering")
    for config in grid:
        if config.axis in {"baseline", "interaction"}:
            continue
        # Compare against the baseline as it would be with this row's chunker,
        # since several axes are measured on structure-aware chunks rather than on
        # the baseline's fixed-size ones. That shared reference is itself a grid
        # row ("retrieval-bm25-struct"), so the comparison stays single-axis.
        reference = replace(BASELINE, chunker=config.chunker, name="ref")
        differing = [f for f in tracked if getattr(config, f) != getattr(reference, f)]
        # `embedding` is not independent: dense and hybrid retrieval require it, so
        # changing retrieval necessarily changes it too.
        if config.axis == "retrieval" and set(differing) <= {"retrieval", "embedding"}:
            continue
        if config.axis == "embedding" and set(differing) <= {"retrieval", "embedding"}:
            continue
        if config.axis == "candidates" and set(differing) <= {"reranker", "candidate_k"}:
            continue
        if len(differing) > 1:
            raise AssertionError(
                f"{config.name} (axis={config.axis}) differs from its reference on "
                f"{differing}; a multi-axis row cannot be attributed to one component"
            )


def group_by_index_key(grid: list[Config]) -> dict[str, list[Config]]:
    """Group configurations that can share chunking and embedding work."""
    groups: dict[str, list[Config]] = {}
    for config in grid:
        groups.setdefault(config.index_key, []).append(config)
    return groups
