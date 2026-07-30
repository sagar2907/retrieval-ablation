"""The published eval-set format, and what each field is allowed to claim.

The labelled benchmark is the artifact this project exists to produce, so its
schema is explicit about provenance. Every query records how it was produced and
whether a human ever looked at it, because a benchmark that cannot distinguish a
verified label from a generated one is not a benchmark.

`verification` is the field that matters. It is not decoration: metrics computed
over `GENERATED` labels and metrics computed over `HUMAN_VERIFIED` labels are
different claims, and the ablation reports them separately rather than pooling
them into one number that overstates the weaker half.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..corpus.models import GoldPassage, Span


class Verification(enum.StrEnum):
    """How much confidence a label has earned."""

    #: Derived programmatically from document structure. Reproducible and
    #: mechanically correct by construction, but nobody has confirmed the query
    #: is answerable or that the gold span is the passage a reader would cite.
    GENERATED = "generated"

    #: A human read the query and the gold passage and confirmed the pairing.
    HUMAN_VERIFIED = "human_verified"

    #: A human read it and rejected it. Retained rather than deleted so the
    #: rejection rate is measurable, which is the only evidence about how
    #: trustworthy the GENERATED labels are in bulk.
    REJECTED = "rejected"


class QueryKind(enum.StrEnum):
    """What retrieval capability a query is designed to exercise."""

    #: A figure that lives in a table cell, identified by row and column.
    TABLE_LOOKUP = "table_lookup"

    #: A fact stated in prose within a named section.
    PROSE_FACT = "prose_fact"

    #: Phrased without reusing the document's own wording. These exist to
    #: separate genuine semantic retrieval from lexical string matching; see
    #: `lexical_overlap`.
    PARAPHRASED = "paraphrased"


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One labelled query.

    `lexical_overlap` records the fraction of the query's content words that also
    appear in the gold passage. It exists because a template-generated query
    inevitably reuses the document's own row labels, which hands BM25 an exact
    string match and would make lexical retrieval look far stronger than it is on
    real questions. Recording the overlap lets the ablation report low-overlap and
    high-overlap subsets separately, turning a confound into a measured variable.
    """

    query_id: str
    text: str
    gold: tuple[GoldPassage, ...]
    kind: QueryKind
    verification: Verification = Verification.GENERATED
    lexical_overlap: float | None = None
    #: Ticker, fiscal year, sector, section path -- used for stratified sampling
    #: and for slicing results by document type.
    metadata: dict[str, str] = field(default_factory=dict)
    #: Set only when a paraphrase was produced, naming the model that wrote it, so
    #: a reader can tell generated text from corpus text.
    paraphrase_source: str | None = None

    def to_json(self) -> dict:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "kind": self.kind.value,
            "verification": self.verification.value,
            "lexical_overlap": self.lexical_overlap,
            "paraphrase_source": self.paraphrase_source,
            "metadata": self.metadata,
            "gold": [
                {
                    "passage_id": g.passage_id,
                    "doc_id": g.doc_id,
                    "start": g.span.start,
                    "end": g.span.end,
                    "gain": g.gain,
                }
                for g in self.gold
            ],
        }

    @classmethod
    def from_json(cls, payload: dict) -> EvalQuery:
        return cls(
            query_id=payload["query_id"],
            text=payload["text"],
            gold=tuple(
                GoldPassage(
                    passage_id=g["passage_id"],
                    doc_id=g["doc_id"],
                    span=Span(g["start"], g["end"]),
                    gain=g["gain"],
                )
                for g in payload["gold"]
            ),
            kind=QueryKind(payload["kind"]),
            verification=Verification(payload["verification"]),
            lexical_overlap=payload["lexical_overlap"],
            metadata=payload.get("metadata", {}),
            paraphrase_source=payload.get("paraphrase_source"),
        )


def write_eval_set(queries: list[EvalQuery], path: Path) -> None:
    """Write one query per line as JSON.

    JSON Lines rather than a single array: the file is appended to across several
    sessions of verification work, and a line-oriented format means an interrupted
    write costs one query rather than the whole set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_eval_set(path: Path) -> list[EvalQuery]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; build the eval set first")
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(EvalQuery.from_json(json.loads(line)))
    return out


def gold_by_query(queries: list[EvalQuery]) -> dict[str, list[GoldPassage]]:
    """Shape the eval set for `relevance.build_qrels`."""
    return {q.query_id: list(q.gold) for q in queries}


def summarise(queries: list[EvalQuery]) -> dict:
    """Counts by kind and verification state, for reporting beside metrics."""
    by_kind: dict[str, int] = {}
    by_verification: dict[str, int] = {}
    for query in queries:
        kind = query.kind.value
        by_kind[kind] = by_kind.get(kind, 0) + 1
        state = query.verification.value
        by_verification[state] = by_verification.get(state, 0) + 1

    overlaps = [q.lexical_overlap for q in queries if q.lexical_overlap is not None]
    return {
        "n_queries": len(queries),
        "by_kind": by_kind,
        "by_verification": by_verification,
        "mean_lexical_overlap": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "n_documents_referenced": len({g.doc_id for q in queries for g in q.gold}),
    }
