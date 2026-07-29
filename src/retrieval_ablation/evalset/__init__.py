"""Construction and use of the labelled retrieval benchmark."""

from .relevance import (
    DEFAULT_MIN_COVERAGE,
    Chunk,
    ReachabilityReport,
    build_qrels,
    common_judgeable_queries,
    judgeable_queries,
    reachability,
    relevant_chunk_ids,
)

__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "Chunk",
    "ReachabilityReport",
    "build_qrels",
    "common_judgeable_queries",
    "judgeable_queries",
    "reachability",
    "relevant_chunk_ids",
]
