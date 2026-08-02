"""FastAPI service exposing retrieval, answers, and the scores behind both.

The service exists to make the pipeline *inspectable*, not to be impressive. Two
design commitments follow from that.

**Scores are first-class in the response, not hidden behind the answer.** Every
returned passage carries its rank, its raw retriever score, the components that
contributed to it, and its section path in the filing. A demo that returns only
prose gives a reader no way to tell a lucky answer from a well-retrieved one,
which is the exact confusion this project is built to dispel.

**Answers are optional.** `/search` needs no API key and no quota, so the
retrieval half is always usable. `/answer` degrades to a clear error when no key
is configured rather than silently returning something weaker.

The index is built once at startup and shared. Chunking and indexing 42,215 chunks
takes about ninety seconds, so doing it per request would make the service useless;
doing it in a background thread would serve wrong results until it finished.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..ablation.runner import make_chunker
from ..config import get_settings
from ..corpus.ingest import load_corpus
from ..evalset.relevance import Chunk
from ..index.bm25 import BM25Index
from .templates import INDEX_HTML

log = logging.getLogger(__name__)

#: Matches the best-measured first stage: structure-aware chunking, BM25.
DEFAULT_CHUNKER = "struct512"


@dataclass
class Pipeline:
    """Everything a request needs, built once."""

    chunks: dict[str, Chunk] = field(default_factory=dict)
    index: BM25Index | None = None
    doc_titles: dict[str, str] = field(default_factory=dict)
    ready: bool = False
    #: Populated instead of raising when the corpus is absent, so the service
    #: starts and can explain itself rather than crash-looping.
    error: str | None = None
    build_seconds: float = 0.0

    def load(self) -> None:
        started = time.monotonic()
        try:
            docs = load_corpus()
        except (FileNotFoundError, ValueError) as exc:
            self.error = (
                f"corpus unavailable: {exc}. Run "
                f"`python -m retrieval_ablation.corpus.ingest` first."
            )
            log.error(self.error)
            return

        chunker = make_chunker(DEFAULT_CHUNKER)
        chunk_list = chunker.chunk_corpus(docs)
        self.chunks = {c.chunk_id: c for c in chunk_list}
        self.index = BM25Index(chunk_list)
        self.doc_titles = {
            d.doc_id: f"{d.metadata.get('company', d.doc_id)} "
            f"{d.metadata.get('form', '')} {d.metadata.get('report_date', '')}".strip()
            for d in docs
        }
        self.build_seconds = time.monotonic() - started
        self.ready = True
        log.info(
            "indexed %d chunks from %d documents in %.1fs",
            len(chunk_list),
            len(docs),
            self.build_seconds,
        )


PIPELINE = Pipeline()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    PIPELINE.load()
    yield


app = FastAPI(
    title="retrieval-ablation",
    description="Retrieval over SEC filings, with the scores behind every result.",
    lifespan=lifespan,
)


class Passage(BaseModel):
    """One retrieved chunk, with everything needed to judge why it was returned."""

    rank: int
    chunk_id: str
    doc_id: str
    document: str
    section: str
    score: float
    #: Score normalised to the top hit, purely for rendering a bar. Explicitly not
    #: a probability or a confidence -- BM25 scores have no upper bound and no
    #: calibration, and labelling this "confidence" would invite exactly the
    #: misreading the project avoids elsewhere.
    score_relative: float
    contains_table: bool
    text: str
    char_start: int
    char_end: int


class SearchResponse(BaseModel):
    query: str
    n_chunks_indexed: int
    took_ms: float
    passages: list[Passage]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    cited_chunk_ids: list[str]
    invalid_citations: list[int]
    passages: list[Passage]
    prompt_tokens: int
    output_tokens: int
    took_ms: float
    from_cache: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)


def _require_ready() -> BM25Index:
    if not PIPELINE.ready or PIPELINE.index is None:
        raise HTTPException(status_code=503, detail=PIPELINE.error or "index still building")
    return PIPELINE.index


def _passages(query: str, top_k: int) -> list[Passage]:
    index = _require_ready()
    hits = index.search(query, top_k=top_k)
    best = max((h.score for h in hits), default=1.0) or 1.0
    out: list[Passage] = []
    for rank, hit in enumerate(hits, start=1):
        chunk = PIPELINE.chunks[hit.chunk_id]
        out.append(
            Passage(
                rank=rank,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                document=PIPELINE.doc_titles.get(chunk.doc_id, chunk.doc_id),
                section=" > ".join(chunk.section_path) or "(no section)",
                score=round(hit.score, 4),
                score_relative=round(hit.score / best, 4),
                contains_table=chunk.contains_table,
                text=chunk.text,
                char_start=chunk.span.start,
                char_end=chunk.span.end,
            )
        )
    return out


@app.get("/health")
def health() -> dict:
    """Readiness and what the service is actually serving."""
    return {
        "ready": PIPELINE.ready,
        "error": PIPELINE.error,
        "n_chunks": len(PIPELINE.chunks),
        "n_documents": len(PIPELINE.doc_titles),
        "chunker": DEFAULT_CHUNKER,
        "retriever": "bm25",
        "index_build_seconds": round(PIPELINE.build_seconds, 1),
        # Stated so nobody reads a demo answer as the project's best result.
        "note": (
            "First stage is lexical only. Dense, hybrid and reranked "
            "configurations require the GPU artifacts; see the README."
        ),
        "answer_endpoint_available": bool(get_settings().gemini_api_key),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Retrieve passages and return the scores behind the ranking."""
    started = time.monotonic()
    passages = _passages(request.query, request.top_k)
    return SearchResponse(
        query=request.query,
        n_chunks_indexed=len(PIPELINE.chunks),
        took_ms=round((time.monotonic() - started) * 1000, 2),
        passages=passages,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: SearchRequest) -> AnswerResponse:
    """Retrieve, then answer with numbered citations back to the passages."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY is not configured. /search works without it; "
                "only answer generation needs a key."
            ),
        )

    from ..generation.answer import generate_answer  # noqa: PLC0415 - optional path
    from ..llm.gemini import GeminiClient, QuotaExhaustedError  # noqa: PLC0415

    started = time.monotonic()
    passages = _passages(request.query, request.top_k)
    chunks = [PIPELINE.chunks[p.chunk_id] for p in passages]

    try:
        with GeminiClient() as client:
            generated = generate_answer(client, "live", request.query, chunks, arm="retrieval")
    except QuotaExhaustedError as exc:
        # 429 rather than 500: this is a quota condition the caller can retry,
        # not a bug in the service.
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return AnswerResponse(
        query=request.query,
        answer=generated.answer,
        refused=generated.refused,
        cited_chunk_ids=list(generated.cited_ids),
        invalid_citations=list(generated.invalid_citations),
        passages=passages,
        prompt_tokens=generated.prompt_tokens,
        output_tokens=generated.output_tokens,
        took_ms=round((time.monotonic() - started) * 1000, 2),
        from_cache=generated.from_cache,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
