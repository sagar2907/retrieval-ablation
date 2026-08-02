"""Tests for the service. No corpus, no key, no network.

The pipeline is replaced with a small in-memory index, so these exercise the real
request handlers and response models rather than mocking them away.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip(
    "fastapi",
    reason="fastapi lives in the optional `service` extra; install with .[service]",
)
from fastapi.testclient import TestClient  # noqa: E402

from retrieval_ablation.corpus.models import Span  # noqa: E402
from retrieval_ablation.evalset.relevance import Chunk  # noqa: E402
from retrieval_ablation.index.bm25 import BM25Index  # noqa: E402
from retrieval_ablation.service import app as service  # noqa: E402

CHUNKS = [
    Chunk(
        chunk_id="c1",
        doc_id="aapl-10-k-2025-09-27",
        span=Span(0, 60),
        text="Research and development expense was 34,550 million in fiscal 2025.",
        section_path=("Part II", "Item 8. Financial Statements"),
        contains_table=True,
    ),
    Chunk(
        chunk_id="c2",
        doc_id="aapl-10-k-2024-09-28",
        span=Span(0, 60),
        text="Research and development expense was 31,370 million in fiscal 2024.",
        section_path=("Part II", "Item 8. Financial Statements"),
    ),
    Chunk(
        chunk_id="c3",
        doc_id="msft-10-k-2025-06-30",
        span=Span(0, 40),
        text="The Company designs and markets cloud services.",
        section_path=("Part I", "Item 1. Business"),
    ),
]


@pytest.fixture
def client(monkeypatch) -> TestClient:
    pipeline = service.Pipeline(
        chunks={c.chunk_id: c for c in CHUNKS},
        index=BM25Index(CHUNKS),
        doc_titles={c.doc_id: c.doc_id.upper() for c in CHUNKS},
        ready=True,
        build_seconds=0.4,
    )
    monkeypatch.setattr(service, "PIPELINE", pipeline)
    # lifespan would rebuild from the real corpus, which is not present in CI.
    monkeypatch.setattr(service.Pipeline, "load", lambda self: None)
    with TestClient(service.app) as c:
        monkeypatch.setattr(service, "PIPELINE", pipeline)
        yield c


class TestHealth:
    def test_reports_readiness_and_contents(self, client):
        body = client.get("/health").json()
        assert body["ready"] is True
        assert body["n_chunks"] == 3
        assert body["retriever"] == "bm25"

    def test_states_that_only_the_lexical_arm_is_served(self, client):
        """A demo answer must not be mistaken for the project's best result."""
        note = client.get("/health").json()["note"]
        assert "lexical" in note.lower()


class TestSearch:
    def test_returns_ranked_passages(self, client):
        body = client.post("/search", json={"query": "research development 2025"}).json()
        assert body["passages"]
        assert body["passages"][0]["chunk_id"] == "c1"
        assert body["passages"][0]["rank"] == 1

    def test_exposes_the_score_behind_each_result(self, client):
        """Scores are the point of the service, not decoration."""
        passage = client.post("/search", json={"query": "research development"}).json()[
            "passages"
        ][0]
        for field in ("score", "score_relative", "section", "char_start", "char_end"):
            assert field in passage
        assert passage["score"] > 0

    def test_relative_score_is_one_for_the_top_hit(self, client):
        passages = client.post("/search", json={"query": "research development"}).json()[
            "passages"
        ]
        assert passages[0]["score_relative"] == pytest.approx(1.0)
        assert all(p["score_relative"] <= 1.0 for p in passages)

    def test_ranks_are_dense_and_ordered(self, client):
        passages = client.post("/search", json={"query": "expense"}).json()["passages"]
        assert [p["rank"] for p in passages] == list(range(1, len(passages) + 1))

    def test_table_flag_is_surfaced(self, client):
        passages = client.post("/search", json={"query": "research development 2025"}).json()[
            "passages"
        ]
        assert passages[0]["contains_table"] is True

    def test_no_match_returns_empty_not_padding(self, client):
        """Padding with zero-scoring chunks would misrepresent retrieval."""
        body = client.post("/search", json={"query": "zebra pharmacology"}).json()
        assert body["passages"] == []

    def test_respects_top_k(self, client):
        body = client.post("/search", json={"query": "expense", "top_k": 1}).json()
        assert len(body["passages"]) <= 1

    def test_empty_query_is_rejected(self, client):
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_absurd_top_k_is_rejected(self, client):
        assert client.post("/search", json={"query": "x", "top_k": 9999}).status_code == 422


class TestAnswerWithoutKey:
    def test_returns_503_and_says_search_still_works(self, client, monkeypatch):
        """Degrading loudly beats returning something weaker in silence."""
        from retrieval_ablation import config

        config.get_settings.cache_clear()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(
            service, "get_settings", lambda: config.Settings(gemini_api_key=None)
        )
        response = client.post("/answer", json={"query": "revenue"})
        assert response.status_code == 503
        assert "search" in response.json()["detail"].lower()


class TestNotReady:
    def test_search_returns_503_with_the_reason(self, monkeypatch):
        """A missing corpus must explain itself, not crash-loop the container."""
        broken = service.Pipeline(ready=False, error="corpus unavailable: run the ingest")
        monkeypatch.setattr(service, "PIPELINE", broken)
        monkeypatch.setattr(service.Pipeline, "load", lambda self: None)
        with TestClient(service.app) as client:
            monkeypatch.setattr(service, "PIPELINE", broken)
            response = client.post("/search", json={"query": "x"})
        assert response.status_code == 503
        assert "ingest" in response.json()["detail"]


class TestUI:
    def test_serves_a_self_contained_page(self, client):
        html = client.get("/").text
        assert "<!doctype html>" in html.lower()
        # No external asset may be fetched: the page must work offline and under
        # a strict content policy.
        for marker in ("http://", "https://", "cdn."):
            assert marker not in html.split("<style>")[0]

    def test_page_wires_up_citation_clicking(self, client):
        html = client.get("/").text
        assert "scrollIntoView" in html
        assert "cite" in html

    def test_page_escapes_untrusted_text(self, client):
        """Passage text is filing content and goes into innerHTML."""
        assert "function esc(" in client.get("/").text
