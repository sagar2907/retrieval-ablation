"""The client must report whether a response came from disk.

Offline: no request is ever sent. Every test either pre-writes a cache entry, so
the client returns before touching the network, or asserts on a pure helper. The
one test that would otherwise reach the API replaces the transport.
"""

from __future__ import annotations

import json

import pytest

from retrieval_ablation.config import Settings
from retrieval_ablation.llm.gemini import GeminiClient


def client(tmp_path) -> GeminiClient:
    return GeminiClient(settings=Settings(gemini_api_key="not-a-real-key"), cache_dir=tmp_path)


def seed_cache(gemini: GeminiClient, prompt: str, model: str, text: str, latency: float) -> None:
    """Write a cache entry exactly as a live call would have written it."""
    payload: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.0},
    }
    path = gemini._cache_path(payload, f"/models/{model}:generateContent")
    path.write_text(
        json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": text}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
                # The live path writes the measured latency into the body, which is
                # the whole reason the old inference was wrong.
                "_latency_seconds": latency,
            }
        ),
        encoding="utf-8",
    )


class TestFromCache:
    def test_a_cached_response_is_reported_as_cached(self, tmp_path):
        """Regression: `from_cache` was False on every answer ever recorded.

        It was inferred as "the response carries no latency", but the client stores
        the measured latency inside the cached body, so a cache hit always has one.
        The flag was therefore never true, and anything downstream that trusted it
        -- latency statistics, in particular -- silently treated timings from an
        earlier session as if they had just been measured.
        """
        gemini = client(tmp_path)
        payload = {
            "contents": [{"parts": [{"text": "hello"}]}],
            "generationConfig": {"maxOutputTokens": 512, "temperature": 0.0},
        }
        seed_cache(gemini, "hello", "m", "hi", latency=1.25)

        # Sanity: the entry is where the client will look for it.
        assert gemini._cache_path(payload, "/models/m:generateContent").exists()

        out = gemini.generate("hello", model="m", max_output_tokens=512)

        assert out.from_cache is True
        assert out.latency_seconds == 1.25
        assert gemini.usage.cached_calls == 1
        gemini.close()

    def test_a_live_response_is_reported_as_live(self, tmp_path, monkeypatch):
        """The other half: a fresh call must not be mistaken for a cache hit."""
        gemini = client(tmp_path)

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "candidates": [{"content": {"parts": [{"text": "fresh"}]}}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
                }

        monkeypatch.setattr(gemini._client, "post", lambda *a, **k: FakeResponse())

        out = gemini.generate("uncached prompt", model="m")

        assert out.from_cache is False
        assert out.latency_seconds is not None
        gemini.close()

    def test_the_stored_entry_is_reused_on_a_second_call(self, tmp_path, monkeypatch):
        """A second identical request must not reach the transport at all."""
        gemini = client(tmp_path)
        calls = {"n": 0}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json() -> dict:
                return {
                    "candidates": [{"content": {"parts": [{"text": "once"}]}}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
                }

        def post(*_a, **_k):
            calls["n"] += 1
            return FakeResponse()

        monkeypatch.setattr(gemini._client, "post", post)

        first = gemini.generate("same prompt", model="m")
        second = gemini.generate("same prompt", model="m")

        assert calls["n"] == 1
        assert first.from_cache is False
        assert second.from_cache is True
        assert second.text == first.text
        gemini.close()


class TestCacheKey:
    def test_a_different_prompt_is_a_different_entry(self, tmp_path):
        """The key covers the whole request body, so nothing is reused wrongly."""
        gemini = client(tmp_path)
        a = gemini._cache_path({"contents": [{"parts": [{"text": "a"}]}]}, "/p")
        b = gemini._cache_path({"contents": [{"parts": [{"text": "b"}]}]}, "/p")

        assert a != b
        gemini.close()

    def test_a_different_endpoint_is_a_different_entry(self, tmp_path):
        gemini = client(tmp_path)
        body = {"contents": [{"parts": [{"text": "a"}]}]}

        assert gemini._cache_path(body, "/one") != gemini._cache_path(body, "/two")
        gemini.close()


class TestConstruction:
    def test_a_missing_key_is_refused_rather_than_deferred(self):
        """Failing at construction beats failing halfway through a paid run."""
        with pytest.raises(Exception, match=r"(?i)gemini_api_key|required|missing"):
            GeminiClient(settings=Settings(gemini_api_key=None))
