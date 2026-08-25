"""A Gemini client built for a tight free tier: cached, resumable, and honest about cost.

Three properties, each forced by something measured rather than guessed.

**Every response is cached on disk, keyed by a hash of the exact request.** The
free tier permits only a few requests per minute, so a full generation-eval pass
takes hours. A run that is interrupted and restarted must not re-pay for work it
already did, and -- more importantly -- must not produce *different* answers the
second time, because that would make the reported metrics depend on when the run
happened to be interrupted.

**Rate limiting is handled by waiting, not by failing.** Measured behaviour on
this tier: `batchEmbedContents` accepts at most 32 texts and returns 429
RESOURCE_EXHAUSTED on the second consecutive call, against quota
`EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`. That is roughly
96 texts per minute. A client that treated 429 as an error would abort constantly;
this one backs off and continues, and the caller sees a slow success instead of a
fast failure.

**Token usage is recorded per call.** The long-context comparison is a *cost*
claim, and a cost claim computed from estimated token counts is not evidence. The
API reports `promptTokenCount` and `candidatesTokenCount`, and those reported
numbers are what the comparison uses.

One behaviour worth knowing about, found by testing rather than reading: Gemini 3.x
flash models are thinking models. Asking for a one-word answer with
`maxOutputTokens=16` returns **empty text**, because 13 of those tokens went to
`thoughtsTokenCount`. `thinkingConfig.thinkingBudget = 0` is rejected outright with
HTTP 400. So the output budget must be generous even for short answers, and
`DEFAULT_MAX_OUTPUT_TOKENS` reflects that rather than the length of the expected
reply.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import CACHE_DIR, Settings, get_settings

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: Verified present on this tier with a 1,048,576-token input window, which is
#: what makes the long-context arm possible at all.
DEFAULT_MODEL = "gemini-3.6-flash"

#: Cheaper sibling for high-volume judging, also with a 1M window.
JUDGE_MODEL = "gemini-3.5-flash-lite"

DEFAULT_EMBED_MODEL = "gemini-embedding-001"

#: Generous because thinking tokens are drawn from this budget. At 16 the model
#: returned nothing at all; at 512 the same prompt answered correctly having spent
#: 81 tokens on thoughts.
DEFAULT_MAX_OUTPUT_TOKENS = 1024

#: Measured ceiling. 64 is rejected with RESOURCE_EXHAUSTED.
MAX_EMBED_BATCH = 32


class QuotaExhaustedError(RuntimeError):
    """Raised when the daily quota, not merely the per-minute rate, is spent."""


@dataclass
class Usage:
    """Token accounting, aggregated across a run."""

    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    rate_limited: int = 0
    seconds_waiting: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, metadata: dict, latency: float) -> None:
        self.calls += 1
        self.prompt_tokens += metadata.get("promptTokenCount", 0) or 0
        self.output_tokens += metadata.get("candidatesTokenCount", 0) or 0
        self.thought_tokens += metadata.get("thoughtsTokenCount", 0) or 0
        self.latencies.append(latency)

    def p95_latency(self) -> float | None:
        """95th-percentile latency, or None when nothing was actually called.

        None rather than 0.0 for an all-cache run: a cached run's latency is a
        property of the disk, not of the API, and reporting it as an API latency
        would be a fabricated measurement.
        """
        live = sorted(self.latencies)
        if not live:
            return None
        index = min(len(live) - 1, int(0.95 * len(live)))
        return live[index]

    def to_json(self) -> dict:
        return {
            "live_calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "rate_limited_responses": self.rate_limited,
            "seconds_spent_waiting": round(self.seconds_waiting, 1),
            "p95_latency_seconds": (round(p, 3) if (p := self.p95_latency()) is not None else None),
        }


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response, with its own token accounting."""

    text: str
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    model: str
    from_cache: bool
    latency_seconds: float | None


class GeminiClient:
    """Cached, rate-limit-tolerant access to the Gemini API."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        max_attempts: int = 8,
        seed: int = 0,
    ) -> None:
        self._settings = settings or get_settings()
        self._key = self._settings.require("gemini_api_key")
        self._cache = (cache_dir or CACHE_DIR) / "gemini"
        self._cache.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            timeout=httpx.Timeout(600.0, connect=30.0),
            # Verified: the API-key header works, while an OAuth bearer header is
            # rejected with 401 for this credential.
            headers={"x-goog-api-key": self._key},
            trust_env=False,
        )
        self._max_attempts = max_attempts
        self.usage = Usage()
        # Jitter comes from a seeded generator so a rerun schedules its retries
        # identically. Retry timing cannot change results -- responses are cached
        # by content -- but a reproducible run log is worth having.
        self._rng = random.Random(seed)

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- caching --------------------------------------------------------------

    def _cache_path(self, payload: dict, path: str) -> Path:
        # The key covers the endpoint and the entire request body, so changing a
        # prompt, a model, or a temperature produces a different key. Reusing a
        # response across a changed prompt would be silent contamination.
        digest = hashlib.sha256(
            json.dumps({"path": path, "payload": payload}, sort_keys=True).encode()
        ).hexdigest()
        return self._cache / f"{digest}.json"

    # -- transport ------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        """POST with caching and quota-aware retry."""
        cache_file = self._cache_path(payload, path)
        if cache_file.exists():
            self.usage.cached_calls += 1
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            # Marked on the returned dict, not in the stored file. Callers need to
            # know a response came from disk, and the only previous signal was a
            # missing latency -- which never happens, because the measured latency
            # is written *into* the cached body. `from_cache` was therefore False
            # on every answer this project ever recorded.
            data["_from_cache"] = True
            return data

        for attempt in range(1, self._max_attempts + 1):
            started = time.monotonic()
            response = self._client.post(f"{BASE_URL}{path}", json=payload)
            latency = time.monotonic() - started

            if response.status_code == 200:
                data = response.json()
                data["_latency_seconds"] = latency
                tmp = cache_file.with_suffix(".partial")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(cache_file)
                data["_from_cache"] = False
                return data

            if response.status_code == 429:
                self.usage.rate_limited += 1
                delay = self._retry_delay(response, attempt)
                if attempt == self._max_attempts:
                    violation = self._quota_violation(response)
                    named = (
                        f"The server names the exceeded quota as {violation}. "
                        if violation
                        else "The response named no specific quota. "
                    )
                    raise QuotaExhaustedError(
                        f"still rate-limited after {attempt} attempts on {path}. "
                        f"{named}Re-run later; cached work is preserved, so the run "
                        f"resumes rather than restarts."
                    )
                log.warning(
                    "429 on %s (attempt %d/%d), waiting %.1fs",
                    path,
                    attempt,
                    self._max_attempts,
                    delay,
                )
                time.sleep(delay)
                self.usage.seconds_waiting += delay
                continue

            if response.status_code in {500, 502, 503, 504}:
                delay = min(60.0, 2.0**attempt) + self._rng.uniform(0, 1)
                log.warning("%d on %s, retrying in %.1fs", response.status_code, path, delay)
                time.sleep(delay)
                self.usage.seconds_waiting += delay
                continue

            # 400/404 and friends are deterministic: retrying cannot help, and
            # swallowing them would hide a malformed request or a wrong model id.
            raise RuntimeError(f"{response.status_code} from {path}: {response.text[:400]}")

        raise QuotaExhaustedError(f"exhausted retries on {path}")

    @staticmethod
    def _quota_violation(response: httpx.Response) -> str:
        """Describe which quota the server says was exceeded.

        A 429 body carries a QuotaFailure detail naming the exact limit -- for this
        project, `GenerateRequestsPerDayPerProjectPerModel-FreeTier` with a value of
        20. The previous message guessed instead: "the daily free-tier quota is
        likely spent". It was hedged, and it was still a guess about a fact the
        response states outright, which meant every run's stopping reason was an
        inference rather than a quotation.

        It matters beyond tidiness. The guess encouraged reasoning about token
        budgets, and the limit the server actually names is counted in *requests*.
        A cheap call and an expensive one cost the same against it.

        Returns an empty string when the body carries no such detail, because
        inventing a reason is what this replaces.
        """
        try:
            details = response.json().get("error", {}).get("details", [])
        except ValueError:
            return ""
        for detail in details:
            for violation in detail.get("violations", []) or []:
                quota_id = violation.get("quotaId")
                if not quota_id:
                    continue
                value = violation.get("quotaValue")
                return f"{quota_id}" + (f" (limit {value})" if value else "")
        return ""

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Honour the server's RetryInfo when present, else exponential backoff."""
        try:
            for detail in response.json().get("error", {}).get("details", []):
                if detail.get("@type", "").endswith("RetryInfo"):
                    raw = str(detail.get("retryDelay", "")).rstrip("s")
                    if raw:
                        return float(raw) + self._rng.uniform(0.5, 2.0)
        except (ValueError, KeyError, TypeError):
            pass
        # The per-minute quota resets on a minute boundary, so waiting less than
        # that just burns another 429.
        return min(90.0, 20.0 * attempt) + self._rng.uniform(0.5, 3.0)

    # -- generation -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> Completion:
        """One completion. Temperature 0 by default so runs are reproducible."""
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data = self._post(f"/models/{model}:generateContent", payload)
        latency = data.get("_latency_seconds")
        metadata = data.get("usageMetadata", {})
        if latency is not None:
            self.usage.add(metadata, latency)

        text = "".join(
            part.get("text", "")
            for candidate in data.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
        )
        return Completion(
            text=text.strip(),
            prompt_tokens=metadata.get("promptTokenCount", 0) or 0,
            output_tokens=metadata.get("candidatesTokenCount", 0) or 0,
            thought_tokens=metadata.get("thoughtsTokenCount", 0) or 0,
            model=model,
            from_cache=bool(data.get("_from_cache", latency is None)),
            latency_seconds=latency,
        )

    # -- embedding ------------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
        model: str = DEFAULT_EMBED_MODEL,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed texts, batching at the measured maximum of 32 per request.

        `task_type` matters and is not decoration: this family of models is
        asymmetric, so passages must be embedded as RETRIEVAL_DOCUMENT and queries
        as RETRIEVAL_QUERY. Using one setting for both silently degrades retrieval
        quality with no error, which an ablation would misread as a property of
        the model.
        """
        out: list[list[float]] = []
        for start in range(0, len(texts), MAX_EMBED_BATCH):
            batch = texts[start : start + MAX_EMBED_BATCH]
            data = self._post(
                f"/models/{model}:batchEmbedContents",
                {
                    "requests": [
                        {
                            "model": f"models/{model}",
                            "content": {"parts": [{"text": text}]},
                            "taskType": task_type,
                        }
                        for text in batch
                    ]
                },
            )
            out.extend(item["values"] for item in data.get("embeddings", []))
        return out

    def embed_queries(self, texts: Sequence[str], model: str = DEFAULT_EMBED_MODEL):
        return self.embed(texts, model=model, task_type="RETRIEVAL_QUERY")
