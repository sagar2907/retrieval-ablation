"""A polite, cached, resumable EDGAR client.

Three properties matter here, in this order:

1.  **Politeness.** SEC publishes a fair-access limit of 10 requests/second and
    requires a User-Agent that names a real contact. Both are enforced here
    rather than left to the caller. Verified empirically: a generic
    `python-httpx/0.27` User-Agent receives HTTP 403 from both `data.sec.gov`
    and `www.sec.gov`; only a contact-style string is served.

2.  **Idempotence.** Every fetch is written to a content-addressed cache before
    parsing. Re-running the ingest re-reads bytes from disk and issues no
    network traffic, so the corpus is stable across runs and a failed run can
    be resumed without re-downloading gigabytes.

3.  **Reproducibility.** The corpus is defined by a committed ticker list plus
    a rule ("the N most recent 10-K filings, oldest first"), not by whatever
    the API happened to return today. The resulting selection is written to a
    checksummed manifest so a third party can verify they built the same corpus.

The rate limiter reads the wall clock. That is the one sanctioned clock read in
the project: it affects *when* requests go out, never which bytes end up in the
corpus, so it cannot make a run non-reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import RAW_DIR, Settings, get_settings

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class EdgarError(RuntimeError):
    """Raised when EDGAR cannot be used as configured."""


@dataclass(frozen=True, slots=True)
class FilingRef:
    """A specific filing document, identified well enough to re-fetch exactly."""

    cik: int
    ticker: str
    company: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    document: str

    @property
    def doc_id(self) -> str:
        """Stable, human-readable, filesystem-safe identifier.

        Built from ticker and report date rather than the accession number
        because it appears in the published eval set, where a reader should be
        able to tell which filing a gold passage came from at a glance.
        Uniqueness still holds: one company files one 10-K per period.
        """
        return f"{self.ticker.lower()}-{self.form.lower().replace('/', '')}-{self.report_date}"

    @property
    def url(self) -> str:
        return DOCUMENT_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.document
        )

    @property
    def cache_path(self) -> Path:
        return RAW_DIR / f"{self.doc_id}.htm"


class _RateLimiter:
    """Token bucket, shared across threads.

    Serialises requests to at most `rate` per second. Deliberately simple and
    conservative: a burst allowance would let a retry storm exceed SEC's limit
    at exactly the moment things are already going wrong.
    """

    def __init__(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class EdgarClient:
    """Fetches EDGAR metadata and filing documents through a local cache."""

    def __init__(self, settings: Settings | None = None, cache_dir: Path | None = None) -> None:
        self._settings = settings or get_settings()
        user_agent = self._settings.edgar_user_agent
        if not user_agent or "@" not in user_agent:
            raise EdgarError(
                "EDGAR_USER_AGENT must be set to a contact string of the form "
                "'Name email@example.com'. SEC returns HTTP 403 to clients that "
                "do not identify themselves; this is a documented access "
                "condition, not an obstacle to work around."
            )
        self._cache_dir = cache_dir or RAW_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _RateLimiter(self._settings.edgar_requests_per_second)
        # trust_env=False so an ambient proxy or CA bundle cannot silently alter
        # what lands in the corpus on a different machine.
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
            trust_env=False,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- transport ------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        self._limiter.acquire()
        response = self._client.get(url)
        if response.status_code == 403:
            # Not retryable: 403 here means the User-Agent was rejected, and
            # hammering the endpoint will not change that.
            raise EdgarError(
                f"EDGAR returned 403 for {url}. The configured User-Agent "
                f"was rejected; it must name a real contact."
            )
        response.raise_for_status()
        return response

    # -- metadata -------------------------------------------------------------

    def ticker_to_cik(self) -> dict[str, int]:
        """EDGAR's official ticker-to-CIK mapping, cached on disk.

        Resolved from the authoritative source rather than hardcoded, because a
        wrong CIK silently fetches a different company's filings -- a corpus
        error that no downstream test would catch.
        """
        cached = self._cache_dir / "company_tickers.json"
        if cached.exists():
            payload = json.loads(cached.read_text(encoding="utf-8"))
        else:
            payload = self._get(TICKER_MAP_URL).json()
            cached.write_text(json.dumps(payload), encoding="utf-8")
        return {row["ticker"].upper(): int(row["cik_str"]) for row in payload.values()}

    def submissions(self, cik: int) -> dict:
        """A company's filing history, cached on disk."""
        cached = self._cache_dir / f"submissions-{cik:010d}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        payload = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def find_filings(
        self,
        ticker: str,
        form: str = "10-K",
        limit: int = 4,
    ) -> list[FilingRef]:
        """The `limit` most recent filings of `form`, returned oldest first.

        Oldest-first ordering is deliberate: it keeps document ids and therefore
        chunk ids stable as newer filings are added, so extending the corpus
        does not renumber the existing one.

        Amended filings (10-K/A) are excluded. They restate parts of an earlier
        filing, so including both would put near-duplicate passages in the
        corpus and make "the" gold passage for a query ambiguous.
        """
        cik = self.ticker_to_cik().get(ticker.upper())
        if cik is None:
            raise EdgarError(f"ticker {ticker!r} is not in EDGAR's ticker map")

        payload = self.submissions(cik)
        company = payload.get("name", ticker)
        recent = payload["filings"]["recent"]

        rows = zip(
            recent["form"],
            recent["filingDate"],
            recent["reportDate"],
            recent["accessionNumber"],
            recent["primaryDocument"],
            strict=True,
        )
        matches = [
            FilingRef(
                cik=cik,
                ticker=ticker.upper(),
                company=company,
                form=f,
                filing_date=fdate,
                report_date=rdate,
                accession=acc,
                document=doc,
            )
            for f, fdate, rdate, acc, doc in rows
            # Exact match excludes amendments, whose form is "10-K/A".
            if f == form and doc.endswith((".htm", ".html"))
        ]
        # `recent` is newest-first; take the newest `limit`, then reverse.
        return list(reversed(matches[:limit]))

    # -- documents ------------------------------------------------------------

    def fetch(self, ref: FilingRef) -> bytes:
        """Return a filing's raw HTML bytes, downloading only on a cache miss.

        Bytes, not str, all the way from the socket to the parser. Inline-XBRL
        filings are XHTML carrying their own `<?xml encoding=...?>` declaration,
        and that declaration is authoritative. Decoding here would mean guessing
        an encoding and then handing lxml a string whose declared encoding
        contradicts how it was already decoded -- which lxml rejects outright,
        and which would silently mojibake the typographic characters filings are
        full of if it did not. Keeping bytes lets exactly one component honour
        the declaration, and makes the cached artifact byte-identical to what SEC
        served, so the manifest checksum means something.
        """
        path = ref.cache_path
        if path.exists():
            log.debug("cache hit %s", ref.doc_id)
            return path.read_bytes()

        log.info("fetching %s (%s)", ref.doc_id, ref.url)
        response = self._get(ref.url)
        # Written via a temporary file then renamed, so an interrupted run
        # cannot leave a truncated document that a later run mistakes for a
        # complete cache entry.
        tmp = path.with_suffix(".partial")
        tmp.write_bytes(response.content)
        tmp.replace(path)
        return response.content


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
