"""Process-wide settings and canonical filesystem layout.

Settings are read from the environment (and `.env`) exactly once, at first use.
Secrets are typed as optional and are *never* given defaults: code that needs a
key must fail loudly when it is absent rather than silently degrading to a
different provider, because a silent fallback would let an unmeasured run be
mistaken for a measured one.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root rather than the current working directory so that a
# script invoked from anywhere writes to the same place. Layout is
# src/retrieval_ablation/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / ".cache"
MODEL_DIR = REPO_ROOT / "models"

# The single source of truth for reproducibility. Every sampling decision in the
# project derives its generator from this value; nothing calls an unseeded RNG
# and nothing reads the clock to make a data-affecting decision.
GLOBAL_SEED = 20260730


class Settings(BaseSettings):
    """Environment-backed configuration.

    All API keys are `None` by default. The offline test suite must pass with an
    empty environment, so no field here may be required.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    hf_token: str | None = None

    # SEC returns 403 to clients that do not declare a contact. Left as None so
    # the corpus fetcher raises a clear error instead of hammering sec.gov with
    # a header they have asked people not to send.
    edgar_user_agent: str | None = None

    # EDGAR's published fair-access ceiling is 10 requests/second. We stay well
    # under it: the corpus fetch is a one-time job and being a good citizen
    # costs us nothing but wall-clock time.
    edgar_requests_per_second: float = Field(default=5.0, gt=0, le=10.0)

    def require(self, field: str) -> str:
        """Return a secret or raise, naming the variable and where to get it.

        Used at the call site of any network operation so a missing key produces
        an actionable message rather than a downstream 401.
        """
        value = getattr(self, field)
        if not value:
            raise RuntimeError(
                f"{field.upper()} is not set. Add it to {REPO_ROOT / '.env'} "
                f"(see .env.example for where to obtain it)."
            )
        return value


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def ensure_dirs() -> None:
    """Create the writable data directories. Safe to call repeatedly."""
    for path in (
        RAW_DIR,
        INTERIM_DIR,
        INDEX_DIR,
        EVAL_DIR,
        MANIFEST_DIR,
        RESULTS_DIR,
        CACHE_DIR,
        MODEL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
