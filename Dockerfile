# Serves the retrieval API and the inspection UI.
#
# CPU-only and deliberately so. The dense and reranking arms need GPU artifacts
# that are produced elsewhere (see notebooks/kaggle_gpu_arms.py) and loaded as
# files; baking torch into this image would multiply its size for a code path the
# service does not currently take.

FROM python:3.12-slim AS base

# Build tools are needed for lxml's C extension, then removed. Keeping them would
# roughly double the image for something only the install step uses.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency metadata first, so a source-only change does not invalidate the
# dependency layer and force a full reinstall on every rebuild.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[service]" \
    && apt-get purge -y --auto-remove build-essential libxml2-dev libxslt1-dev

# The corpus and eval set are mounted rather than copied. They are ~70 MB of
# derived data reproducible from the committed manifest, and baking them in would
# make the image stale the moment the corpus is rebuilt.
VOLUME ["/app/data"]

EXPOSE 8000

# The index takes about ninety seconds to build at startup, so the healthcheck
# allows for that before reporting the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
    r=json.load(urllib.request.urlopen('http://localhost:8000/health')); \
    sys.exit(0 if r['ready'] else 1)"

# Single worker on purpose. Each worker builds its own 42,000-chunk index, so
# adding workers multiplies memory rather than throughput; scale with replicas
# behind a proxy if that is ever needed.
CMD ["uvicorn", "retrieval_ablation.service.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
