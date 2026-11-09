# Reproducible MemCheck environment.
#
# Build:  docker build -t memcheck .
# Smoke:  docker run --rm memcheck bash scripts/reproduce_smoke.sh
# Shell:  docker run --rm -it -v "$PWD/results:/app/results" memcheck bash
#
# The image contains no credentials and runs everything with the deterministic mock reader.
# To run a paid experiment you must pass BOTH controls explicitly, e.g.:
#   docker run --rm -e ALLOW_PAID_RUNS=true -e MAX_EXPERIMENT_COST_USD=5 \
#     -e ANTHROPIC_API_KEY=... memcheck uv run memcheck run --config configs/longmemeval.yaml

FROM python:3.12-slim-bookworm

# texlive-* is what lets `memcheck paper --build` work inside the image; without it the CLI
# falls back to source validation, which is what happens on a host with no LaTeX engine.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        make \
        ca-certificates \
        latexmk \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-publishers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY requirements.lock.txt ./
RUN uv venv --python 3.12 \
    && uv pip install --require-hashes=false -r requirements.lock.txt || uv pip install -e ".[dev,parquet]"

COPY . .
RUN uv pip install -e ".[dev,parquet]"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ALLOW_PAID_RUNS=false \
    MAX_EXPERIMENT_COST_USD=0

# Fail the build if the environment is not usable.
RUN memcheck doctor && memcheck validate --backend guarded_memory

CMD ["bash", "scripts/reproduce_smoke.sh"]
