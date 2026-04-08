# Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${POETRY_VENV}/bin:${PATH}"

# Set workdir
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./
# poetry.lock is optional — include if present
# COPY poetry.lock ./

# Install dependencies (no dev deps in prod)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy full project
COPY . .

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
