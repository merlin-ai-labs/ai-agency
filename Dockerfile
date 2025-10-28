# Lean Dockerfile for Cloud Run deployment
# TODO: Multi-stage build for smaller image size

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY alembic.ini ./

# Expose port for Cloud Run (8080 is the default)
ENV PORT=8080
EXPOSE 8080

# Health check (optional)
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#   CMD python -c "import httpx; httpx.get('http://localhost:8080/healthz')"

# Run the FastAPI app
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
