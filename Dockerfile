FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY ./app ./app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

# Health check. PORT-aware so it works both locally/compose (PORT unset → 8000)
# and on a platform that injects PORT (Cloud Run → 8080). Cloud Run ignores the
# Docker HEALTHCHECK and runs its own probes, but this keeps compose correct.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# --proxy-headers/--forwarded-allow-ips: the platform (Render / Cloud Run)
# terminates TLS at its proxy and rewrites X-Forwarded-For with the real client
# IP. Without these flags every request reports the proxy's IP, collapsing all
# per-IP rate-limit buckets (app/rate_limit.py) into one shared bucket.
#
# Shell form (not exec/JSON) so ${PORT} expands. Cloud Run injects PORT=8080 and
# the container MUST listen on it; locally/compose PORT is unset → 8000. '*' is
# single-quoted so the shell passes it literally to uvicorn instead of globbing
# it against /code. `exec` makes uvicorn replace the shell as PID 1 so it
# receives SIGTERM directly (clean shutdown on Cloud Run scale-down) instead of
# the shell swallowing it.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'
