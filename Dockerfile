FROM python:3.10-slim

WORKDIR /app

# Install system deps needed by psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Liveness only (/healthz never depends on juris.ph/OpenAI/DB), so a slow dependency can't get a
# healthy process marked unhealthy. Informational under plain `docker run`; anything that acts on
# health status (autoheal, compose, an orchestrator) should key off this, not /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3   CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

# --timeout-graceful-shutdown: a legal turn holds one WebSocket open for 4-5 minutes. Without a
# bound, `docker stop` (deploys, restarts) waits on that socket until Docker's own 10 s SIGKILL
# hits, and the client just sees the connection vanish. 30 s lets short turns finish and makes
# the long ones fail fast and cleanly instead of hanging the restart.
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
