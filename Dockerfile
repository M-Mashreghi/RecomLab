# syntax=docker/dockerfile:1
# Multi-stage, slim, non-root demo image for the RecomLab API and dashboard.
# Data and artifacts are mounted at runtime; no secrets are baked into the image.

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".[api,dashboard]"

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RECOMLAB_ENV=production \
    RECOMLAB_ROOT=/app \
    RECOMLAB_DB_URL=sqlite:////app/artifacts/recomlab.db
RUN useradd --create-home --uid 10001 recomlab
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY configs ./configs
COPY data/fixtures ./data/fixtures
RUN mkdir -p /app/artifacts /app/data/processed && chown -R recomlab:recomlab /app
USER recomlab
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["sh", "-c", "recomlab bootstrap-demo && uvicorn recomlab.api.main:app --host 0.0.0.0 --port 8000"]
