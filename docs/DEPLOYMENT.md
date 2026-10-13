# RecomLab Deployment

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Minimal Demo Services

- FastAPI application
- Streamlit dashboard
- SQLite metadata registry
- local Parquet/model artifacts

No Redis, Kafka, Kubernetes, external vector DB, or feature store is required for v1.

## Environment

See `.env.example` for every supported `RECOMLAB_*` variable, for example:

```env
RECOMLAB_ENV=development
RECOMLAB_DB_URL=sqlite:///./data/recomlab.db
RECOMLAB_LLM_PROVIDER=none
RECOMLAB_LLM_API_KEY=
```

The LLM provider, model, and key are optional and only required for the Phase 7 LLM path
(`openai`/`anthropic`); `mock` is used in tests.

## CI Release Gates

Mandatory:
- unit/integration/E2E fixture tests
- temporal leakage tests
- metric correctness tests
- full-vs-sampled evaluation labeling tests
- secret/dependency audit

## Demo Bootstrap (no Docker)

```bash
pip install -e ".[api,dashboard]"
recomlab bootstrap-demo                       # snapshot + train + evaluate + deploy demo bundle
uvicorn recomlab.api.main:app --port 8000     # API: /health, /ready, /recommendations, ...
streamlit run src/recomlab/dashboard/app.py   # dashboard on :8501
```

`bootstrap-demo` is idempotent: when a bundle is already active it does nothing unless
`--force` is given. The first deployment activates the trained bundle explicitly (recorded as
`initial demo deployment`); later `recomlab retrain` runs never auto-promote.

## Docker

```bash
docker compose up --build          # api on :8000, dashboard on :8501
```

- `Dockerfile`: multi-stage build on `python:3.12-slim`; the runtime stage installs only the
  built wheels, runs as the non-root `recomlab` user (uid 10001), exposes 8000/8501, and has a
  `HEALTHCHECK` against `/health`. The container runs `recomlab bootstrap-demo` then uvicorn.
- `docker-compose.yml`: `api` + `dashboard` only; both mount `./artifacts` and
  `./data/processed` from the host, read optional settings from `.env`, and have
  healthchecks; the dashboard waits for a healthy API.
- `.dockerignore` excludes `.env`, raw/processed data, artifacts, tests and docs, so no
  secrets or datasets are baked into the image. Provide secrets only at runtime via `.env`
  or environment variables.
