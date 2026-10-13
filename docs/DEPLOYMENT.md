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

## Docker

The demo image should run non-root where practical, expose health/readiness endpoints, and mount datasets/artifacts outside the image.
