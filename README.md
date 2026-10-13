# RecomLab — Recommendation & Ranking Engine

**RecomLab** is a research-oriented recommender-system platform that separates **candidate retrieval**, **ranking**, **offline evaluation**, **serving**, and an optional **human-centered LLM explanation** research track.

It is designed as a portfolio project that demonstrates more than “train a recommender and report NDCG.” The project explicitly handles temporal leakage, exposure logs, negative sampling, candidate recall, full-catalog versus sampled evaluation, uncertainty in metric differences, cold-start behavior, model promotion, and explanation fidelity.

## Core Workflow

```text
Raw User / Item / Interaction / Impression Data
                    ↓
          Validation + Data Contract
                    ↓
      Label Policy + Leakage-Safe Split
                    ↓
       Point-in-Time Feature Builder
                    ↓
          Candidate Generation
                    ↓
    Fusion + Eligibility Filtering
                    ↓
                Ranking
                    ↓
        Diversification / Fallback
                    ↓
          Top-K Recommendations
                    ↓
    Offline Evaluation + Statistics
                    ↓
       Champion / Challenger Gate
                    ↓
           API + Dashboard
                    ↓
   Impression + Feedback Logging Loop
```

## Human-AI Research Extension

```text
Recommendation
      ↓
Grounding Evidence
      ↓
Explanation Condition
None / Template / Generic LLM / Contextualized LLM
      ↓
Fidelity Validation
      ↓
Controlled Experiment
      ↓
Trust / Acceptance / Usefulness / Satisfaction
      ↓
Statistical Analysis + Research Report
```

The LLM **never changes the ranking** in this track.

## Recommended Demo Data

Start with a public MovieLens dataset through a dataset adapter and keep a tiny synthetic fixture in-repo for CI. A second dataset can be added only after the full pipeline is stable.

## Suggested Stack

- Python 3.12+
- pandas or Polars
- NumPy / SciPy
- scikit-learn
- PyTorch when needed
- optional `implicit` for collaborative baselines
- optional LightGBM/XGBoost for learning-to-rank
- FastAPI
- Streamlit
- SQLite + Parquet / DuckDB
- pytest
- Docker
- GitHub Actions

Heavy dependencies are optional adapters, not architectural requirements.

## Research Integrity Principles

- No test-set or future-event leakage.
- Training negative sampling is not the same thing as evaluation sampling.
- Candidate recall and ranking quality are reported separately.
- Full-catalog evaluation is preferred when computationally feasible.
- Sampled offline metrics must be clearly labeled.
- Offline relevance, diversity, fidelity, trust, and persuasion are different outcomes.
- Statistical uncertainty is reported for model comparisons.
- Human-participant recruitment is outside the software workflow unless appropriate institutional approval exists.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,api,dashboard]"
recomlab smoke                 # install check
sh scripts/check.sh            # ruff + format + mypy + full test suite
recomlab bootstrap-demo        # train, evaluate and deploy the demo bundle (simulated fixture)
uvicorn recomlab.api.main:app --port 8000
streamlit run src/recomlab/dashboard/app.py
```

Or with Docker: `docker compose up --build` (API on :8000, dashboard on :8501).

Mandatory dependencies are kept minimal (NumPy only). API, dashboard and LLM libraries are
optional extras (`.[api]`, `.[dashboard]`, `.[llm]`).

## Documentation

- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · data contract:
  [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)
- Evaluation protocol: [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)
- Benchmark / evaluation report: [`docs/reports/benchmark_tiny.md`](docs/reports/benchmark_tiny.md)
  (regenerate with `python scripts/run_benchmark.py`; simulated fixture)
- Human-AI study extension: [`docs/HUMAN_AI_STUDY.md`](docs/HUMAN_AI_STUDY.md)
- Operations: [`docs/RUNBOOK.md`](docs/RUNBOOK.md) · [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) ·
  [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md) ·
  [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)

## v1 Limitations

- The in-repo dataset is a 7-user simulated fixture; its benchmark numbers validate the
  software, not recommender quality. Run the MovieLens adapter for meaningful results.
- Offline metrics reflect logged exposure; off-policy correction is available only when
  impression logs carry real propensities.
- Pure-NumPy models (ALS, linear pairwise LTR, exact retrieval) target portfolio-scale data;
  ANN and boosted rankers are optional adapters, not included by default.
- The explanation fidelity validator is rule-based (titles, categories, history, popularity,
  ungroundable claims); it does not parse arbitrary free-form semantics.
- The Human-AI study module supports design, randomization, simulation and analysis only; it
  does not recruit participants.

## Post-v1 Research Backlog

Session/sequential recommendation, contextual bandits, slate optimization, graph neural
recommenders, multimodal item encoders, large-scale ANN benchmarking, causal/counterfactual
estimators beyond IPS/SNIPS, and federated recommendation (see `docs/REVIEW_NOTES_v2.md`).

## Project Plan

The implementation is split into **8 phases and 52 atomic tasks**. See `docs/PROJECT_PHASES_AND_TASKS.md` (status and verification evidence per task) and `docs/TASKS.md`.

## Start with Claude Code

```text
Read CLAUDE.md and all project governance documents.

Inspect Task 1.1 in:
docs/PROJECT_PHASES_AND_TASKS.md

Implement only Task 1.1.
Follow its Definition of Done.
Run all required tests.
Update TASKS.md and PROJECT_PHASES_AND_TASKS.md only after verification succeeds.
Do not start Task 1.2.
```
