# RecomLab — Recommendation & Ranking Engine

**RecomLab** is a research-grade recommender-system platform. Given users, items and their
interactions (ratings, clicks, purchases, …), it **retrieves candidate items, ranks them,
serves top-K recommendations through an API and a dashboard, and measures — honestly — how
good those recommendations are**. An optional research track adds **grounded LLM
explanations** ("why was this recommended?") and the tooling to study how people react to
them.

It goes beyond "train a model and report NDCG": the pipeline guards against data leakage,
evaluates retrieval and ranking separately, reports uncertainty, handles cold-start users and
items, and only promotes a new model when predefined quality guardrails pass.

> The repository ships with a tiny **simulated** dataset (7 users, 12 items) so everything
> runs offline in seconds. Plug in MovieLens or your own data for real results.

---

## Table of Contents

- [What it does](#what-it-does)
- [What you get out of it (outputs)](#what-you-get-out-of-it-outputs)
- [Quickstart](#quickstart)
- [How to use it](#how-to-use-it)
- [Use cases](#use-cases)
- [Bring your own data](#bring-your-own-data)
- [Architecture](#architecture)
- [Research integrity principles](#research-integrity-principles)
- [Tech stack](#tech-stack)
- [Documentation](#documentation)
- [Limitations & roadmap](#limitations--roadmap)

---

## What it does

```text
Raw users / items / interactions / impressions
   → validation & cleaning (data contract, reason codes)
   → label policy (what counts as "relevant") + leakage-safe train/validation/test split
   → point-in-time features (no future information)
   → candidate retrieval: popularity · item-to-item similarity · collaborative (ALS) · embeddings
   → fusion + eligibility filtering (unavailable / already-consumed items removed)
   → ranking: heuristic · collaborative · learning-to-rank
   → re-ranking: diversity (MMR) · calibration · cold-start fallbacks
   → top-K recommendations  ──►  API + dashboard  ──►  impression & feedback log
   → offline evaluation + statistics  →  champion/challenger promotion gate  →  retraining
```

| Capability | Details |
|---|---|
| **Recommenders** | Random & popularity baselines, implicit-feedback ALS (matrix factorization), item-item similarity, embedding retrieval (exact search, optional ANN adapter), heuristic ranker, collaborative ranker, pairwise learning-to-rank |
| **Evaluation** | Precision/Recall/NDCG/MAP/MRR/HitRate@K, coverage, diversity, novelty, popularity bias (Gini), calibration, candidate recall, cold/sparse/head-tail slices |
| **Statistics** | User-level bootstrap confidence intervals, paired comparisons (bootstrap + Wilcoxon), multiple-comparison correction |
| **Protocols** | Full-catalog evaluation by default; sampled evaluation only as a clearly labeled diagnostic; optional IPS/SNIPS off-policy evaluation when real propensities are logged |
| **Operations** | Model registry with fail-closed activation and rollback, retraining workflow, promotion gate, structured logs, drift & train/serving-skew checks |
| **Explanations (optional)** | Evidence payload traced to real data, template and LLM explanations, fidelity validator that rejects unsupported claims, Human-AI study design & analysis tools |

---

## What you get out of it (outputs)

**1. Ranked recommendations (API / Python).** For each user: top-K item ids with scores, the
retrieval sources that proposed each item, and the per-feature contributions behind the score.
Real response from the demo (`POST /recommendations` with `{"user_id": "u2", "k": 3}`):

```json
{
  "trace_id": "demo-1",
  "user_id": "u2",
  "bundle_id": "bundle-28a81ba1d1cf",
  "items": [
    {
      "item_id": "i07", "rank": 1, "score": 0.614, "fallback": null,
      "sources": {"collaborative": {"rank": 1}, "item_item": {"rank": 1}, "popularity": {"rank": 7}},
      "contributions": {"retrieval": 0.4, "popularity": 0.155, "recency": 0.043, "content": 0.017}
    },
    {"item_id": "i08", "rank": 2, "score": 0.579, "...": "..."},
    {"item_id": "i09", "rank": 3, "score": 0.565, "...": "..."}
  ],
  "metadata": {"n_candidates": 5, "candidate_counts": {"pre_dedup": 22, "post_dedup": 12, "post_filter": 5, "final": 5}}
}
```

**2. Evaluation reports (Markdown + JSON).** Metrics per model with 95% confidence intervals,
paired comparisons against a baseline, retrieval diagnostics, slice results, and automatic
caveats (e.g. "sampled evaluation", "simulated data"). Example:
[`docs/reports/benchmark_tiny.md`](docs/reports/benchmark_tiny.md).

**3. Experiment registry (SQLite).** Every run stores a reproducibility manifest: dataset
version + checksum, split id, label policy, model parameters, metrics, seeds, software versions
and git commit — so any number in a report can be traced and regenerated.

**4. Interactive dashboard (Streamlit).** Experiment leaderboard, run details
(candidate-stage vs ranking-stage metrics), a recommendation explorer per user, and
cold-start / beyond-accuracy views.

**5. Deployable model bundles.** Versioned, checksummed bundles (pipeline config + trained
artifacts) that can be activated, swapped and rolled back without downtime logic in your code.

**6. Promotion decisions & monitoring.** JSON decision records ("promote / reject, and why"),
`/monitoring` serving metrics, and drift/skew warnings.

**7. (Optional) Explanations & study reports.** Grounded explanation text with cited evidence,
fidelity verdicts, and research reports for explanation experiments.

---

## Quickstart

Requires Python 3.12+.

**Windows (PowerShell)**

```powershell
cd "D:\path\to\RecomLab"
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,api,dashboard]"
recomlab bootstrap-demo                        # train, evaluate and deploy the demo model
uvicorn recomlab.api.main:app --port 8000      # API  → http://127.0.0.1:8000/docs
streamlit run src/recomlab/dashboard/app.py    # UI   → http://localhost:8501
```

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,dashboard]"
recomlab bootstrap-demo
uvicorn recomlab.api.main:app --port 8000
streamlit run src/recomlab/dashboard/app.py
```

**Docker:** `docker compose up --build` (API on :8000, dashboard on :8501).

Check everything works: `pytest` (300+ deterministic tests, no internet needed) or
`sh scripts/check.sh` (lint + format + type check + tests).

---

## How to use it

### 1. Call the HTTP API

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
     -H "Content-Type: application/json" \
     -d '{"user_id": "u1", "k": 5}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /ready` | liveness / is a model deployed |
| `POST /recommendations` | top-K recommendations for a user |
| `GET /experiments/runs`, `GET /experiments/runs/{id}` | browse experiment results |
| `GET /models/bundles`, `GET /models/active` | deployed model info |
| `POST /models/activate` | switch model (needs `X-Admin-Token` = `RECOMLAB_API_ADMIN_TOKEN`) |
| `GET /monitoring` | latency, empty-candidate rate, fallbacks |

Interactive docs: `http://127.0.0.1:8000/docs`.

### 2. Use it as a Python library

```python
from pathlib import Path

from recomlab.datasets import FixtureAdapter
from recomlab.datasets.labels import LabelPolicy
from recomlab.datasets.splits import SplitConfig, make_split
from recomlab.experiments.runner import run_offline_experiment
from recomlab.models import PopularityRecommender
from recomlab.models.als import ALSParams, ALSRecommender

data = FixtureAdapter().load(Path("data/fixtures/tiny"))      # or your own CSV folder
split = make_split(data.interactions, SplitConfig())          # leave-last-1-out, leakage-checked

for model in (PopularityRecommender(), ALSRecommender(ALSParams(factors=8))):
    manifest, result = run_offline_experiment(
        experiment_id="my-experiment", model=model, dataset=data,
        dataset_ref={"dataset_id": "tiny", "version": "1"}, split=split,
        label_policy=LabelPolicy(), ks=(5, 10), partition="validation",
    )
    print(model.name, {k: round(v, 3) for k, v in result.metrics.items() if "@5" in k})
```

### 3. Command line

| Command | What it does |
|---|---|
| `recomlab smoke` | installation check |
| `recomlab bootstrap-demo` | snapshot data, train, evaluate and deploy the demo bundle |
| `recomlab retrain --spec configs/demo_bundle.json` | retrain a *challenger* (never auto-deployed) |
| `python scripts/run_benchmark.py` | regenerate the benchmark report |
| `python scripts/fetch_movielens.py` | download MovieLens into `data/raw/` (not committed) |
| `python scripts/scan_secrets.py` | scan the repo for committed secrets |

### 4. Configure

All settings come from `RECOMLAB_*` environment variables (see [`.env.example`](.env.example)):
dataset id, evaluation K values and mode, database path, optional LLM provider and model.
The pipeline itself (which retrievers, which ranker, candidate budget, diversity, cold-start
slots) is a small JSON spec — see [`configs/demo_bundle.json`](configs/demo_bundle.json).

---

## Use cases

| Domain | How RecomLab fits |
|---|---|
| **E-commerce / retail** | "Customers like you also bought…" — map purchases/add-to-cart to interactions, use item categories for diversity, popularity fallback for new visitors, and the promotion gate to ship new models safely. |
| **Movies, music, video, podcasts** | Ratings or watch completion as feedback (configurable thresholds), item-item similarity for "because you watched…", calibration so recommendations match each user's genre mix. |
| **News, blogs, content feeds** | Time-aware splits and recency features for fast-changing catalogs; impression logging to learn from what users were *shown*, not only what they clicked. |
| **Online courses / e-learning** | Recommend next courses from enrollments/completions; slice analysis to check new learners (cold-start) get useful suggestions. |
| **Jobs, real estate, marketplaces** | Eligibility rules (unavailable listings, already applied) are first-class; diversity re-ranking avoids showing near-duplicates. |
| **Research & teaching** | A reference implementation of correct offline evaluation: leakage-safe splits, full-catalog vs sampled metrics, uncertainty, and reproducible manifests — useful for papers, theses and courses. |
| **Explainable AI / Human-AI studies** | Compare no explanation vs template vs LLM explanations, verify explanation fidelity, randomize conditions, and analyze trust/acceptance with proper statistics (simulated data for development; real studies need ethics approval). |
| **ML engineering portfolio / interviews** | End-to-end system design: data contracts, two-stage retrieval & ranking, model registry, CI, Docker, monitoring and runbooks. |

---

## Bring your own data

Provide three CSV files (see [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md)):

```text
users.csv         user_id[, non-sensitive attributes]
items.csv         item_id, title, categories (a|b|c), available (true/false)
interactions.csv  event_id, user_id, item_id, event_type, value, event_time (ISO-8601 with timezone), session_id
```

`event_type` is one of `rating`, `click`, `view`, `watch`, `like`, `add_to_cart`, `purchase`.
What counts as "relevant" (e.g. rating ≥ 4, watched ≥ 80%) is a versioned **label policy**, not
hard-coded. Put the three files in `data/fixtures/<your_id>/` (or
`data/processed/<your_id>/<version>/`), set `RECOMLAB_DATASET_ID=<your_id>`, and run
`recomlab bootstrap-demo` — or load the folder in Python with `FixtureAdapter` and use
`python scripts/run_benchmark.py --data <folder> --dataset-id <your_id>`. A MovieLens adapter is
included.

Privacy by design: pseudonymous ids only; columns like `name`, `email`, `phone` are rejected;
free-text feedback is not stored.

---

## Architecture

```text
src/recomlab/
├── domain/        canonical records, exposure/feedback contract, eligibility rules
├── ingestion/     CSV loading, validation & cleaning with reason codes
├── datasets/      adapters, registry & data cards, labels, splits, negative sampling
├── features/      point-in-time feature builder
├── candidates/    retrieval generators, embeddings/ANN adapter, fusion
├── models/        baselines and ALS
├── rankers/       heuristic, collaborative, learning-to-rank
├── reranking/     diversity, calibration, cold-start policy
├── evaluation/    metrics, protocols, slices, statistics, off-policy, leaderboard, reports
├── experiments/   run manifests, SQLite registry, seeds, retraining workflow
├── registry/      model bundles, activation/rollback, promotion gate
├── services/      recommendation pipeline & service, event log
├── api/           FastAPI app          dashboard/   Streamlit app
├── monitoring/    structured logs, drift & skew checks
├── explanations/  evidence, conditions, LLM generator, fidelity
└── research/      Human-AI study design, analysis, persuasion stress test
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Human-AI research track:
[`docs/HUMAN_AI_STUDY.md`](docs/HUMAN_AI_STUDY.md) — the LLM **never changes the ranking**.

---

## Research integrity principles

- No test-set or future-event leakage (enforced by tests that run in CI).
- Training negative sampling is never used as evaluation truth.
- Candidate recall and ranking quality are reported separately.
- Full-catalog evaluation is the default; sampled metrics are clearly labeled and never mixed.
- Models are selected on validation data; the test set is reported once.
- Statistical uncertainty is reported for model comparisons.
- Relevance, diversity, explanation fidelity, trust and persuasion are separate outcomes.
- Simulated study data is always labeled; real participant recruitment requires institutional approval.

---

## Tech stack

Python 3.12+, NumPy (the only mandatory dependency), FastAPI, Streamlit, SQLite, pytest, ruff,
mypy, Docker, GitHub Actions. Optional: `anthropic` for LLM explanations, `hnswlib` for ANN
search. Models are implemented in pure NumPy, so no GPU or heavy ML framework is needed.

---

## Documentation

| Topic | File |
|---|---|
| Architecture & module contracts | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Data contract | [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) |
| Evaluation protocol | [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) |
| Benchmark report | [`docs/reports/benchmark_tiny.md`](docs/reports/benchmark_tiny.md) |
| Human-AI study extension | [`docs/HUMAN_AI_STUDY.md`](docs/HUMAN_AI_STUDY.md) |
| Operations runbook | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) |
| Deployment, Docker & CI | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| Security & privacy | [`docs/SECURITY_PRIVACY.md`](docs/SECURITY_PRIVACY.md) |
| Release checklist | [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) |
| Project plan (52 tasks, with test evidence) | [`docs/PROJECT_PHASES_AND_TASKS.md`](docs/PROJECT_PHASES_AND_TASKS.md) |

---

## Limitations & roadmap

**v1 limitations**

- The in-repo dataset is a 7-user simulated fixture; its numbers validate the software, not
  recommender quality. Use MovieLens or your own data for meaningful results.
- Offline metrics reflect what users were exposed to; off-policy correction needs real logged
  propensities.
- Pure-NumPy models target small-to-medium data; ANN and boosted rankers are optional adapters.
- The fidelity validator is rule-based (titles, categories, history, popularity, ungroundable
  claims) and does not understand arbitrary free-form text.
- The study module designs, simulates and analyzes experiments; it does not recruit participants.

**Post-v1 backlog:** session/sequential recommendation, contextual bandits, slate
optimization, graph neural recommenders, multimodal item encoders, large-scale ANN
benchmarking, advanced causal estimators, federated recommendation
(see [`docs/REVIEW_NOTES_v2.md`](docs/REVIEW_NOTES_v2.md)).
