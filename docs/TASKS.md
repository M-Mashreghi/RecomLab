# RecomLab — Task Progress Ledger

Update this ledger only after verification.

## Status Legend

- `[ ]` Pending
- `[-]` In Progress
- `[x]` Completed

## Phase 1 — Foundation, Data Contracts & Reproducibility

- [x] **Task 1.1** — Python Project Scaffold & Quality Tooling
  - Evidence: pytest tests/unit/test_smoke.py (4 passed: import, in-process CLI, `python -m recomlab smoke`, logging idempotence); ruff and mypy clean; editable install verified in a fresh Python 3.14 venv.
- [x] **Task 1.2** — Canonical User, Item & Interaction Schemas
  - Evidence: tests/unit/test_schemas.py (25 tests): tiny fixture validates; explicit and implicit examples; 14 invalid cases fail with stable codes; UTC normalization; duplicate policy. Full suite 29 passed; ruff/mypy clean.
- [x] **Task 1.3** — Impression, Exposure & Feedback Event Contract
  - Evidence: tests/unit/test_exposure.py (9 tests): explicit linkage verified incl. 3 invalid reasons; inference off by default; missing propensity counted not imputed; free text/PII rejected; schema version 1.0. Suite 38 passed; ruff/mypy clean.
- [x] **Task 1.4** — Configuration, Environment & Dependency Policy
  - Evidence: tests/unit/test_config.py (15 tests): defaults, file→env→override precedence, 9 invalid configurations rejected, unknown keys rejected, secret masked, .env.example placeholders-only, only config/settings.py reads env vars. Suite 53 passed; ruff/mypy clean.
- [x] **Task 1.5** — Dataset Registry, Data Cards & Licensing Metadata
  - Evidence: tests/unit/test_dataset_registry.py (6 tests): fixture materializes with identical checksums across runs and resolves deterministically; raw dir untouched; modified file detected; same version/different checksum rejected; MovieLens adapter tested on a synthetic in-test sample (no download). Suite 59 passed; ruff/mypy clean.
- [x] **Task 1.6** — Experiment Registry, Seed Control & Reproducibility Manifest
  - Evidence: tests/unit/test_experiments.py (6 tests): same config + fixture reproduces identical toy baseline result and config hash; 7 invalid manifests rejected by schema validation; SQLite registry reopened and compares runs; software state recorded. Suite 65 passed; ruff/mypy clean.

## Phase 2 — Dataset Protocol, Labels, Features & Baselines

- [x] **Task 2.1** — Interaction Validation & Cleaning Pipeline
  - Evidence: tests/unit/test_cleaning.py (6 tests): 11 known-bad rows yield 9 expected reason codes; report reconciles; no removal without reason; repeated and reversed-order runs give identical checksum. Suite 71 passed; ruff/mypy clean.
- [x] **Task 2.2** — Explicit-to-Implicit Label Policy & Relevance Semantics
  - Evidence: tests/unit/test_labels.py (7 tests): rating 4/3.99 and watch 0.8/0.79 boundaries; graded relevance; raw events untouched; any policy change changes label_version; policy round-trips through RunManifest and is required for evaluation. Suite 78 passed; ruff/mypy clean.
- [ ] **Task 2.3** — Negative Sampling Engine for Training
- [ ] **Task 2.4** — Leakage-Safe Train / Validation / Test Split Engine
- [ ] **Task 2.5** — Point-in-Time User, Item & Interaction Feature Builder
- [ ] **Task 2.6** — Random & Popularity Recommendation Baselines
- [ ] **Task 2.7** — Collaborative Filtering Baseline

## Phase 3 — Candidate Generation & Retrieval

- [ ] **Task 3.1** — Candidate Generator Interface & Retrieval Contract
- [ ] **Task 3.2** — Item-Item Similarity Candidate Generator
- [ ] **Task 3.3** — Collaborative Retrieval Candidate Generator
- [ ] **Task 3.4** — Embedding Retrieval & ANN Adapter
- [ ] **Task 3.5** — Candidate Fusion, Deduplication & Eligibility Filtering
- [ ] **Task 3.6** — Candidate Recall, Coverage & Failure Diagnostics

## Phase 4 — Ranking, Re-ranking & Cold-Start Policy

- [ ] **Task 4.1** — Ranker Interface & Scoring Contract
- [ ] **Task 4.2** — Heuristic / Weighted Feature Ranker
- [ ] **Task 4.3** — Collaborative Score Ranker
- [ ] **Task 4.4** — Learning-to-Rank Dataset Builder
- [ ] **Task 4.5** — Learning-to-Rank Baseline
- [ ] **Task 4.6** — Diversification, Calibration & Cold-Start Re-ranking Policy

## Phase 5 — Evaluation, Statistical Testing & Model Selection

- [ ] **Task 5.1** — Top-K Ranking Metrics Library
- [ ] **Task 5.2** — Beyond-Accuracy Metrics: Coverage, Diversity, Novelty, Calibration & Popularity Bias
- [ ] **Task 5.3** — Full-Catalog vs Sampled Evaluation Protocol
- [ ] **Task 5.4** — Slice Analysis for Cold, Sparse, Popularity & Activity Segments
- [ ] **Task 5.5** — Bootstrap Confidence Intervals & Paired Significance Testing
- [ ] **Task 5.6** — Exposure-Aware / Off-Policy Evaluation (Optional When Logged Propensities Exist)
- [ ] **Task 5.7** — Multi-Metric Model Selection & Champion Leaderboard
- [ ] **Task 5.8** — Evaluation Report & Reproducibility Manifest

## Phase 6 — API, Dashboard & Automated Recommendation Workflow

- [ ] **Task 6.1** — Recommendation Service & Model Registry
- [ ] **Task 6.2** — FastAPI Recommendation & Experiment API
- [ ] **Task 6.3** — Streamlit Experiment Dashboard & Recommendation Explorer
- [ ] **Task 6.4** — Impression & Feedback Logging Workflow
- [ ] **Task 6.5** — Scheduled Retraining & Data-Snapshot Workflow
- [ ] **Task 6.6** — Champion / Challenger Promotion Gate

## Phase 7 — LLM Explanations & Human-AI Research Extension

- [ ] **Task 7.1** — Recommendation Grounding Evidence Builder
- [ ] **Task 7.2** — Explanation Conditions: None, Template, Generic LLM & Contextualized LLM
- [ ] **Task 7.3** — Structured LLM Explanation Generator
- [ ] **Task 7.4** — Explanation Fidelity & Unsupported-Claim Validator
- [ ] **Task 7.5** — Human-AI Study Protocol, Randomization & Ethics Boundary
- [ ] **Task 7.6** — Trust, Acceptance, Usefulness & Satisfaction Analysis
- [ ] **Task 7.7** — Explanation Bias / Persuasiveness Stress Test & Research Report

## Phase 8 — Hardening, Monitoring, CI/CD & Release

- [ ] **Task 8.1** — End-to-End Regression Suite
- [ ] **Task 8.2** — Observability, Drift & Train-Serving Skew Checks
- [ ] **Task 8.3** — Dockerization & Reproducible Demo Stack
- [ ] **Task 8.4** — GitHub Actions CI & Quality Gates
- [ ] **Task 8.5** — Security, Privacy & Dependency Audit
- [ ] **Task 8.6** — Operational Runbook, Release Checklist & Portfolio Demo
