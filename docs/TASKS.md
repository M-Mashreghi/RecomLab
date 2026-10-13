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
- [x] **Task 2.3** — Negative Sampling Engine for Training
  - Evidence: tests/unit/test_negative_sampling.py (7 tests): no negative is a training-horizon positive (uniform and popularity); post-horizon events have no influence; fixed seed reproduces samples regardless of input order; sparse users, cold users and exhausted small catalogs covered. Suite 85 passed; ruff/mypy clean.
- [x] **Task 2.4** — Leakage-Safe Train / Validation / Test Split Engine
  - Evidence: tests/unit/test_splits.py (10 tests, marked leakage): per-user and global temporal order; same dataset+config → same split_id; no test event in train/validation for all 3 protocols; repeat interactions dropped not moved; random split labeled diagnostic; injected leaks raise LeakageError; boundaries persisted. Suite 95 passed; ruff/mypy clean.
- [x] **Task 2.5** — Point-in-Time User, Item & Interaction Feature Builder
  - Evidence: tests/unit/test_features.py (7 tests): hand-computed point-in-time counts, profiles and recency decay; injected future events do not change features and strict mode raises; assert_point_in_time fails when a future event is used; split train-only aggregates match; feature manifest round-trips in RunManifest. Suite 102 passed; ruff/mypy clean.
- [x] **Task 2.6** — Random & Popularity Recommendation Baselines
  - Evidence: tests/unit/test_baselines.py (6 tests): deterministic popularity top-K with id tie-breaks; popularity counts equal training-only counts; seeded random independent of call order; eligibility filters; both baselines logged to the SQLite registry with coverage, latency and metrics; rerun reproduces metrics. Suite 108 passed; ruff/mypy clean.
- [x] **Task 2.7** — Collaborative Filtering Baseline
  - Evidence: tests/unit/test_als.py (5 tests): ALS recovers the held-back cluster item where popularity fails; seeded/deterministic factors; artifact save/load with version check (serving separate from training); label-policy feedback and params recorded; evaluated under the same protocol as popularity in the registry. Suite 113 passed; ruff/mypy clean.

## Phase 3 — Candidate Generation & Retrieval

- [x] **Task 3.1** — Candidate Generator Interface & Retrieval Contract
  - Evidence: tests/unit/test_candidate_contract.py (6 tests): budget enforced for several budgets and invalid budgets rejected; unique items with deterministic ties; validator catches duplicates/over-budget/mixed sources; generate() takes only the request and fit() rejects future history; multiple generators supported. Suite 119 passed; ruff/mypy clean.
- [x] **Task 3.2** — Item-Item Similarity Candidate Generator
  - Evidence: tests/unit/test_item_item.py (6 tests): hand-checked cosine neighbors; recency-weighted aggregation and seed-item evidence; explicit cold-user fallback flag; train-only fit differs from full-data fit and never contains held-out pairs; metadata blend; tiny fixture cluster neighbors. Suite 125 passed; ruff/mypy clean.
- [x] **Task 3.3** — Collaborative Retrieval Candidate Generator
  - Evidence: tests/unit/test_collaborative_candidates.py (5 tests): consumed-item filtering on/off; score order and model version exposed; batch and single retrieval agree (items and scores) incl. unknown-user fallback; artifact injection matches fit; candidate recall evaluated independently of ranking. Suite 130 passed; ruff/mypy clean.
- [x] **Task 3.4** — Embedding Retrieval & ANN Adapter
  - Evidence: tests/unit/test_embedding_retrieval.py (6 tests): exact cosine/dot neighbors match hand-computed values with id tie-breaks; category and ALS embedding providers retrieve on the tiny fixture; stale index vs new ALS artifact raises IndexVersionMismatch; no ANN/vector-DB module imported. Suite 136 passed; ruff/mypy clean.
- [x] **Task 3.5** — Candidate Fusion, Deduplication & Eligibility Filtering
  - Evidence: tests/unit/test_fusion.py (5 tests): RRF and weighted-score fusion match hand computation; provenance lists every contributing source; pre/post-filter counts and removal reasons recorded; output identical under reversed source order; fused real generators never contain duplicates or ineligible items under a budget. Suite 141 passed; ruff/mypy clean.
- [x] **Task 3.6** — Candidate Recall, Coverage & Failure Diagnostics
  - Evidence: tests/unit/test_retrieval_diagnostics.py (5 tests): hand-checkable recall, hit rate, empty rate, count distribution, segment and source breakdowns; failure breakdown separates retrieval vs ranking misses; fused tiny-fixture diagnostics written to JSON and stored in a validated RunManifest. Suite 146 passed; ruff/mypy clean.

## Phase 4 — Ranking, Re-ranking & Cold-Start Policy

- [x] **Task 4.1** — Ranker Interface & Scoring Contract
  - Evidence: tests/unit/test_ranker_contract.py (4 tests): ordering with id tie-breaks; scoring a non-candidate raises and outputs never exceed candidates or k; requests reject duplicates and carry no label fields; ranked lists round-trip through JSON without presentation text. Suite 150 passed; ruff/mypy clean.
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
