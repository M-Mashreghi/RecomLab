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
- [x] **Task 4.2** — Heuristic / Weighted Feature Ranker
  - Evidence: tests/unit/test_heuristic_ranker.py (5 tests): hand-computed fixture scores and order; single-feature weights reorder predictably; contributions sum to the raw score; future events do not change the ranking and a contaminated snapshot is refused; weights and feature manifest exposed for the run manifest. Suite 155 passed; ruff/mypy clean.
- [x] **Task 4.3** — Collaborative Score Ranker
  - Evidence: tests/unit/test_collaborative_ranker.py (6 tests): ranking equals ALS score order; unknown user falls back to deterministic popularity regardless of input order; unknown items ranked below known ones; a spy confirms only candidate ids reach the model; batch equals single scoring with latency recorded. Suite 161 passed; ruff/mypy clean.
- [x] **Task 4.4** — Learning-to-Rank Dataset Builder
  - Evidence: tests/unit/test_ltr_data.py (5 tests): group sizes reconcile with candidate rows and unique items per group; history-after-as_of and target-before-as_of raise LeakageError; training rows contain no validation/test target pairs and forbidden ids are refused; schema/feature manifest recorded. Suite 166 passed; ruff/mypy clean.
- [x] **Task 4.5** — Learning-to-Rank Baseline
  - Evidence: tests/unit/test_ltr.py (3 tests): deterministic train/eval smoke on the tiny fixture; grid search selects by validation NDCG and test-tagged data is refused for tuning and training; LTR, heuristic and collaborative rankers compared on identical validation candidates, serving scores equal dataset scores. Suite 169 passed; ruff/mypy clean.
- [x] **Task 4.6** — Diversification, Calibration & Cold-Start Re-ranking Policy
  - Evidence: tests/unit/test_reranking.py (5 tests): MMR increases ILD without adding items; calibration follows the profile and records skips; MMR/calibration on real fused candidates never leave the eligible candidate set; cold-user and cold-item fallbacks are non-empty, eligible, deterministic and labeled; before/after relevance and beyond-accuracy report with deltas. Suite 174 passed; ruff/mypy clean.

## Phase 5 — Evaluation, Statistical Testing & Model Selection

- [x] **Task 5.1** — Top-K Ranking Metrics Library
  - Evidence: tests/unit/test_metrics.py (5 tests, marked metrics): Precision/Recall/HitRate/MAP/MRR and binary + linear/exponential graded NDCG match hand computations; edge cases (no positives, k<=0, empty/short lists); aggregation excludes and reports users without positives; metric config round-trips in RunManifest. Suite 179 passed; ruff/mypy clean.
- [x] **Task 5.2** — Beyond-Accuracy Metrics: Coverage, Diversity, Novelty, Calibration & Popularity Bias
  - Evidence: tests/unit/test_beyond_accuracy.py (3 tests, marked metrics): coverage, ILD, novelty, Gini, head share and calibration KL match deterministic hand-computed fixture values; helper edge cases; undefined metrics reported as None with reasons rather than zero. Suite 182 passed; ruff/mypy clean.
- [x] **Task 5.3** — Full-Catalog vs Sampled Evaluation Protocol
  - Evidence: tests/unit/test_eval_protocol.py (4 tests, marked metrics): full-catalog evaluation deterministic; sampled mode labeled with warning, seed, sampler, n_negatives and universe size, reproducible; full vs sampled and different sampling policies raise MixedEvaluationModeError; training negatives refused as universe. Suite 186 passed; ruff/mypy clean.
- [x] **Task 5.4** — Slice Analysis for Cold, Sparse, Popularity & Activity Segments
  - Evidence: tests/unit/test_slices.py (3 tests, marked metrics): user and item bucket memberships deterministic under reordering; slice counts reconcile with the evaluation population and item-bucket relevant counts with held-out truth; slice report id stored in the run manifest artifacts. Suite 189 passed; ruff/mypy clean.
- [x] **Task 5.5** — Bootstrap Confidence Intervals & Paired Significance Testing
  - Evidence: tests/unit/test_statistics.py (5 tests, marked metrics): CI brackets the mean and the same seed reproduces it; synthetic clearly-better/worse/noise comparisons classified better/worse/uncertain; paired intervals reproducible with stored seed and resamples; Wilcoxon statistic and p-value hand-checked incl. ties; mismatched populations rejected. Suite 194 passed; ruff/mypy clean.
- [x] **Task 5.6** — Exposure-Aware / Off-Policy Evaluation (Optional When Logged Propensities Exist)
  - Evidence: tests/unit/test_off_policy.py (3 tests, marked metrics): IPS, SNIPS and ESS match a hand-computed fixture using only linked rewards; extreme weights and clipping are flagged; logs with a missing propensity or no impressions are refused. Suite 197 passed; ruff/mypy clean.
- [x] **Task 5.7** — Multi-Metric Model Selection & Champion Leaderboard
  - Evidence: tests/unit/test_leaderboard.py (5 tests): test runs refused for selection; decision records policy id, rationale, metrics and guardrails, test report attached only afterwards; decision identical when rebuilt from registry manifests or reordered; failing/missing guardrails block a champion; Pareto front hand-checked and mixed full/sampled runs refused. Suite 202 passed; ruff/mypy clean.
- [x] **Task 5.8** — Evaluation Report & Reproducibility Manifest
  - Evidence: tests/integration/test_evaluation_report.py (3 tests): report rebuilt from a reopened registry and stored per-user artifacts is byte-identical (Markdown and JSON); contains protocol, ranking-stage CIs, separate retrieval section, paired comparison, slice links, limitations and simulated/exposure caveats; sampled runs carry the sampled warning. Suite 205 passed; ruff/mypy clean.

## Phase 6 — API, Dashboard & Automated Recommendation Workflow

- [x] **Task 6.1** — Recommendation Service & Model Registry
  - Evidence: tests/integration/test_recommendation_service.py (4 tests): heuristic→collaborative bundle swap, rollback and restart-from-registry; tampered artifact, catalog mismatch and invalid spec fail closed with the active bundle untouched; service output equals the manually assembled offline pipeline for warm and cold users; responses carry serializable provenance. Suite 209 passed; ruff/mypy clean.
- [x] **Task 6.2** — FastAPI Recommendation & Experiment API
  - Evidence: tests/integration/test_api.py (11 tests): health/readiness; recommendation contract incl. no-active-model 503 and evidence stripping; 4 invalid payloads return structured validation errors; experiment list/detail/404; activation requires the admin token, unknown bundle keeps the active model, mutation disabled without a configured token; no secrets in responses; app builds from Settings. Suite 220 passed; ruff/mypy clean.
- [x] **Task 6.3** — Streamlit Experiment Dashboard & Recommendation Explorer
  - Evidence: tests/integration/test_dashboard.py (7 tests): all four Streamlit pages load without exceptions via streamlit AppTest on fixture data; view models built from the experiment registry and service only (source check forbids fit/training/metric calls); no user attributes in rendered data. Suite 227 passed; ruff/mypy clean.
- [x] **Task 6.4** — Impression & Feedback Logging Workflow
  - Evidence: tests/integration/test_event_log.py (3 tests): re-logging a served slate and feedback is idempotent while conflicting ids are rejected; linked/unlinked feedback queryable per request and convertible to interactions; stored impressions are contract Impressions and propensities stay missing unless supplied. Suite 230 passed; ruff/mypy clean.
- [x] **Task 6.5** — Scheduled Retraining & Data-Snapshot Workflow
  - Evidence: tests/integration/test_retraining.py (4 tests): end-to-end workflow on the demo fixture freezes a verified snapshot (base + logged events), logs a validation run and registers a challenger linking workflow, dataset version and run id without activation; snapshots deterministic; a failing spec logs a failed run and leaves the active bundle untouched; `recomlab retrain` CLI works. Suite 234 passed; ruff/mypy clean.
- [x] **Task 6.6** — Champion / Challenger Promotion Gate
  - Evidence: tests/integration/test_promotion.py (5 tests): primary improvement rule with configurable minimum; failing coverage, slice and missing-metric guardrails each block promotion; stored decision replays identically from registry + artifacts; mismatched split and test runs refused; blocked decision keeps the champion, promotion activates challenger and rollback restores the previous champion with reason. Suite 239 passed; ruff/mypy clean.

## Phase 7 — LLM Explanations & Human-AI Research Extension

- [x] **Task 7.1** — Recommendation Grounding Evidence Builder
  - Evidence: tests/unit/test_evidence.py (5 tests): payload schema with ranking vs presentation roles; every entry traces to stored interactions/catalog and seed items stay as ranking evidence; unknown facts listed as unavailable rather than synthesized; fabricated, other-user and wrong-fact evidence rejected; history window bounds the payload. Suite 244 passed; ruff/mypy clean.
- [x] **Task 7.2** — Explanation Conditions: None, Template, Generic LLM & Contextualized LLM
  - Evidence: tests/unit/test_explanation_conditions.py (5 tests): four versioned condition ids and unknown ids rejected; router dispatches none/template/generic/contextual to the requested strategy with template fallback when no LLM is configured; template deterministic, cites payload evidence and imports no LLM module; generic sees only item facts and non-personal features while contextual sees history and provenance; template abstains without evidence. Suite 249 passed; ruff/mypy clean.
- [x] **Task 7.3** — Structured LLM Explanation Generator
  - Evidence: tests/unit/test_llm_explainer.py (10 tests): valid mocked output accepted with prompt/model version logged; malformed output retried; 4 invalid/unsupported outputs fall back to the template after bounded retries; abstain respected; transient vs permanent provider errors; a failing provider leaves served recommendations unchanged. Suite 258 passed; ruff/mypy clean; no live API calls.
- [x] **Task 7.4** — Explanation Fidelity & Unsupported-Claim Validator
  - Evidence: tests/unit/test_fidelity.py (10 tests): template and hand-written grounded explanations accepted; synthetic hallucinations (unknown item, category, popularity, fabricated citation, history claim for a cold user) rejected; threshold behavior; feature ablation on the heuristic ranker reports score/rank effects and flags no-effect features; fidelity log refuses perception records. Suite 267 passed; ruff/mypy clean.
- [x] **Task 7.5** — Human-AI Study Protocol, Randomization & Ethics Boundary
  - Evidence: tests/unit/test_study.py (6 tests): block randomization deterministic under the test seed and balanced within one block; config versioned and validated; real mode blocked without approved ethics, consent and retention, and simulation refused in real mode; simulated records cannot enter a real store or masquerade as real ids; exclusions auditable; power helper (d=0.5 → 63/group). Suite 273 passed; ruff/mypy clean.
- [x] **Task 7.6** — Trust, Acceptance, Usefulness & Satisfaction Analysis
  - Evidence: tests/unit/test_study_analysis.py (4 tests): simulated known effect (+1.0) recovered with CI above zero and Holm p<0.001 while a null effect's CI spans zero; exclusions reconcile per condition and descriptive n equals analyzed n; offline metrics refused as outcomes; t-distribution p-values and Holm adjustment hand-checked. Suite 277 passed; ruff/mypy clean.
- [x] **Task 7.7** — Explanation Bias / Persuasiveness Stress Test & Research Report
  - Evidence: tests/unit/test_persuasion.py (3 tests): wording variants require an explicitly enabled research mode and never alter production/template explanations; simulated stress test shows persuasive variants raising acceptance while failing fidelity, and the report separates persuasion from correctness with a SIMULATED banner; free text sanitized before (spy) coding and in the CSV export. Suite 280 passed; ruff/mypy clean.

## Phase 8 — Hardening, Monitoring, CI/CD & Release

- [ ] **Task 8.1** — End-to-End Regression Suite
- [ ] **Task 8.2** — Observability, Drift & Train-Serving Skew Checks
- [ ] **Task 8.3** — Dockerization & Reproducible Demo Stack
- [ ] **Task 8.4** — GitHub Actions CI & Quality Gates
- [ ] **Task 8.5** — Security, Privacy & Dependency Audit
- [ ] **Task 8.6** — Operational Runbook, Release Checklist & Portfolio Demo
