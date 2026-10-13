# RecomLab — Project Phases & Atomic Tasks
**Single source of truth** for implementation scope and completion.
Status legend: `[ ]` Not Started · `[-]` In Progress · `[x]` Completed
## Global Definition of Done
A task may be marked complete only when:
1. Its scoped implementation exists and no unrelated work was added.
2. Required unit/integration tests pass.
3. Relevant leakage/reproducibility invariants pass.
4. No secrets or unnecessary PII are committed.
5. `TASKS.md` and relevant docs are synchronized.
6. New research results are reproducible from a stored manifest.
7. Optional human-subject features do not imply institutional approval or real-participant recruitment.

---

## Phase 1 — Foundation, Data Contracts & Reproducibility

### Task 1.1: Python Project Scaffold & Quality Tooling

**Status:** `[x]` Completed

**Business / Research Value**  
Create a small, testable Python package with consistent tooling so every later recommender experiment runs from the same reproducible foundation.

**Technical Specification**
- Create `src/recomlab`, `tests/unit`, `tests/integration`, `scripts`, `data`, and `artifacts` layout.
- Add `pyproject.toml` with Python 3.12+, pytest, lint/type-check hooks, and editable install support.
- Add `.env.example`, `.gitignore`, deterministic logging/config bootstrap, and a CLI smoke command.
- Keep mandatory CI dependencies minimal; heavy recommender libraries require task-level justification.

**Definition of Done (DoD)**
- [x] Fresh environment can install the package and import `recomlab`.
- [x] Unit smoke test and CLI smoke test pass.
- [x] No secrets or raw datasets are committed.
- [x] Relevant tests pass and documentation is synchronized.

**Verification Evidence**  
pytest tests/unit/test_smoke.py (4 passed: import, in-process CLI, `python -m recomlab smoke`, logging idempotence); ruff and mypy clean; editable install verified in a fresh Python 3.14 venv.

### Task 1.2: Canonical User, Item & Interaction Schemas

**Status:** `[x]` Completed

**Business / Research Value**  
Define explicit contracts for the data the recommender consumes so model code does not silently depend on ad-hoc notebook columns.

**Technical Specification**
- Define canonical user, item, and interaction records with stable IDs and timestamps.
- Support explicit feedback (e.g., ratings) and implicit feedback (e.g., click/watch/purchase) without conflating their semantics.
- Specify required vs optional metadata and validation errors.
- Document timestamp timezone and duplicate-event policy.

**Definition of Done (DoD)**
- [x] Schema fixtures validate and invalid fixtures fail predictably.
- [x] Explicit and implicit examples are covered by tests.
- [x] Data contract is documented in `docs/DATA_CONTRACT.md`.
- [x] Relevant tests pass and documentation is synchronized.

**Verification Evidence**  
tests/unit/test_schemas.py (25 tests): tiny fixture validates; explicit and implicit examples; 14 invalid cases fail with stable codes; UTC normalization; duplicate policy. Full suite 29 passed; ruff/mypy clean.

### Task 1.3: Impression, Exposure & Feedback Event Contract

**Status:** `[ ]` Not Started

**Business / Research Value**  
Capture what users were shown—not only what they clicked—so later evaluation can reason about exposure bias and online-style feedback.

**Technical Specification**
- Define impression/exposure event schema with request/session id, user, item, rank position, model id, timestamp, and optional propensity.
- Define click/like/watch/purchase feedback linkage back to the exposure when available.
- Allow datasets without exposure logs while marking exposure-aware evaluation unavailable.
- Prevent raw free-text or unnecessary PII from entering analytics tables by default.

**Definition of Done (DoD)**
- [ ] Exposure-to-feedback linkage tests pass.
- [ ] Missing propensity is handled explicitly rather than guessed.
- [ ] Schema is documented and versioned.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 1.4: Configuration, Environment & Dependency Policy

**Status:** `[ ]` Not Started

**Business / Research Value**  
Centralize configuration so datasets, evaluation protocols, models, and optional LLM providers can change without hardcoded paths or credentials.

**Technical Specification**
- Implement typed configuration loading for environment, paths, dataset, evaluation, and optional provider settings.
- Support local defaults and environment overrides.
- Reject missing mandatory settings with clear errors.
- Document dependency-approval rule for heavy ANN, ranking, and LLM libraries.

**Definition of Done (DoD)**
- [ ] Config tests cover defaults, overrides, and invalid values.
- [ ] Secrets appear only as placeholders in `.env.example`.
- [ ] No model logic reads environment variables directly.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 1.5: Dataset Registry, Data Cards & Licensing Metadata

**Status:** `[ ]` Not Started

**Business / Research Value**  
Make every experiment traceable to a known dataset version and usage context instead of an unnamed CSV snapshot.

**Technical Specification**
- Create dataset registry entries with id, version, source, checksum, schema version, license/terms note, and retrieval date.
- Support a local fixture dataset and a MovieLens adapter as the recommended public demo path.
- Store processed data separately from immutable raw snapshots.
- Create a machine-readable dataset manifest and human-readable data card.

**Definition of Done (DoD)**
- [ ] Dataset checksum/version changes are detected.
- [ ] Fixture dataset can be registered and resolved deterministically.
- [ ] No raw downloaded dataset is required in git.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 1.6: Experiment Registry, Seed Control & Reproducibility Manifest

**Status:** `[ ]` Not Started

**Business / Research Value**  
Ensure each reported result can be traced back to the exact dataset, split, features, model, metric configuration, and code state.

**Technical Specification**
- Define experiment id and run manifest.
- Persist dataset version, split id, feature version, model config, metric config, seeds, timestamps, and optional git commit.
- Provide deterministic seed helper for Python/NumPy/PyTorch when used.
- Record software versions relevant to a run.

**Definition of Done (DoD)**
- [ ] Same config and fixture reproduce the same deterministic baseline result.
- [ ] Experiment manifests validate against a schema.
- [ ] Registry supports comparing runs without reading notebook state.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 2 — Dataset Protocol, Labels, Features & Baselines

### Task 2.1: Interaction Validation & Cleaning Pipeline

**Status:** `[ ]` Not Started

**Business / Research Value**  
Turn raw interaction data into an auditable canonical dataset while reporting—not hiding—data quality decisions.

**Technical Specification**
- Validate IDs, timestamps, event types, duplicate policy, invalid values, and item/user references.
- Produce a cleaning report with counts before/after each rule.
- Do not silently remove events without a reason code.
- Keep cleaning deterministic.

**Definition of Done (DoD)**
- [ ] Known bad fixtures yield expected reason codes.
- [ ] Cleaning report totals reconcile.
- [ ] Repeated runs produce identical processed output.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.2: Explicit-to-Implicit Label Policy & Relevance Semantics

**Status:** `[ ]` Not Started

**Business / Research Value**  
Make “relevant item” semantics explicit because ranking metrics are meaningless if ratings/clicks are converted inconsistently.

**Technical Specification**
- Define configurable rules for ratings-to-positive conversion, watch thresholds, purchases, and event weights.
- Keep raw feedback untouched; derive labels as versioned outputs.
- Document binary vs graded relevance support.
- Ensure evaluation reads the label policy from the experiment manifest.

**Definition of Done (DoD)**
- [ ] Label fixtures cover boundary values.
- [ ] Policy version is stored with datasets/runs.
- [ ] Changing policy changes the derived dataset version or label-version field.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.3: Negative Sampling Engine for Training

**Status:** `[ ]` Not Started

**Business / Research Value**  
Provide controlled negative sampling for models that need negatives without leaking future positives or confusing training sampling with evaluation sampling.

**Technical Specification**
- Implement uniform and popularity-aware training negative samplers.
- Exclude known positive interactions according to the training horizon.
- Seed sampling deterministically.
- Mark sampled training negatives as synthetic and never mix them with full-catalog evaluation truth.

**Definition of Done (DoD)**
- [ ] No sampled negative is a known training-horizon positive.
- [ ] Sampler is deterministic for a fixed seed.
- [ ] Tests cover sparse users and small catalogs.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.4: Leakage-Safe Train / Validation / Test Split Engine

**Status:** `[ ]` Not Started

**Business / Research Value**  
Create evaluation splits that preserve the intended causal/time ordering and prevent future interactions from influencing earlier predictions.

**Technical Specification**
- Support leave-last-N-out and global time-based splits as primary protocols.
- Support random split only as an explicitly labeled diagnostic baseline.
- Persist split boundaries and excluded users/items.
- Guard against users with insufficient history and duplicate event leakage.

**Definition of Done (DoD)**
- [ ] Temporal order invariants are tested.
- [ ] Same dataset + split config creates same split id.
- [ ] No test event appears in train/validation.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.5: Point-in-Time User, Item & Interaction Feature Builder

**Status:** `[ ]` Not Started

**Business / Research Value**  
Create reusable features without computing popularity or behavioral summaries from future information.

**Technical Specification**
- Implement feature builder interface with `as_of` semantics.
- Add baseline features such as history length, recency, item popularity, genre/category metadata, and simple interaction statistics.
- Version feature definitions.
- Prevent validation/test-only events from contributing to training-time aggregates.

**Definition of Done (DoD)**
- [ ] Hand-built fixtures verify point-in-time counts.
- [ ] Feature leakage tests fail when future data are injected.
- [ ] Feature manifest is stored with runs.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.6: Random & Popularity Recommendation Baselines

**Status:** `[ ]` Not Started

**Business / Research Value**  
Establish honest lower-bound baselines before adding personalized models.

**Technical Specification**
- Implement seeded random recommender.
- Implement global popularity recommender using training data only.
- Support top-K generation and eligibility filters.
- Record latency and catalog coverage along with ranking metrics.

**Definition of Done (DoD)**
- [ ] Baselines produce deterministic top-K on fixtures.
- [ ] Popularity counts exclude validation/test data.
- [ ] Baseline results are saved to experiment registry.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 2.7: Collaborative Filtering Baseline

**Status:** `[ ]` Not Started

**Business / Research Value**  
Add a conventional personalized baseline strong enough that advanced models cannot claim gains over a strawman.

**Technical Specification**
- Implement or wrap one reproducible matrix-factorization / implicit-feedback baseline behind a project interface.
- Separate training from serving score calls.
- Document objective, regularization, and feedback assumptions.
- Provide CPU-friendly fixture settings for CI.

**Definition of Done (DoD)**
- [ ] Model trains on tiny fixture and ranks expected items.
- [ ] Seed/config are recorded.
- [ ] Baseline is evaluated under the same protocol as popularity.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 3 — Candidate Generation & Retrieval

### Task 3.1: Candidate Generator Interface & Retrieval Contract

**Status:** `[ ]` Not Started

**Business / Research Value**  
Separate recall-oriented retrieval from precision-oriented ranking so bottlenecks can be diagnosed correctly.

**Technical Specification**
- Define candidate request, candidate item, source score, source name, and retrieval metadata.
- Require configurable candidate budget.
- Do not expose validation/test labels to candidate generators.
- Support multiple generators for later fusion.

**Definition of Done (DoD)**
- [ ] Interface contract tests pass.
- [ ] Candidate outputs contain unique item ids per source or documented duplicates before fusion.
- [ ] Candidate budget is enforced.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 3.2: Item-Item Similarity Candidate Generator

**Status:** `[ ]` Not Started

**Business / Research Value**  
Provide an interpretable content/co-occurrence retrieval path useful for warm and sparse users.

**Technical Specification**
- Build item-item similarity from training-only interactions and/or item metadata.
- Aggregate candidates from recent/high-value user history.
- Store source evidence such as seed item and similarity score.
- Handle empty history gracefully.

**Definition of Done (DoD)**
- [ ] No future interactions are used.
- [ ] Known fixture neighbors are retrieved.
- [ ] Cold-user fallback is explicit.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 3.3: Collaborative Retrieval Candidate Generator

**Status:** `[ ]` Not Started

**Business / Research Value**  
Use collaborative latent signals for candidate recall while keeping retrieval independently measurable.

**Technical Specification**
- Adapt the collaborative baseline or a dedicated retrieval model to return top-N candidates.
- Exclude already-consumed items when policy requires it.
- Expose retrieval score and model version.
- Support batch retrieval for evaluation.

**Definition of Done (DoD)**
- [ ] Candidate recall can be evaluated independently.
- [ ] Consumed-item filtering is tested.
- [ ] Batch and single-user outputs agree.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 3.4: Embedding Retrieval & ANN Adapter

**Status:** `[ ]` Not Started

**Business / Research Value**  
Support modern vector retrieval without making a vector database mandatory for the project.

**Technical Specification**
- Define embedding provider and nearest-neighbor index interfaces.
- Provide an exact/brute-force small-data implementation as the mandatory baseline.
- Allow optional ANN backend behind an adapter.
- Version item embeddings and index build metadata.

**Definition of Done (DoD)**
- [ ] Exact retrieval fixture matches expected neighbors.
- [ ] Index/model version mismatch is detected.
- [ ] Core tests do not require external vector infrastructure.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 3.5: Candidate Fusion, Deduplication & Eligibility Filtering

**Status:** `[ ]` Not Started

**Business / Research Value**  
Combine multiple retrieval sources into a single valid candidate set while preserving source provenance.

**Technical Specification**
- Implement deterministic fusion with configurable source weights or rank fusion.
- Deduplicate items while preserving contributing source scores.
- Apply eligibility rules (availability, already-consumed policy, excluded catalog entries).
- Record pre/post-filter counts.

**Definition of Done (DoD)**
- [ ] Fusion output is deterministic.
- [ ] Eligibility filters cannot reintroduce duplicates.
- [ ] Provenance identifies every contributing generator.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 3.6: Candidate Recall, Coverage & Failure Diagnostics

**Status:** `[ ]` Not Started

**Business / Research Value**  
Measure whether the retrieval stage contains the relevant items before blaming the ranker.

**Technical Specification**
- Compute Recall@CandidateN / hit coverage against held-out truth.
- Report diagnostics by warm/cold/sparse users and candidate source.
- Track empty-candidate rate and average candidate count.
- Persist retrieval diagnostics per experiment.

**Definition of Done (DoD)**
- [ ] Synthetic fixtures produce hand-checkable candidate recall.
- [ ] Report separates missing-from-candidates from ranking misses.
- [ ] Diagnostics are visible in experiment output.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 4 — Ranking, Re-ranking & Cold-Start Policy

### Task 4.1: Ranker Interface & Scoring Contract

**Status:** `[ ]` Not Started

**Business / Research Value**  
Standardize ranking so heuristic, collaborative, and learned rankers can be compared on the exact same candidate sets.

**Technical Specification**
- Define rank request, feature input, raw score, final score, rank position, and model metadata.
- Require deterministic tie-breaking.
- Prevent rankers from querying held-out labels.
- Keep ranker output independent from presentation/explanation text.

**Definition of Done (DoD)**
- [ ] Contract tests cover ordering/ties.
- [ ] Rankers cannot exceed candidate set.
- [ ] Score/rank metadata is serializable.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 4.2: Heuristic / Weighted Feature Ranker

**Status:** `[ ]` Not Started

**Business / Research Value**  
Create an interpretable personalized ranking baseline before learning-to-rank.

**Technical Specification**
- Combine normalized retrieval score, recency, popularity, and content/profile similarity.
- Make weights configurable and stored in run manifest.
- Avoid using post-event features.
- Expose per-feature contribution for explanation grounding.

**Definition of Done (DoD)**
- [ ] Hand-computed fixture ranking matches implementation.
- [ ] Changing weights changes ranking predictably.
- [ ] Contributions sum/trace to ranking score where applicable.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 4.3: Collaborative Score Ranker

**Status:** `[ ]` Not Started

**Business / Research Value**  
Use collaborative model scores as a direct ranker for the common retrieval-and-rank baseline.

**Technical Specification**
- Score fused candidates with the trained collaborative model.
- Handle unknown users/items with explicit fallback behavior.
- Batch score for offline evaluation.
- Record model version and inference latency.

**Definition of Done (DoD)**
- [ ] Known score ordering is tested.
- [ ] Unknown entity behavior is deterministic.
- [ ] No catalog-wide label access occurs during ranking.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 4.4: Learning-to-Rank Dataset Builder

**Status:** `[ ]` Not Started

**Business / Research Value**  
Construct query-grouped ranking examples without leaking evaluation labels or incorrectly sampling negatives.

**Technical Specification**
- Build one query/group per user/request context.
- Join point-in-time features and candidate provenance.
- Create training labels only from the appropriate future/target window.
- Store group boundaries and feature schema version.

**Definition of Done (DoD)**
- [ ] Group sizes reconcile with candidates.
- [ ] Feature/label timestamp ordering is tested.
- [ ] No validation/test target leaks into training rows.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 4.5: Learning-to-Rank Baseline

**Status:** `[ ]` Not Started

**Business / Research Value**  
Add a practical learned ranker that can exploit candidate and context features while remaining reproducible.

**Technical Specification**
- Implement an approved gradient-boosted or pairwise/listwise ranking baseline behind the ranker interface.
- Tune on validation only.
- Persist hyperparameters and feature importance where supported.
- Provide a small CPU configuration for CI.

**Definition of Done (DoD)**
- [ ] Training/evaluation smoke test passes.
- [ ] Test set is not used for tuning.
- [ ] Result is compared with heuristic and collaborative baselines.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 4.6: Diversification, Calibration & Cold-Start Re-ranking Policy

**Status:** `[ ]` Not Started

**Business / Research Value**  
Improve final recommendation quality beyond relevance while providing explicit fallbacks for new/sparse users.

**Technical Specification**
- Implement optional diversity-aware or calibration-aware re-ranking.
- Define cold-user and cold-item fallback policy using popularity/content signals.
- Keep re-ranking constraints transparent and configurable.
- Measure relevance trade-off rather than hiding it.

**Definition of Done (DoD)**
- [ ] Re-ranking never adds items outside eligible candidates.
- [ ] Cold-start fixtures return non-empty deterministic results.
- [ ] Before/after relevance and beyond-accuracy metrics are reported.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 5 — Evaluation, Statistical Testing & Model Selection

### Task 5.1: Top-K Ranking Metrics Library

**Status:** `[ ]` Not Started

**Business / Research Value**  
Provide trustworthy metric implementations for model comparison.

**Technical Specification**
- Implement Precision@K, Recall@K, NDCG@K, MAP@K, MRR, and HitRate where meaningful.
- Define graded vs binary relevance behavior.
- Aggregate per-user metrics transparently.
- Handle users with no evaluable positives explicitly.

**Definition of Done (DoD)**
- [ ] Metrics match hand-computed fixtures.
- [ ] Edge cases are covered.
- [ ] Metric configuration is stored in manifests.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.2: Beyond-Accuracy Metrics: Coverage, Diversity, Novelty, Calibration & Popularity Bias

**Status:** `[ ]` Not Started

**Business / Research Value**  
Measure whether a recommender is useful beyond raw relevance and whether gains simply amplify popular items.

**Technical Specification**
- Implement catalog/user coverage.
- Implement intra-list diversity with documented item representation.
- Implement novelty/self-information and popularity concentration.
- Implement profile-to-list calibration when category metadata exists.

**Definition of Done (DoD)**
- [ ] Metrics pass deterministic fixtures.
- [ ] Undefined metrics are reported as unavailable—not zero.
- [ ] Metric assumptions are documented.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.3: Full-Catalog vs Sampled Evaluation Protocol

**Status:** `[ ]` Not Started

**Business / Research Value**  
Prevent misleading gains caused by ranking one positive against a small arbitrary negative sample.

**Technical Specification**
- Implement a preferred full-eligible-catalog evaluation mode for feasible datasets.
- Allow sampled evaluation only as an explicitly labeled fast diagnostic.
- Persist evaluation universe size and sampling policy.
- Warn in reports when sampled metrics are used.

**Definition of Done (DoD)**
- [ ] Full-catalog fixture is deterministic.
- [ ] Sampled and full-catalog results cannot be silently mixed on the leaderboard.
- [ ] Sampling seed/policy are recorded.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.4: Slice Analysis for Cold, Sparse, Popularity & Activity Segments

**Status:** `[ ]` Not Started

**Business / Research Value**  
Reveal where a model helps or hurts instead of hiding subgroup failures inside global means.

**Technical Specification**
- Define reproducible user/item activity buckets.
- Report warm vs cold and sparse vs dense users.
- Report head/mid/tail item performance.
- Allow dataset-specific slices through configuration.

**Definition of Done (DoD)**
- [ ] Slice memberships are deterministic.
- [ ] Slice counts reconcile with evaluation population.
- [ ] Leaderboard links to slice report.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.5: Bootstrap Confidence Intervals & Paired Significance Testing

**Status:** `[ ]` Not Started

**Business / Research Value**  
Quantify uncertainty in model differences so tiny metric changes are not overclaimed.

**Technical Specification**
- Implement user-level bootstrap confidence intervals for supported metrics.
- Implement paired comparison between two recommenders using a documented nonparametric test or bootstrap difference interval.
- Store seed and number of resamples.
- Report effect size / metric delta with uncertainty.

**Definition of Done (DoD)**
- [ ] Known synthetic comparison behaves as expected.
- [ ] Same seed reproduces intervals.
- [ ] Reports distinguish statistically uncertain differences.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.6: Exposure-Aware / Off-Policy Evaluation (Optional When Logged Propensities Exist)

**Status:** `[ ]` Not Started

**Business / Research Value**  
Use impression and propensity logs when available to reduce exposure bias without fabricating counterfactual estimates when they are not available.

**Technical Specification**
- Gate the module on valid exposure/propensity data.
- Implement a simple IPS/SNIPS-style estimator or approved equivalent with clipping diagnostics.
- Report effective sample size / weight concentration diagnostics.
- Never infer propensities from clicks alone in the mandatory path.

**Definition of Done (DoD)**
- [ ] Module refuses incompatible datasets.
- [ ] Estimator passes small hand-computed fixture.
- [ ] Diagnostics flag extreme weights.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.7: Multi-Metric Model Selection & Champion Leaderboard

**Status:** `[ ]` Not Started

**Business / Research Value**  
Select models using explicit trade-offs rather than a single cherry-picked NDCG value.

**Technical Specification**
- Create leaderboard across relevance, beyond-accuracy, latency, and cold-start slices.
- Define primary metric and guardrail metrics before test evaluation.
- Support Pareto-style comparison where no single model dominates.
- Separate validation selection from final test reporting.

**Definition of Done (DoD)**
- [ ] Test set cannot choose hyperparameters.
- [ ] Champion decision records rationale and metrics.
- [ ] Leaderboard is reproducible from manifests.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 5.8: Evaluation Report & Reproducibility Manifest

**Status:** `[ ]` Not Started

**Business / Research Value**  
Turn experiments into a reviewable research artifact with protocol, metrics, uncertainty, caveats, and exact configuration.

**Technical Specification**
- Generate Markdown/JSON report from registry.
- Include dataset/split/model versions, evaluation universe, metric definitions, CI/significance, slice results, and caveats.
- Include candidate-stage and ranking-stage diagnostics separately.
- Include warnings for sampled evaluation or unavailable exposure correction.

**Definition of Done (DoD)**
- [ ] Report can be regenerated from stored artifacts.
- [ ] No manual notebook copy/paste is required.
- [ ] Report clearly states limitations.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 6 — API, Dashboard & Automated Recommendation Workflow

### Task 6.1: Recommendation Service & Model Registry

**Status:** `[ ]` Not Started

**Business / Research Value**  
Create one application-layer service for production/demo recommendations and an auditable registry of active model artifacts.

**Technical Specification**
- Define active model/candidate/ranker bundle.
- Validate artifact/config compatibility before activation.
- Expose recommendation request with top-K, eligible catalog policy, and trace id.
- Return provenance needed for debugging and explanation grounding.

**Definition of Done (DoD)**
- [ ] Active model swap is tested.
- [ ] Incompatible artifacts fail closed.
- [ ] Service output matches offline ranker on fixture context.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 6.2: FastAPI Recommendation & Experiment API

**Status:** `[ ]` Not Started

**Business / Research Value**  
Expose stable endpoints for demos and automated workflows without embedding model logic in controllers.

**Technical Specification**
- Add health/readiness endpoints.
- Add recommendation endpoint.
- Add experiment/run query endpoints.
- Add model registry/activation endpoint with safe local auth placeholder if mutation is exposed.

**Definition of Done (DoD)**
- [ ] API contract tests pass.
- [ ] Validation errors use structured responses.
- [ ] No secrets or raw private data are returned.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 6.3: Streamlit Experiment Dashboard & Recommendation Explorer

**Status:** `[ ]` Not Started

**Business / Research Value**  
Provide a portfolio-friendly interface to inspect models, metrics, slices, candidates, rankings, and recommendation explanations.

**Technical Specification**
- Show experiment leaderboard and run details.
- Show candidate-stage vs rank-stage diagnostics.
- Add user recommendation explorer using demo-safe ids.
- Display cold-start and beyond-accuracy views.

**Definition of Done (DoD)**
- [ ] Critical pages load with fixture data.
- [ ] Dashboard calls services/registry rather than training models inline.
- [ ] No raw PII is displayed.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 6.4: Impression & Feedback Logging Workflow

**Status:** `[ ]` Not Started

**Business / Research Value**  
Close the loop by logging what was recommended and what response occurred, enabling future evaluation and retraining.

**Technical Specification**
- Log request id, model id, ranked impressions, positions, timestamp, and optional propensity.
- Link subsequent feedback to impressions when available.
- Make logging idempotent by event id.
- Keep demo mode local and privacy-minimal.

**Definition of Done (DoD)**
- [ ] Duplicate event handling is tested.
- [ ] Feedback linkage is queryable.
- [ ] Exposure data conforms to Task 1.3 contract.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 6.5: Scheduled Retraining & Data-Snapshot Workflow

**Status:** `[ ]` Not Started

**Business / Research Value**  
Automate the reproducible path from newly logged events to a candidate model without auto-promoting unverified models.

**Technical Specification**
- Create retraining command/workflow that freezes a dataset snapshot.
- Run validation, training, and evaluation stages with one run id.
- Store artifacts and report.
- Do not promote automatically unless release gates pass.

**Definition of Done (DoD)**
- [ ] Workflow can run end-to-end on demo fixture.
- [ ] Failure leaves previous active model untouched.
- [ ] Dataset and model versions are linked.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 6.6: Champion / Challenger Promotion Gate

**Status:** `[ ]` Not Started

**Business / Research Value**  
Make model promotion a testable decision based on predefined primary and guardrail metrics.

**Technical Specification**
- Compare challenger against current champion using Phase 5 protocol.
- Require minimum primary improvement or documented acceptance rule.
- Enforce guardrails for coverage, cold-start, latency, and major slice regressions.
- Record promotion/rejection decision.

**Definition of Done (DoD)**
- [ ] A failing guardrail blocks promotion.
- [ ] Decision is reproducible from stored report.
- [ ] Rollback to previous champion is supported.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 7 — LLM Explanations & Human-AI Research Extension

### Task 7.1: Recommendation Grounding Evidence Builder

**Status:** `[ ]` Not Started

**Business / Research Value**  
Translate recommender evidence into a structured, bounded payload that can support explanations without exposing raw hidden data or inventing preferences.

**Technical Specification**
- Build evidence from prior interacted items, item attributes, retrieval sources, and ranker contributions that are actually available.
- Limit history by a configurable privacy-safe window.
- Mark unavailable evidence explicitly.
- Separate evidence used for ranking from evidence used only for presentation.

**Definition of Done (DoD)**
- [ ] Evidence can be traced to real stored interactions/metadata.
- [ ] Unknown facts are never synthesized.
- [ ] Payload schema is tested.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.2: Explanation Conditions: None, Template, Generic LLM & Contextualized LLM

**Status:** `[ ]` Not Started

**Business / Research Value**  
Support controlled comparison of explanation strategies rather than evaluating one LLM prompt in isolation.

**Technical Specification**
- Define versioned conditions with stable ids.
- Implement deterministic template explanation.
- Define generic LLM condition using item-level evidence.
- Define contextualized LLM condition using bounded user-history evidence.

**Definition of Done (DoD)**
- [ ] Condition assignment produces the requested strategy.
- [ ] Template has no LLM dependency.
- [ ] Generic/contextual evidence boundaries are tested.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.3: Structured LLM Explanation Generator

**Status:** `[ ]` Not Started

**Business / Research Value**  
Generate user-facing explanations with machine-readable provenance and safe failure behavior.

**Technical Specification**
- Use structured output schema including explanation text, cited evidence ids, uncertainty/abstain field, and prompt/model version.
- Use low-variance configuration for research reproducibility where supported.
- Retry only bounded transient/schema failures.
- Fall back to template or no explanation on provider failure.

**Definition of Done (DoD)**
- [ ] Mocked valid/malformed/unsupported outputs are tested.
- [ ] Provider failure never blocks recommendation delivery.
- [ ] Prompt/model version is logged.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.4: Explanation Fidelity & Unsupported-Claim Validator

**Status:** `[ ]` Not Started

**Business / Research Value**  
Distinguish fluent persuasion from faithful explanation by checking every generated claim against actual recommender evidence.

**Technical Specification**
- Validate referenced items/attributes against evidence payload.
- Add perturbation/counterfactual fidelity check where feasible for the selected ranker.
- Score or label unsupported claims.
- Reject explanations above a configured unsupported-claim threshold.

**Definition of Done (DoD)**
- [ ] Synthetic hallucinations are detected.
- [ ] Valid grounded explanations pass.
- [ ] Fidelity result is stored separately from user-perception outcomes.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.5: Human-AI Study Protocol, Randomization & Ethics Boundary

**Status:** `[ ]` Not Started

**Business / Research Value**  
Create a reproducible experiment design suitable for a research demo while preventing the codebase from treating unapproved human-subject recruitment as a normal engineering step.

**Technical Specification**
- Define research questions, hypotheses, primary/secondary outcomes, conditions, exclusion rules, and randomization.
- Add sample-size/power planning helper or documented analysis.
- Define consent/privacy/data-retention fields but do not recruit participants without applicable institutional approval.
- Support synthetic/demo responses for software testing only and label them as simulated.

**Definition of Done (DoD)**
- [ ] Random assignment is deterministic under test seed and balanced within tolerance.
- [ ] Study config is versioned.
- [ ] Synthetic results cannot be confused with real participant data.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.6: Trust, Acceptance, Usefulness & Satisfaction Analysis

**Status:** `[ ]` Not Started

**Business / Research Value**  
Analyze user-facing outcomes with proper statistical separation from offline ranking quality.

**Technical Specification**
- Implement descriptive statistics by condition.
- Implement pre-specified comparisons and confidence intervals for key outcomes.
- Support recommendation acceptance/intention, trust, usefulness, satisfaction, and decision time when collected.
- Report multiple-comparison policy when many outcomes are tested.

**Definition of Done (DoD)**
- [ ] Analysis passes synthetic known-effect fixtures.
- [ ] Condition labels and exclusions are auditable.
- [ ] Offline recommender metrics are not merged into perception scales.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 7.7: Explanation Bias / Persuasiveness Stress Test & Research Report

**Status:** `[ ]` Not Started

**Business / Research Value**  
Evaluate whether wording can influence users independently of recommendation quality and summarize the full human-centered experiment transparently.

**Technical Specification**
- Create controlled wording variants only in a clearly labeled research mode.
- Compare fidelity and persuasion outcomes separately.
- Add qualitative feedback coding/export path when free-text responses exist.
- Generate a report with hypotheses, methods, results, uncertainty, fidelity results, limitations, and simulated-vs-real status.

**Definition of Done (DoD)**
- [ ] Research mode cannot silently modify production/demo explanations.
- [ ] Report distinguishes persuasion from correctness.
- [ ] Free-text is sanitized/minimized before LLM-assisted coding if used.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Phase 8 — Hardening, Monitoring, CI/CD & Release

### Task 8.1: End-to-End Regression Suite

**Status:** `[ ]` Not Started

**Business / Research Value**  
Prove the full data→retrieval→ranking→evaluation→serving path works and core methodological invariants remain intact.

**Technical Specification**
- Add deterministic E2E fixture pipeline.
- Cover warm user, cold user, empty candidate, and model fallback paths.
- Assert no test leakage and reproducible metrics.
- Include optional explanation path with mocked provider.

**Definition of Done (DoD)**
- [ ] E2E suite passes in CI.
- [ ] No live external API is required.
- [ ] Regression artifacts are bounded and deterministic.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 8.2: Observability, Drift & Train-Serving Skew Checks

**Status:** `[ ]` Not Started

**Business / Research Value**  
Detect operational and data changes that can invalidate recommendation quality after deployment.

**Technical Specification**
- Add structured logs with request/run/model ids.
- Track event volume, catalog size, user/item activity distribution, empty-candidate rate, and latency.
- Compare serving feature schema/distributions with training snapshot.
- Add configurable warnings for major drift/skew.

**Definition of Done (DoD)**
- [ ] Health/readiness tests pass.
- [ ] Synthetic drift fixture triggers warning.
- [ ] Sensitive fields are scrubbed from logs.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 8.3: Dockerization & Reproducible Demo Stack

**Status:** `[ ]` Not Started

**Business / Research Value**  
Make the portfolio demo easy to run without introducing unnecessary distributed infrastructure.

**Technical Specification**
- Create multi-stage or slim Python Docker image.
- Compose API + dashboard only as needed.
- Mount local artifacts/data directories explicitly.
- Run as non-root where practical and include healthcheck.

**Definition of Done (DoD)**
- [ ] Docker build succeeds.
- [ ] Demo stack serves health endpoint and dashboard.
- [ ] Secrets are not baked into image.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 8.4: GitHub Actions CI & Quality Gates

**Status:** `[ ]` Not Started

**Business / Research Value**  
Block regressions before merge and make repository quality visible.

**Technical Specification**
- Run install, lint/type checks, unit/integration/E2E fixture tests.
- Run leakage and metric correctness tests as mandatory gates.
- Cache dependencies safely.
- Optionally build Docker image after tests.

**Definition of Done (DoD)**
- [ ] CI config is syntax-valid.
- [ ] Intentional failing leakage test blocks pipeline.
- [ ] No deploy secrets are available to untrusted PRs.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 8.5: Security, Privacy & Dependency Audit

**Status:** `[ ]` Not Started

**Business / Research Value**  
Review the project for secret leakage, unsafe participant data handling, and vulnerable/unnecessary dependencies.

**Technical Specification**
- Scan for committed secrets.
- Review PII minimization and demo identifiers.
- Document human-study data separation and retention policy placeholders.
- Run dependency vulnerability audit and remove unjustified packages.

**Definition of Done (DoD)**
- [ ] No known committed secrets.
- [ ] Privacy checklist completed.
- [ ] Critical dependency findings block release or receive documented waiver.
- [ ] Relevant tests pass and documentation is synchronized.

### Task 8.6: Operational Runbook, Release Checklist & Portfolio Demo

**Status:** `[ ]` Not Started

**Business / Research Value**  
Finish the project with a reproducible demo path and documented failure recovery rather than an unmaintainable notebook collection.

**Technical Specification**
- Finalize runbook for reproducibility, leakage, candidate recall collapse, model regression, LLM failure, and rollback.
- Create one-command demo bootstrap where feasible.
- Publish representative benchmark report and screenshots without private data.
- Document v1 limitations and post-v1 research backlog.

**Definition of Done (DoD)**
- [ ] Fresh-machine walkthrough succeeds.
- [ ] Release checklist is complete.
- [ ] README links to benchmark/evaluation report and architecture docs.
- [ ] Relevant tests pass and documentation is synchronized.

---

## Progress Summary

| Phase | Tasks | Completed |
|---|---:|---:|
| 1 — Foundation, Data Contracts & Reproducibility | 6 | 2 |
| 2 — Dataset Protocol, Labels, Features & Baselines | 7 | 0 |
| 3 — Candidate Generation & Retrieval | 6 | 0 |
| 4 — Ranking, Re-ranking & Cold-Start Policy | 6 | 0 |
| 5 — Evaluation, Statistical Testing & Model Selection | 8 | 0 |
| 6 — API, Dashboard & Automated Recommendation Workflow | 6 | 0 |
| 7 — LLM Explanations & Human-AI Research Extension | 7 | 0 |
| 8 — Hardening, Monitoring, CI/CD & Release | 6 | 0 |
| **Total** | **52** | **2** |
