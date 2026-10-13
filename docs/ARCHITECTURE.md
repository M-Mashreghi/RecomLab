# RecomLab Architecture

## 1. Goal

RecomLab is a local-first, modular recommender platform. It intentionally separates **retrieval**, **ranking**, **re-ranking**, **evaluation**, and **human-facing explanation** so errors at one stage are not hidden by another.

## 2. High-Level Flow

```mermaid
flowchart LR
    A[Users / Items / Interactions / Impressions] --> B[Validation + Dataset Registry]
    B --> C[Label Policy + Split Engine]
    C --> D[Point-in-Time Features]
    D --> E[Candidate Generators]
    E --> F[Fusion + Eligibility]
    F --> G[Ranker]
    G --> H[Re-ranker / Cold-start Policy]
    H --> I[Top-K]
    I --> J[Offline Evaluation]
    J --> K[Experiment + Model Registry]
    K --> L[API / Dashboard]
    L --> M[Impression + Feedback Log]
    M --> B
    I --> X[Optional Explanation Evidence]
    X --> Y[LLM / Template Explanation]
    Y --> Z[Fidelity + Human-AI Study]
```

## 3. Dataset Layer

Canonical entities:
- `User`
- `Item`
- `Interaction`
- `Impression`
- `FeedbackEvent`
- `DatasetManifest`

Raw data is immutable. Cleaning, labels, splits, and features are versioned derived artifacts.

## 4. Retrieval Layer

Candidate generators optimize recall under a candidate budget. Every candidate carries source provenance. Candidate fusion deduplicates and applies eligibility constraints before ranking.

## 5. Ranking Layer

Rankers score only eligible candidate items. Re-ranking may optimize diversity/calibration or apply cold-start policy, but must report relevance trade-offs.

## 6. Evaluation Layer

Evaluation has three explicit levels:
1. candidate recall,
2. ranking relevance conditional on candidates,
3. end-to-end recommendation/system quality.

Full-catalog evaluation is the preferred reference when feasible. Sampled evaluation is a diagnostic mode and is never silently mixed with full-catalog results.

## 7. Statistical Comparison

Model comparison includes uncertainty (e.g., user-level bootstrap intervals) and paired comparisons, not only point estimates.

## 8. Exposure-Aware Path

When impression logs and valid propensities exist, an optional off-policy evaluation module may estimate exposure-corrected outcomes. It must fail closed when required logging is absent.

## 9. Serving & Workflow

A model registry holds compatible retrieval/ranking bundles. The serving layer logs ranked impressions. Retraining creates a challenger; promotion happens only after predefined primary/guardrail criteria pass.

## 10. Human-AI Explanation Layer

Explanations consume a **grounding evidence payload** derived from the recommender. They never alter ranking. Fidelity and user perception are evaluated separately.

## 11. Persistence

- SQLite: run/model/study metadata
- Parquet: processed analytical datasets
- filesystem/object-like local artifact directory: models and reports
- no mandatory external vector database or feature store for v1

## 12. Invariants

1. No future/test leakage.
2. Retrieval and ranking remain separable.
3. Training negative sampling is not evaluation truth.
4. Evaluation protocol is recorded with every result.
5. Baselines are mandatory.
6. LLMs do not fabricate recommender evidence or metrics.
7. Real human-participant data is separate from demo/synthetic data.

## 13. Configuration

`recomlab.config.load_settings()` returns a frozen, validated `Settings` object
(environment, DB URL, paths, dataset, evaluation, optional LLM provider). Precedence:
defaults → optional TOML file → `RECOMLAB_*` environment variables → explicit overrides.
Invalid values and missing mandatory settings raise `ConfigError`. API keys are wrapped
in `SecretStr` and never appear in `repr`. `config/settings.py` is the only module that
reads environment variables (enforced by a test); all other code receives `Settings`.

## 14. Dependency Policy

- Mandatory runtime dependency: NumPy. Everything else is an optional extra.
- Heavy ANN engines (FAISS, hnswlib, ...), learning-to-rank libraries (LightGBM, XGBoost),
  deep-learning frameworks, and LLM SDKs require a task-level justification recorded in
  the task's verification evidence, must sit behind a project interface/adapter, and must
  never be required by mandatory CI tests.
- Every such adapter needs a pure-Python/NumPy reference implementation (e.g. exact
  brute-force retrieval) that the tests use.
- New dependencies are added to an optional extra in `pyproject.toml`, not to core.


## 15. Experiment Registry & Reproducibility

- `RunManifest` (`recomlab.experiments.manifest`, schema 1.0) stores dataset id/version/
  checksum, split id, feature version, label policy, model name/version/params, metric
  configuration, evaluation universe/mode, seeds, metrics, artifact paths, software state
  (Python, platform, package versions) and the optional git commit.
- `config_hash` covers only the configuration (not metrics or timestamps); `run_id` combines
  experiment id, config hash and creation time so repeated executions are distinct but
  traceable to the same configuration.
- `validate_manifest` rejects missing/unknown fields, wrong types, non-integer seeds,
  unsupported schema versions and tampered config hashes.
- `ExperimentRegistry` persists manifests in SQLite and supports `list_runs`,
  `find_by_config` and `compare_runs` without any notebook state.
- Seeds: `make_rng(seed, *names)` returns isolated NumPy generators (preferred);
  `set_global_seed` seeds Python/NumPy and PyTorch when it is installed.

## 16. Point-in-Time Features (`recomlab.features`, version `pit-1.0`)

`PointInTimeFeatures(events, items, as_of)` uses only events strictly before `as_of`.

| Group | Features |
|---|---|
| user | `history_length`, `distinct_items`, `days_since_last_event`, `days_active`, `mean_rating`, category profile |
| item | `popularity`, `unique_users`, `recent_popularity` (exponential decay, half-life param), `days_since_last_event`, `mean_rating`, categories |
| user-item | `category_affinity`, `seen_before` |

Undefined values (e.g. no history) are `None`, not zero. `strict=True` raises
`FutureDataError` if the input contains events at/after `as_of`; `assert_point_in_time`
verifies the built object. The `FeatureManifest` (version, `as_of`, events used, latest event
time used, params) is stored in `RunManifest.features`. Training-time aggregates are built
from the training partition only; validation/test events never contribute.

## 17. Retrieval Contract (`recomlab.candidates`)

- `CandidateRequest(user_id, budget, exclude_items, request_id)`: the budget is a mandatory
  positive integer. Requests carry no labels and no held-out data.
- `CandidateGenerator.fit(history, items, as_of)` fails closed when history contains events
  at/after `as_of`; `generate(request)` returns a `CandidateSet` from a single source with
  unique items, ranks `1..n`, source score, per-candidate evidence, source version, and
  metadata (`n_scored`, `n_after_exclusions`, `budget`, `truncated`, `latency_ms`).
- Ordering is deterministic: descending score, ties by ascending item id.
- `validate_candidate_set` enforces budget, uniqueness, rank sequence, single source, and
  exclusions. Duplicates across sources are expected and resolved only by fusion (§ 3.5).
- `generate_all` runs several generators for later fusion; generator names must be unique.
- `PopularityCandidates` (training-history counts) is the reference generator and the
  explicit cold-user fallback source.

### 17.1 Item-Item Generator (`candidates.item_item.ItemItemCandidates`)

- Similarity = `(1 - metadata_weight) * co-occurrence cosine + metadata_weight * category Jaccard`,
  built from training interactions and item metadata only; top `neighbors_per_item` kept.
- Candidates sum neighbor similarities of the user's last `history_window` unique items with
  recency weights `decay ** age_rank`; consumed items are skipped.
- Evidence per candidate: up to 3 seed items with their similarity.
- Users without history return an empty set with `metadata.fallback_required = True` and
  `reason = "cold_user_no_history"`; the caller must apply an explicit fallback source.

### 17.2 Collaborative Generator (`candidates.collaborative.CollaborativeCandidates`)

- Top-N items by ALS score `x_u · y_i` from an `ALSArtifact` trained on the visible history
  (`fit`) or injected (`from_artifact`); `source_version` is the artifact content version.
- `exclude_consumed=True` (default) removes items the user interacted with in that history.
- Unknown users return an empty set with `fallback_required = True` (`reason = "unknown_user"`).
- `generate_batch` scores all requested users with one matrix product; tests assert it
  matches per-user `generate` exactly in items and scores.
- Candidate recall is measured independently of ranking with
  `evaluation.retrieval.candidate_recall` / `candidate_hit`.

### 17.3 Embedding Retrieval (`candidates.embedding`)

- `EmbeddingProvider` exposes a versioned `EmbeddingTable` of item vectors and a user vector:
  `ALSEmbeddingProvider` (latent factors, version tied to the ALS artifact) and
  `CategoryEmbeddingProvider` (L2-normalized multi-hot categories; user = mean of history).
- `NearestNeighborIndex` implementations record `IndexMetadata` (type, embedding version,
  metric, size, dim). `ExactIndex` (brute force, cosine or dot, id tie-break) is the
  mandatory baseline. `HnswlibIndex` is an optional adapter that raises
  `OptionalDependencyError` when `hnswlib` is absent; no core test needs vector infrastructure.
- `EmbeddingCandidates` refuses an index whose embedding version differs from the provider
  (`IndexVersionMismatch`), excludes consumed items, and flags users without an embedding
  (`fallback_required`). Availability/eligibility is applied later during fusion.

### 17.4 Fusion & Eligibility (`candidates.fusion`)

- `FusionConfig(method="rrf" | "weighted_score", weights, rrf_k, budget)`; unspecified source
  weights default to 1.0. RRF adds `w / (rrf_k + rank)`; weighted score adds `w * minmax(score)`
  per source (constant-score source → 1.0).
- Items are deduplicated by id; `FusedCandidate.sources` keeps each contributing source's
  score, rank, contribution and evidence. Sources are processed in sorted order and output
  is sorted by `(-score, item_id)`, so fusion is deterministic and input-order independent.
- `EligibilityFilter` (availability, excluded catalog entries, consumed-item policy) runs
  after deduplication and can only remove items. `counts` records
  `pre_dedup`, `post_dedup`, `post_filter`, `final`; `removed` records
  `not_eligible_catalog` and `consumed`.

## 18. Ranking Contract (`recomlab.rankers`)

- `RankRequest(user_id, candidates, k, request_id)` holds one user's deduplicated candidate
  set (`RankCandidate`: item id, fused retrieval score, source provenance). It contains no
  labels; `RankRequest.from_fused` builds it from fusion output.
- `Ranker._score` returns `{item: (raw_score, contributions)}`. `Ranker.rank` rejects any
  scored item outside the candidate set (`RankerContractError`), applies `_finalize`
  (calibration hook, identity by default), orders by `(-final_score, item_id)`, truncates to
  `k`, and records `n_candidates`, `n_unscored`, and `latency_ms`.
- `RankedList` / `RankedItem` (raw score, final score, rank, per-feature contributions,
  model name/version) are JSON-serializable and contain no presentation or explanation text.
- `RetrievalScoreRanker` ranks by fused retrieval score and serves as the identity baseline.
