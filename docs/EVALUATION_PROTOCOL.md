# RecomLab Evaluation Protocol

## 1. Default Protocol

For time-stamped interaction data, prefer leave-last-N-out or a documented temporal split. Random splits are diagnostic only unless the research question explicitly justifies them.

## 2. Retrieval Evaluation

Measure whether held-out relevant items appear in the candidate set before applying the ranker.

Required outputs:
- candidate Recall@N / HitRate@N
- empty-candidate rate
- candidate count distribution
- source-level contribution
- cold/sparse-user slices

## 3. Ranking Evaluation

Primary metrics may include Recall@K and NDCG@K. Precision@K, MAP@K, MRR, and HitRate may be added when appropriate.

## 4. Beyond-Accuracy

Report at least a subset of:
- catalog coverage
- intra-list diversity
- novelty
- popularity concentration/bias
- calibration where item categories permit it

## 5. Evaluation Universe

**Preferred:** all eligible catalog items when feasible.

**Diagnostic:** sampled negatives for speed. Sampled results must record the sampler and seed and must not be compared as if they were full-catalog results.

## 6. Uncertainty

Report uncertainty for model comparisons using a documented paired/user-level bootstrap or equivalent appropriate method. Avoid claiming practical superiority from tiny point-estimate differences alone.

## 7. Slice Analysis

At minimum inspect:
- cold vs warm users/items
- sparse vs active users
- head vs mid/tail catalog

## 8. Exposure Bias

Logged interactions reflect what users were exposed to. When valid impression propensities exist, optional IPS/SNIPS-style evaluation may be used with diagnostics. Without propensities, do not fabricate counterfactual certainty.

## 9. Test-Set Governance

Test data is evaluated only after model/hyperparameter selection using training/validation data. Test results must not drive iterative tuning.

## 10. Relevance Semantics (Label Policy)

Relevance is derived from raw feedback by a versioned `LabelPolicy`
(`recomlab.datasets.labels`); raw events are never modified.

| Event | Default rule |
|---|---|
| `rating` | positive iff `value >= 4.0` (boundary inclusive); lower ratings are explicit negatives |
| `watch` | positive iff fraction watched `>= 0.8`; missing value is not positive |
| `purchase`, `like` | positive |
| `click`, `view`, `add_to_cart` | not positive by default (configurable) |

- `binary` mode: relevance 1/0. `graded` mode: rating grade `value - threshold + 1`;
  implicit events use configured weights (purchase 3, like/watch 2, click/add_to_cart 1).
  Per user-item relevance is the maximum over that pair's positive events.
- `label_version` is a content hash of the policy; any change produces a new version.
- The policy is stored in `RunManifest.label_policy`; evaluation rebuilds it with
  `label_policy_from_manifest` and refuses to run when it is missing or tampered.

## 11. Training Negative Sampling (not evaluation)

`TrainingNegativeSampler` (`recomlab.datasets.negative_sampling`) supports `uniform` and
`popularity` (weight `(count + 1) ** 0.75`) strategies.

- Only events at or before the training `horizon` are used, both to exclude known
  positives (every item the user interacted with) and to compute popularity.
- Sampling is per-user seeded (`derive_seed(seed, "negatives", strategy, user)`), so output
  is independent of input order. Sampling is without replacement; users with fewer eligible
  items than requested are listed in `exhausted_users`.
- Results are wrapped in `TrainingNegatives(purpose="training_only")` with every sample
  marked `synthetic=True`. Evaluation code rejects this type: training negatives are never
  full-catalog evaluation truth and never the sampled-evaluation universe.

## 12. Split Engine (`recomlab.datasets.splits`)

| Protocol | Use | Rule |
|---|---|---|
| `leave_last_n` | primary | per user: last `n_test` events → test, previous `n_validation` → validation |
| `global_time` | primary | `< validation_cutoff` → train, `< test_cutoff` → validation, rest → test |
| `random` | **diagnostic only** | seeded; `Split.is_diagnostic` is true and labels carry `(DIAGNOSTIC)` |

- Users with fewer than `min_train_events + n_validation + n_test` events stay entirely in
  train and are listed in `excluded_users` (`insufficient_history`); under `global_time`,
  held-out users without training history are flagged `no_training_history`.
- Held-out events whose (user, item) pair already appeared earlier are dropped from the
  held-out sets (`dropped_events`, reason `repeat_interaction`) and never moved into train.
- `split_id` hashes the config and the event fingerprint; the same dataset + config always
  yields the same id. `Split.save()` persists boundaries, excluded users, dropped events,
  cold test items, and per-partition event ids.
- `assert_no_leakage()` runs on every split: partitions are disjoint and, for temporal
  protocols, training/validation events strictly precede test events (globally or per user).
- When predicting the test partition, visible history is train + validation
  (`Split.train_until("test")`); model selection uses validation only.

## 13. Baselines & Offline Runner

- `RandomRecommender(seed)` ranks eligible items with a per-user seeded stream (independent
  of call order). `PopularityRecommender` counts training events only
  (`split.train_until(partition)`); ties are broken by ascending item id.
- `EligibilityFilter` (`recomlab.domain.eligibility`) removes unavailable items, explicitly
  excluded items, and (by default) items the user already consumed in the visible history.
- `evaluate_recommender` ranks the **full eligible catalog** for every held-out user with at
  least one positive label (users without positives are counted, not scored) and reports
  Recall/NDCG/HitRate@K, catalog coverage@K, and recommendation latency (mean/p50/max ms).
- `run_offline_experiment` fits on the visible history, evaluates, and logs a `RunManifest`
  with dataset, split, label policy, model params, evaluation universe, metrics, latency,
  and software state. Latency is excluded from determinism checks.

## 14. Collaborative Filtering Baseline (`recomlab.models.als`)

Weighted ALS for implicit feedback (Hu, Koren & Volinsky, 2008), pure NumPy:

- Objective: `sum c_ui (p_ui - x_u·y_i)^2 + reg(|X|^2 + |Y|^2)`, `p_ui = 1` for observed pairs,
  confidence `c_ui = 1 + alpha * r_ui`.
- Feedback assumption: by default every training event is an implicit positive of strength 1
  (aggregated per user-item pair, including low ratings). With a `LabelPolicy`, only
  label-positive events contribute, with their relevance as strength; the label version is
  stored in the artifact.
- Defaults `factors=8, regularization=0.1, alpha=10, iterations=15, seed=0`; CI uses
  `factors=4, iterations=10`. Initialization is seeded, so artifacts are reproducible.
- `ImplicitALS.fit` produces an `ALSArtifact` (saved as `factors.npz` + `meta.json` with a
  content version); `ALSRecommender` serves from the artifact only and returns `None` scores
  for unknown users/items so callers apply an explicit fallback.
- Evaluated with `run_offline_experiment` under exactly the same split, label policy, and
  full-catalog universe as the popularity baseline.

## 15. Retrieval Diagnostics (`recomlab.evaluation.retrieval`)

`retrieval_diagnostics(candidates_by_user, truth, segments, candidate_budget)` reports, for
users with held-out positives: Recall@CandidateN, HitRate@CandidateN, empty-candidate rate,
mean and min/median/max candidate count, per-segment recall/empty rate (segments from
`activity_segments`: `cold` = no history, `sparse` = fewer than 3 events, `warm`), and
per-source recall and contribution (from fusion provenance). `failure_breakdown` classifies
every relevant held-out item as `hit`, `ranking_failure` (retrieved but outside top-K), or
`retrieval_failure` (never retrieved). Diagnostics serialize to JSON (`write_json`) and are
stored in `RunManifest.metrics["retrieval"]`.

## 16. Learning-to-Rank Data (`recomlab.rankers.ltr_data`)

- One query group per user; rows are that user's fused candidates (default: popularity +
  item-item, RRF, fitted on the group's history).
- Features (schema `ltr-features-1`): the four normalized heuristic features, user history
  length, number of contributing sources, and per-source reciprocal rank.
- Time order: features/candidates use `history` strictly before `as_of`; labels use `target`
  events at/after `as_of` (label policy relevance, 0 for non-positive candidates). Violations
  raise `LeakageError`, as does any target event listed in `forbidden_event_ids`.
- Training groups use an inner cutoff inside the training partition
  (`inner_training_cutoff`); validation groups use train history + validation targets; test
  groups are built only for final reporting.
- `LTRDataset` validates that group sizes reconcile with rows and records schema version,
  feature names, `as_of`, label version, group boundaries, and skipped users.

## 17. Learning-to-Rank Baseline (`recomlab.rankers.ltr`)

- Model: linear pairwise RankNet (`log(1 + exp(-(s_i - s_j)))` over pairs with
  `label_i > label_j` in a group), standardized features, L2, full-batch gradient descent;
  pure NumPy and deterministic. Dependency rationale: no LightGBM/XGBoost needed for v1
  (dependency policy); a boosted ranker can be added later behind the same interface.
- `tune_on_validation(train, validation, grid, k)` fits on the training dataset and selects
  by validation NDCG@k; datasets tagged `partition="test"` raise `HeldOutTuningError`
  (training as well). The chosen model stores params, all trials, validation score,
  feature importance (normalized |w| on standardized features), schema and label versions.
- `LTRRanker` serves the model through the ranker interface with exactly the training
  feature construction (`candidate_feature_rows`); per-feature contributions sum to the
  score. A CI test compares LTR, heuristic, and collaborative rankers on identical
  validation candidates (CPU config: 50–100 epochs, tiny fixture).

## 18. Re-ranking & Cold-Start Policy (`recomlab.reranking`)

- `MMRDiversifier(categories, lam)`: greedy `lam · rel_norm − (1 − lam) · max category-Jaccard
  similarity to already selected items`; `lam = 1` keeps the ranker order.
- `CalibrationReranker(categories, lam, alpha)`: greedy calibrated list (Steck, 2018), maximizing
  `(1 − lam) Σ rel − lam · KL(p_user ‖ q_list)` with the point-in-time category profile;
  skipped (and recorded) when the user has no profile.
- Both only reorder/select items already in the ranked (eligible candidate) list.
- `ColdStartPolicy(cold_item_slots)`: the only step that may add items, and only eligible
  ones. Short/empty lists are filled by training popularity (`fallback = "popularity"`),
  then by content affinity (`"content"`); optional tail slots go to items without training
  interactions ranked by content affinity (`"cold_item_content"`). Every added item is
  listed in `metadata.fallbacks`; output is deterministic.
- `rerank_report(before, after, truth, categories, k)` reports NDCG@k, ILD@k, and distinct
  items@k before/after plus deltas, so relevance trade-offs are visible.

## 19. Metric Library (`recomlab.evaluation.metrics`)

| Metric | Definition (per user, top-K) | Relevance |
|---|---|---|
| Precision@K | hits / K (short lists are not rewarded) | binary |
| Recall@K | hits / number of relevant items | binary |
| HitRate@K | 1 if any hit | binary |
| MAP@K | Σ precision@hit-position / min(|relevant|, K) | binary |
| MRR@K | 1 / rank of first hit within K, else 0 | binary |
| NDCG@K | DCG / ideal DCG, gain `rel` (`linear`) or `2^rel − 1` (`exponential`), `log2(rank+1)` discount | graded |

Users without relevant items are undefined (`UndefinedMetricError`); `aggregate()` excludes
and lists them (`users_without_positives`) instead of scoring zero. Users with relevant items
but no ranking are scored as an empty list (a recommender failure) and listed in
`users_without_ranking`. Means are unweighted over evaluable users. `MetricConfig.to_dict()`
is stored in `RunManifest.metric_config`.

## 20. Beyond-Accuracy Metrics (`recomlab.evaluation.beyond_accuracy`)

`beyond_accuracy(rankings, k, catalog, train_item_users, n_train_users, categories, profiles)`:

| Metric | Definition / assumption |
|---|---|
| `catalog_coverage@k` | distinct recommended items ∩ eligible catalog / catalog size |
| `user_coverage@k` | share of users with a non-empty list |
| `ild@k` | mean pairwise `1 − Jaccard(category sets)` per list (item representation = categories) |
| `novelty@k` | mean `−log2((train_users_i + 1) / (n_train_users + 1))` over recommended slots |
| `gini@k` | Gini of recommendation counts over the eligible catalog |
| `head_share@k` | share of slots taken by the top 20% items by training popularity |
| `calibration_kl@k` | mean `KL(p_user ‖ q_list)` over categories, smoothing `alpha = 0.01` |

Popularity statistics use training interactions only. Metrics that cannot be computed
(no category metadata, no recommendations, no profiles, lists shorter than 2) are reported
as `None` with a reason in `unavailable`, never as zero.
