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
