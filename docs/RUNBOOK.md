# RecomLab Operational Runbook

## 1. Implausibly High Offline Metrics
Check temporal split, point-in-time features, popularity statistics, negative sampling, duplicate users/items, and evaluation universe. Invalidate affected experiment ids if leakage is confirmed.

## 2. Candidate Recall Collapse
Inspect candidate budget, item eligibility filters, index/model version, cold-user rate, and per-source candidate contribution before changing the ranker.

## 3. Ranker Regression
Run the same candidate set through champion and challenger; compare user-level metric deltas, confidence intervals, slices, and latency.

## 4. Sampled vs Full-Catalog Mismatch
Do not merge the results. Verify negative-sampling policy and use full-catalog evaluation as the reference when feasible.

## 5. Drift / Train-Serving Skew
Compare feature schema, catalog size, user/item activity, event-type mix, empty-candidate rate, and scoring latency against the training snapshot.

## 6. LLM Explanation Failure
Disable/fallback the explanation layer without disabling recommendations. Inspect evidence payload, schema errors, unsupported claims, and provider status. The LLM must never become a ranking fallback.

## 7. Human-AI Study Safety
Do not mix simulated and real participant records. Do not start real recruitment merely because the study software is ready. Follow applicable institutional ethics/privacy procedures.

## 8. Rollback
Model registry must retain the previous champion. If a challenger causes operational or guardrail regression, reactivate the prior compatible bundle and record the rollback reason.
