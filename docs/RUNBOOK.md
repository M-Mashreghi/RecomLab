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

## 9. Scheduled Retraining (`recomlab retrain`)

```bash
recomlab retrain --spec configs/demo_bundle.json [--events data/events.db]
```

One workflow id links every stage (`recomlab.experiments.retraining.run_retraining`):

1. **Snapshot**: base dataset + logged feedback (as interactions) are cleaned and frozen to
   `data/processed/<dataset>/snap-<checksum>/` with manifest + data card, and registered.
2. **Validate/split**: leakage-safe split of the frozen snapshot.
3. **Train + evaluate**: the bundle spec is trained on the training partition and evaluated
   on validation with the standard protocol; the run (with per-user artifacts) is logged and
   its notes carry the workflow id.
4. **Artifacts**: final artifacts are trained on the full snapshot and registered as a
   **challenger** bundle whose metadata links workflow id, dataset version/checksum, and the
   validation run id.
5. **No auto-promotion**: the workflow never changes the active bundle; promotion goes
   through the champion/challenger gate. On any failure a `failed` run manifest is logged
   and the previous active bundle stays untouched.

Schedule it with the host's scheduler (cron, Windows Task Scheduler, CI schedule); no queue
infrastructure is required.

## 10. End-to-End Regression Suite

`pytest -m e2e` (`tests/e2e/test_pipeline_e2e.py`) runs the full deterministic fixture
pipeline twice: ingestion + cleaning → leakage-checked split → offline evaluation of the
popularity baseline and the two-stage pipeline (registry + per-user artifacts + report) →
bundle registration and serving (warm user, cold user, empty-candidate cold-start fallback,
collaborative-ranker unknown-user fallback) → grounded explanation with a mocked LLM provider
and fidelity validation. It asserts identical metrics across runs, no held-out events in
serving history, bounded artifacts (< 60 files, < 5 MB), and needs no live external API.

## 11. Observability, Drift & Skew (`recomlab.monitoring.observability`)

- **Structured logs**: `JsonFormatter` emits one JSON object per line with `request_id`,
  `trace_id`, `run_id`, `model_id`, `bundle_id`, `workflow_id` lifted to top level. Sensitive
  keys (`api_key`, `token`, `password`, `secret`, `email`, `text`, `history`, `free_text`,
  auth headers) are replaced by `[REDACTED]` and email-like strings by `[EMAIL]`. The API logs
  every served recommendation with trace and bundle ids.
- **Serving metrics**: `ServingMonitor` (exposed at `GET /monitoring`) tracks requests,
  empty-candidate rate, fallback counts, latency p50/p95/max, and distinct items served.
- **Drift / train-serving skew**: `profile()` summarizes event volume, catalog size,
  event-type mix, user-activity and item-popularity bucket distributions, and the feature
  schema/means. `check_drift(train, serving, DriftThresholds, serving_metrics)` warns on
  catalog-size or volume change, event-mix total variation distance, activity/popularity
  PSI, feature-schema mismatch, feature-mean shifts, empty-candidate rate, and p95 latency.
  All thresholds are configurable; any warning should trigger the checks in §5.
