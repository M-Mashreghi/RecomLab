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
