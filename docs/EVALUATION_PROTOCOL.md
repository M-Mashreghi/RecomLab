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
