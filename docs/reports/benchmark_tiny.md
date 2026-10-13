# RecomLab Benchmark (tiny)

## Warnings & Caveats

- **Dataset is SIMULATED (synthetic fixture); results are not research evidence.**
- **Exposure-aware (off-policy) correction unavailable: no valid logged propensities.**

## Protocol

- Evaluation protocol: `full`
- Primary metric: `ndcg@5`
- Label policy: `labels-default-c30149fe93`
- Metric definitions: `{"aggregation": "mean over users with >=1 relevant item", "gain": "linear", "ks": [5, 10], "metrics": ["precision", "recall", "ndcg", "map", "mrr", "hit_rate"], "relevance": "binary except graded NDCG"}`
- Exposure correction: `unavailable`

## Ranking-Stage Results

| Model | Partition | Dataset | Split | ndcg@5 | 95% CI | Users |
|---|---|---|---|---|---|---|
| random@1 | test | tiny@1 | `split-leave_last_n-a23c80a2c699` | 0.5000 | [0.0000, 1.0000] | 3 |
| popularity@1 | test | tiny@1 | `split-leave_last_n-a23c80a2c699` | 0.4392 | [0.3869, 0.5000] | 3 |
| als@als-3e33e9e7624d | test | tiny@1 | `split-leave_last_n-a23c80a2c699` | 0.5436 | [0.0000, 1.0000] | 3 |
| pipeline@h0a34f359 | test | tiny@1 | `split-leave_last_n-a23c80a2c699` | 0.8333 | [0.5000, 1.0000] | 3 |

## Candidate-Stage (Retrieval) Diagnostics

- Not recorded for these runs (full-catalog recommenders without a retrieval stage).

## Paired Comparisons

| Comparison | Δ | 95% CI | Wilcoxon p | Conclusion | Users |
|---|---|---|---|---|---|
| random@1 vs popularity@1 | +0.0608 | [-0.5000, +0.5693] | 0.593 | uncertain | 3 |
| als@als-3e33e9e7624d vs popularity@1 | +0.1045 | [-0.5000, +0.6131] | 0.593 | uncertain | 3 |
| pipeline@h0a34f359 vs popularity@1 | +0.3942 | [+0.1131, +0.5693] | 0.109 | uncertain | 3 |

## Slice Reports

- random@1: none
- popularity@1: none
- als@als-3e33e9e7624d: none
- pipeline@h0a34f359: none

## Limitations

- Offline metrics measure agreement with logged held-out interactions, which reflect what users were exposed to (exposure bias); they are not causal estimates of online impact.
- Uncertainty intervals resample users and do not cover dataset or split variability.
- Results on the simulated fixture are for software validation only.

_Uncertainty: user-level bootstrap, 1000 resamples, seed 0._

## Model Selection (validation only)

- Policy `policy-62fd6799344b`: primary `ndcg@5`, guardrails ['catalog_coverage@5 >= 0.05']
- Decision: als@als-14dc13247975 has the best validation ndcg@5=0.612115 among 4 guardrail-passing runs
- Note: multiple Pareto-optimal runs: no single model dominates
- Champion test result (reported once): als@als-3e33e9e7624d
