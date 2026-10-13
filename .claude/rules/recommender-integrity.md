# Recommender Integrity

- No future/test leakage.
- Popularity/features/indexes use only the allowed historical horizon.
- Training negative sampling is synthetic training data, not evaluation truth.
- Baselines are mandatory.
- Cold-start behavior is explicit.
- Candidate recall and ranking quality are reported separately.
