# RecomLab Roadmap

## Vision

Build a rigorous, explainable recommendation workflow that is strong enough for a data-science portfolio and extensible into Human-AI research.

## M1 — Reproducible Data Foundation
Phases 1–2: data contracts, exposure logging schema, split/feature safety, baselines.

## M2 — Two-Stage Recommender
Phases 3–4: candidate retrieval, fusion, ranking, re-ranking, cold-start policy.

## M3 — Evaluation-Grade Research Platform
Phase 5: full-catalog/sampled evaluation distinction, uncertainty, slices, exposure-aware optional evaluation, leaderboard.

## M4 — Usable Product Workflow
Phase 6: API/dashboard, feedback logging, retraining, champion/challenger promotion.

## M5 — Human-Centered AI Extension
Phase 7: grounded LLM explanations, fidelity, controlled experiment framework, statistical analysis.

## M6 — Release Quality
Phase 8: E2E, drift/observability, Docker, CI, privacy/security, runbook.

## Success Metrics

| Area | Target |
|---|---|
| Reproducibility | Same fixture/config/seed reproduces deterministic baseline results |
| Leakage | 0 known test/future leakage incidents |
| Retrieval | Candidate recall reported separately from ranking |
| Evaluation | Full-catalog reference when feasible; sampled mode visibly labeled |
| Statistics | Uncertainty reported for champion/challenger metric differences |
| Cold start | Dedicated user/item slices reported |
| Workflow | Challenger cannot auto-promote when guardrails fail |
| Explanations | Unsupported LLM claims rejected in labeled fidelity fixtures |
| Human-AI | Synthetic/demo data never presented as real participant evidence |
