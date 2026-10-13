# RecomLab — Claude Code Instructions

RecomLab is a research-oriented Recommendation & Ranking Engine with an optional Human-AI explanation study track.

## Read Before Coding

For every task, read:
1. `docs/PROJECT_PHASES_AND_TASKS.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DATA_CONTRACT.md`
4. `docs/EVALUATION_PROTOCOL.md`
5. `docs/TASKS.md`
6. applicable `.claude/rules/*.md`
7. the exact task scope

Do not invent requirements that conflict with these documents.

## Task Workflow

1. Inspect current repository state.
2. Implement only the requested task.
3. Add/update tests.
4. Run relevant tests and quality checks.
5. Fix failures before completion.
6. Update documentation and ledgers only after verification.
7. Do not start the next task unless asked.

## Non-Negotiable Recommender Rules

- Retrieval and ranking are separate stages.
- Candidate recall is measured before ranking quality.
- Validation/test information never enters training features, popularity counts, negative sampling, or retrieval indexes.
- Time-aware experiments preserve event order.
- Training negative sampling is never presented as full-catalog evaluation.
- Full-catalog evaluation is preferred when feasible; sampled evaluation is explicitly labeled.
- Simple baselines are mandatory.
- Test data do not tune models or choose hyperparameters.
- Experiment manifests track dataset, split, features, model, metrics, seed, and software state.

## Human-AI / LLM Rules

LLMs may generate explanations and assist qualitative analysis. They must not invent user preferences, item facts, interactions, metrics, or experiment results.

The explanation path never modifies ranking scores.

Separate:
- recommendation relevance,
- explanation fidelity,
- trust,
- acceptance,
- usefulness,
- satisfaction,
- persuasion.

Human-subject recruitment is not an engineering task. Demo/synthetic study data must be labeled simulated. Real participant workflows require applicable institutional approval and privacy controls.

## Architecture

Keep separate modules for:
- ingestion/data contracts,
- dataset/splits/features,
- candidate retrieval,
- ranking/re-ranking,
- evaluation/statistics,
- experiment/model registry,
- API/dashboard,
- feedback workflow,
- explanation/research extension.

Do not put training or metric logic in UI/API controllers.

## Dependencies

Prefer simple local implementations first. External ANN engines, distributed queues, vector databases, feature stores, and heavy MLOps platforms are not required for v1 unless a task explicitly justifies them.

## Security & Privacy

- Never commit secrets.
- Minimize PII; use pseudonymous demo IDs.
- Do not log raw private user histories unnecessarily.
- Free-text research responses require extra care and are not sent to an LLM by default.

## Completion

A task is complete only when its implementation exists, tests pass, methodological invariants hold, and documentation is synchronized.
