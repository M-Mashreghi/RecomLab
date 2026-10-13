# RecomLab — Project Structure

```text
RecomLab/
├── CLAUDE.md
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .claude/rules/
│   ├── architecture.md
│   ├── recommender-integrity.md
│   ├── offline-evaluation.md
│   ├── experimentation.md
│   ├── human-ai-research.md
│   ├── llm-analysis.md
│   ├── testing.md
│   └── documentation.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_CONTRACT.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── HUMAN_AI_STUDY.md
│   ├── ROADMAP.md
│   ├── PHASES.md
│   ├── PROJECT_PHASES_AND_TASKS.md
│   ├── TASKS.md
│   ├── DIAGRAMS.md
│   ├── DEPLOYMENT.md
│   ├── RUNBOOK.md
│   └── REVIEW_NOTES_v2.md
├── src/recomlab/
│   ├── config/
│   ├── domain/
│   ├── ingestion/
│   ├── datasets/
│   ├── features/
│   ├── candidates/
│   ├── rankers/
│   ├── reranking/
│   ├── evaluation/
│   ├── experiments/
│   ├── registry/
│   ├── explanations/
│   ├── services/
│   ├── api/
│   └── dashboard/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/{raw,processed,fixtures}/
├── artifacts/{models,metrics,reports}/
├── notebooks/exploration/
├── scripts/
└── .github/workflows/
```

Reusable logic belongs in `src/recomlab/`; notebooks are exploratory only.
