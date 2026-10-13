# RecomLab Security, Privacy & Dependency Audit

Audit date: 2026-10-13 (Task 8.5). Re-run before every release.

## Secret Scanning

- `python scripts/scan_secrets.py` scans every git-tracked file for Anthropic/OpenAI-style
  keys, AWS access keys, GitHub tokens, private keys, and `*_API_KEY / *SECRET / *PASSWORD /
  *TOKEN = <value>` assignments; it exits non-zero on findings.
- Result: **0 findings**. `.env.example` contains placeholders only (enforced by a test);
  `.env` is git- and docker-ignored. API keys are wrapped in `SecretStr` and never logged
  (`JsonFormatter` redacts `api_key`, `token`, `password`, `secret`, auth headers).

## Dependency Audit

- Runtime core dependency: `numpy` only. Optional extras: `api` (fastapi, pydantic,
  uvicorn, httpx), `dashboard` (streamlit), `llm` (anthropic), `dev` (pytest, ruff, mypy,
  pyyaml). Each heavy library is optional and justified by its task; none is imported by
  core modules.
- `pip-audit` 2.10.1 on the development environment (all extras installed):
  **No known vulnerabilities found** (only the local `recomlab` package itself is skipped
  because it is not on PyPI).
- Release rule: a critical finding blocks release unless a documented waiver (package,
  advisory id, reason, expiry) is added below.

Waivers: none.

## Privacy Checklist

- [x] Pseudonymous ids only (`u1`, `ml_u<id>`); user attributes with identifier-like keys
      (`name`, `email`, `phone`, `address`, `ip`, `device_id`, ...) are rejected by the schema.
- [x] Feedback events reject free text and identifier fields (`free_text_not_allowed`).
- [x] Event logs store ids, positions, timestamps and numeric values only.
- [x] API responses never include secrets, raw histories, or candidate evidence; run
      manifests are served without platform details.
- [x] Dashboard shows pseudonymous ids only, never user attributes.
- [x] Structured logs scrub sensitive keys and email addresses.
- [x] Explanation evidence uses a bounded history window and is traceable to stored data.
- [x] Free-text research responses are sanitized before any coding and are not sent to an
      LLM by default.
- [x] Raw downloaded datasets (MovieLens) live in git-ignored `data/raw/`; only the simulated
      fixture is committed.

## Human-Study Data

- Simulated and real participant data are separated by `StudyConfig.data_mode`;
  `StudyDataStore` refuses to mix them, and simulated records carry `simulated=True` and
  `SIM-` ids. Reports print a SIMULATED banner.
- Real mode requires a recorded, approved ethics/IRB decision, a consent version, and a
  retention period. The repository does **not** authorize participant recruitment.
- Retention policy placeholders (to be set by the approving body before any real study):
  - raw responses: `<retention_days>` days, then deletion;
  - pseudonymous analysis tables: `<retention_days>` days;
  - consent records: per institutional policy;
  - storage location and access list: `<to be defined>`.
- Fidelity results are stored separately from perception outcomes (`FidelityLog`).
