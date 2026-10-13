# RecomLab v0.1.0 Release Checklist

Status as of 2026-10-13. Unchecked items are not yet verified and block a public release.

## Build & Quality
- [x] `sh scripts/check.sh`: ruff, ruff format, mypy, full test suite (301 tests) pass locally.
- [x] Mandatory gate `pytest -m "leakage or metrics"` passes.
- [x] E2E fixture pipeline (`pytest -m e2e`) passes and is deterministic.
- [ ] CI workflow green on GitHub (not yet pushed; workflow validated locally).
- [ ] `docker compose up --build` builds and serves `/health` and the dashboard (Docker not
      available on the development machine).

## Reproducibility
- [x] Fresh-machine walkthrough: fresh clone + fresh Python 3.14 venv + non-editable install of
      `.[api,dashboard,dev]` → `recomlab smoke`, `recomlab bootstrap-demo`, API `/health` and
      `/ready` 200, recommendations served, 300/300 tests passed.
- [x] Benchmark report regenerated from code (`python scripts/run_benchmark.py`) →
      `docs/reports/benchmark_tiny.md` (simulated fixture, labeled as such).

## Methodology
- [x] Retrieval and ranking evaluated separately; full-catalog protocol by default; sampled
      results labeled and never mixed.
- [x] Champion chosen on validation only; test reported once.
- [x] Promotion gate with guardrails; rollback tested.

## Security & Privacy
- [x] Secret scan: 0 findings. pip-audit: no known vulnerabilities.
- [x] Privacy checklist complete (`docs/SECURITY_PRIVACY.md`).
- [x] No real participant data; simulated study data labeled; recruitment not authorized.

## Portfolio Demo
- [x] One-command demo bootstrap (`recomlab bootstrap-demo`).
- [x] README links to architecture, evaluation protocol, benchmark report, runbook.
- [ ] Dashboard screenshots (to capture after the Docker/demo stack runs on a desktop).
- [ ] Benchmark on MovieLens (requires local download via `scripts/fetch_movielens.py`; the
      in-repo report uses the 7-user simulated fixture, so its intervals are wide).
