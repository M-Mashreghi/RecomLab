#!/usr/bin/env sh
# Run the local quality gate: lint, format check, type check, tests.
set -e
PY="${PYTHON:-python}"
"$PY" -m ruff check src tests scripts
"$PY" -m ruff format --check src tests scripts
"$PY" -m mypy
"$PY" -m pytest "$@"
