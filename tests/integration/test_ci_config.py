import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_ci_config_is_syntax_valid(ci):
    triggers = ci.get("on", ci.get(True))  # YAML 1.1 parses the bare key `on` as True
    assert set(triggers) == {"push", "pull_request"}
    assert set(ci["jobs"]) == {"quality", "tests", "docker"}
    for job in ci["jobs"].values():
        assert job["runs-on"] and job["steps"]
        for step in job["steps"]:
            assert "uses" in step or "run" in step
            if "uses" in step:
                assert "@" in step["uses"]  # pinned action versions


def test_quality_gates_present_and_blocking(ci):
    runs = [s.get("run", "") for j in ci["jobs"].values() for s in j["steps"]]
    joined = "\n".join(runs)
    for cmd in (
        "ruff check",
        "ruff format --check",
        "mypy",
        'pytest -m "leakage or metrics"',
        "pytest tests/unit",
        "pytest tests/integration",
        "pytest tests/e2e",
    ):
        assert cmd in joined, cmd
    for job in ci["jobs"].values():
        assert not job.get("continue-on-error")
        assert not any(s.get("continue-on-error") for s in job["steps"])
    assert ci["jobs"]["tests"]["needs"] == "quality"
    assert ci["jobs"]["docker"]["needs"] == "tests"


def test_no_deploy_secrets_for_untrusted_prs(ci):
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request_target" not in text
    assert "${{ secrets" not in text and "secrets.GITHUB" not in text
    assert ci["permissions"] == {"contents": "read"}
    assert ci["jobs"]["docker"]["if"] == "github.event_name == 'push'"
    assert "docker push" not in text


def test_intentional_failing_leakage_test_blocks_pipeline(tmp_path):
    """The CI gate command exits non-zero when any leakage test fails."""
    (tmp_path / "test_injected.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.leakage\n"
        "def test_injected_leak():\n"
        "    assert False, 'test event leaked into training'\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    leakage: x\n    metrics: x\n")
    gate = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "leakage or metrics", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert gate.returncode != 0
    assert "injected" in gate.stdout
