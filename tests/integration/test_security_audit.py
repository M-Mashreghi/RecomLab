import csv
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scan_secrets", ROOT / "scripts" / "scan_secrets.py")
assert spec and spec.loader
scan_secrets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_secrets)


def test_no_known_committed_secrets():
    assert scan_secrets.scan(ROOT) == []


def test_scanner_detects_planted_secrets(tmp_path):
    # Built at runtime so this test file itself never contains a secret-shaped literal.
    fake_key = "sk-" + "ant-api03-" + "abcdefghijklmnopqrstuvwxyz012345"
    fake_aws = "AKIA" + "ABCDEFGHIJKLMNOP"
    (tmp_path / "leak.py").write_text(
        f'ANTHROPIC_API_KEY = "{fake_key}"\naws = "{fake_aws}"\n', encoding="utf-8"
    )
    kinds = {k for _, _, k in scan_secrets.scan(tmp_path)}
    assert {"anthropic_key", "aws_access_key"} <= kinds


def test_demo_identifiers_are_pseudonymous():
    email = re.compile(r"[\w.+-]+@[\w-]+\.\w+")
    for name in ("users.csv", "items.csv", "interactions.csv"):
        with (ROOT / "data" / "fixtures" / "tiny" / name).open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        text = repr(rows)
        assert not email.search(text)
        assert not {"name", "email", "phone", "address", "ip"} & set(rows[0])
    with (ROOT / "data" / "fixtures" / "tiny" / "users.csv").open(encoding="utf-8") as fh:
        assert all(re.fullmatch(r"u\d+", r["user_id"]) for r in csv.DictReader(fh))


def test_core_dependencies_minimal_and_heavy_ones_optional():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == ["numpy>=1.26"]
    extras = project["optional-dependencies"]
    heavy = {"torch", "faiss", "hnswlib", "lightgbm", "xgboost", "openai", "anthropic"}
    core_names = {re.split(r"[<>=\[]", d)[0] for d in project["dependencies"]}
    assert not heavy & core_names
    assert {re.split(r"[<>=\[]", d)[0] for d in extras["llm"]} == {"anthropic"}


def test_privacy_checklist_completed():
    doc = (ROOT / "docs" / "SECURITY_PRIVACY.md").read_text(encoding="utf-8")
    assert "- [ ]" not in doc.split("## Privacy Checklist")[1].split("##")[0]
    for heading in ("## Secret Scanning", "## Dependency Audit", "## Human-Study Data"):
        assert heading in doc
