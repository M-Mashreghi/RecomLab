import subprocess
import sys

import recomlab
from recomlab.cli import main
from recomlab.logging_utils import configure_logging, get_logger


def test_package_imports_with_version():
    assert recomlab.__version__ == "0.1.0"


def test_cli_smoke_in_process(capsys):
    assert main(["smoke"]) == 0
    assert "ok" in capsys.readouterr().out


def test_cli_smoke_as_module():
    result = subprocess.run(
        [sys.executable, "-m", "recomlab", "smoke"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "recomlab 0.1.0 ok" in result.stdout


def test_logging_is_idempotent():
    first = configure_logging()
    count = len(first.handlers)
    second = configure_logging()
    assert first is second
    assert len(second.handlers) == count
    assert get_logger("x").name == "recomlab.x"
