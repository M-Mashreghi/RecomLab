"""Scan git-tracked files (or a directory) for committed secrets.

Usage: python scripts/scan_secrets.py [PATH]   (exit code 1 when findings exist)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)\b[A-Z_]*(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*[=:]\s*['\"]?([A-Za-z0-9/+_\-]{16,})"
    ),
}
ALLOWLIST_VALUES = {"test-admin-token-not-a-secret"}
SKIP_SUFFIXES = {".png", ".jpg", ".npz", ".db", ".sqlite"}


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        return [root / p for p in out]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def scan(root: Path) -> list[tuple[str, int, str]]:
    findings = []
    for path in tracked_files(root):
        if path.suffix in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(lines, 1):
            for name, pattern in PATTERNS.items():
                m = pattern.search(line)
                if not m:
                    continue
                value = m.group(1) if m.groups() else m.group(0)
                if value in ALLOWLIST_VALUES:
                    continue
                findings.append((str(path.relative_to(root)), n, name))
    return findings


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    findings = scan(root)
    for path, line, kind in findings:
        print(f"{path}:{line}: possible {kind}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
