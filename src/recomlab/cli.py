"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from recomlab import __version__
from recomlab.logging_utils import configure_logging


def _cmd_smoke(_: argparse.Namespace) -> int:
    logger = configure_logging()
    logger.info("recomlab smoke check ok (version %s)", __version__)
    print(f"recomlab {__version__} ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recomlab", description="RecomLab CLI")
    parser.add_argument("--version", action="version", version=f"recomlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="verify the installation")
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
