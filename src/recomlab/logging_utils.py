"""Deterministic logging bootstrap.

Logging is configured once, with a fixed format and no wall-clock-dependent
handlers, so that log output is stable across runs and environments.
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure the ``recomlab`` logger idempotently and return it."""
    global _CONFIGURED
    logger = logging.getLogger("recomlab")
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``recomlab`` namespace."""
    if not name.startswith("recomlab"):
        name = f"recomlab.{name}"
    return logging.getLogger(name)
