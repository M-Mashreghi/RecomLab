"""Deterministic seed control."""

from __future__ import annotations

import hashlib
import importlib.util
import random

import numpy as np


def derive_seed(base_seed: int, *names: str) -> int:
    """Derive a stable 32-bit sub-seed from a base seed and component names."""
    payload = ":".join([str(base_seed), *names]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_rng(seed: int, *names: str) -> np.random.Generator:
    """Return an isolated NumPy generator; prefer this over global state."""
    return np.random.default_rng(derive_seed(seed, *names) if names else seed)


def set_global_seed(seed: int) -> dict[str, int]:
    """Seed Python, NumPy and (when installed) PyTorch global RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    seeded = {"python": seed, "numpy": seed % (2**32)}
    if importlib.util.find_spec("torch") is not None:  # pragma: no cover - optional
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = seed
    return seeded
