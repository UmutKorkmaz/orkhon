"""Deterministic seeding and RNG state capture for exact checkpoint resume."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch (CPU/CUDA/MPS).

    ``deterministic=True`` requests deterministic kernels (warn-only so it does not
    hard-fail on ops without a deterministic implementation, e.g. on MPS).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state() -> dict[str, Any]:
    """Snapshot all RNG states so training can resume bit-for-bit."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`get_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """torch RNG state must be a uint8 CPU tensor; coerce loaded values."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.uint8, device="cpu")
    return torch.tensor(value, dtype=torch.uint8)
