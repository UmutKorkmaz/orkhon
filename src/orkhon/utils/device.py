"""Device resolution (CUDA > MPS > CPU) with an explicit override."""

from __future__ import annotations

import torch


def resolve_device(pref: str = "auto") -> torch.device:
    """Resolve a device preference string to a concrete ``torch.device``.

    ``auto`` picks CUDA, then MPS, then CPU. Any other value is passed through
    (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``).
    """
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def device_type(device: torch.device | str) -> str:
    """Return the bare device type string (``cuda``/``mps``/``cpu``)."""
    return torch.device(device).type
