"""dtype resolution and autocast context with device-aware guards.

MPS autocast and reduced precision are fragile; the smoke path defaults to float32.
CUDA prefers bfloat16. CPU stays in float32.
"""

from __future__ import annotations

import contextlib
from typing import ContextManager

import torch

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(name: str, device: torch.device | str = "cpu") -> torch.dtype:
    """Map a dtype name to ``torch.dtype``, downgrading unsafe combos.

    ``auto`` -> bfloat16 on CUDA, float32 elsewhere. float16/bfloat16 on CPU is
    forced to float32 (CPU autocast support is limited and slow).
    """
    dev = torch.device(device).type
    if name == "auto":
        return torch.bfloat16 if dev == "cuda" else torch.float32
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; choose from {sorted(_DTYPES)} or 'auto'")
    dtype = _DTYPES[name]
    if dev == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def autocast_ctx(device: torch.device | str, dtype: torch.dtype) -> ContextManager:
    """Return an autocast context, or a no-op when autocast does not apply.

    Autocast is only engaged for reduced precision on CUDA/MPS. On CPU, or when
    dtype is float32, this is a null context so the same training code runs everywhere.
    """
    dev = torch.device(device).type
    if dtype == torch.float32 or dev == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev, dtype=dtype)
