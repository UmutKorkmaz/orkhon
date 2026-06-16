"""Packed-sequence dataset over a flat uint16 token stream.

``PackedDataset`` memory-maps a ``.bin`` produced by
:func:`orkhon.data.tokenize.prepare_pretrain` and serves random fixed-length
windows for autoregressive pretraining. Each sample is a window of ``seq_len + 1``
tokens sliced from the stream; the model input is the first ``seq_len`` and the
target ``y`` is the same window shifted by one (next-token prediction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_DTYPE = np.uint16


def sample_starts(
    max_start: int,
    batch_size: int,
    *,
    seed: int | None = None,
    step: int | None = None,
    rank: int = 0,
) -> np.ndarray:
    """Sample window starts, optionally keyed by seed/step/rank.

    The legacy path (no ``seed``/``step``) uses NumPy's global RNG. Scale runs can
    pass both values so a resumed run sees the same data stream for a given step,
    while distributed ranks get disjoint deterministic streams.
    """
    if seed is None or step is None:
        return np.random.randint(0, max_start + 1, size=batch_size)
    rng = np.random.default_rng([int(seed), int(step), int(rank)])
    return rng.integers(0, max_start + 1, size=batch_size)


class PackedDataset:
    """Random fixed-length windows over a packed token-id ``.bin`` stream.

    Args:
        bin_path: path to a uint16 ``.bin`` token stream.
        seq_len: window length (model context); each batch row spans seq_len+1
            tokens of the underlying stream.
    """

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(f"packed bin not found: {bin_path}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        # memmap keeps memory low for large corpora; copied per-batch into tensors.
        self._data = np.memmap(bin_path, dtype=_DTYPE, mode="r")
        self.seq_len = seq_len

        # We need seq_len+1 tokens per window; the last valid start index is
        # len - (seq_len + 1). Require at least one full window.
        self._max_start = self._data.shape[0] - (seq_len + 1)
        if self._max_start < 0:
            raise ValueError(
                f"stream too short ({self._data.shape[0]} tokens) for seq_len "
                f"{seq_len}; need at least {seq_len + 1}"
            )

    def __len__(self) -> int:
        """Number of distinct window start positions."""
        return self._max_start + 1

    def get_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        *,
        seed: int | None = None,
        step: int | None = None,
        rank: int = 0,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Sample ``batch_size`` random windows.

        Returns:
            (x, y) each of shape ``[batch_size, seq_len]`` (long). ``y`` is ``x``
            shifted left by one token (the next-token targets).
        """
        # Random start indices in [0, max_start].
        starts = sample_starts(self._max_start, batch_size, seed=seed, step=step, rank=rank)

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for row, s in enumerate(starts):
            window = self._data[s : s + self.seq_len + 1].astype(np.int64)
            x[row] = window[:-1]
            y[row] = window[1:]

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if str(device) != "cpu":
            xt = xt.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
        return xt, yt  # type: ignore[return-value]
