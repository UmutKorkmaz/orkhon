"""Streaming, sharded pretraining-tokenize — no RAM ceiling.

``prepare_pretrain`` (data/tokenize.py) holds the whole corpus in two Python lists
and writes a single ``train.bin`` — it dies at ~300–500M tokens. This module is the
scale fix: it streams documents one at a time, tokenizes each, and flushes
fixed-size shards (~256M tokens each) to ``train/shard_NNNNN.bin`` so peak memory is
one shard, not the corpus. Validation is usually tiny, so it stays a single file.

``ShardedPackedDataset`` memory-maps all shards as if concatenated and serves random
windows — a drop-in for :class:`~orkhon.data.PackedDataset` (same ``get_batch``).

Manifest ``manifest.json`` records per-shard token counts + totals, so a resumable
run can skip already-tokenized shards and a dataset can be sized without remapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from orkhon.data.normalize import iter_documents
from orkhon.data.pack import sample_starts
from orkhon.data.tokenize import _split_for_doc
from orkhon.tokenizer.tokenizer import load_tokenizer

_DTYPE = np.uint16
_DEFAULT_SHARD_TOKENS = 1 << 28  # ~268M tokens / ~512MB per shard.


def prepare_pretrain_sharded(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.01,
    seed: int = 1337,
    shard_tokens: int = _DEFAULT_SHARD_TOKENS,
) -> dict:
    """Stream-tokenize ``corpus_path`` into sharded train bins + a single val bin.

    Peak memory is one shard (``shard_tokens`` ids), not the whole corpus.
    """
    tokenizer = load_tokenizer(tokenizer_dir)
    eos = tokenizer.special.eos

    out_dir = Path(out_dir)
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    buf = np.empty(shard_tokens, dtype=_DTYPE)
    buf_pos = 0
    shard_idx = 0
    train_total = 0
    val_chunks: list[np.ndarray] = []
    written_shards: list[dict] = []

    def _flush():
        nonlocal buf_pos, shard_idx, train_total
        if buf_pos == 0:
            return
        name = f"shard_{shard_idx:05d}.bin"
        buf[:buf_pos].tofile(train_dir / name)
        written_shards.append({"shard": name, "tokens": int(buf_pos)})
        train_total += buf_pos
        shard_idx += 1
        buf_pos = 0

    for doc in iter_documents(corpus_path):
        ids = tokenizer.encode(doc)
        ids.append(eos)
        bucket = _split_for_doc(doc, val_fraction, seed)
        if bucket == "val":
            val_chunks.append(np.asarray(ids, dtype=_DTYPE))
            continue
        i = 0
        # A single document may straddle shard boundaries; that is fine — windows
        # already cross <eos> in PackedDataset, so splitting a doc across shards
        # is no worse than the existing packing behavior.
        while i < len(ids):
            room = shard_tokens - buf_pos
            take = min(room, len(ids) - i)
            buf[buf_pos : buf_pos + take] = ids[i : i + take]
            buf_pos += take
            i += take
            if buf_pos >= shard_tokens:
                _flush()
    _flush()

    val_total = 0
    if val_chunks:
        val_arr = np.concatenate(val_chunks)
        val_arr.tofile(out_dir / "val.bin")
        val_total = int(val_arr.size)

    manifest = {
        "dtype": str(_DTYPE),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_total,
        "val_tokens": val_total,
        "shard_tokens": shard_tokens,
        "shards": written_shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ShardedPackedDataset:
    """Random fixed-length windows over sharded ``train/shard_*.bin`` files.

    Memmaps each shard; a global token index maps to ``(shard, local)`` via prefix
    sums. ``get_batch`` is identical to :class:`PackedDataset.get_batch`.
    """

    def __init__(self, prepared_dir: str | Path, seq_len: int) -> None:
        prepared_dir = Path(prepared_dir)
        manifest_path = prepared_dir / "manifest.json"
        if not manifest_path.exists():
            # Back-compat: a flat train.bin directory (from prepare_pretrain).
            flat = prepared_dir / "train.bin"
            if not flat.exists():
                raise FileNotFoundError(f"no manifest.json or train.bin in {prepared_dir}")
            self._arrays = [np.memmap(flat, dtype=_DTYPE, mode="r")]
            self._offsets = [0]
            self._total = int(self._arrays[0].shape[0])
        else:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._arrays, self._offsets, self._total = [], [], 0
            for s in man["shards"]:
                arr = np.memmap(prepared_dir / "train" / s["shard"], dtype=_DTYPE, mode="r")
                self._offsets.append(self._total)
                self._arrays.append(arr)
                self._total += int(s["tokens"])

        self.seq_len = seq_len
        self._max_start = self._total - (seq_len + 1)
        if self._max_start < 0:
            raise ValueError(
                f"stream too short ({self._total} tokens) for seq_len {seq_len}"
            )

    def __len__(self) -> int:
        return self._max_start + 1

    def _window(self, start: int) -> np.ndarray:
        end = start + self.seq_len + 1
        out = np.empty(self.seq_len + 1, dtype=np.int64)
        # Resolve the global [start, end) range across shards.
        g = start
        for arr, off in zip(self._arrays, self._offsets):
            shard_len = arr.shape[0]
            lo = max(g, off)
            hi = min(end, off + shard_len)
            if lo < hi:
                out[lo - start : hi - start] = arr[lo - off : hi - off]
            if end <= off + shard_len:
                break
        return out

    def get_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        *,
        seed: int | None = None,
        step: int | None = None,
        rank: int = 0,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        starts = sample_starts(self._max_start, batch_size, seed=seed, step=step, rank=rank)
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for row, s in enumerate(starts):
            w = self._window(s)
            x[row] = w[:-1]
            y[row] = w[1:]
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        if str(device) != "cpu":
            xt, yt = xt.to(device, non_blocking=True), yt.to(device, non_blocking=True)
        return xt, yt  # type: ignore[return-value]
