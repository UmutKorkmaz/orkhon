"""Pretraining corpus tokenization + deterministic train/val split.

``prepare_pretrain`` reads documents, tokenizes each with the trained tokenizer,
appends an ``<eos>`` separator, and concatenates into a flat token stream written
as little-endian ``uint16`` (valid because pretraining vocab < 65536). The split is
by stable document hash so a document never lands in both train and val, and the
split is identical across runs.

Outputs (under ``out_dir``):
    - ``train.bin`` / ``val.bin``  flat uint16 token-id streams
    - ``meta.json``                {dtype, vocab_size, train_tokens, val_tokens}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orkhon.data.normalize import iter_documents
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.utils.hashing import stable_unit_interval

# uint16 holds vocab ids up to 65535; pretraining vocab must stay below this.
_DTYPE = np.uint16
_DTYPE_NAME = "uint16"
_MAX_UINT16_VOCAB = 1 << 16


def _split_for_doc(doc: str, val_fraction: float, seed: int) -> str:
    """Return 'val' or 'train' for a document via stable hashing.

    Keyed by ``seed`` + document text so the assignment is deterministic and a
    document's split never changes between runs (no leakage across train/val).
    """
    u = stable_unit_interval(f"{seed}:{doc}")
    return "val" if u < val_fraction else "train"


def prepare_pretrain(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 1337,
) -> dict:
    """Tokenize ``corpus_path`` into packed train/val bins under ``out_dir``.

    Args:
        corpus_path: .txt (one doc per line) or .jsonl ({"text": ...}).
        tokenizer_dir: directory containing tokenizer.json.
        out_dir: destination for train.bin / val.bin / meta.json.
        val_fraction: fraction of documents (by hash) routed to validation.
        seed: hash seed for the deterministic split.

    Returns:
        The meta dict that was written to meta.json.
    """
    tokenizer = load_tokenizer(tokenizer_dir)
    if tokenizer.vocab_size > _MAX_UINT16_VOCAB:
        raise ValueError(
            f"vocab_size {tokenizer.vocab_size} exceeds uint16 capacity "
            f"({_MAX_UINT16_VOCAB}); pretraining .bin format requires < 65536"
        )
    eos = tokenizer.special.eos

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens: list[int] = []
    val_tokens: list[int] = []

    for doc in iter_documents(corpus_path):
        ids = tokenizer.encode(doc)
        ids.append(eos)  # document separator
        bucket = _split_for_doc(doc, val_fraction, seed)
        (val_tokens if bucket == "val" else train_tokens).extend(ids)

    train_arr = np.asarray(train_tokens, dtype=_DTYPE)
    val_arr = np.asarray(val_tokens, dtype=_DTYPE)

    train_arr.tofile(out_dir / "train.bin")
    val_arr.tofile(out_dir / "val.bin")

    meta = {
        "dtype": _DTYPE_NAME,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_arr.size),
        "val_tokens": int(val_arr.size),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta
