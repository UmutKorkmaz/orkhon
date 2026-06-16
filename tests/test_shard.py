"""The sharded tokenize pipeline is byte-identical to the single-.bin one."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orkhon.data.pack import PackedDataset
from orkhon.data.shard import ShardedPackedDataset, prepare_pretrain_sharded
from orkhon.data.tokenize import prepare_pretrain
from orkhon.tokenizer.train import train_tokenizer


def _make_corpus(tmp_path: Path) -> Path:
    docs = [
        "the little fox jumped over the fence",
        "a robot learned to dance in the rain",
        "türkiye'nin başkenti ankara'dır",
        "göktürkler orta asya'da büyük bir devlet kurdu",
    ] * 200
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(docs), encoding="utf-8")
    return p


def _tokenizer(tmp_path: Path, corpus: Path) -> Path:
    tok = tmp_path / "tok"
    train_tokenizer(corpus_paths=str(corpus), out_dir=str(tok), vocab_size=400)
    return tok


def test_sharded_matches_single_bin(tmp_path):
    corpus = _make_corpus(tmp_path)
    tok = _tokenizer(tmp_path, corpus)

    flat = tmp_path / "flat"
    prepare_pretrain(corpus, tok, flat, val_fraction=0.0, seed=1337)
    sharded = tmp_path / "sharded"
    # Tiny shard size forces multiple shards, exercising the straddle logic.
    prepare_pretrain_sharded(corpus, tok, sharded, val_fraction=0.0, seed=1337,
                             shard_tokens=256)

    man = json.loads((sharded / "manifest.json").read_text())
    assert man["train_tokens"] > 0
    assert man["shards"], "expected >=1 shard written"

    flat_arr = np.memmap(flat / "train.bin", dtype=np.uint16, mode="r")
    # Reassemble shards in order and compare to the flat stream.
    recon = np.concatenate([
        np.memmap(sharded / "train" / s["shard"], dtype=np.uint16, mode="r")
        for s in man["shards"]
    ])
    assert recon.shape == flat_arr.shape, (recon.shape, flat_arr.shape)
    assert np.array_equal(recon, flat_arr), "sharded stream != flat stream"


def test_sharded_dataset_serves_windows(tmp_path):
    corpus = _make_corpus(tmp_path)
    tok = _tokenizer(tmp_path, corpus)
    sharded = tmp_path / "sharded"
    prepare_pretrain_sharded(corpus, tok, sharded, val_fraction=0.0, seed=1337,
                             shard_tokens=256)

    ds = ShardedPackedDataset(sharded, seq_len=16)
    assert len(ds) > 0
    x, y = ds.get_batch(8, "cpu")
    assert x.shape == (8, 16) and y.shape == (8, 16)
    # next-token shift: y[:, :-1] == x[:, 1:]
    assert np.array_equal(y[:, :-1].numpy(), x[:, 1:].numpy())


def test_sharded_dataset_step_keyed_sampling_is_repeatable(tmp_path):
    corpus = _make_corpus(tmp_path)
    tok = _tokenizer(tmp_path, corpus)
    sharded = tmp_path / "sharded"
    prepare_pretrain_sharded(corpus, tok, sharded, val_fraction=0.0, seed=1337,
                             shard_tokens=128)

    ds = ShardedPackedDataset(sharded, seq_len=16)
    x1, y1 = ds.get_batch(8, "cpu", seed=11, step=5, rank=0)
    x2, y2 = ds.get_batch(8, "cpu", seed=11, step=5, rank=0)
    x3, _ = ds.get_batch(8, "cpu", seed=11, step=5, rank=1)

    assert np.array_equal(x1.numpy(), x2.numpy())
    assert np.array_equal(y1.numpy(), y2.numpy())
    assert not np.array_equal(x1.numpy(), x3.numpy())
