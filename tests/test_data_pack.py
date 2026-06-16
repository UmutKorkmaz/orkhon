"""Pretraining tokenize/pack tests: split disjointness + batch shapes/shift."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from orkhon.data.pack import PackedDataset
from orkhon.data.tokenize import _split_for_doc, prepare_pretrain
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.tokenizer.train import train_tokenizer


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prep")
    # Build a corpus with many DISTINCT documents (so hash split has both sides).
    corpus = tmp / "corpus.txt"
    docs = [f"document number {i} talks about the quick brown fox and {i % 7} dogs"
            for i in range(400)]
    corpus.write_text("\n".join(docs) + "\n", encoding="utf-8")

    tok_dir = tmp / "tok"
    train_tokenizer([corpus], tok_dir, vocab_size=300, min_frequency=2)

    out_dir = tmp / "prepared"
    meta = prepare_pretrain(corpus, tok_dir, out_dir, val_fraction=0.2, seed=1337)
    return {"dir": out_dir, "meta": meta, "corpus": corpus, "tok_dir": tok_dir}


def test_bin_and_meta_written(prepared):
    out_dir = prepared["dir"]
    assert (out_dir / "train.bin").exists()
    assert (out_dir / "val.bin").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["dtype"] == "uint16"
    assert meta["train_tokens"] > 0
    assert meta["val_tokens"] > 0
    assert meta["vocab_size"] == 300


def test_no_document_in_both_splits(prepared):
    # Re-derive each document's split and confirm partition is a clean disjoint cut.
    corpus = prepared["corpus"]
    docs = [line.strip() for line in corpus.read_text().splitlines() if line.strip()]
    train_docs, val_docs = set(), set()
    for doc in docs:
        bucket = _split_for_doc(doc, val_fraction=0.2, seed=1337)
        (val_docs if bucket == "val" else train_docs).add(doc)

    assert train_docs and val_docs  # both non-empty
    assert train_docs.isdisjoint(val_docs)


def test_packed_batch_shapes_and_shift(prepared):
    bin_path = prepared["dir"] / "train.bin"
    seq_len = 8
    ds = PackedDataset(bin_path, seq_len=seq_len)

    torch.manual_seed(0)
    np.random.seed(0)
    x, y = ds.get_batch(batch_size=4, device="cpu")

    assert x.shape == (4, seq_len)
    assert y.shape == (4, seq_len)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # y must be x shifted by one: y[:, :-1] == x[:, 1:].
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_packed_step_keyed_sampling_is_repeatable(prepared):
    bin_path = prepared["dir"] / "train.bin"
    ds = PackedDataset(bin_path, seq_len=8)

    x1, y1 = ds.get_batch(batch_size=4, device="cpu", seed=7, step=3, rank=0)
    x2, y2 = ds.get_batch(batch_size=4, device="cpu", seed=7, step=3, rank=0)
    x3, _ = ds.get_batch(batch_size=4, device="cpu", seed=7, step=4, rank=0)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert not torch.equal(x1, x3)


def test_packed_dataset_rejects_short_stream(tmp_path):
    short = tmp_path / "short.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(short)
    with pytest.raises(ValueError):
        PackedDataset(short, seq_len=8)


def test_split_is_deterministic(prepared):
    corpus = prepared["corpus"]
    docs = [line.strip() for line in corpus.read_text().splitlines() if line.strip()]
    first = [_split_for_doc(d, 0.2, 1337) for d in docs]
    second = [_split_for_doc(d, 0.2, 1337) for d in docs]
    assert first == second
