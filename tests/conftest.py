"""Shared fixtures for the eval/serve/export test suites.

Builds a tiny, real tokenizer and a tiny saved checkpoint so the API and export
tests exercise the full load path (no mocking of the model/tokenizer).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orkhon.config.schema import OptimConfig, TrainConfig
from orkhon.model.config import ModelConfig
from orkhon.model.transformer import Transformer
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.tokenizer.train import train_tokenizer
from orkhon.train.checkpoint import save_checkpoint
from orkhon.train.optim import build_optimizer
from orkhon.utils.seed import get_rng_state

# Corpus rich enough to train a small byte-level BPE with a usable vocab.
_CORPUS_LINES = [
    "hello world this is a tiny corpus for the orkhon tokenizer",
    "the quick brown fox jumps over the lazy dog again and again",
    "two plus two equals four and four plus four equals eight",
    "say something nice to the friendly assistant please",
    "orkhon is a from scratch transformer language model in pytorch",
    "the assistant replies with helpful and concise answers",
    "numbers like one two three four five six seven eight nine ten",
    "a quick test of generation sampling and chat completions here",
] * 4


def _tiny_model_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        dropout=0.0,
        attn_impl="manual",
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> Path:
    """Train a tiny tokenizer once per session; return its directory."""
    base = tmp_path_factory.mktemp("tokenizer")
    corpus = base / "corpus.txt"
    corpus.write_text("\n".join(_CORPUS_LINES) + "\n", encoding="utf-8")
    out_dir = base / "tok"
    train_tokenizer([corpus], out_dir, vocab_size=256, min_frequency=1)
    return out_dir


@pytest.fixture(scope="session")
def tiny_tokenizer(tokenizer_dir):
    return load_tokenizer(tokenizer_dir)


@pytest.fixture
def checkpoint_dir(tmp_path, tiny_tokenizer) -> Path:
    """Build a tiny model matching the tokenizer vocab and save a checkpoint."""
    cfg = _tiny_model_config(tiny_tokenizer.vocab_size)
    torch.manual_seed(0)
    model = Transformer(cfg).eval()

    optim = build_optimizer(model, OptimConfig())
    out_dir = tmp_path / "ckpt"
    save_checkpoint(
        out_dir=out_dir,
        model=model,
        optimizer=optim,
        scheduler_state=None,
        step=0,
        rng_state=get_rng_state(),
        model_cfg=cfg,
        train_cfg=TrainConfig(),
        tag="last",
    )
    return out_dir
