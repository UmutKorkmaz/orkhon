"""Byte-level BPE tokenizer training for Orkhon.

We use HuggingFace ``tokenizers`` with a ByteLevel pre-tokenizer + BPE model. The
canonical special tokens (see :data:`orkhon.tokenizer.special_tokens.SPECIAL_TOKENS`)
are passed to the trainer FIRST so they receive ids ``0..7`` deterministically,
before the byte alphabet and learned merges. This ordering is a hard contract:
every checkpoint, the render/masking logic, and the export step all assume those
fixed ids. We assert it after training and refuse to write a broken tokenizer.

Outputs written to ``out_dir``:
    - ``tokenizer.json``            the full HF tokenizer (model + pre-tokenizer + decoder)
    - ``special_tokens_map.json``   role/control token -> string map
    - ``tokenizer_config.json``     metadata incl. the embedded chat template
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

from orkhon.tokenizer.special_tokens import (
    BOS,
    EOS,
    PAD,
    SPECIAL_TOKENS,
    UNK,
)

_CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")


def _read_chat_template() -> str:
    """Read the canonical Jinja chat template bundled with the package."""
    return _CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _iter_corpus_lines(corpus_paths: Sequence[str | Path]) -> Iterable[str]:
    """Yield non-empty lines from every corpus file (streaming, low memory)."""
    for path in corpus_paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


def train_tokenizer(
    corpus_paths: Sequence[str | Path] | str | Path,
    out_dir: str | Path,
    vocab_size: int,
    min_frequency: int = 2,
) -> None:
    """Train a ByteLevel-BPE tokenizer and write it to ``out_dir``.

    Args:
        corpus_paths: one path or a sequence of text files (one document per line).
        out_dir: destination directory (created if missing).
        vocab_size: target vocabulary size INCLUDING the 8 special tokens.
        min_frequency: minimum merge frequency for BPE.

    Raises:
        AssertionError: if the trained special-token ids are not exactly ``0..7``.
    """
    if isinstance(corpus_paths, (str, Path)):
        corpus_paths = [corpus_paths]
    corpus_paths = [Path(p) for p in corpus_paths]
    for p in corpus_paths:
        if not p.exists():
            raise FileNotFoundError(f"corpus file not found: {p}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ByteLevel BPE: lossless byte coverage, no <unk> needed for arbitrary text, but
    # we still register <unk> as a special control id for downstream code paths.
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # Special tokens FIRST => ids 0..7 (this is the load-bearing ordering).
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(_iter_corpus_lines(corpus_paths), trainer=trainer)

    # ByteLevel post-processor trims offsets correctly on decode; no auto specials.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    _assert_special_ids(tokenizer)

    tokenizer.save(str(out_dir / "tokenizer.json"))
    _write_special_tokens_map(out_dir)
    _write_tokenizer_config(out_dir, vocab_size=tokenizer.get_vocab_size())


def _assert_special_ids(tokenizer: Tokenizer) -> None:
    """Verify the special tokens occupy ids 0..7 in declared order."""
    for expected_id, tok in enumerate(SPECIAL_TOKENS):
        got = tokenizer.token_to_id(tok)
        assert got == expected_id, (
            f"special token {tok!r} got id {got}, expected {expected_id}. "
            "The trainer must receive SPECIAL_TOKENS first; ordering is a hard "
            "contract for checkpoint/render compatibility."
        )


def _write_special_tokens_map(out_dir: Path) -> None:
    """Write an HF-style special_tokens_map.json (control tokens + role markers)."""
    mapping = {
        "pad_token": PAD,
        "bos_token": BOS,
        "eos_token": EOS,
        "unk_token": UNK,
        # The role/control markers are additional special tokens.
        "additional_special_tokens": list(SPECIAL_TOKENS[4:]),
    }
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_tokenizer_config(out_dir: Path, vocab_size: int) -> None:
    """Write tokenizer_config.json embedding the canonical chat template."""
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1_000_000,
        "vocab_size": vocab_size,
        "pad_token": PAD,
        "bos_token": BOS,
        "eos_token": EOS,
        "unk_token": UNK,
        "clean_up_tokenization_spaces": False,
        "chat_template": _read_chat_template(),
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
