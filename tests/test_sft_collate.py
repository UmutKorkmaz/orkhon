"""SFT/DPO dataset collation tests: padding, IGNORE_INDEX, shared prompt prefix."""

from __future__ import annotations

import json

import pytest
import torch

from orkhon.data.dataset import DPODataset, SFTDataset
from orkhon.tokenizer.render import IGNORE_INDEX
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.tokenizer.train import train_tokenizer


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("collate_tok")
    corpus = tmp / "corpus.txt"
    lines = []
    for i in range(300):
        lines.append(f"what is {i % 10} plus {i % 7} answer is some number here")
        lines.append("hello there i am a helpful assistant for you today")
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp / "tok"
    train_tokenizer([corpus], out, vocab_size=320, min_frequency=2)
    return load_tokenizer(out)


@pytest.fixture()
def sft_path(tmp_path, tokenizer):
    rows = [
        {"messages": [
            {"role": "user", "content": "what is 2 plus 2"},
            {"role": "assistant", "content": "Answer: 4."},
        ]},
        {"messages": [
            {"role": "user", "content": "what is 1 plus 9 and tell me more please"},
            {"role": "assistant", "content": "Answer: 10."},
        ]},
    ]
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_sft_collate_pads_to_longest(tokenizer, sft_path):
    ds = SFTDataset(sft_path, tokenizer)
    batch = [ds[0], ds[1]]
    out = ds.collate(batch)

    pad_id = tokenizer.special.pad
    lens = [len(item["input_ids"]) for item in batch]
    max_len = max(lens)

    assert out["input_ids"].shape == (2, max_len)
    assert out["labels"].shape == (2, max_len)
    assert out["attention_mask"].shape == (2, max_len)

    # Shorter row is padded with pad id and IGNORE_INDEX on labels.
    short_idx = lens.index(min(lens))
    real_len = lens[short_idx]
    pad_region_ids = out["input_ids"][short_idx, real_len:]
    pad_region_labels = out["labels"][short_idx, real_len:]
    assert torch.all(pad_region_ids == pad_id)
    assert torch.all(pad_region_labels == IGNORE_INDEX)


def test_sft_attention_mask_matches_non_pad(tokenizer, sft_path):
    ds = SFTDataset(sft_path, tokenizer)
    batch = [ds[0], ds[1]]
    out = ds.collate(batch)
    lens = [len(item["input_ids"]) for item in batch]

    for i, real_len in enumerate(lens):
        assert torch.all(out["attention_mask"][i, :real_len])  # real tokens True
        assert not torch.any(out["attention_mask"][i, real_len:])  # pad False


def test_sft_labels_only_assistant_trained(tokenizer, sft_path):
    ds = SFTDataset(sft_path, tokenizer)
    item = ds[0]
    # Trainable labels equal assistant content ids + <|end|>.
    expected = tokenizer.encode("Answer: 4.") + [tokenizer.special.end]
    trainable = [l for l in item["labels"] if l != IGNORE_INDEX]
    assert trainable == expected


@pytest.fixture()
def dpo_path(tmp_path):
    rows = [
        {"prompt": [{"role": "user", "content": "what is 2 plus 2"}],
         "chosen": "Answer: 4.",
         "rejected": "hmm maybe four i am not so sure about this one really"},
        {"prompt": "what is 1 plus 1",
         "chosen": "Answer: 2.",
         "rejected": "could be two or something else entirely who knows"},
    ]
    p = tmp_path / "dpo.jsonl"
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_dpo_chosen_rejected_share_prompt_prefix(tokenizer, dpo_path):
    ds = DPODataset(dpo_path, tokenizer)
    item = ds[0]
    plen = item["prompt_len"]

    chosen_prefix = item["chosen_input_ids"][:plen]
    rejected_prefix = item["rejected_input_ids"][:plen]
    assert chosen_prefix == rejected_prefix

    # Prompt positions are masked out in both label sets.
    assert all(l == IGNORE_INDEX for l in item["chosen_labels"][:plen])
    assert all(l == IGNORE_INDEX for l in item["rejected_labels"][:plen])
    # Completion positions are supervised (not all ignored).
    assert any(l != IGNORE_INDEX for l in item["chosen_labels"][plen:])


def test_dpo_collate_shapes_and_masks(tokenizer, dpo_path):
    ds = DPODataset(dpo_path, tokenizer)
    batch = [ds[0], ds[1]]
    out = ds.collate(batch)

    assert out["prompt_len"].shape == (2,)
    for side in ("chosen", "rejected"):
        ids = out[f"{side}_input_ids"]
        labels = out[f"{side}_labels"]
        attn = out[f"{side}_attention_mask"]
        assert ids.shape == labels.shape == attn.shape
        assert ids.shape[0] == 2
        # attention mask marks real tokens; pad positions are False.
        for i in range(2):
            real = int(attn[i].sum())
            assert torch.all(attn[i, :real])
            assert not torch.any(attn[i, real:])


def test_dpo_collate_shared_prompt_after_padding(tokenizer, dpo_path):
    ds = DPODataset(dpo_path, tokenizer)
    batch = [ds[0], ds[1]]
    out = ds.collate(batch)
    for i in range(2):
        plen = int(out["prompt_len"][i])
        assert torch.equal(
            out["chosen_input_ids"][i, :plen],
            out["rejected_input_ids"][i, :plen],
        )
