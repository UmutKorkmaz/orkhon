"""SFT and DPO datasets with padding-aware collation.

Both datasets read JSONL and lean on :mod:`orkhon.tokenizer.render` for the exact,
contract-correct encoding/masking:

* :class:`SFTDataset` reads ``{"messages": [...]}`` and trains only on assistant
  content + closing ``<|end|>`` (everything else is ``IGNORE_INDEX``).
* :class:`DPODataset` reads ``{"prompt", "chosen", "rejected"}`` and encodes the
  prompt ONCE so chosen/rejected share an identical prompt prefix — required for a
  correct DPO objective. Completion-only logprobs are recoverable from the labels
  (prompt positions are ``IGNORE_INDEX``).

Collation pads right to the longest sequence in the batch using the pad id; label
padding uses ``IGNORE_INDEX`` and ``attention_mask`` marks real (non-pad) tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from orkhon.tokenizer.render import (
    IGNORE_INDEX,
    encode_for_training,
    encode_prompt_and_completion,
)
from orkhon.tokenizer.tokenizer import OrkhonTokenizer


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pad_rows(
    rows: list[list[int]], pad_value: int, max_len: int
) -> torch.Tensor:
    """Right-pad a list of int rows to ``max_len`` into a [N, max_len] long tensor."""
    out = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
class SFTDataset:
    """Supervised fine-tuning dataset over ``{"messages": [...]}`` JSONL.

    Each item is encoded with :func:`encode_for_training`, yielding
    ``(input_ids, labels)`` where only assistant content + closing ``<|end|>`` are
    supervised.
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: OrkhonTokenizer) -> None:
        self._tok = tokenizer
        self._special = tokenizer.special
        self._rows = _read_jsonl(jsonl_path)
        if not self._rows:
            raise ValueError(f"no examples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        messages = self._rows[idx]["messages"]
        input_ids, labels = encode_for_training(
            messages, self._tok.encode, self._special
        )
        return {"input_ids": input_ids, "labels": labels}

    def collate(self, batch: list[dict]) -> dict:
        """Pad a batch to its longest sequence.

        Returns dict with:
            input_ids: [B, T] long, pad id on padding.
            labels: [B, T] long, IGNORE_INDEX on padding (and on non-assistant spans).
            attention_mask: [B, T] bool, True for real (non-pad) tokens.
        """
        pad_id = self._special.pad
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = _pad_rows([item["input_ids"] for item in batch], pad_id, max_len)
        labels = _pad_rows([item["labels"] for item in batch], IGNORE_INDEX, max_len)

        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            attention_mask[i, : len(item["input_ids"])] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
class DPODataset:
    """Preference dataset over ``{"prompt", "chosen", "rejected"}`` JSONL.

    ``prompt`` may be a list of messages or a single user string. Chosen and
    rejected reuse the SAME prompt encoding (a hard DPO requirement). For each
    side we build a full sequence ``prompt_ids + completion_ids`` and labels that
    are ``IGNORE_INDEX`` over the prompt and the real ids over the completion, so a
    trainer can compute completion-only log-probabilities directly.
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: OrkhonTokenizer) -> None:
        self._tok = tokenizer
        self._special = tokenizer.special
        self._rows = _read_jsonl(jsonl_path)
        if not self._rows:
            raise ValueError(f"no examples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def _prompt_messages(self, prompt) -> list[dict]:
        """Normalize a prompt field to a messages list."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return list(prompt)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        prompt_messages = self._prompt_messages(row["prompt"])

        # Encode the prompt ONCE; reuse for both completions (identical prefix).
        prompt_ids, chosen_completion = encode_prompt_and_completion(
            prompt_messages, row["chosen"], self._tok.encode, self._special
        )
        _, rejected_completion = encode_prompt_and_completion(
            prompt_messages, row["rejected"], self._tok.encode, self._special
        )

        prompt_len = len(prompt_ids)
        chosen_input = prompt_ids + chosen_completion
        rejected_input = prompt_ids + rejected_completion

        # Labels: prompt masked, completion supervised.
        chosen_labels = [IGNORE_INDEX] * prompt_len + list(chosen_completion)
        rejected_labels = [IGNORE_INDEX] * prompt_len + list(rejected_completion)

        return {
            "prompt_len": prompt_len,
            "chosen_input_ids": chosen_input,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input,
            "rejected_labels": rejected_labels,
        }

    def collate(self, batch: list[dict]) -> dict:
        """Pad chosen and rejected sequences independently to their batch maxima.

        Returns dict with (all long/bool tensors):
            prompt_len: [B] long, shared prompt length per example.
            chosen_input_ids / rejected_input_ids: [B, T*] padded with pad id.
            chosen_labels / rejected_labels: [B, T*] padded with IGNORE_INDEX.
            chosen_attention_mask / rejected_attention_mask: [B, T*] bool.
        """
        pad_id = self._special.pad

        def pack(side: str) -> dict:
            id_rows = [item[f"{side}_input_ids"] for item in batch]
            label_rows = [item[f"{side}_labels"] for item in batch]
            max_len = max(len(r) for r in id_rows)
            input_ids = _pad_rows(id_rows, pad_id, max_len)
            labels = _pad_rows(label_rows, IGNORE_INDEX, max_len)
            attn = torch.zeros((len(batch), max_len), dtype=torch.bool)
            for i, r in enumerate(id_rows):
                attn[i, : len(r)] = True
            return {
                f"{side}_input_ids": input_ids,
                f"{side}_labels": labels,
                f"{side}_attention_mask": attn,
            }

        out = {
            "prompt_len": torch.tensor(
                [item["prompt_len"] for item in batch], dtype=torch.long
            ),
        }
        out.update(pack("chosen"))
        out.update(pack("rejected"))
        return out
