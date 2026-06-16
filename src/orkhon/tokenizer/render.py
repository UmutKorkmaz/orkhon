"""Authoritative chat rendering + SFT/DPO label masking.

Design decision (correctness over prettiness): the chat format places **no text
between a role marker and its content** —

    <bos><|system|>SYS<|end|><|user|>MSG<|end|><|assistant|>REPLY<|end|>...

Because the special tokens are registered with the tokenizer, they are hard
segmentation boundaries: BPE never merges across them. Therefore each message's
content is byte-pair-encoded identically whether tokenized standalone or in
context, which makes piecewise tokenization for label masking exact. This avoids
the classic SFT bug where label spans drift because content was re-tokenized in a
different neighborhood.

Masking rule (the other classic trap): only **assistant content and its closing
``<|end|>``** are trainable. Role markers (including ``<|assistant|>``) and
user/system content are ``IGNORE_INDEX``. ``labels`` are returned aligned to
``input_ids`` (same length); the training loss performs the single next-token
shift (``logits[:, :-1]`` vs ``labels[:, 1:]``). Keeping ``<|end|>`` trainable is
what teaches the model to stop.
"""

from __future__ import annotations

from typing import Callable, Sequence

from orkhon.tokenizer.special_tokens import (
    ASSISTANT,
    BOS,
    END,
    ROLE_TOKENS,
    SYSTEM,
    USER,
    SpecialIds,
)

IGNORE_INDEX = -100

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}
EncodeFn = Callable[[str], Sequence[int]]  # text -> token ids, WITHOUT specials


def render_chat(messages: Sequence[Message], add_generation_prompt: bool = False) -> str:
    """Render messages to the canonical Orkhon chat string (for display / HF template).

    Mirrors :func:`encode_for_training` / :func:`encode_for_inference` exactly.
    """
    parts: list[str] = [BOS]
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role!r}")
        parts.append(ROLE_TOKENS[role])
        parts.append(m["content"])
        parts.append(END)
    if add_generation_prompt:
        parts.append(ASSISTANT)
    return "".join(parts)


def encode_for_training(
    messages: Sequence[Message],
    encode_fn: EncodeFn,
    special: SpecialIds,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Encode a conversation into ``(input_ids, labels)`` for SFT.

    Only assistant content + closing ``<|end|>`` carry real labels; everything else
    is ``ignore_index``. ``input_ids`` and ``labels`` have equal length.
    """
    input_ids: list[int] = [special.bos]
    labels: list[int] = [ignore_index]

    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role!r}")
        trainable = role == "assistant"

        # Role marker: never trained (it is part of the prompt scaffold).
        input_ids.append(special.role_id(role))
        labels.append(ignore_index)

        # Content: trained only for the assistant.
        content_ids = list(encode_fn(m["content"]))
        input_ids.extend(content_ids)
        labels.extend(content_ids if trainable else [ignore_index] * len(content_ids))

        # Closing <|end|>: trained for the assistant so the model learns to stop.
        input_ids.append(special.end)
        labels.append(special.end if trainable else ignore_index)

    return input_ids, labels


def encode_for_inference(
    messages: Sequence[Message],
    encode_fn: EncodeFn,
    special: SpecialIds,
) -> list[int]:
    """Encode a conversation and append the ``<|assistant|>`` generation prompt."""
    input_ids: list[int] = [special.bos]
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role!r}")
        input_ids.append(special.role_id(role))
        input_ids.extend(encode_fn(m["content"]))
        input_ids.append(special.end)
    input_ids.append(special.assistant)  # generation prompt
    return input_ids


def encode_prompt_and_completion(
    prompt_messages: Sequence[Message],
    completion: str,
    encode_fn: EncodeFn,
    special: SpecialIds,
) -> tuple[list[int], list[int]]:
    """Encode a DPO/preference example into ``(prompt_ids, completion_ids)``.

    ``prompt_ids`` ends with the ``<|assistant|>`` marker; ``completion_ids`` is the
    assistant reply plus closing ``<|end|>``. Chosen and rejected share an identical
    ``prompt_ids`` (the caller encodes the prompt once), which is required for a
    correct DPO objective.
    """
    prompt_ids = encode_for_inference(prompt_messages, encode_fn, special)
    completion_ids = list(encode_fn(completion)) + [special.end]
    return prompt_ids, completion_ids
