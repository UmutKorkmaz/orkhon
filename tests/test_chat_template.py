"""Chat rendering + SFT label-masking tests against a real trained tokenizer."""

from __future__ import annotations

import pytest

from orkhon.tokenizer.render import (
    IGNORE_INDEX,
    encode_for_training,
    render_chat,
)
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.tokenizer.train import train_tokenizer


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chat_tok")
    corpus = tmp / "corpus.txt"
    lines = []
    for i in range(300):
        lines.append(f"hello there how are you doing today number {i % 10}")
        lines.append("I am a helpful assistant ready to help you")
        lines.append("question what is the weather like answer it is sunny")
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp / "tok"
    train_tokenizer([corpus], out, vocab_size=320, min_frequency=2)
    return load_tokenizer(out)


@pytest.fixture()
def two_turn_messages():
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello to you"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you are welcome"},
    ]


def test_labels_ignore_non_assistant_tokens(tokenizer, two_turn_messages):
    special = tokenizer.special
    input_ids, labels = encode_for_training(
        two_turn_messages, tokenizer.encode, special
    )

    assert len(input_ids) == len(labels)

    role_marker_ids = {special.system, special.user, special.assistant}

    # Every role marker position must be IGNORE_INDEX.
    for tok_id, label in zip(input_ids, labels):
        if tok_id in role_marker_ids:
            assert label == IGNORE_INDEX

    # bos at position 0 is never trained.
    assert input_ids[0] == special.bos
    assert labels[0] == IGNORE_INDEX


def test_assistant_content_and_end_are_trained(tokenizer, two_turn_messages):
    special = tokenizer.special
    input_ids, labels = encode_for_training(
        two_turn_messages, tokenizer.encode, special
    )

    # Reconstruct expected trainable spans: each assistant content + its <|end|>.
    expected_trainable: list[int] = []
    for m in two_turn_messages:
        if m["role"] == "assistant":
            expected_trainable.extend(tokenizer.encode(m["content"]))
            expected_trainable.append(special.end)

    actual_trainable = [lab for lab in labels if lab != IGNORE_INDEX]
    assert actual_trainable == expected_trainable


def test_user_system_content_is_ignored(tokenizer):
    special = tokenizer.special
    messages = [
        {"role": "user", "content": "secret user words here"},
        {"role": "assistant", "content": "ok"},
    ]
    input_ids, labels = encode_for_training(messages, tokenizer.encode, special)

    user_content_ids = tokenizer.encode("secret user words here")
    # Locate user content span (right after the <|user|> marker) and assert ignored.
    user_pos = input_ids.index(special.user)
    span = labels[user_pos + 1 : user_pos + 1 + len(user_content_ids)]
    assert all(lab == IGNORE_INDEX for lab in span)


def test_render_chat_structure_matches_encoding(tokenizer, two_turn_messages):
    # render_chat (no generation prompt) re-encodes to the same input_ids that
    # encode_for_training produces, since specials are hard BPE boundaries.
    special = tokenizer.special
    input_ids, _ = encode_for_training(
        two_turn_messages, tokenizer.encode, special
    )

    rendered = render_chat(two_turn_messages, add_generation_prompt=False)
    # The rendered string starts with <bos> and contains each role marker once+.
    assert rendered.startswith("<bos>")
    assert rendered.count("<|assistant|>") == 2
    assert rendered.count("<|end|>") == len(two_turn_messages)

    # Structural parity: number of <|end|> ids in input_ids equals message count.
    assert input_ids.count(special.end) == len(two_turn_messages)
