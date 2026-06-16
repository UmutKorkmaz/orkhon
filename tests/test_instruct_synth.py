"""Tests for story-instruction dataset synthesis (offline, deterministic).

Builds a few fake stories and asserts the SFT/DPO outputs are well-formed,
the val split is held out, counts are internally consistent, and re-running
produces byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orkhon.data.instruct_synth import (
    _instruction_for,
    _pick_subject,
    make_story_instructions,
)

_FAKE_STORIES = [
    "Once upon a time, Lily found a red ball. She kicked it high. Then she laughed.",
    "Tom and Ben built a sandcastle. The waves came. They ran away giggling.",
    "A small fox wanted honey. The bees buzzed loudly. The fox went home sad.",
    "Mia painted a blue sky. Her brush was wet. The picture made her smile.",
    "The little dog dug a deep hole. It hid a bone there. Later it forgot where.",
    "Sara and her cat watched the rain. Drops hit the glass. Soon the sun came out.",
    "A green frog hopped onto a log. It sang a funny song. All the birds clapped.",
    "Max baked a warm cake. The kitchen smelled sweet. Everyone wanted a slice.",
    "Anna lost her toy train. She looked under the bed. There it was, dusty.",
    "The owl flew over the quiet town. Stars filled the sky. It hooted softly.",
]


def _write_corpus(tmp_path: Path, stories: list[str]) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(stories) + "\n", encoding="utf-8")
    return corpus


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _synth(tmp_path: Path, stories: list[str] | None = None, **kw) -> dict:
    stories = stories if stories is not None else _FAKE_STORIES
    corpus = _write_corpus(tmp_path, stories)
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    counts = make_story_instructions(corpus, sft, dpo, **kw)
    return {"counts": counts, "sft": sft, "dpo": dpo, "stories": stories}


# --------------------------------------------------------------------------- #
# subject heuristic
# --------------------------------------------------------------------------- #
def test_picks_character_name_over_sentence_starter():
    # "Once" is a stop-cap, so the subject should be the real name "Lily".
    assert _pick_subject(_FAKE_STORIES[0]) == "Lily"


def test_falls_back_to_salient_noun_when_no_name():
    story = "the small red ball rolled down the long hill quietly today"
    subject = _pick_subject(story)
    assert subject == subject.lower()
    assert subject not in {"the", "a", "an"}
    assert len(subject) > 2


def test_falls_back_to_generic_when_empty_storyish():
    assert _pick_subject("a an of to") == "a little adventure"


def test_instruction_is_non_empty_and_mentions_subject():
    instr = _instruction_for(_FAKE_STORIES[1])
    assert instr.startswith("Write a short story about ")
    assert instr.endswith(".")
    assert len(instr) > len("Write a short story about .")


# --------------------------------------------------------------------------- #
# SFT well-formedness
# --------------------------------------------------------------------------- #
def test_sft_messages_well_formed(tmp_path):
    out = _synth(tmp_path)
    rows = _read_jsonl(out["sft"])
    assert len(rows) == out["counts"]["sft_train"]

    train_stories = out["stories"][: out["counts"]["sft_train"]]
    for row, story in zip(rows, train_stories):
        messages = row["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        user = messages[0]["content"]
        assistant = messages[1]["content"]
        assert user.strip(), "user instruction must be non-empty"
        assert user.startswith("Write a short story about ")
        assert assistant == story, "assistant content must equal the source story"


# --------------------------------------------------------------------------- #
# DPO well-formedness
# --------------------------------------------------------------------------- #
def test_dpo_pairs_well_formed(tmp_path):
    out = _synth(tmp_path)
    rows = _read_jsonl(out["dpo"])
    assert len(rows) == out["counts"]["dpo_train"]

    train_stories = out["stories"][: out["counts"]["dpo_train"]]
    for row, story in zip(rows, train_stories):
        prompt = row["prompt"]
        assert isinstance(prompt, list) and len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert prompt[0]["content"].startswith("Write a short story about ")
        assert row["chosen"] == story, "chosen must equal the source story"
        assert row["rejected"] != row["chosen"], "rejected must differ from chosen"
        assert isinstance(row["rejected"], str) and row["rejected"].strip()


def test_dpo_prompt_matches_sft_user_instruction(tmp_path):
    out = _synth(tmp_path)
    sft_rows = _read_jsonl(out["sft"])
    dpo_rows = _read_jsonl(out["dpo"])
    for sft_row, dpo_row in zip(sft_rows, dpo_rows):
        assert sft_row["messages"][0]["content"] == dpo_row["prompt"][0]["content"]


# --------------------------------------------------------------------------- #
# counts + val split
# --------------------------------------------------------------------------- #
def test_counts_are_consistent_and_val_split_written(tmp_path):
    out = _synth(tmp_path)
    counts = out["counts"]
    n = len(_FAKE_STORIES)

    assert counts["stories"] == n
    assert counts["sft_train"] + counts["sft_val"] == n
    assert counts["dpo_train"] + counts["dpo_val"] == n
    assert counts["sft_val"] >= 1, "a small val split must be held out"
    assert counts["sft_val"] == counts["dpo_val"]
    assert counts["sft_train"] == counts["dpo_train"]

    sft_val = out["sft"].with_name("sft_val.jsonl")
    dpo_val = out["dpo"].with_name("dpo_val.jsonl")
    assert sft_val.exists() and dpo_val.exists()
    assert len(_read_jsonl(sft_val)) == counts["sft_val"]
    assert len(_read_jsonl(dpo_val)) == counts["dpo_val"]

    # Train and val do not overlap (stories are partitioned in order).
    train_stories = {
        r["messages"][1]["content"] for r in _read_jsonl(out["sft"])
    }
    val_stories = {r["messages"][1]["content"] for r in _read_jsonl(sft_val)}
    assert train_stories.isdisjoint(val_stories)


def test_max_examples_caps_consumption(tmp_path):
    out = _synth(tmp_path, max_examples=4)
    assert out["counts"]["stories"] == 4


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_deterministic_across_two_runs(tmp_path):
    corpus = _write_corpus(tmp_path, _FAKE_STORIES)

    a_sft, a_dpo = tmp_path / "a_sft.jsonl", tmp_path / "a_dpo.jsonl"
    b_sft, b_dpo = tmp_path / "b_sft.jsonl", tmp_path / "b_dpo.jsonl"

    counts_a = make_story_instructions(corpus, a_sft, a_dpo, seed=1337)
    counts_b = make_story_instructions(corpus, b_sft, b_dpo, seed=1337)

    assert counts_a == counts_b
    assert a_sft.read_bytes() == b_sft.read_bytes()
    assert a_dpo.read_bytes() == b_dpo.read_bytes()
    assert (
        a_sft.with_name("a_sft_val.jsonl").read_bytes()
        == b_sft.with_name("b_sft_val.jsonl").read_bytes()
    )
    assert (
        a_dpo.with_name("a_dpo_val.jsonl").read_bytes()
        == b_dpo.with_name("b_dpo_val.jsonl").read_bytes()
    )


def test_different_seed_changes_rejected(tmp_path):
    corpus = _write_corpus(tmp_path, _FAKE_STORIES)
    a_dpo = tmp_path / "a_dpo.jsonl"
    b_dpo = tmp_path / "b_dpo.jsonl"
    make_story_instructions(corpus, tmp_path / "a_sft.jsonl", a_dpo, seed=1)
    make_story_instructions(corpus, tmp_path / "b_sft.jsonl", b_dpo, seed=999)

    rejected_a = [r["rejected"] for r in _read_jsonl(a_dpo)]
    rejected_b = [r["rejected"] for r in _read_jsonl(b_dpo)]
    # Seeds drive which degradation strategy is used; at least one differs.
    assert rejected_a != rejected_b


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_missing_corpus_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_story_instructions(
            tmp_path / "nope.txt", tmp_path / "s.jsonl", tmp_path / "d.jsonl"
        )


def test_empty_corpus_raises(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        make_story_instructions(corpus, tmp_path / "s.jsonl", tmp_path / "d.jsonl")
