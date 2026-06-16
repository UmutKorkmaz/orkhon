"""Tests for the code (FIM) and math (verifiable-reward) verticals."""

from __future__ import annotations

import random

from orkhon.data.code_synth import fim_example, make_fim_sft
from orkhon.train.rewards import code_reward, extract_boxed, math_equal, math_reward


def test_fim_transform_produces_valid_split():
    code = "def add(a, b):\n    return a + b\n\nresult = add(1, 2)\nprint(result)\n# done"
    # any seed that yields a non-empty middle works; find one.
    ex = None
    for s in range(10):
        ex = fim_example(code, random.Random(s))
        if ex is not None:
            break
    assert ex is not None
    assert ex["messages"][0]["role"] == "user"
    assert "<fim_prefix>" in ex["messages"][0]["content"]
    assert ex["messages"][1]["content"] in code  # the middle is a real line


def test_fim_skips_short_code():
    assert fim_example("x = 1", random.Random(0)) is None  # too short / too few lines


def test_make_fim_sft(tmp_path):
    doc = "def f():\n    return 1\n\ng = f()\nprint(g)\nh = g + 1\nreturn h"
    n = make_fim_sft([doc] * 20, tmp_path / "fim.jsonl", max_examples=10, seed=1)
    assert n >= 1
    lines = (tmp_path / "fim.jsonl").read_text().splitlines()
    assert len(lines) == n


def test_extract_boxed_and_trailing_number():
    assert extract_boxed("blah \\boxed{42} done") == "42"
    assert extract_boxed("the answer is 7") == "7"
    assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_math_equal_variants():
    assert math_equal("42", "42")
    assert math_equal("0.5", "1/2")
    assert math_equal("100", "100.0")
    assert not math_equal("42", "43")
    assert not math_equal(None, "1")


def test_math_reward():
    assert math_reward("thinking... \\boxed{8}", "8") == 1.0
    assert math_reward("the answer is 8", "8") == 1.0
    assert math_reward("\\boxed{9}", "8") == 0.0


def test_code_reward_runs_and_passes():
    good = "def sq(x):\n    return x * x"
    assert code_reward(good, tests=["sq(3) == 9"]) == 1.0
    assert code_reward("def sq(x):\n    return x", tests=["sq(3) == 9"]) == 0.0  # fails assert
    assert code_reward("raise RuntimeError", tests=[]) == 0.0  # errors
