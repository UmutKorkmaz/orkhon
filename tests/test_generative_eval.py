"""Generative-eval tests: pass@k math, sandbox, graders, end-to-end."""

from __future__ import annotations

from orkhon.eval.code_sandbox import SubprocessCodeSandbox
from orkhon.eval.generative import (
    estimate_pass_at_k,
    extract_code,
    run_generative_task,
    score_gsm8k,
)


def test_pass_at_k_math():
    assert estimate_pass_at_k(1, 1, 1) == 1.0      # all correct
    assert estimate_pass_at_k(5, 0, 1) == 0.0      # none correct
    # n=4, c=2, k=1 -> 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
    assert abs(estimate_pass_at_k(4, 2, 1) - 0.5) < 1e-9
    # c covers k -> 1.0
    assert estimate_pass_at_k(4, 4, 2) == 1.0
    assert estimate_pass_at_k(2, 1, 3) == 0.0       # k > n


def test_score_gsm8k_extraction():
    assert score_gsm8k("thinking... \\boxed{42}", "42") == 1.0
    assert score_gsm8k("the answer is 5", "5") == 1.0
    assert score_gsm8k("\\boxed{9}", "8") == 0.0


def test_extract_code_fenced():
    assert extract_code("here:\n```python\nx = 1\n```\n") == "x = 1"
    assert extract_code("plain code") == "plain code"


def test_extract_code_tag_variants():
    assert extract_code("```py\ny = 2\n```") == "y = 2"
    assert extract_code("```python3\nz = 3\n```") == "z = 3"
    # unclosed fence -> take to end of text
    assert extract_code("```\nw = 4") == "w = 4"


def test_sandbox_pass_fail_timeout_block():
    sb = SubprocessCodeSandbox(timeout_s=3)
    assert sb.run("def sq(x):\n    return x*x", ["sq(3) == 9"]).passed
    assert not sb.run("def sq(x):\n    return x", ["sq(3) == 9"]).passed  # assertion fails
    # safe stdlib imports ARE allowed (a legit MBPP solution using math must work)
    assert sb.run("import math\nresult = math.sqrt(16)", ["math.sqrt(16) == 4"]).passed
    # dangerous calls/imports blocked
    assert not sb.run("import os", []).passed
    assert not sb.run("import socket", []).passed
    assert not sb.run("open('/etc/passwd').read()", []).passed
    # timeout
    sb_to = SubprocessCodeSandbox(timeout_s=1)
    assert not sb_to.run("import time; time.sleep(5)", []).passed


def test_run_generative_task_gsm8k_scripted():
    from orkhon.eval.generative_tasks import load_generative_task

    examples = load_generative_task("gsm8k", fixture=True)
    # scripted generator: always answers the boxed gold of example 0 ("5")
    turns = iter(["\\boxed{5}", "\\boxed{42}", "\\boxed{120}"])

    def gen(_prompt):
        return next(turns)

    res = run_generative_task(examples, gen, grader="gsm8k", n_samples=1, k=1)
    assert res.task == "gsm8k" and res.n == 3
    assert res.pass_at_1 == 1.0  # all three scripted answers are correct
    assert res.mean_reward == 1.0
    assert len(res.per_example) == 3 and all(p["correct"] == 1 for p in res.per_example)


def test_run_generative_task_mbpp_scripted():
    from orkhon.eval.generative_tasks import load_generative_task

    examples = load_generative_task("mbpp", fixture=True)
    turns = iter([
        "```python\ndef add(a, b):\n    return a + b\n```",
        "```python\ndef sq(x):\n    return x * x\n```",
    ])
    res = run_generative_task(examples, lambda p: next(turns), grader="mbpp", n_samples=1, k=1)
    assert res.pass_at_1 == 1.0
