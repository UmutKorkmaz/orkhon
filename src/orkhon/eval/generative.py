"""Generative evaluation: pass@k over GSM8K (math) and MBPP (code).

Unlike the loglikelihood multiple-choice path, generative eval samples ``n`` free-form
completions per problem and checks each with a deterministic grader (math_reward for
GSM8K, the code sandbox for MBPP). pass@k is the standard HumanEval metric:

    pass@k = 1 - C(n - c, k) / C(n, k)

computed as a stable product to avoid huge binomials.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from orkhon.eval.code_sandbox import SubprocessCodeSandbox
from orkhon.train.rewards import extract_boxed, math_equal


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k: 1 - C(n-c, k)/C(n, k). Returns 0 if k > n; 1.0 if all c correct & k<=c edge."""
    if n <= 0 or k > n:
        return 0.0
    if n - c < k:  # correct samples already cover all k picks
        return 1.0
    # 1 - prod_{i=0..k-1} (n-c-i)/(n-i)
    num = 1.0
    for i in range(k):
        num *= (n - c - i) / (n - i)
    return 1.0 - num


def score_gsm8k(output: str, gold: str) -> float:
    """1.0 if the model's boxed/trailing answer math-equals the gold answer."""
    return 1.0 if math_equal(extract_boxed(output), gold) else 0.0


_FENCE_OPEN = re.compile(r"```[a-zA-Z0-9_]*\s*\n?", re.M)


def extract_code(output: str) -> str:
    """Pull the first fenced code block from a model answer (else the whole text).

    Handles `````py`` / `````python3`` / tagged / untagged fences, and an unclosed
    fence (take to end-of-text).
    """
    if "```" not in output:
        return output.strip()
    first = output.find("```")
    rest = output[first + 3 :]
    # strip an optional language tag on the opening fence line
    nl = rest.find("\n")
    body = rest[nl + 1 :] if nl >= 0 else rest
    close = body.find("```")
    return (body if close < 0 else body[:close]).strip()


@dataclass
class GenerativeResult:
    task: str
    n: int
    pass_at_1: float
    pass_at_k: float
    k: int
    mean_reward: float
    per_example: list[dict]


def run_generative_task(
    examples: Sequence[dict],
    generate_fn: Callable[[str], str],
    *,
    grader: str = "gsm8k",            # "gsm8k" | "mbpp"
    n_samples: int = 1,
    k: int = 1,
    max_new_tokens: int = 256,
    sandbox: SubprocessCodeSandbox | None = None,
) -> GenerativeResult:
    """Sample ``n_samples`` per example, grade, and aggregate pass@k.

    ``generate_fn(prompt) -> text`` is model-agnostic (scriptable for tests).
    """
    per_example = []
    rewards = []
    sandbox = sandbox or SubprocessCodeSandbox()
    for ex in examples:
        gold = ex.get("gold")
        tests = ex.get("tests", [])
        prompt = ex["prompt"]
        outs = [generate_fn(prompt) for _ in range(n_samples)]
        if grader == "gsm8k":
            scores = [score_gsm8k(o, gold) for o in outs]
        else:  # mbpp
            scores = [
                float(sandbox.run(extract_code(o), tests).passed) for o in outs
            ]
        c = sum(1 for s in scores if s >= 1.0)
        rewards.extend(scores)
        per_example.append({
            "id": ex.get("id"), "n": n_samples, "correct": c,
            "pass@1": estimate_pass_at_k(n_samples, c, 1),
            f"pass@{k}": estimate_pass_at_k(n_samples, c, k),
        })
    n_ex = max(len(examples), 1)
    return GenerativeResult(
        task=grader, n=n_ex,
        pass_at_1=math.fsum(p["pass@1"] for p in per_example) / n_ex,
        pass_at_k=math.fsum(p[f"pass@{k}"] for p in per_example) / n_ex,
        k=k, mean_reward=sum(rewards) / max(len(rewards), 1),
        per_example=per_example,
    )
