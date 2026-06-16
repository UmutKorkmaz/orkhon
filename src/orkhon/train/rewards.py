"""Verifiable rewards for math/code — the core of GRPO/RLVR.

A reward function returns 1.0 if the model's answer is *verifiably* correct and 0.0
otherwise, with no learned judge (no reward-hacking via style). This is the
DeepSeek-R1 pattern: extract the final answer from the model's trace and compare it
to the ground truth with a deterministic checker.

- :func:`extract_boxed`  — pull the last ``\\boxed{...}`` (or a trailing number) from text.
- :func:`math_equal`     — compare two answers (float, fraction, symbolic via sympy).
- :func:`math_reward`    — 1.0 if ``extract_boxed(output)`` math-equals ``gold``.
- :func:`code_reward`    — 1.0 if the code runs and unit tests pass (verifiable).
"""

from __future__ import annotations

import re
from typing import Iterable

_BOXED = re.compile(r"\\boxed\{(.+?)\}", re.S)
_TRAILING_NUM = re.compile(r"[-+]?\d+(?:[.,/]\d+)*")


def extract_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` content, else a trailing number, else None.

    Handles nested braces (e.g. ``\\boxed{\\frac{1}{2}}``) via balanced matching.
    """
    out = None
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            break
        depth, k = 0, j + len("\\boxed{")
        start = k
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                if depth == 0:
                    break
                depth -= 1
            k += 1
        out = text[start:k]
        i = k + 1
    if out is not None:
        return out.strip().rstrip(".")
    nums = _TRAILING_NUM.findall(text.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def _to_number(s: str):
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def math_equal(a: str | None, b: str | None, *, atol: float = 1e-4) -> bool:
    """True if answers ``a`` and ``b`` are numerically equal (fraction-aware)."""
    if a is None or b is None:
        return False
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    # Fraction forms like "3/4".
    def _frac(x):
        if "/" in x:
            try:
                n, d = x.split("/")
                return float(n) / float(d)
            except (ValueError, ZeroDivisionError):
                return None
        return None
    fa = _frac(a) or _to_number(a)
    fb = _frac(b) or _to_number(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= atol * max(1.0, abs(fb))
    # Last resort: symbolic equality via sympy (optional).
    try:
        import sympy

        return sympy.simplify(sympy.sympify(a) - sympy.sympify(b)) == 0
    except Exception:
        return False


def math_reward(output: str, gold: str) -> float:
    """1.0 if the model's boxed/trailing answer equals ``gold``."""
    return 1.0 if math_equal(extract_boxed(output), gold) else 0.0


def copy_digit_reward(output: str, gold: str) -> float:
    """1.0 if the output is exactly the gold digit; 0.1 if it's any single digit;
    0.0 otherwise. A learnable-but-trivially-verifiable skill: a constant policy
    caps near 0.55 on balanced binary data, so >0.85 requires attending to TARGET.
    """
    import re

    s = output.strip()
    if s == gold:
        return 1.0
    if re.fullmatch(r"\d", s):
        return 0.1
    return 0.0


def code_reward(code: str, *, tests: Iterable[str] = (), timeout: float = 2.0) -> float:
    """1.0 if ``code`` execs without error and every assertion in ``tests`` passes.

    Runs in a subprocess with a hard timeout (a minimal sandbox; for untrusted code
    use the tools layer's full container sandbox, not this).
    """
    import subprocess, sys

    body = code + "\n\n" + "\n".join(f"assert {t}" for t in tests)
    try:
        subprocess.run(
            [sys.executable, "-c", body],
            timeout=timeout, capture_output=True, check=True,
        )
        return 1.0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, SyntaxError):
        return 0.0
