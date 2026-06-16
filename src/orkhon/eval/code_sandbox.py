"""A stronger code-execution sandbox for MBPP-style generative eval.

``train.rewards.code_reward`` is a minimal subprocess checker; eval needs a real
sandbox: temp working dir, minimal environment, a hard timeout, an output cap, and
blocking of dangerous builtins/modules (``open``, ``socket``, ``subprocess``,
``os.system``, ``os.remove`` …). This is still single-process isolation (good enough
for trusted local eval; NOT for hostile multi-tenant code — that needs containers).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CodeSandboxResult:
    passed: bool
    output: str
    error: str = ""


_BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit",
}
# Modules a legitimate MBPP-style solution is allowed to import. Everything else
# is denied by the import hook below.
_ALLOWED_MODULES = {
    "math", "cmath", "re", "collections", "itertools", "functools", "string",
    "statistics", "fractions", "decimal", "heapq", "bisect", "typing", "json",
    "datetime", "copy", "operator", "random", "textwrap", "numbers", "enum",
    "dataclasses", "abc",
}


def _guard_prelude() -> str:
    """A prologue restricting builtins + imports before user code runs.

    SECURITY: this is a TRUSTED-LOCAL sandbox, NOT a security boundary. It stops
    accidental damage and obvious ``os.system``/``open`` calls; it can be bypassed
    by object-graph tricks (``().__class__.__bases__[0].__subclasses__()`` …). For
    UNTRUSTED code, run in a real OS sandbox (container / nsjail / seccomp, no
    network, read-only FS, low-priv user) — never this layer alone.
    """
    blocked = ", ".join(f"'{n}'" for n in sorted(_BLOCKED_NAMES))
    allowed = ", ".join(f"'{n}'" for n in sorted(_ALLOWED_MODULES))
    return (
        "import sys as _sys\n"
        "import builtins as _b\n"
        f"_BL = {{{blocked}}}\n"
        "for _n in _BL:\n"
        "    setattr(_b, _n, lambda *a, **k: (_ for _ in ()).throw("
        "PermissionError('blocked: ' + _n)))\n"
        f"_ALLOW = {{{allowed}}}\n"
        "_orig_import = _b.__import__\n"
        "def _safe_import(name, *a, **k):\n"
        "    top = name.split('.')[0]\n"
        "    if top not in _ALLOW:\n"
        "        raise ImportError(f'blocked import: {name}')\n"
        "    return _orig_import(name, *a, **k)\n"
        "_b.__import__ = _safe_import\n"
    )


class SubprocessCodeSandbox:
    def __init__(self, *, timeout_s: float = 2.0, max_output_chars: int = 4096,
                 memory_mb: int = 512) -> None:
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars
        self.memory_mb = memory_mb

    def run(self, code: str, tests: Iterable[str] = ()) -> CodeSandboxResult:
        body = _guard_prelude() + "\n" + code + "\n\n"
        for t in tests:
            body += f"assert {t}\n"
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR"}}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "_sol.py"
            script.write_text(body, encoding="utf-8")
            preexec = self._preexec if sys.platform != "win32" else None
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", str(script)],
                    capture_output=True, timeout=self.timeout_s, env=env,
                    cwd=tmp, preexec_fn=preexec,
                )
            except subprocess.TimeoutExpired:
                return CodeSandboxResult(False, "", "timeout")
            out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
            out = out[: self.max_output_chars]
            if proc.returncode == 0:
                return CodeSandboxResult(True, out)
            return CodeSandboxResult(False, out, error=f"exit {proc.returncode}")

    def _preexec(self):
        # Best-effort rlimits (POSIX). Ignored on failure.
        try:
            mem = self.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_CPU, (int(self.timeout_s) + 1,) * 2)
        except Exception:
            pass
