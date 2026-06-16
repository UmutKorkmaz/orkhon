"""Read-file tool — jailed to configured roots, traversal/symlink hardened."""

from __future__ import annotations

from pathlib import Path

from orkhon.serve.tools.base import ToolResult

_MAX_BYTES = 64 * 1024  # cap output so a model can't pull a huge file into context


class ReadFile:
    name = "read_file"
    description = "Read a text file (jailed to an allowed root). Returns up to 64KB."
    parameters = {"path": {"type": "string"}}

    def __init__(self, roots: list[str] | None = None, *, max_bytes: int = _MAX_BYTES) -> None:
        self.roots = [Path(r).resolve() for r in (roots or ["."])]
        self.max_bytes = max_bytes

    def _resolve_safe(self, path: str) -> Path | None:
        """Resolve ``path`` under one of the roots, denying traversal/symlink escapes."""
        p = Path(path)
        p = p.resolve() if p.is_absolute() else None
        for root in self.roots:
            cand = (root / path).resolve()
            # Must stay under `root` after resolution (defeats ../ and symlinks).
            if root in cand.parents or cand == root:
                if cand.exists() and cand.is_file():
                    return cand
        return None

    def __call__(self, *, path: str | None = None, **_) -> ToolResult:
        try:
            if not isinstance(path, str):
                return ToolResult(output="read_file error: 'path' must be a string", error=True)
            safe = self._resolve_safe(path)
            if safe is None:
                return ToolResult(output=f"read_file error: '{path}' not in an allowed root", error=True)
            # Stream-read only up to the cap (a huge allowed file must not be fully loaded).
            with safe.open("rb") as f:
                data = f.read(self.max_bytes)
            return ToolResult(output=data.decode("utf-8", errors="replace"))
        except Exception as e:
            return ToolResult(output=f"read_file error: {e}", error=True)
