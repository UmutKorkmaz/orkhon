"""The tool registry — name -> Tool, with safe construction from a name list."""

from __future__ import annotations

from typing import Iterable

from orkhon.serve.tools.base import Tool, ToolSpec


class ToolRegistry:
    """A bag of named tools plus their declarative specs (for prompts/schemas)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError("tool must have a non-empty name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec.from_tool(t) for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


def build_default_registry(
    names: Iterable[str] | None = None,
    *,
    file_roots: list[str] | None = None,
    allow_python: bool = False,
) -> ToolRegistry:
    """Construct a registry from a name list (defaults to calculator only).

    ``read_file`` requires EXPLICIT ``file_roots`` (never the cwd by default — a
    model reading arbitrary files in a server cwd is a secret-leak footgun).
    ``python_exec`` is intentionally unavailable in this release; local code
    execution needs container isolation before it can be exposed safely.
    """
    from orkhon.serve.tools.calculator import Calculator
    from orkhon.serve.tools.read_file import ReadFile

    reg = ToolRegistry()
    want = set(names) if names is not None else {"calculator"}
    if "calculator" in want:
        reg.register(Calculator())
    if "read_file" in want:
        if not file_roots:
            raise ValueError("read_file requires explicit file_roots (not the cwd by default)")
        reg.register(ReadFile(roots=file_roots))
    if allow_python or "python_exec" in want:
        raise ValueError("python_exec is not implemented in this release")
    return reg
