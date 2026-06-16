"""Tool primitives — the contract tools and the runtime agree on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """The outcome of running a tool: an observation string for the model."""

    output: str
    error: bool = False


@runtime_checkable
class Tool(Protocol):
    """A callable tool with a name, a JSON-schema-ish argument spec, and a description."""

    name: str
    description: str

    def __call__(self, **arguments: Any) -> ToolResult: ...


@dataclass
class ToolSpec:
    """A declarative description of a tool, for the model's prompt / OpenAI schema."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool(cls, tool: Tool) -> "ToolSpec":
        return cls(name=tool.name, description=tool.description,
                   parameters=getattr(tool, "parameters", {}))
