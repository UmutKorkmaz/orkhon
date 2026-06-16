"""Agent data types: config, per-step record, and run result."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    max_steps: int = 5            # hard cap on act/observe rounds (no post-budget gen)
    max_tool_errors: int = 2      # consecutive tool errors before stopping
    max_observation_chars: int = 4000


@dataclass
class AgentStep:
    index: int                    # 1-based
    plan: str                     # the assistant's reasoning text before any tool call
    assistant_text: str           # the full generated turn (raw)
    tool_call: dict | None        # {"name":..., "arguments":...} or None if it answered
    observation: str              # tool output (or "" if no call)
    error: bool = False


@dataclass
class AgentRunResult:
    status: str                   # "completed" | "max_steps" | "tool_errors" | "no_answer"
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full reproducible transcript
