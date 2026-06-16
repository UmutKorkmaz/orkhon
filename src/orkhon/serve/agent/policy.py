"""Agent policy — gate which tool calls are allowed at runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""


class AgentPolicy:
    """Day-one policy: only registered tools are allowed; everything else denied.

    Designed to be extended (approval gates, allow/deny lists, network/exec gating)
    without changing the loop.
    """

    def __init__(self, *, deny: list[str] | None = None) -> None:
        self.deny = set(deny or [])

    def check(self, call_name: str, registry) -> PolicyDecision:
        if call_name in self.deny:
            return PolicyDecision(False, f"tool {call_name!r} denied by policy")
        if call_name not in registry:
            return PolicyDecision(False, f"unknown tool {call_name!r}")
        return PolicyDecision(True)
