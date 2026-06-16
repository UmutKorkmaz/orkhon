"""Tool-call wire format — JSON the model emits after ``<|tool|>``.

The model is prompted to end an assistant turn with ``<|tool|>`` followed by a JSON
object ``{"name": ..., "arguments": {...}}``. We parse that out of generated text,
falling back to scraping a JSON object if the marker is absent. Robust to trailing
prose after the JSON (we take the first balanced ``{...}``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from orkhon.serve.tools.base import ToolSpec

_TOOL_MARKER = "<|tool|>"


@dataclass
class ToolCall:
    name: str
    arguments: dict


def _object_after_marker(text: str) -> dict | None:
    """Parse the single JSON object immediately following the ``<|tool|>`` marker.

    Requires the marker (a plain answer containing ``{...}`` is NOT a tool call)
    and uses ``raw_decode`` so string contents with braces are handled and a
    trailing second object is rejected.
    """
    idx = text.find(_TOOL_MARKER)
    if idx < 0:
        return None
    tail = text[idx + len(_TOOL_MARKER):].lstrip()
    if not tail.startswith("{"):
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # Reject a second non-whitespace object / trailing junk after the call.
    if tail[end:].strip():
        return None
    return obj


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract a tool call from model output, or None if it didn't make one."""
    obj = _object_after_marker(text)
    if not obj:
        return None
    name = obj.get("name")
    if not isinstance(name, str):
        return None
    # ``arguments`` must be ABSENT (defaults to {}) or a dict; a present non-dict
    # (list/str/number/bool) is a malformed call, not silently coerced to {}.
    if "arguments" in obj:
        args = obj["arguments"]
        if args is None:
            args = {}
        elif not isinstance(args, dict):
            return None
    else:
        args = obj.get("args")
        if args is None:
            args = {}
        elif not isinstance(args, dict):
            return None
    return ToolCall(name=name, arguments=args)


def format_tool_prompt(specs: list[ToolSpec]) -> str:
    """The system instruction describing the available tools + the call format."""
    if not specs:
        return ""
    lines = [
        "You can call tools to help. To call one, end your reply with "
        "<|tool|> followed by exactly one JSON object on the pattern:",
        '{"name": "<tool>", "arguments": {<...>}}',
        "Available tools:",
    ]
    for s in specs:
        params = ", ".join(s.parameters) if s.parameters else "(no arguments)"
        lines.append(f'- {s.name}({params}): {s.description}')
    return "\n".join(lines)
