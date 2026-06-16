"""The tool-use runtime loop: generate -> parse tool call -> execute -> observe -> answer.

A bounded loop (``max_rounds``) that gives the model tools, lets it emit a call,
executes the call via the registry, appends the observation, and asks again. Stops
when the model produces a final answer with no tool call, or the round budget is hit.

Tool output is UNTRUSTED data: it is neutralized (:func:`escape_tool_output`) before
being placed in history so a malicious indexed document containing literal special
tokens (``<|end|>``, ``<|system|>``) cannot structurally escape the tool message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from orkhon.model import generate
from orkhon.serve.tool_protocol import ToolCall, format_tool_prompt, parse_tool_call
from orkhon.tokenizer.special_tokens import SPECIAL_TOKENS

Message = dict  # {"role": ..., "content": ...}

# A zero-width space inserted after the first char of each special-token literal so
# it no longer matches the trained special token at encode time (prompt-injection
# neutralization for tool/retrieved output).
_ZWSP = "​"


def escape_tool_output(text: str) -> str:
    """Break any literal Orkhon special-token strings in ``text`` so they cannot
    be tokenized as control tokens when the conversation is re-encoded."""
    out = text
    for tok in SPECIAL_TOKENS:
        if len(tok) > 1 and tok in out:
            out = out.replace(tok, tok[0] + _ZWSP + tok[1:])
    return out


@dataclass
class ToolLoopResult:
    messages: list[Message] = field(default_factory=list)  # full transcript
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    rounds: int = 0


def run_tool_loop(
    generate_fn: Callable[[list[Message]], str],
    messages: list[Message],
    specs,
    registry,
    *,
    max_rounds: int = 3,
) -> ToolLoopResult:
    """Run a tool loop using an injected ``generate_fn(messages) -> text``.

    ``generate_fn`` is supplied by the caller (chat CLI / API) so this stays
    model-agnostic and unit-testable with a scripted generator.
    """
    transcript = [dict(m) for m in messages]
    if specs:
        # Prepend/merge tool instructions into a system turn.
        instr = format_tool_prompt(specs)
        if transcript and transcript[0]["role"] == "system":
            transcript[0]["content"] = transcript[0]["content"] + "\n\n" + instr
        else:
            transcript.insert(0, {"role": "system", "content": instr})

    res = ToolLoopResult(messages=transcript)
    for rnd in range(max_rounds):
        res.rounds = rnd + 1
        text = generate_fn(transcript).strip()
        call = parse_tool_call(text)
        if call is None:
            res.final_answer = text
            return res
        res.tool_calls.append(call)
        # Record the assistant turn (the call) and the tool observation.
        transcript.append({"role": "assistant", "content": text})
        tool = registry.get(call.name)
        if tool is None:
            obs = f"tool error: unknown tool {call.name!r}"
        else:
            try:
                obs = tool(**call.arguments).output
            except Exception as e:  # bad arguments / tool crash must not kill the loop
                obs = f"tool error: {type(e).__name__}: {e}"
        # Tool output is UNTRUSTED data, never instructions (prompt-injection guard).
        transcript.append({"role": "tool", "content": escape_tool_output(obs)})
    # Round budget exhausted: take the last generation as the answer.
    res.final_answer = generate_fn(transcript).strip()
    return res
