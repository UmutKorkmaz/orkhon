"""The agent loop: bounded plan -> act -> observe with a hard step budget.

Unlike :func:`orkhon.serve.tool_loop.run_tool_loop` (which does one extra
generation after its round budget), the agent stops HARD at ``max_steps``: if the
model hasn't produced a plain answer by then, the run ends with status
``"max_steps"`` and no post-budget generation. Every step is recorded as an
:class:`AgentStep` for reproducible transcripts (useful for later tool-use SFT,
distillation, and GRPO rollouts).
"""

from __future__ import annotations

from typing import Callable

from orkhon.serve.agent.policy import AgentPolicy
from orkhon.serve.agent.types import AgentConfig, AgentRunResult, AgentStep
from orkhon.serve.tool_loop import escape_tool_output
from orkhon.serve.tool_protocol import format_tool_prompt, parse_tool_call

Message = dict


def build_agent_prompt(specs, config: AgentConfig) -> str:
    base = format_tool_prompt(specs)
    return (base + "\n\nThink step by step. Each turn: give a short plan, then either call a "
            "tool (end the turn with <|tool|> + JSON) or give the final answer with no tool call.")


def run_agent(
    generate_fn: Callable[[list[Message]], str],
    messages: list[Message],
    registry,
    *,
    config: AgentConfig = AgentConfig(),
    policy: AgentPolicy = AgentPolicy(),
) -> AgentRunResult:
    """Run a bounded agent loop; ``generate_fn`` is model-agnostic (scriptable)."""
    transcript = [dict(m) for m in messages]
    if registry and registry.names():
        instr = build_agent_prompt(registry.specs(), config)
        if transcript and transcript[0]["role"] == "system":
            transcript[0]["content"] = transcript[0]["content"] + "\n\n" + instr
        else:
            transcript.insert(0, {"role": "system", "content": instr})

    res = AgentRunResult(status="no_answer", final_answer="", messages=transcript)
    consecutive_errors = 0

    def _obs_safe(raw: str) -> str:
        """One observation treatment, used in BOTH the step and the transcript."""
        if len(raw) > config.max_observation_chars:
            raw = raw[: config.max_observation_chars] + " …[truncated]"
        return escape_tool_output(raw)

    for i in range(config.max_steps):
        text = (generate_fn(transcript) or "").strip()
        # Parse the LAST <|tool|> marker so plan reasoning that happens to mention
        # "<|tool|>" can't shadow a real call that comes after it.
        marker_idx = text.rfind("<|tool|>")
        if marker_idx >= 0:
            call = parse_tool_call(text[marker_idx:])
            plan = text[:marker_idx].strip()
        else:
            call = parse_tool_call(text)
            plan = text

        if call is None:
            # A plain (or empty) answer ends the run.
            transcript.append({"role": "assistant", "content": text})
            res.steps.append(AgentStep(index=i + 1, plan=plan, assistant_text=text,
                                       tool_call=None, observation=""))
            res.final_answer = text
            res.status = "completed" if text else "no_answer"
            return res

        # Policy gate.
        decision = policy.check(call.name, registry)
        if not decision.allowed:
            obs = _obs_safe(f"tool denied: {decision.reason}")
            res.steps.append(AgentStep(index=i + 1, plan=plan, assistant_text=text,
                                       tool_call={"name": call.name, "arguments": call.arguments},
                                       observation=obs, error=True))
            transcript.append({"role": "assistant", "content": text})
            transcript.append({"role": "tool", "content": obs})
            consecutive_errors += 1
            if consecutive_errors >= config.max_tool_errors:
                res.status = "tool_errors"
                return res
            continue

        tool = registry.get(call.name)
        try:
            result = tool(**call.arguments)
            err = bool(getattr(result, "error", False))
            obs = _obs_safe(result.output)
        except Exception as e:
            err = True
            obs = _obs_safe(f"tool error: {type(e).__name__}: {e}")
        consecutive_errors = consecutive_errors + 1 if err else 0

        res.steps.append(AgentStep(index=i + 1, plan=plan, assistant_text=text,
                                   tool_call={"name": call.name, "arguments": call.arguments},
                                   observation=obs, error=err))
        transcript.append({"role": "assistant", "content": text})
        transcript.append({"role": "tool", "content": obs})  # escaped + capped, consistent

        if consecutive_errors >= config.max_tool_errors:
            res.status = "tool_errors"
            return res

    # Hard stop: step budget exhausted with no plain answer.
    res.status = "max_steps"
    return res
