"""Interactive chat REPL over a trained Orkhon checkpoint.

``chat`` loads a model + tokenizer once, then loops reading user input, appending
to a running conversation history, rendering with the canonical inference encoder,
generating with :func:`orkhon.model.generate` (stopping on ``<|end|>``), and
printing the decoded assistant reply.

The pure :func:`reply` helper is the testable core — it takes an explicit history
and already-loaded model/tokenizer and returns the assistant's reply string, with
no I/O. The interactive ``chat`` is a thin wrapper around it.
"""

from __future__ import annotations

from pathlib import Path

from orkhon.model.generation import generate
from orkhon.tokenizer.render import encode_for_inference
from orkhon.tokenizer.tokenizer import OrkhonTokenizer, load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.utils.device import resolve_device

Message = dict  # {"role": ..., "content": ...}


def reply(
    history: list[Message],
    model,
    tokenizer: OrkhonTokenizer,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.3,
    device=None,
) -> str:
    """Generate the assistant reply for ``history`` (no mutation, no I/O).

    Args:
        history: conversation so far (system/user/assistant turns). The last turn
            is typically a user message; a ``<|assistant|>`` generation prompt is
            appended internally.
        model: a trained :class:`orkhon.model.Transformer`.
        tokenizer: the matching :class:`OrkhonTokenizer`.
        max_new_tokens / temperature / top_k / top_p: generation controls.
        device: device override (defaults to the model's device).

    Returns:
        The decoded assistant reply string (special tokens stripped).
    """
    prompt_ids = encode_for_inference(history, tokenizer.encode, tokenizer.special)
    new_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_ids=(tokenizer.special.end,),
        device=device,
    )
    return tokenizer.decode(new_ids, skip_special=True)


def reply_with_tools(
    history: list[Message],
    model,
    tokenizer: OrkhonTokenizer,
    registry,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.3,
    max_rounds: int = 3,
    device=None,
) -> str:
    """Run a tool loop for ``history`` and return the final answer text.

    The per-turn generator uses a marker-preserving decode so the ``<|tool|>``
    token (a special token that ``skip_special`` would otherwise strip) survives
    for :func:`~orkhon.serve.tool_protocol.parse_tool_call`.
    """
    from orkhon.serve.tool_loop import run_tool_loop

    def gen(msgs):
        # Like reply(), but decode keeps the <|tool|> marker so tool calls parse.
        prompt_ids = encode_for_inference(msgs, tokenizer.encode, tokenizer.special)
        bs = getattr(getattr(model, "cfg", None), "block_size", None)
        if bs and len(prompt_ids) > bs:
            prompt_ids = prompt_ids[-bs:]
        new_ids = generate(model, prompt_ids, max_new_tokens=max_new_tokens,
                           temperature=temperature, top_k=top_k, top_p=top_p,
                           repetition_penalty=repetition_penalty,
                           eos_ids=(tokenizer.special.end,), device=device)
        return _decode_keep_tool(new_ids, tokenizer)

    res = run_tool_loop(gen, history, registry.specs(), registry, max_rounds=max_rounds)
    return res.final_answer


# Specials to strip from decoded text EXCEPT <|tool|> (needed for call parsing).
_STRIP_EXCEPT_TOOL = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>",
                      "<|assistant|>", "<|end|>", "<image>"]


def _decode_keep_tool(ids, tokenizer) -> str:
    """Decode keeping ``<|tool|>`` (filter other special IDS first, not strings).

    String-replacement would wrongly strip legitimate text containing e.g.
    ``<|assistant|>`` as plain prose. Instead drop the unwanted special-token IDS
    (keeping the tool id) before decoding with skip_special=False.
    """
    keep = {tokenizer.special.tool} if tokenizer.special.tool is not None else set()
    drop = {i for i in tokenizer.special.all if i not in keep}
    filtered = [i for i in ids if i not in drop]
    return tokenizer.decode(filtered, skip_special=False).strip()


def chat(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str = "auto",
    system_prompt: str | None = None,
    tag: str = "last",
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.3,
    tools: list[str] | None = None,
    file_roots: list[str] | None = None,
    allow_python: bool = False,
    rag_index: str | None = None,
) -> None:
    """Run an interactive chat REPL.

    Loads the model+tokenizer once, maintains conversation history, and loops over
    stdin. Type ``/exit`` (or send EOF) to quit; ``/reset`` clears the history.

    Args:
        checkpoint_dir: directory holding ``ckpt_<tag>.pt`` + ``model_config.json``.
        tokenizer_dir: directory with the trained tokenizer.
        device: device preference string.
        system_prompt: optional leading system message.
        tag: checkpoint tag ("last" or "best").
        max_new_tokens / temperature / top_k / top_p: generation controls.
    """
    dev = resolve_device(device)
    model, _cfg = load_model_from_checkpoint(checkpoint_dir, device=dev, tag=tag)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    # Optional tool registry (None => plain chat).
    registry = None
    if tools:
        from orkhon.serve.tools.registry import build_default_registry
        registry = build_default_registry(tools, file_roots=file_roots,
                                          allow_python=allow_python)
    if rag_index:
        from orkhon.serve.tools.registry import ToolRegistry
        from orkhon.serve.tools.retrieve import Retrieve
        registry = registry or ToolRegistry()
        registry.register(Retrieve(rag_index))

    def _fresh_history() -> list[Message]:
        if system_prompt:
            return [{"role": "system", "content": system_prompt}]
        return []

    history: list[Message] = _fresh_history()
    print("Orkhon chat. Type /exit to quit, /reset to clear history.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            history = _fresh_history()
            print("[history cleared]")
            continue

        # Build the next history immutably (append user turn).
        turn_history = history + [{"role": "user", "content": user_text}]
        if registry is not None:
            answer = reply_with_tools(
                turn_history, model, tokenizer, registry,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, device=dev,
            )
        else:
            answer = reply(
                turn_history,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=dev,
            )
        print(f"bot> {answer}")

        # Commit both turns to the running history.
        history = turn_history + [{"role": "assistant", "content": answer}]


def agent_repl(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str = "auto",
    tag: str = "last",
    tools: list[str] | None = None,
    file_roots: list[str] | None = None,
    rag_index: str | None = None,
    max_steps: int = 5,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    top_k: int | None = 40,
) -> None:
    """Interactive agent REPL: plan -> act -> observe, printing each step."""
    from rich.console import Console
    from orkhon.serve.agent import AgentConfig, run_agent
    from orkhon.serve.tools.registry import ToolRegistry, build_default_registry

    dev = resolve_device(device)
    model, _ = load_model_from_checkpoint(checkpoint_dir, device=dev, tag=tag)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    registry = build_default_registry(tools, file_roots=file_roots) if tools else ToolRegistry()
    if rag_index:
        from orkhon.serve.tools.retrieve import Retrieve
        registry.register(Retrieve(rag_index))
    if not registry.names():
        print("agent: no tools/RAG index enabled (use --tools / --rag-index); nothing to act with.")

    console = Console()
    history: list[Message] = []
    print("Orkhon agent. Type /exit to quit, /reset to clear history.")

    def gen(msgs):
        return reply(msgs, model, tokenizer, max_new_tokens=max_new_tokens,
                     temperature=temperature, top_k=top_k, device=dev)

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            history = []
            print("[history cleared]")
            continue
        turn = history + [{"role": "user", "content": user_text}]
        res = run_agent(gen, turn, registry, config=AgentConfig(max_steps=max_steps))
        for s in res.steps:
            if s.tool_call:
                console.print(f"  [cyan]step {s.index}[/cyan] {s.plan} "
                              f"-> [yellow]{s.tool_call['name']}[/yellow]({s.tool_call['arguments']})")
                console.print(f"    [dim]{s.observation[:200]}[/dim]")
            else:
                console.print(f"  [cyan]step {s.index}[/cyan] {s.plan}")
        console.print(f"  [({res.status})] bot> {res.final_answer or '(no final answer)'}")
        # Only commit a clean final answer; failed runs leave the turn unanswered
        # rather than poisoning context with a raw tool-call assistant turn.
        if res.status == "completed" and res.final_answer:
            history = turn + [{"role": "assistant", "content": res.final_answer}]
        else:
            history = turn
