"""Orkhon demo — a Gradio chat/agent UI for Hugging Face Spaces or local use.

Env:
    ORKHON_CHECKPOINT, ORKHON_TOKENIZER  (required)
    ORKHON_TOOLS, ORKHON_RAG_INDEX        (optional; enable agent mode)
    ORKHON_DEVICE                         (default: cpu)

Local:
    uv run --extra demo python spaces/orkhon-demo/app.py
"""

from __future__ import annotations

import os

CHECKPOINT = os.environ.get("ORKHON_CHECKPOINT", "runs/sft_smoke")
TOKENIZER = os.environ.get("ORKHON_TOKENIZER", "artifacts/tokenizer/smoke")
TOOLS = [t for t in os.environ.get("ORKHON_TOOLS", "").split(",") if t]
FILE_ROOTS = [r for r in os.environ.get("ORKHON_FILE_ROOTS", "").split(",") if r]
RAG_INDEX = os.environ.get("ORKHON_RAG_INDEX") or None
DEVICE = os.environ.get("ORKHON_DEVICE", "cpu")


def _build_reply():
    """Load the model once; return a stateful reply function + whether agent mode is on."""
    from orkhon.serve.tools.registry import ToolRegistry, build_default_registry
    from orkhon.tokenizer import load_tokenizer
    from orkhon.train.checkpoint import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(CHECKPOINT, device=DEVICE)
    model.eval()
    tok = load_tokenizer(TOKENIZER)
    # read_file needs explicit roots; never enable python_exec on a public Space.
    registry = (build_default_registry(TOOLS, file_roots=FILE_ROOTS or None) if TOOLS
                else ToolRegistry())
    if RAG_INDEX:
        from orkhon.serve.tools.retrieve import Retrieve
        registry.register(Retrieve(RAG_INDEX))
    agent = bool(registry.names())

    def reply(history: list[dict]) -> str:
        from orkhon.serve.chat_cli import reply_with_tools, reply
        if agent:
            return reply_with_tools(history, model, tok, registry, max_new_tokens=160, device=DEVICE)
        return reply(history, model, tok, max_new_tokens=160, device=DEVICE)

    return reply, agent


def main():
    import gradio as gr

    reply_fn, agent = _build_reply()
    title = f"Orkhon {'agent' if agent else 'chat'} demo"

    def respond(message, history):
        msgs = [{"role": "user" if r["role"] == "user" else "assistant", "content": r["content"]}
                for r in history]
        msgs.append({"role": "user", "content": message})
        return reply_fn(msgs)

    gr.ChatInterface(respond, type="messages", title=title).launch()


if __name__ == "__main__":
    main()
