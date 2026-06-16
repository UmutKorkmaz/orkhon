"""OpenAI-compatible FastAPI server for Orkhon chat completions.

``create_app`` loads the model + tokenizer ONCE at startup and exposes:

* ``GET  /health``                — liveness probe.
* ``GET  /v1/models``             — the single served model id.
* ``POST /v1/chat/completions``   — OpenAI chat completions. ``stream=False``
  returns a full :class:`ChatCompletionResponse`; ``stream=True`` returns an
  ``text/event-stream`` of ``chat.completion.chunk`` SSE events terminated by
  ``data: [DONE]``.

Generation reuses :func:`orkhon.model.generate` for the non-streaming path and a
token-by-token KV-cache decode loop for streaming, both stopping on ``<|end|>``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from orkhon.model.generation import generate
from orkhon.model.kv_cache import KVCache
from orkhon.serve.sampling import sample_next
from orkhon.serve.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    Choice,
    DeltaMessage,
    ModelCard,
    ModelList,
    StreamChoice,
    Usage,
)
from orkhon.serve.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentRunStep,
    ChatMessage,
)
from orkhon.tokenizer.render import encode_for_inference
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.utils.device import resolve_device

MODEL_ID = "orkhon"


def _render_prompt(messages, tokenizer) -> list[int]:
    """Render OpenAI messages to inference token ids ending in ``<|assistant|>``."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    return encode_for_inference(msg_dicts, tokenizer.encode, tokenizer.special)


@torch.no_grad()
def _stream_token_ids(
    model,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_id: int,
    device,
) -> Iterator[tuple[int, bool]]:
    """Yield ``(token_id, is_eos)`` one decode step at a time using a KV cache.

    Mirrors :func:`orkhon.model.generate` but yields incrementally so the API can
    stream. Stops after ``max_new_tokens`` or when ``eos_id`` is produced (the eos
    token itself is yielded with ``is_eos=True`` so the caller can finalize).
    """
    block_size = model.cfg.block_size
    cache = KVCache(model.cfg.n_layers)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache = model(input_ids, past=cache, use_cache=True)
    next_logits = logits[0, -1, :]

    for _ in range(max_new_tokens):
        next_id = sample_next(
            next_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        is_eos = next_id == eos_id
        yield next_id, is_eos
        if is_eos or cache.length >= block_size:
            break
        step_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, cache = model(step_input, past=cache, use_cache=True)
        next_logits = logits[0, -1, :]


def create_app(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str = "auto",
    tag: str = "last",
    tools: list[str] | None = None,
    file_roots: list[str] | None = None,
    allow_python: bool = False,
    rag_index: str | None = None,
    agent_generate_fn=None,
) -> FastAPI:
    """Build a FastAPI app serving the given checkpoint + tokenizer.

    The model and tokenizer are loaded once here and captured in the route
    closures, so every request reuses the same in-memory objects. When ``tools``
    and/or ``rag_index`` are given, a server-side tool registry is built at startup
    and exposed via ``POST /v1/agent/run``.
    """
    dev = resolve_device(device)
    model, _cfg = load_model_from_checkpoint(checkpoint_dir, device=dev, tag=tag)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_dir)
    eos_id = tokenizer.special.end

    # Build the server-side tool registry once (None if no tools/RAG configured).
    registry = None
    if tools or rag_index:
        from orkhon.serve.tools.registry import build_default_registry
        registry = build_default_registry(tools, file_roots=file_roots,
                                          allow_python=allow_python) if tools else None
        if rag_index:
            from orkhon.serve.tools.registry import ToolRegistry
            from orkhon.serve.tools.retrieve import Retrieve
            registry = registry or ToolRegistry()
            registry.register(Retrieve(rag_index))

    app = FastAPI(title="Orkhon", version="1.0")

    def _generate_text(messages: list[dict], gen_kwargs: dict) -> str:
        """Shared decode path: render messages -> generate -> decode (no specials)."""
        # Old tokenizers have no <|tool|> id; render tool observations as user text
        # so they encode safely instead of raising KeyError.
        if tokenizer.special.tool is None:
            messages = [{"role": "user",
                         "content": "[observation] " + m["content"]} if m["role"] == "tool"
                        else m for m in messages]
        prompt_ids = encode_for_inference(messages, tokenizer.encode, tokenizer.special)
        # Cap the prompt to the context window, RESERVING decode space (so a long
        # transcript can't fill the whole window and leave no room to generate).
        bs = getattr(getattr(model, "cfg", None), "block_size", None)
        reserve = min(int(gen_kwargs.get("max_new_tokens", 1)), max(1, (bs or 1) // 4))
        cap = (bs - reserve) if bs else None
        if cap and len(prompt_ids) > cap:
            prompt_ids = prompt_ids[-cap:]
        new_ids = generate(model, prompt_ids, eos_ids=(eos_id,), device=dev, **gen_kwargs)
        visible = new_ids[:-1] if (new_ids and new_ids[-1] == eos_id) else new_ids
        return tokenizer.decode(visible, skip_special=True)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": MODEL_ID, "device": str(dev),
                "tools": registry.names() if registry else []}

    @app.get("/v1/models")
    def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=MODEL_ID)])

    @app.post("/v1/agent/run")
    def agent_run(req: AgentRunRequest):
        if registry is None or not registry.names():
            from fastapi import HTTPException
            raise HTTPException(422, "no tools/RAG index configured on this server")
        from orkhon.serve.agent import AgentConfig, run_agent
        from orkhon.serve.tool_loop import escape_tool_output

        gen_kwargs = dict(max_new_tokens=req.max_tokens, temperature=req.temperature,
                          top_k=req.top_k, top_p=None)
        # External message content is untrusted: neutralize any literal Orkhon
        # special-token strings so a client cannot inject control tokens.
        msgs = [{"role": m.role, "content": escape_tool_output(m.content)} for m in req.messages]
        # Use the injected generator (for testing) or the real model decode path.
        driver = ((lambda ms: agent_generate_fn(ms, gen_kwargs))
                  if agent_generate_fn is not None
                  else (lambda ms: _generate_text(ms, gen_kwargs)))
        res = run_agent(driver, msgs, registry,
                        config=AgentConfig(max_steps=req.max_steps))
        return AgentRunResponse(
            status=res.status, final_answer=res.final_answer,
            steps=[AgentRunStep(index=s.index, plan=s.plan, tool_call=s.tool_call,
                                observation=s.observation, error=s.error) for s in res.steps],
            messages=[ChatMessage(role="tool" if m["role"] == "tool" else m["role"],
                                  content=m["content"]) for m in res.messages]
            if req.include_messages else [],
        )

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt_ids = _render_prompt(req.messages, tokenizer)
        gen_kwargs = dict(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

        if req.stream:
            return StreamingResponse(
                _sse_stream(prompt_ids, gen_kwargs),
                media_type="text/event-stream",
            )

        # Non-streaming: generate all new ids at once.
        new_ids = generate(
            model,
            prompt_ids,
            eos_ids=(eos_id,),
            device=dev,
            **gen_kwargs,
        )
        finish_reason = "stop" if (new_ids and new_ids[-1] == eos_id) else "length"
        # Drop a trailing eos before decoding the visible content.
        visible = new_ids[:-1] if (new_ids and new_ids[-1] == eos_id) else new_ids
        text = tokenizer.decode(visible, skip_special=True)

        return ChatCompletionResponse(
            model=MODEL_ID,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionResponseMessage(content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(visible),
                total_tokens=len(prompt_ids) + len(visible),
            ),
        )

    def _sse_stream(prompt_ids: list[int], gen_kwargs: dict) -> Iterator[str]:
        """Yield OpenAI-style SSE chunks for a streaming completion."""
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"

        def _chunk(delta: DeltaMessage, finish=None) -> str:
            payload = ChatCompletionChunk(
                id=chunk_id,
                model=MODEL_ID,
                choices=[StreamChoice(index=0, delta=delta, finish_reason=finish)],
            )
            return f"data: {payload.model_dump_json()}\n\n"

        # First chunk announces the assistant role.
        yield _chunk(DeltaMessage(role="assistant"))

        finish_reason = "length"
        emitted_ids: list[int] = []
        for token_id, is_eos in _stream_token_ids(
            model,
            prompt_ids,
            eos_id=eos_id,
            device=dev,
            **gen_kwargs,
        ):
            if is_eos:
                finish_reason = "stop"
                break
            # Incremental decode: decode the running buffer and emit the new suffix
            # so multi-byte / merged tokens render correctly.
            prev_text = tokenizer.decode(emitted_ids, skip_special=True)
            emitted_ids.append(token_id)
            new_text = tokenizer.decode(emitted_ids, skip_special=True)
            piece = new_text[len(prev_text):]
            if piece:
                yield _chunk(DeltaMessage(content=piece))

        yield _chunk(DeltaMessage(), finish=finish_reason)
        yield "data: [DONE]\n\n"

    return app
