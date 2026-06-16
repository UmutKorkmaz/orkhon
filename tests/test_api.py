"""Tests for the OpenAI-compatible FastAPI server (orkhon.serve.api)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from orkhon.serve.api import create_app


def _client(checkpoint_dir, tokenizer_dir) -> TestClient:
    app = create_app(checkpoint_dir, tokenizer_dir, device="cpu")
    return TestClient(app)


def test_health_ok(checkpoint_dir, tokenizer_dir):
    client = _client(checkpoint_dir, tokenizer_dir)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models(checkpoint_dir, tokenizer_dir):
    client = _client(checkpoint_dir, tokenizer_dir)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert any(m["id"] == "orkhon" for m in body["data"])


def test_chat_completion_non_stream(checkpoint_dir, tokenizer_dir):
    client = _client(checkpoint_dir, tokenizer_dir)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "orkhon",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert body["usage"]["prompt_tokens"] > 0


def test_chat_completion_stop_token_honored(checkpoint_dir, tiny_tokenizer, tokenizer_dir):
    """A 1-token greedy generation whose token is <|end|> must finish as 'stop'.

    We can't know the model's first token a priori, so instead we assert the
    contract: when the greedy first token equals <|end|>, finish_reason is 'stop'
    and content is empty. We discover that token via a tiny generate call mirror.
    """
    from orkhon.model.generation import generate
    from orkhon.tokenizer.render import encode_for_inference
    from orkhon.train.checkpoint import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(checkpoint_dir, device="cpu")
    end_id = tiny_tokenizer.special.end
    messages = [{"role": "user", "content": "Hello"}]
    prompt_ids = encode_for_inference(messages, tiny_tokenizer.encode, tiny_tokenizer.special)
    first = generate(model, prompt_ids, max_new_tokens=1, temperature=0.0)[0]

    client = _client(checkpoint_dir, tokenizer_dir)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    if first == end_id:
        # Stop token produced immediately => empty content, finished as 'stop'.
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["content"] == ""
    else:
        # Otherwise we still get a valid completion.
        assert isinstance(body["choices"][0]["message"]["content"], str)


def test_chat_completion_streaming(checkpoint_dir, tokenizer_dir):
    client = _client(checkpoint_dir, tokenizer_dir)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 6,
            "temperature": 0.0,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    text = resp.text
    # SSE frames terminated by a [DONE] sentinel.
    assert "data:" in text
    assert "[DONE]" in text

    # Parse the chunk frames (excluding the [DONE] sentinel) as valid JSON.
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and "[DONE]" not in line:
            chunks.append(json.loads(line[len("data:"):].strip()))
    assert chunks, "expected at least one streamed chunk"
    assert chunks[0]["object"] == "chat.completion.chunk"
    # First chunk announces the assistant role.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Final non-DONE chunk carries a finish_reason.
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
