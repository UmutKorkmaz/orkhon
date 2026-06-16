# Agents, Tools & RAG

Orkhon's agent substrate: a model + a bounded tool/RAG loop. Three surfaces share one core.

> **Caveat:** the current checkpoints (`bumin`…`bengü-göktürk`) were trained *before* the `<|tool|>` token
> was added, so they do not *natively* emit structured tool calls — tool use works via the loop's prompt
> + parser, not learned tool-calling. A tool-trained checkpoint (capability C6) is the next step for
> reliable native tool selection.

## The pieces

- **Tools** (`serve/tools/`): `calculator` (safe-AST), `read_file` (jailed to explicit roots),
  `retrieve` (RAG). Local code execution is not implemented in this release.
- **RAG** (`rag/`): `orkhon rag ingest PATH... --out IDX` → `orkhon rag search "query" --index IDX`.
- **Agent loop** (`serve/agent/`): bounded plan → act → observe, hard `max_steps`, a policy gate,
  escaped+truncated observations, a reproducible transcript.

## CLI

```bash
# Plain chat (no tools)
orkhon chat --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke

# Tools + RAG
orkhon rag ingest README.md docs --out data/rag/repo
orkhon agent --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke \
  --tools calculator --rag-index data/rag/repo --max-steps 5
```

## HTTP server

```bash
orkhon serve --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke \
  --tools calculator --rag-index data/rag/repo --port 8000
```

```bash
curl -s localhost:8000/v1/agent/run -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"What is 6*7?"}],
  "max_steps": 5, "max_tokens": 64, "temperature": 0
}'
# -> {"object":"agent.run","status":"completed","final_answer":"...","steps":[...]}
```

`POST /v1/chat/completions` remains OpenAI-compatible (streaming + non-streaming); tool use is via
`/v1/agent/run`. See [`SECURITY.md`](../SECURITY.md) for the trust boundaries (bind to localhost; do not add
public code execution without container isolation).

## Safety boundaries (summary)

- `read_file` needs `--file-root`; `retrieve` needs `--rag-index`.
- All external (client) message content is escaped; inbound `role="tool"` is rejected.
- Tool/RAG observations are escaped + length-capped and treated as untrusted data.
