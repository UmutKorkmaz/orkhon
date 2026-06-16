# Orkhon demo (Hugging Face Space)

A one-file Gradio chat/agent UI over an Orkhon checkpoint. Drop into a HF Space (SDK: Gradio,
hardware: CPU or ZeroGPU) and set the secrets/env below.

## Environment

| Var | Required | Meaning |
|---|---|---|
| `ORKHON_CHECKPOINT` | yes | a checkpoint dir (or a HF repo id once published) |
| `ORKHON_TOKENIZER` | yes | tokenizer dir |
| `ORKHON_TOOLS` | no | comma-separated, e.g. `calculator,read_file` (enables agent mode) |
| `ORKHON_RAG_INDEX` | no | a RAG index dir (enables `retrieve`) |
| `ORKHON_DEVICE` | no | `cpu` (default) / `cuda` / `mps` |

## Local

```bash
uv sync --extra demo
ORKHON_CHECKPOINT=runs/sft_smoke ORKHON_TOKENIZER=artifacts/tokenizer/smoke \
  uv run python spaces/orkhon-demo/app.py
```

> Do **not** enable `python_exec` on a public Space — the sandbox is trusted-local only
> (see [`SECURITY.md`](../../SECURITY.md)). The demo never enables it.
