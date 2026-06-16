# Orkhon demosu (Hugging Face Space)

[English](README.md) | [Türkçe](README.tr.md)

Bir Orkhon checkpoint'i üzerinde tek dosyalık Gradio chat/agent UI. HF Space içine koyun (SDK: Gradio, hardware:
CPU veya ZeroGPU) ve aşağıdaki secret/env değerlerini ayarlayın.

## Ortam

| Değişken | Zorunlu | Anlamı |
|---|---|---|
| `ORKHON_CHECKPOINT` | evet | checkpoint dizini veya yayınlandıktan sonra HF repo id |
| `ORKHON_TOKENIZER` | evet | tokenizer dizini |
| `ORKHON_TOOLS` | hayır | virgülle ayrılmış liste, örn. `calculator,read_file`; agent mode'u açar |
| `ORKHON_RAG_INDEX` | hayır | RAG index dizini; `retrieve` aracını açar |
| `ORKHON_DEVICE` | hayır | `cpu` (varsayılan) / `cuda` / `mps` |

## Yerel çalışma

```bash
uv sync --extra demo
ORKHON_CHECKPOINT=runs/sft_smoke ORKHON_TOKENIZER=artifacts/tokenizer/smoke \
  uv run python spaces/orkhon-demo/app.py
```

> Public Space üzerinde `python_exec` açmayın. Sandbox yalnızca trusted-local kullanım içindir
> (bkz. [`SECURITY.tr.md`](../../SECURITY.tr.md)). Demo bunu asla etkinleştirmez.
