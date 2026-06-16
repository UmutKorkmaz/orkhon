# Agents, Tools ve RAG

[English](agents.md) | [Türkçe](agents.tr.md)

Orkhon'un agent zemini: bir model + sınırlı tool/RAG döngüsü. Üç yüzey aynı çekirdeği paylaşır.

> **Not:** mevcut checkpoint'ler (`bumin`...`bengü-göktürk`) `<|tool|>` token'ı eklenmeden önce eğitildi.
> Bu yüzden native structured tool call üretmezler. Tool use bugün loop'un prompt + parser mekanizmasıyla
> çalışır; öğrenilmiş tool-calling değildir. Güvenilir native tool seçimi için sıradaki adım tool-trained
> checkpoint'tir (capability C6).

## Parçalar

- **Tools** (`serve/tools/`): `calculator` (safe-AST), `read_file` (açık root'lara hapsedilmiş),
  `retrieve` (RAG). Bu release'te yerel code execution yoktur.
- **RAG** (`rag/`): `orkhon rag ingest PATH... --out IDX` → `orkhon rag search "query" --index IDX`.
- **Agent loop** (`serve/agent/`): sınırlı plan → act → observe, sert `max_steps`, policy gate,
  escape edilmiş ve kısaltılmış observations, yeniden üretilebilir transcript.

## CLI

```bash
# Düz chat (tool yok)
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

`POST /v1/chat/completions` OpenAI uyumlu kalır (streaming + non-streaming). Tool use ayrı olarak
`/v1/agent/run` üzerinden yapılır. Güven sınırları için [`SECURITY.tr.md`](../SECURITY.tr.md): localhost'a bind
edin, container isolation olmadan public code execution eklemeyin.

## Güvenlik sınırları

- `read_file` için `--file-root`, `retrieve` için `--rag-index` gerekir.
- Tüm dış client mesaj içeriği escape edilir; inbound `role="tool"` reddedilir.
- Tool/RAG observation'ları escape edilir, uzunlukları kısıtlanır ve güvenilmeyen veri sayılır.
