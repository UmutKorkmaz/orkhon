# Orkhon Kabiliyet Roadmap'i — Tools, RAG, Agents, Vision

[English](capability-roadmap.md) | [Türkçe](capability-roadmap.tr.md)

> [`docs/roadmap.tr.md`](roadmap.tr.md) ölçek eksenini yönetir: R0→R6, yani model boyutu × token.
> Bu doküman kabiliyet eksenini yönetir: C0→C8, yani tools, retrieval, search, agents ve vision. Orkhon'un
> chat modelinden agentic ve multimodal sisteme nasıl büyüyeceğini ve bu eksenin scale ladder ile nasıl
> iç içe geçtiğini anlatır.
>
> **Durum notu (2026-06-16):** İlk serve-time zemin artık uygulanmış durumda: tool parsing,
> calculator/read-file/retrieve tools, RAG ingest/search, bounded agent
> loop, HTTP agent endpoint, tool-SFT traces ve GRPO smoke code. Kalan iş: validation, benchmark gates, public
> surfaces ve pre-R3 tokenizer freeze.

---

## 1. Karar kuralı

> **Algı → modele işlenir.** Vision/audio encoder + projector pahalıdır ve C8'e kalır.
> **Bilgi / eylem → tool verilir.** Calculator, file-read, retrieval, web-search, image-as-tool C1-C4'tür ve
> GPU'suz başlar. **Tekrarlı tool use → agent loop** (C5). **Routing / MoE → v1 değil.**

"Model mi agent mı tool mu?" sorusu isim meselesi değildir; kabiliyetin türü meselesidir. İnternet ve dosya
sistemi weights içine gömülmez; stale olur, kaynak veremez ve izinlendirme yapılamaz. Algı bile önce tool olarak
başlayabilir; ancak sahiplenmek gerektiğinde modele işlenir.

## 2. Serve-time önce, train-time sonra

En ucuz ve en yüksek kaldıraçlı kabiliyetler **weights dışında**, `serve/` orchestration katmanındadır.
Tool-calling, RAG injection ve agent loop öğrenilmiş davranış olmak zorunda değildir; prompt construction +
output parsing ile çalışır. Bu yüzden C0-C5 bugün laptop üzerinde ve $0 GPU ile kurulabilir.

```text
SERVE-TIME (laptop, $0)          TRAIN-TIME (cloud, gated)
C0  infra spine                  C6 tool-use SFT      -> R3 gerekir
C1  <|tool|> token + schema      C7 GRPO/RLVR         -> R4 gerekir
C2  tool executors               C8 native orkhon-vl  -> R3/R4
C3  RAG / files
C4  web search + image-as-tool
C5  agent loop
```

## 3. İki eksen arasındaki tek sert bağ

Tokenizer geri dönüşsüzdür. R3 tokenizer freeze edildikten sonra control token eklemek yeni pretrain demektir.

> R3 (`tengri`, 350M) tokenizer'ı eğitilmeden önce `<|tool|>` id 8 olarak eklenmeli ve `<image>` id 9 olarak
> rezerve edilmelidir. Id 0-7 asla yeniden sıralanmaz.

Capability timeline ile scale timeline arasındaki gerçek coupling budur. Geri kalan serve-time işler scale
rung beklemeden inşa edilir.

---

## 4. Birleşik merdiven

| C# | Milestone | Ana dosyalar | Gate | Efor |
|----|-----------|--------------|------|------|
| **C0** | GitHub + CI + repo hijyeni | `.github/workflows/ci.yml`, `.gitignore` | R0 | 0.5g |
| **C0.1** | `orkhon publish` | `publish.py`, `cli.py` | R0 | 1g |
| **C0.2** | MetricsSink / W&B | `train/monitor.py`, `train/engine.py` | R1 | 1g |
| **C1** | `<\|tool\|>` id 8 + OpenAI tool schema + parser | tokenizer, render, serve schemas | R3 freeze öncesi | 2g |
| **C2** | Tool registry: calculator, read_file, retrieve | `serve/tools/`, `serve/tool_loop.py` | yok | 1.5g |
| **C3** | RAG / file understanding | `rag/`, `serve/tools/retrieve.py`, `cli.py` | kalite R3 | 7g |
| **C4** | Web search + image-as-tool | `serve/tools/web_search.py`, `vision/describe.py` | yok | 3g |
| **C5** | Bounded agent loop + `/v1/agent/run` | `serve/agent/`, `serve/api.py`, `cli.py` | güvenilirlik R4 | 4g |
| **C-EVAL** | Tool/RAG/agent eval harness | `eval/{tool_eval,rag_eval,agent_eval}.py` | R3/R4'te blocking | 3g |
| **C6** | Native tool-use SFT | `data/tool_synth.py`, `train/sft.py` | R3 | 3g |
| **C7** | Rejection sampling → GRPO/RLVR | `train/{rejection_sample,grpo,rewards}.py` | R4 | roadmap P1 |
| **C8** | Native `orkhon-vl` | `vision/`, `train/vl_*`, `model/transformer.py` | R3/R4 | haftalar |

### Mevcut implementation status

- **Uygulandı:** C0 CI skeleton, C1 special-token append + parser plumbing, C2 calculator/read-file/retrieve,
  C3 hash-based RAG, C5 bounded agent loop + `/v1/agent/run`, C6 tool-trace synthesis, C7 GRPO smoke path.
- **Kısmi / doğrulanmadı:** OpenAI-native `tools/tool_calls` compatibility, trained checkpoint tool reliability,
  RAG recall/citation benchmark, scoreboard reports, HF publish flow, deterministic sharded scale proof.
- **Yok:** C4 web search, image-as-tool, native `orkhon-vl`, public code-execution tool.

---

## 5. Katmanlar

### 5.1 Tools / function-calling

`<|tool|>` id 8 ve `<image>` id 9 append-only kalır. `serve/schemas.py` OpenAI `tools`/`tool_choice`/`tool_calls`,
`tool` role ve `finish_reason: "tool_calls"` alanlarını taşır. Tool loop, modelin call üretmesi → runtime'ın
execute etmesi → observation'ın modele dönmesi şeklinde `serve/` altında yaşar. İlk tools: calculator, read_file
ve retrieve. `python_exec` bu release'te yoktur; container isolation kanıtlanmadan açılmaz.

### 5.2 RAG / files + web search

Embedding modeli eğitmeyin; BGE-M3/e5 gibi hazır multilingual modelleri kullanın. Parse → chunk → embed →
persisted NumPy/JSONL store → retrieve → cite. Citation yalnız context'e gerçekten giren chunk için verilir.
Web search bir search-API tool'u olarak eklenir.

### 5.3 Vision

Phase A: `describe_image` tool'u mevcut açık VLM çağırır; Orkhon dönen metin üzerinde reasoning yapar.
Phase B: native `orkhon-vl`, frozen SigLIP/CLIP encoder + trainable MLP projector + `<image>` token plumbing.
Bu R3 sonrası GPU ve veri işidir.

### 5.4 Agents

Tek tool call'dan bounded plan→act→observe döngüsüne gidilir. Safety: allow-list, approval gates, max steps,
retry budget, reproducible transcripts, denial sonrası tool call yok.

### 5.5 Infra / accounts

Önce GitHub/CI, sonra Hugging Face, W&B, opportunistic free GPU, paid GPU provider ve object storage.

---

## 6. Sıra

**Phase 0 — laptop, $0:**
1. Truth surface'i stabilize et: implemented / partial / validated-at-scale ayrımı.
2. Benchmark scoreboard'u çalıştır; JSON rapor olmadan capability claim yok.
3. Tokenizer freeze gate: 48k multilingual, `<|tool|>`@8, `<image>`@9, fertility pass.
4. Scale-readiness proof: deterministic sharded sampling, source-mix reader, exact resume.
5. `orkhon publish` + W&B MetricsSink.
6. C4 + C-EVAL: web search, image-as-tool, capability evals.

**Phase B — R2→R3:** `tengri` gelince C6 native tool-use SFT, HF Space demo ve capability eval'leri blocking gate.

**Phase C — R4:** `otuken` ile C7 GRPO ve C8 native vision başlangıcı.

### Defer set

Native video, internal MoE, learned routers, multi-agent orchestration ve public code execution ertelenir.
Capability work scale spine'ı bloklamamalı: sharded data, eval bridge, tokenizer reconciliation ve R1/R2/R3
proof run'ları önce gelir.

---

## 7. Accounts & infra timeline

| Sıra | Hesap | Ne için |
|---|---|---|
| 1 | **GitHub** | code, CI, releases |
| 2 | **Hugging Face** | models/datasets, Spaces, Jobs |
| 3 | **Weights & Biases** | deney takibi |
| 4 | **Kaggle / Colab** | fırsatçı free GPU smoke |
| 5 | **RunPod** | ilk cloud GPU burst |
| 6 | **Modal** | serverless jobs/evals/data prep |
| 7 | **Lambda / Together / Replicate / Vast** | daha büyük cluster ve hosted inference |
| — | **R2 / S3 / Backblaze** | shards + DCP checkpoint storage |

---

## 8. Yeni dosyalar

```text
.github/workflows/ci.yml
src/orkhon/publish.py
src/orkhon/train/monitor.py
src/orkhon/serve/tool_protocol.py
src/orkhon/serve/tool_loop.py
src/orkhon/serve/tools/{base,registry,calculator,read_file,web_search,retrieve,vision}.py
src/orkhon/serve/agent/{loop,policy}.py
src/orkhon/rag/{loaders,chunk,embed,store,retrieve,citations,pipeline}.py
src/orkhon/vision/{describe,encoder,projector,processor,dataset,eval}.py
src/orkhon/eval/{tool_eval,rag_eval,agent_eval}.py
src/orkhon/data/tool_synth.py
src/orkhon/train/vl_{align,sft}.py
configs/{tools,rag,agents}/*.yaml
```

---

*Algı modele girer. Bilgi ve eylem tool'a gider. Tekrarlı tool use agent loop olur. Routing altyapıdır. MoE geç
ölçek trick'idir.*
