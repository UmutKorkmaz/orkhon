# Orkhon Capability Roadmap — Tools, RAG, Agents, Vision

> The companion to [`docs/roadmap.md`](roadmap.md). That document owns the **scale axis** (R0→R6: model size ×
> tokens). This one owns the **capability axis** (C0→C8: tools, retrieval, search, agents, vision) — how Orkhon
> grows from a chatbot into a capable, agentic, multimodal system — and the rule for interleaving the two.
>
> **Status note (2026-06-16):** the first serve-time substrate is now implemented: tool parsing, calculator/read-file/retrieve
> tools, RAG ingest/search, bounded agent loop, HTTP agent endpoint, tool-SFT traces, and GRPO smoke code. The
> remaining work is validation, benchmark gates, missing public surfaces, and the pre-R3 tokenizer freeze.

---

## 1. The decision rule (the spine)

> **Perception → bake into the model.** A vision/audio encoder + projector (expensive, gated — C8).
> **Knowledge / action → give it a tool.** Calculator, file-read, retrieval, web-search, image-as-tool (C1–C4;
> now, $0 GPU). **Repeated tool use → an agent loop** (bounded plan→act→observe — C5). **Routing / MoE → not in v1.**

This dissolves the "model vs agent vs tool" question: it's never about the *label*, only about *what kind of
capability* it is. An "agent" is just an LLM with tools and a loop. You never train "the internet" or "the
filesystem" into weights — that goes stale, can't cite, and can't be permissioned. Even **perception can start as
a tool** (call an existing open VLM) and only becomes baked-in when you want to own it.

## 2. The key insight — serve-time first, train-time later

**The cheapest, highest-leverage capabilities live *outside the weights*,** in `serve/` orchestration around the
existing OpenAI-compatible chat API. Tool-calling, RAG injection, and the agent loop are **prompt construction +
output parsing** — not learned behavior — so **C0 through C5 work *today*, for $0 GPU, on the already-imported
`kashgari` (SmolLM2-135M) checkpoint.** Build the entire serve-side scaffold on the laptop, in parallel with the
scale spine. The model that makes it *reliable* is bought by the scale ladder (R3/R4), not by this layer.

```
SERVE-TIME (laptop, $0, now)         TRAIN-TIME (gated on scale, cloud $)
C0  infra spine                      C6  tool-use SFT        → needs R3 (350M)
C1  <|tool|> token + schema          C7  GRPO/RLVR rewards   → needs R4 (1B)
C2  tool executors                   C8  native orkhon-vl    → R3 floor / R4 real
C3  RAG / files
C4  web search + image-as-tool
C5  agent loop
```

## 3. The ONE hard coupling between the two axes ⚠️

The tokenizer is **irreversible** (guide §11; enforced by `tokenizer/train.py::_assert_special_ids`, which
validates `SPECIAL_TOKENS` land at ids `0..len-1`). A control token added *after* the R3 tokenizer is frozen
forces a fresh pretrain. Therefore:

> **Append `<|tool|>` (id 8) and reserve `<image>` (id 9) in `tokenizer/special_tokens.py` — append-only, never
> reorder ids 0–7 — BEFORE the R3 (350M `tengri`) tokenizer is trained.** Decide the index-8/9 assignment once.

This is the *single* point where capability work couples to the scale timeline. Everything else on the capability
axis is independent of the scale rungs for building and testing.

---

## 4. The unified ladder (C0…C8)

"Effort" = focused solo dev-days on the M5 Pro. "Gates-on" names the scale rung an *acceptance* gate needs; most
rows need **no rung to build**.

| C# | Milestone (what it adds) | Primary Orkhon files | Gates-on | Effort |
|----|--------------------------|----------------------|----------|--------|
| **C0** | **Infra spine** — `git init` + public GitHub repo + CI (`pytest -m "not slow"`) | `.github/workflows/ci.yml`, `.gitignore` | R0, do first | 0.5d |
| **C0.1** | **`orkhon publish`** — push zoo models to HF (composes `export/to_hf.py` + `registry.py` card, parity-checked) | `publish.py`, `cli.py` | R0 | 1d |
| **C0.2** | **MetricsSink / W&B** — one-line `engine.py` swap; JSONL always on, W&B/TB fan-out | `train/monitor.py`, `train/engine.py`, `config/schema.py` | R1 | 1d |
| **C1** | **`<\|tool\|>` role @ id 8** + OpenAI `tools`/`tool_calls` schema + serve-time parser + `generate()` stop-strings | `tokenizer/special_tokens.py`, `render.py`, `chat_template.jinja`, `serve/schemas.py`, `serve/tool_protocol.py`, `model/generation.py` | **must precede R3 freeze**; *reliable* use needs R3 | 2d |
| **C2** | **Tool executors + registry** — calculator, read_file (jailed), retrieve/RAG. `python_exec` is deliberately unavailable until container isolation exists. | `serve/tools/{base,registry,calculator,read_file,retrieve}.py`, `serve/tool_loop.py` | none | 1.5d |
| **C3** | **RAG / file-understanding** — parse→chunk→embed (BGE/e5)→store→retrieve→cite; `orkhon rag` flow | `rag/{loaders,chunk,embed,store,retrieve,citations,pipeline}.py`, `serve/tools/retrieve.py`, `cli.py` | pipeline: none; answer quality: R3 | 7d |
| **C4** | **Web search tool** + **image-as-a-tool** (call an open VLM via `transformers`) | `serve/tools/web_search.py`, `vision/describe.py`, `serve/tools/vision.py`, `serve/schemas.py` | none | 3d |
| **C5** | **Agent loop** — bounded plan→act→observe + approval policy + `/v1/agent/run` + `orkhon agent` REPL | `serve/agent/loop.py`, `serve/agent_policy.py`, `serve/api.py`, `cli.py` | mechanical now; *credible* R3; *reliable* R4 | 4d |
| **C-EVAL** | **Capability eval harness** — tool-call validity, RAG citation/recall, agent task-success | `eval/{tool_eval,rag_eval,agent_eval,chat_eval}.py` | run free on kashgari; blocking at R3/R4 | 3d |
| **C0.3** | **Spaces demo + dataset repos + object storage** — ZeroGPU chat/tools/RAG demo; R2/S3 for shards+DCP | `spaces/chat/`, `infra/storage.py` | Spaces post-R3; storage R4 | 3d |
| **C6** | **Tool-use SFT** — teach *native* tool-calling (not few-shot) + multi-turn packing fix + trace synth | `data/tool_synth.py`, `train/sft.py`, `data/dataset.py` | **R3** (`tengri`) | 3d |
| **C7** | **Rejection-sampling → GRPO/RLVR** for tool/format/verifiable rewards | `train/{rejection_sample,grpo,rewards}.py`, `model/generation.py::generate_batch` | **R4** (`otuken`) | folds into roadmap P1 |
| **C8** | **Native `orkhon-vl`** — frozen SigLIP encoder + trainable MLP projector + `<image>` splice + 2-stage train | `model/transformer.py` (`inputs_embeds` path), `vision/{encoder,projector,processor,dataset,eval}.py`, `train/vl_{align,sft}.py` | **R3 floor / R4 real** | weeks (cloud) |

Everything **C0–C5 + C-EVAL is laptop, $0, no rung dependency** — build it in parallel with the roadmap's NEXT
scale-spine phase. C6/C7/C8 are the **train-time** capabilities that wait for a base worth training.

### Current implementation status

- **Implemented locally:** C0 git/CI skeleton, C1 special-token append + parser plumbing, C2 calculator/read-file/retrieve tools, C3 hash-based RAG, C5 bounded agent loop + `/v1/agent/run`, C6 tool-trace synthesis, C7 GRPO smoke path.
- **Partial / not yet validated:** OpenAI-native `tools/tool_calls` compatibility, tool-call reliability on a trained checkpoint, RAG recall/citation benchmarks, benchmark scoreboard reports, publish-to-HF flow, deterministic sharded scale proof.
- **Not implemented:** C4 web search, image-as-tool, native `orkhon-vl`, and any public code-execution tool.

---

## 5. The layers (folded, file-ownership resolved)

### 5.1 Tools / function-calling (C1–C2) — the agent substrate
Append `TOOL = "<|tool|>"` at index 8 (and reserve `<image>` at 9); `_write_special_tokens_map` already slices
`SPECIAL_TOKENS[4:]` so the next retrain produces them automatically. Extend `serve/schemas.py` with OpenAI
`tools`/`tool_choice`/`tool_calls` + a `tool` role + `finish_reason: "tool_calls"`. The tool *loop* (model emits a
call → runtime executes → result fed back → model answers) lives in `serve/` and is wired into **both** the chat
CLI and the API — `serve/api.py::chat_completions` is the single render+decode chokepoint, so the loop has one
home. First tools: **calculator** (safe-AST), **read_file** (jailed to configured roots, traversal-guarded),
and **retrieve** (RAG index lookup). **`python_exec` is not implemented in this release**; do not expose local code
execution until container isolation, timeout, output caps, and no-network guarantees are proven. *Acceptance:* ≥95%
valid tool-call rate on an overfit smoke model; `read_file` cannot escape roots; API round-trips tool use.

### 5.2 RAG / files + web search (C3–C4) — highest-ROI, GPU-free
The guide's §15 says a domain embedding model is the highest-ROI extension; "Orkhon with files" feels more useful
per engineering-hour than vision. **Reuse** BGE-M3/e5 via `sentence-transformers` (do **not** train embeddings);
parse → chunk → embed → a simple persisted NumPy/JSONL store (FAISS/Qdrant only after the API stabilizes) →
retrieve → cite. Expose retrieval as both a tool and an `orkhon rag ingest docs/` flow. Web search = a search-API
tool (fetch + rank + cite). *Acceptance:* recall@5 ≥ 80% on a 50-question local set; **no citation unless the
cited chunk was actually in context**; RAG beats the no-RAG baseline.

### 5.3 Vision (C4 image-as-tool → C8 native orkhon-vl)
**Phase A (now, no training):** a `describe_image` tool that calls an open VLM (SmolVLM / Qwen2.5-VL); Orkhon
reasons over the returned text, clearly separating "VLM observation" from its own reasoning. **Phase B (after R3):**
native `orkhon-vl` — a **frozen** SigLIP/CLIP encoder + a trainable MLP **projector** + `<image>` token plumbing.
This needs a new `inputs_embeds` path in `model/transformer.py::forward` (it has none today) so continuous
projected patch embeddings splice in at placeholder positions — *not* thousands of discrete image tokens. Two-stage
train: (1) projector alignment on image-caption pairs (reuses the pretrain engine), (2) visual-instruction SFT
(LLaVA-style). Honest: 4–8 weeks + data + GPU; the goal is learning + a credible demo, **not** beating Qwen-VL.

### 5.4 Agents (C5) — the control loop on tools
Progression: single tool call → a bounded multi-step loop (plan→act→observe, with a **max-steps + retry budget +
stop condition**) → later memory/approval gates/task queues. Safety model: sandboxing, allow-lists,
human-in-the-loop for write/exec/network actions, reproducible transcripts, **no tool call after a denial**.
Agent *quality* is gated on model capability — mechanical now, credible at R3, reliable at R4.

### 5.5 Infra / accounts — the platform under every rung
The public project surface should stay release-ready: GitHub for code/CI/releases, Hugging Face for models,
datasets and Spaces, W&B before paid runs, and compute providers per rung. See the timeline below.

---

## 6. Do next, in order

**Phase 0 — laptop, $0, in parallel with the roadmap's scale spine:**
1. **Stabilize the truth surface** — docs/status must separate implemented, partial, and validated-at-scale.
2. **Run the benchmark scoreboard** — write JSON reports before claiming model capability.
3. **Tokenizer freeze gate** — 48k multilingual candidate + `<|tool|>`@8 + `<image>`@9 + fertility pass.
4. **Scale-readiness proof** — deterministic sharded sampling, source-mix reader, exact resume, checkpoint safety.
5. **C0.1 / C0.2** — `orkhon publish` to HF + the W&B MetricsSink (so the first cloud run is observable).
6. **C4 + C-EVAL** — web search, image-as-tool, and capability evals after the base substrate is measured.

**Phase B — R2→R3 ($190 → $1.35k), once `tengri` (350M) arrives:** **C6** tool-use SFT (native tool-calling),
the HF Space demo, promote capability evals to blocking gates.

**Phase C — R4 (~$17k), `otuken` (1B):** **C7** rejection-sampling → GRPO; start **C8** native `orkhon-vl`.

### The explicit DEFER set (with the trigger to revisit)
- **Native video** — after `orkhon-vl` works + a real video-QA need + R4 compute. Until then: sampled frames + transcript + VLM tool.
- **Internal MoE** — after dense R4/R5 works and dense scaling cost is the *actual* blocker.
- **Learned routers** — start with simple rules; revisit only when logs show rules are repeatedly wrong.
- **Multi-agent orchestration** — after the single-agent loop has evals, approval gates, and real demand.
- **Public code execution** — never on a public Space until real container isolation exists.
- **Don't let capability work block the scale spine** — sharded data, the eval bridge, tokenizer reconciliation, and the R1/R2/R3 proof-runs still come first.

---

## 7. Accounts & infrastructure timeline

Set up in this order (free first). GPU pricing changes quickly; re-price before every paid launch.

| Order | Account | Free? | For | Note |
|---|---|---|---|---|
| 1 | **GitHub** | free | code, CI, releases | public source of truth for releases and checks |
| 2 | **Hugging Face** | free + paid | publish models/datasets, **Spaces** demos, Jobs, Inference | the demo surface; ZeroGPU = demos not training |
| 3 | **Weights & Biases** | free personal | experiment tracking before any paid run | wire `train/monitor.py` |
| 4 | **Kaggle / Colab** | free, limited | opportunistic free GPU smoke tests | not guaranteed |
| 5 | **RunPod** | paid | first cloud GPU (R1/R2 burst) | **H100 ~$2.89–3.29/hr**, A100 80GB ~$1.39–1.49/hr |
| 6 | **Modal** | paid | serverless jobs / evals / data-prep | H100 ~$0.0011/sec |
| 7 | **Lambda / Together / Replicate / Vast** | paid | bigger clusters · hosted inference · demos · cheap-spot | Lambda H100 ~$3.29–4.29/GPU/hr |
| — | **Cloudflare R2 / S3 / Backblaze** | paid (cheap) | shards + DCP checkpoints at R4 | object storage |

---

## 8. New files, by package (single ownership)

```
.github/workflows/ci.yml
src/orkhon/publish.py                      # orkhon publish -> HF Hub
src/orkhon/train/monitor.py                # MetricsSink (JSONL + W&B/TB)
src/orkhon/serve/tool_protocol.py          # tool-call parse/format
src/orkhon/serve/tool_loop.py              # the model<->tool runtime loop
src/orkhon/serve/tools/{base,registry,calculator,read_file,web_search,retrieve,vision}.py
src/orkhon/serve/agent/{loop,policy}.py    # bounded plan->act->observe + approval
src/orkhon/rag/{loaders,chunk,embed,store,retrieve,citations,pipeline}.py
src/orkhon/vision/{describe,encoder,projector,processor,dataset,eval}.py
src/orkhon/eval/{tool_eval,rag_eval,agent_eval}.py   # + chat_eval \boxed{} extractor
src/orkhon/data/tool_synth.py              # SFT tool-call trace synthesis (C6)
src/orkhon/train/vl_{align,sft}.py         # native VLM training (C8)
configs/{tools,rag,agents}/*.yaml · configs/model/orkhon_vl_350m.yaml · spaces/chat/
```
Plus targeted edits to: `tokenizer/special_tokens.py` · `render.py` · `chat_template.jinja` ·
`serve/schemas.py` · `serve/api.py` · `serve/chat_cli.py` · `model/generation.py` · `model/transformer.py` ·
`train/{engine,sft}.py` · `data/dataset.py` · `cli.py` · `registry.py` · `pyproject.toml`.

---

*Perception gets a model. Knowledge and actions get tools. Repeated tool use gets an agent loop. Routing is
infrastructure. MoE is a late scaling trick.*
