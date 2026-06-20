# Orkhon

[English](README.md) | [Türkçe](README.tr.md)

![Orkhon hero: an inscription stone connected to a modern neural network](docs/assets/orkhon-hero.png)

**Orkhon** is a complete, auditable, **from-scratch LLM stack** — it takes you from raw text all the way to a chat-capable language model you can train and talk to on your own machine. The whole pipeline (tokenizer → pretrain → SFT → DPO → eval → serve → export) is implemented by hand in PyTorch for clarity, and proven by a **Mac/MPS smoke run** that trains a tiny model end-to-end in about a minute.

The name comes from the [**Orkhon inscriptions**](https://en.wikipedia.org/wiki/Orkhon_inscriptions) (Göktürk, 8th century CE) — the **oldest known Turkic writing**, commissioned by Bilge Kağan. The project’s thesis is intentionally literal: from the first written words of a language to a system that writes language.

> Not a frontier framework. Orkhon is a compact, inspectable reference implementation with production habits: typed configs, deterministic seeds, resumable training, eval gates, an adversarially-verified model core, tests, and a chat CLI + OpenAI-compatible API.

## What's inside

A hand-written **decoder-only Transformer** — GQA, RoPE, RMSNorm, SwiGLU, KV-cache — plus everything around it:

| Stage | What it does |
|-------|--------------|
| **Tokenizer** | Byte-level BPE with stable special-token ids and a chat template |
| **Data** | Synthetic smoke corpora, normalize → tokenize → pack (`.bin` shards), SFT/DPO formatters |
| **Pretrain** | AdamW, cosine+warmup LR, grad-accum, grad-clip, AMP, checkpoint/resume |
| **Post-training** | SFT, DPO, GRPO/RLVR smoke path, distillation |
| **Eval** | Perplexity, multiple-choice benchmarks, generative pass@k |
| **Tools/RAG/Agent** | Calculator/read-file/retrieve tools, local RAG index, bounded agent loop |
| **Serve** | `orkhon chat`, `orkhon agent`, `orkhon serve` (OpenAI-compatible chat + `/v1/agent/run`) |
| **Export** | HuggingFace-style `safetensors` folder with reload-parity check |

## Quickstart

```bash
uv sync --extra dev            # create the env (PyTorch, tokenizers, …)
bash scripts/smoke_all.sh      # full pipeline on a tiny model, ~1 min on Apple Silicon
```

`smoke_all.sh` runs the entire stack end-to-end: synth data → train tokenizer → prepare shards →
pretrain (600 steps) → SFT (200) → DPO → eval → one chat turn → HF export. After it finishes you can chat:

```bash
uv run orkhon chat --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke
```

### Real smoke result (Apple M5 Pro, MPS, float32)

A ~4M-parameter model trained from scratch in about a minute learns the chat format and answers cleanly:

```
you> What is 2 plus 2?
bot> Answer: 4.
you> What is 9 plus 1?
bot> Answer: 10.
```

Pretrain best val loss ≈ 1.17 · SFT loss ≈ 0.29 · perplexity ≈ 6.8 · ~48k tokens/sec on MPS.
(The smoke corpus is small and templated, so the model learns the *format* and stopping reliably; exact
arithmetic beyond the training range is not expected from a 4M toy — scale the config up for real capability.)

## Train a real model on real text (TinyStories)

A ~22M model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) writes fluent,
coherent English — ~43 min on Apple Silicon (MPS), no cloud needed:

```bash
orkhon data download --split train --out data/tinystories/train.txt --max-stories 250000
orkhon tokenizer train --config configs/tokenizer/tinystories.yaml      # 8K-vocab BPE
orkhon data prepare    --config configs/data/tinystories.yaml           # -> 47.4M train tokens
orkhon train pretrain  --config configs/train/pretrain_tinystories.yaml # 4000 steps, ~19.3k tok/s
orkhon generate --checkpoint runs/tinystories --tokenizer artifacts/tokenizer/tinystories \
  -p "One day, a little girl named Mia found a magic paintbrush."
```

Result (val loss **1.55**, held-out perplexity **5.3**):

> One day, a little girl named Mia found a magic paintbrush. She was very excited... Mia's friend, Tom, came
> over to play. He saw the paint and said, *"Wow, Mia! Your paint is so pretty!"* Mia smiled and said, *"Thank
> you, Tom! I love to paint too!"* They both used the magic paint to make their art fun and pretty. They had a
> great day painting together.

## Beyond TinyStories: real data, open bases, and scale

The same stack scales four ways (all wired + tested; the laptop-bound ones run as-is):

- **Real, diverse web text.** `orkhon data download --dataset fineweb-edu --out data/fineweb/train.txt --max-docs N`
  streams [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (or any HF text dataset) into the
  corpus format. A **51M** model (`tiny_50m`) trains on it on MPS — beyond TinyStories' closed vocabulary.
- **Instruction tuning.** `orkhon data instruct --corpus ...` synthesizes SFT + DPO instruction data; the story base
  becomes an instruction-follower (`orkhon train sft/dpo` → `orkhon chat "Write a short story about ..."`).
- **Load an open base.** `orkhon import-hf --repo HuggingFaceTB/SmolLM2-135M --out runs/base` loads any
  **Llama-architecture** model (SmolLM2, Llama-3.2, TinyLlama, …) into Orkhon's hand-written Transformer with
  **exact logit parity** vs `transformers` — then fine-tune/serve it with Orkhon's machinery.
- **Multi-GPU + cloud.** DDP/FSDP2 is wired (single-process stays byte-identical). `scripts/train_cloud.sh` +
  [`docs/scaling.md`](docs/scaling.md) give turnkey commands and cost/time for 125M–1B on A100/H100
  (`torchrun --nproc_per_node=$N -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml`).

## CLI

```bash
orkhon data download   --dataset fineweb-edu --out data/fineweb/train.txt --max-docs 150000  # fetch a corpus
orkhon data download   --dataset wikipedia-tr --out data/turkish/train.txt --max-docs 150000  # Turkish
orkhon tokenizer train --config configs/tokenizer/smoke.yaml
orkhon data prepare    --config configs/data/smoke.yaml [--sharded]
orkhon train pretrain  --config configs/train/pretrain_smoke.yaml [--set train.max_steps=2000]
orkhon train sft       --config configs/train/sft_smoke.yaml
orkhon train dpo       --config configs/train/dpo_smoke.yaml
orkhon train grpo      --config configs/train/grpo_copy_digit_stable.yaml   # RL with verifiable rewards
orkhon rag ingest      README.md docs --out data/rag/repo                    # build a RAG index
orkhon rag search      "What is C3?" --index data/rag/repo
orkhon bench           --checkpoint MODEL --tokenizer TOK --task builtin --json-out report.json
orkhon bench           --checkpoint MODEL --tokenizer TOK --task gsm8k --samples 4 --pass-at 4 --temperature 0.8
orkhon generate        --checkpoint MODEL --tokenizer TOK -p "Once upon a time"
orkhon chat            --checkpoint MODEL --tokenizer TOK [--tools calculator] [--rag-index IDX]
orkhon agent           --checkpoint MODEL --tokenizer TOK --tools calculator --rag-index IDX --max-steps 5
orkhon serve           --checkpoint MODEL --tokenizer TOK --tools calculator --rag-index IDX --port 8000
orkhon register        --checkpoint runs/tinystories --tokenizer TOK --kind base --lineage "..."
orkhon export hf       --checkpoint MODEL --out exports/orkhon --tokenizer TOK
orkhon import-hf       --repo HuggingFaceTB/SmolLM2-135M --out runs/smollm2  # load an open base
```

`generate` continues raw text (base model); `chat` uses the chat template (SFT/instruct checkpoints).

Any config value can be overridden on the command line with `--set dotted.key=value`.

## Model zoo

![Orkhon model lineage: progressively larger inscription stones connected by training signals](docs/assets/model-lineage.png)

Every trained model is archived into a named, dated, self-contained folder under `models/`. The codenames are
not random release labels: they form a Turkic-history lineage that mirrors the technical ladder.

- **Clean current line:** `bumin-mini`, `tonyuk`, `tegin`, `istem`, `kashgar`, `bunghu`, `tangri`.
- **Unified rule:** every normal public model is expected to handle English, Turkish, and Kokturk/Old Turkic transliteration; there is no separate `-gokturk` product branch.
- **Next scale rung:** `qaghan`, then `otuken`, `balasagun`, `kutadgu`.
- **Frontier reserve:** `oguz`, `manas`, `tarkan`, `ergenekon`, `atilla`, `timur`, `korkut`.

The full naming story is in [**`docs/lineage.md`**](docs/lineage.md).

```bash
orkhon register --checkpoint runs/tinystories --tokenizer artifacts/tokenizer/tinystories \
  --kind base --mode complete --lineage "22M TinyStories base" --eval-prepared data/prepared/tinystories
orkhon registry        # rebuild + print the index
```

Each `models/<name>-<YYYYMMDD>/` holds the **weights** (`checkpoint/`, runnable via `run.sh`), the `tokenizer/`,
generated **outputs** (`samples.txt`), `eval.json`, a **code snapshot** of the exact src + configs that produced
it (`code_snapshot.tgz`), and a `model_card.md` + `manifest.json`. `next_name` hands out the next unused codename
so future models slot in automatically.

| Name | Namesake | What it is | Metric / status |
|------|----------|------------|-----------------|
| `bumin-mini` | Bumin Qaghan, founder of the first Göktürk Khaganate | 4M compact unified assistant | backport target |
| `tonyuk` | Tonyukuk, Orkhon-era strategist and inscription author | 22M unified assistant from the story base | backport target |
| `tegin` | Kül Tigin, Göktürk prince commemorated by an Orkhon inscription | 22M unified assistant from the old story-instruct model | backport target |
| `istem` | Istemi Qaghan, westward-expanding co-founder | 51M unified assistant from FineWeb-Edu base | backport target |
| `kashgar` | Mahmud al-Kashgari, author of the first major Turkic dictionary | imported/open-base slot | weights pending |
| `bunghu` | *Bengü Taş*, eternal inscription stone, ASCII product spelling | 57M unified EN/TR/Kokturk assistant from bilingual branch | backport target |
| `tangri` | Tengri/Tangri, sky-scale rung | 100M unified EN/TR/Kokturk assistant from mixed base | training/eval target |

## Model sizes

| Config | Params | Layers · d_model · heads (kv) | Context | Target |
|--------|--------|-------------------------------|---------|--------|
| `smoke_6m`  | ~4–6M | 4 · 256 · 4 (2)  | 256  | Mac/CPU/MPS smoke run |
| `tiny_24m`  | ~22M  | 6 · 512 · 8 (4)  | 512  | **Real text on MPS** (TinyStories) |
| `tiny_50m`  | ~51M  | 8 · 640 · 10 (5) | 512  | Real web text on MPS (FineWeb-Edu) |
| `small_125m`| ~125M | 12 · 768 · 12 (4) | 1024 | Single GPU (4090/A100) |
| `base_350m` | ~350M | 24 · 1024 · 16 (4) | 2048 | Serious single-GPU run |
| `orkhon_1b` | ~1B   | 24 · 2048 · 16 (4) | 2048 | Multi-GPU / FSDP2 |

Same code path for all sizes; smoke defaults to float32 on MPS/CPU, scale configs use bf16 on CUDA.

## Layout

```
src/orkhon/
  config/      typed configs + YAML loader with --set overrides
  tokenizer/   byte-level BPE, stable specials, chat template, fertility gate
  data/        download, normalize, tokenize, shard, pack, synth, Old Turkic tools
  model/       RoPE · RMSNorm · GQA attention · SwiGLU · KV-cache · generation
  train/       pretrain · SFT · DPO · GRPO · distillation · checkpoint/resume
  eval/        perplexity · loglikelihood benchmarks · generative pass@k · reports
  rag/         ingest · chunk · embed · store · retrieve
  serve/       chat CLI · tool loop · bounded agent · OpenAI-compatible API
  export/      HuggingFace safetensors export + reload parity
configs/   model/ + train/ + tokenizer/ + data/ presets
models/    named zoo archives (weights, tokenizer, samples, cards, manifests)
reports/   benchmark and tokenizer-fertility JSON reports
tests/     260 tests (model core, data, eval, tools, RAG, agent, export, resume)
docs/      roadmaps, lineage, eval, Turkic-language plan, scaling guide
```

## Testing & verification

```bash
UV_CACHE_DIR=/tmp/uv-cache-bilge uv run pytest -q
make smoke                      # full end-to-end pipeline
```

The correctness-critical model core was **adversarially verified** by independent reviewers against the
classic LLM traps (RoPE cached-decode offset, GQA repeat, SFT label masking, DPO objective, grad-accum,
exact resume, perplexity masking). That pass found and fixed a real RoPE position bug in cached decoding —
now guarded by `tests/test_kv_cache_rope_regression.py`.

## Decision guide

The design is grounded in [`docs/build-your-own-llm-guide.md`](docs/build-your-own-llm-guide.md) — a 2026
decision guide covering every path from RAG to frontier pretraining — and a concrete
[`docs/implementation-plan.md`](docs/implementation-plan.md). Two roadmaps map the future:
[`docs/roadmap.md`](docs/roadmap.md) is the **scale axis** (the ladder R0→R6 from 51M to 7B, code gaps, GPU
economics); [`docs/capability-roadmap.md`](docs/capability-roadmap.md) is the **capability axis** (C0→C8: tools,
RAG, web search, agents, and native vision); and
[`docs/turkic-languages.md`](docs/turkic-languages.md) plans **Turkish + Göktürk/Old Turkic** — the model named
for the Orkhon inscriptions learning to read them (the `bengü` branch). Additional research directions are ranked
in [`docs/research-additions.md`](docs/research-additions.md).

## Status

- ✅ End-to-end pipeline runs and trains coherent models on Apple Silicon (MPS) and CPU.
- ✅ **260 tests passing**; model core adversarially verified; tool/RAG/agent/HTTP layers security-reviewed.
- ✅ Full training ladder: pretrain → SFT → DPO → **GRPO/RLVR** (proven to learn) → distillation.
- ✅ Agent substrate: tools (calculator/read_file) + **RAG** (ingest/retrieve) + bounded agent loop (CLI + HTTP `/v1/agent/run`).
- ✅ Dual eval: loglikelihood multiple-choice (HellaSwag/ARC) **and generative pass@k** (GSM8K/MBPP sandbox).
- ✅ Turkish + Göktürk/Old Turkic scaffolding: Turkish data path + Old Turkic transliteration and demo fine-tune.
- 🚧 Göktürk is a sourced transliteration/translation/RAG target, not a fluent free-form language-generation claim.
- 🚧 Scale configs (125M–1B) are provided and validated; running them needs a GPU/budget.
- 🚧 Out of scope for v1: MoE/MLA, native multimodal (vision), multi-agent orchestration.
- ⚠️ Note: current checkpoints predate the `<|tool|>` token, so tool use is loop/prompt-based until a tool-trained checkpoint (C6).

## License

Apache-2.0. The decision guide is research synthesis, not legal advice — re-verify model IDs, licenses, and
costs before production use.
