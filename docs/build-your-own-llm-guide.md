# Build Your Own LLM — Decision Guide (2026)

Research synthesis for the **Orkhon** project. Covers the main commercially realistic paths to a custom language model: from prompt engineering and RAG through fine-tuning, continued pretraining, distillation, alignment, and training from scratch.

> **Last updated:** June 2026 · **Verify before relying on:** model IDs, licenses, cloud pricing, and legal obligations change quickly. Cost figures below are **compute-only estimates** unless noted; add engineering, data labeling, and eval overhead.

---

## Table of contents

0. [Start here](#start-here)

1. [What "build your own LLM" means](#1-what-build-your-own-llm-means)
2. [The modern training stack](#2-the-modern-training-stack)
3. [Seven paths ranked by ambition](#3-seven-paths-ranked-by-ambition)
4. [Fine-tuning methods](#4-fine-tuning-methods)
5. [Alignment and post-training](#5-alignment-and-post-training)
6. [Distillation and compression](#6-distillation-and-compression)
7. [Synthetic data and reasoning paradigms](#7-synthetic-data-and-reasoning-paradigms)
8. [Open-weight base models](#8-open-weight-base-models)
9. [Data pipelines](#9-data-pipelines)
10. [Architecture (from scratch)](#10-architecture-from-scratch)
11. [Tokenizers](#11-tokenizers)
12. [Continued pretraining](#12-continued-pretraining)
13. [Frameworks and tooling](#13-frameworks-and-tooling)
14. [Hardware and cloud](#14-hardware-and-cloud)
15. [Multimodal extensions](#15-multimodal-extensions)
16. [Evaluation](#16-evaluation)
17. [Deployment](#17-deployment)
18. [Legal and licensing](#18-legal-and-licensing)
19. [Decision trees and recipes](#19-decision-trees-and-recipes)
20. [Essential links](#20-essential-links)
21. [Glossary](#glossary)

---

## Start here

### Choose your path (30 seconds)

| If you need… | Path | Recipe |
|--------------|------|--------|
| Q&A on your documents | 0 — RAG | [Recipe A](#recipe-a-solo-dev-12-weeks-500) (+ RAG layer) |
| Custom tone or format | 1 — LoRA | [Recipe A](#recipe-a-solo-dev-12-weeks-500) |
| Full custom assistant | 2 — SFT + DPO | [Recipe B](#recipe-b-startup-mvp-1-month-5k20k) |
| Domain knowledge in weights | 3 — CPT | [Recipe C](#recipe-c-domain-expert-36-months-50k300k) |
| Math/code reasoning | Distill + GRPO | [Recipe D](#recipe-d-reasoning-on-budget-2002k) |
| Learn the stack | 4 — small scratch | [Recipe E](#recipe-e-learn-the-stack-1005k) |

### Minimum eval suite (before you fine-tune)

Do not start SFT or alignment until you have:

1. **50–200 golden examples** from real user tasks (not synthetic only)
2. **One automated metric** — IFEval for instruction-following, RAGAS for RAG, or domain accuracy
3. **One human spot-check** — 20 outputs graded pass/fail
4. **A baseline** — same prompts on the base model + RAG-only

Re-run after every training stage. Escalate path level only when the current path fails this suite.

### Master decision tree

See [§19 Decision trees and recipes](#19-decision-trees-and-recipes) for the full tree and costed recipes.

---

## 1. What "build your own LLM" means

In 2026, the phrase covers seven distinct ambition levels:

| Level | What you own | Typical audience |
|-------|--------------|------------------|
| **0** | Prompts + RAG pipeline | Most products (~90%) |
| **1** | LoRA adapters on open base | Domain chatbots, format control |
| **2** | Full fine-tuned instruct weights | Custom assistant |
| **3** | Continued pretraining (knowledge in weights) | Legal, medical, scientific corpora |
| **4** | Small model from scratch (100M–1.5B) | Learning, edge deployment |
| **5** | Medium model from scratch (7B) | Funded AI startups |
| **6** | Large model from scratch (70B+) | Labs, sovereign AI |

**Default recommendation:** Start at Level 0–2. Escalate only when RAG + fine-tuning fail on your eval suite, or you have hard IP/sovereignty requirements.

**2026 economic reality:** Inference costs fall sharply year over year ([Epoch AI](https://epoch.ai)). Training efficiency continues to improve rapidly. Fine-tuning + RAG beats training from scratch for most product use cases.

---

## 2. The modern training stack

```
Data curation → (Optional) Pretrain/CPT → SFT → Alignment (DPO/GRPO) → Eval → Quantize → Deploy
```

Post-training (SFT + alignment) often matters more than pretraining for user-facing quality. Reasoning models add: synthetic CoT data → GRPO with verifiable rewards → rejection sampling → optional distillation to smaller models.

---

## 3. Seven paths ranked by ambition

### Path 0: Prompt + RAG

| | |
|---|---|
| **Cost** | $20K–$150K engineering |
| **Time** | 2–8 weeks |
| **GPU** | None (API or local inference) |
| **Quality** | Competitive on narrow domain Q&A when retrieval is good |

**Stack (examples):** any RAG framework (LangChain, LlamaIndex, custom) + vector store (Qdrant, pgvector) + API or vLLM + RAGAS eval.

---

### Path 1: LoRA / QLoRA adapters

| | |
|---|---|
| **Cost** | $2K–$55K |
| **Time** | 2–4 weeks |
| **GPU** | 1× 12–24GB (7B QLoRA from ~5GB) |
| **Quality** | Often within a few points of full SFT on specialized tasks |

Trains 0.01–0.5% of parameters. Adapters can merge at inference with zero latency overhead.

---

### Path 2: Full supervised fine-tuning (SFT)

| | |
|---|---|
| **Cost** | $25K–$600K |
| **Time** | 2–12 weeks |
| **GPU** | 4–8× H100 for 70B |
| **Quality** | Near open SOTA for model size |

Often followed by DPO, SimPO, or GRPO. Requires 50K–500K high-quality instruction examples.

---

### Path 3: Continued pretraining (CPT)

| | |
|---|---|
| **Cost** | $30K–$2M |
| **Time** | 1–6 months |
| **When** | Domain vocabulary/knowledge gap RAG cannot fix |

Further pretrain an open **base** model on domain text before SFT. Not the same as true pretraining.

---

### Path 4: Train small model from scratch (100M–1.5B)

| | |
|---|---|
| **Cost** | $500–$50K |
| **Time** | 1–3 months |
| **Tools** | nanoGPT, nanochat, lit-gpt |
| **Example** | TinyLlama 1.1B: 3T tokens, ~$50–70K cloud |

Educational, edge deployment, extreme cost sensitivity.

---

### Path 5: Train medium model from scratch (7B)

| | |
|---|---|
| **Cost** | $100K–$10M |
| **Time** | 6–12 months |
| **Tokens** | 500B–15T (2026 models massively overtrain vs Chinchilla) |
| **Reference** | OLMo 3 7B: ~234K H100-hrs (~$470K compute; [Ai2 paper](https://arxiv.org/abs/2512.13961)) |

Requires 8–30 ML engineers + data infra.

---

### Path 6: Train large model from scratch (70B+)

| | |
|---|---|
| **Cost** | $10M–$200M+ |
| **Time** | 18–36 months |
| **Team** | 50–500+ people |

Frontier lab territory only.

---

### Cost comparison matrix

*Compute + engineering estimates as of June 2026. Re-check provider pricing before budgeting.*

| Path | Timeline | Cost | Own weights? |
|------|----------|------|--------------|
| RAG | 2–8 wks | $20K–$150K | No |
| LoRA | 2–4 wks | $2K–$55K | Adapters only |
| Full SFT | 2–12 wks | $25K–$600K | Derivative |
| CPT | 1–6 mo | $30K–$2M | Derivative |
| 1B scratch | 1–3 mo | $10K–$150K | Full |
| 7B scratch | 6–12 mo | $100K–$10M | Full |
| 70B+ scratch | 18–36 mo | $10M–$200M+ | Full |

---

## 4. Fine-tuning methods

### Master comparison (7B reference)

| Method | Trainable % | 7B VRAM | vs FFT quality | Best for |
|--------|-------------|---------|----------------|----------|
| Full FT | 100% | 67–120 GB | 100% | Max quality, multi-GPU |
| LoRA | 0.01–0.5% | 15–28 GB | 90–95% | Production default |
| QLoRA | 0.01–0.5% | 5–9 GB | 80–90% | Consumer GPU |
| DoRA | ~0.5% | 15–30 GB | 92–97% | LoRA underperforms |
| GaLore | 100% weights* | 24–48 GB | 95–99% | Near-full FT, one GPU |
| PiSSA | Same as LoRA | 15–28 GB | 91–96% | Better LoRA init |
| rsLoRA | Same as LoRA | 15–28 GB | 90–96% | Rank 64+ |
| OFT/BOFT | 0.1–1% | 16–30 GB | 90–96% | Preserve base knowledge |

*GaLore trains all weights but saves optimizer memory via gradient projection.

### Recommended defaults (2026)

*Axolotl / LLaMA-Factory-style config. Pin versions in production (`trl`, `peft`, `unsloth`).*

```yaml
adapter: qlora
load_in_4bit: true
lora_r: 16
lora_alpha: 32
use_rslora: true
init_lora_weights: pissa
use_dora: false  # set true if LoRA underperforms
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 2e-4
bf16: true
```

### Escalation ladder

1. QLoRA or LoRA (Unsloth / PEFT)
2. DoRA + PiSSA + rsLoRA
3. GaLore (7B single-node)
4. Full fine-tune (FSDP ZeRO-3)

### Libraries

| Library | Best for |
|---------|----------|
| **PEFT** | Universal integration |
| **Unsloth** | 2–5× speed, ~60% less VRAM |
| **Axolotl** | YAML production pipelines |
| **LLaMA-Factory** | Web UI, 100+ models |
| **TRL** | SFT + DPO + GRPO |

---

## 5. Alignment and post-training

### Two paradigms (2026)

1. **Preference alignment** (chat, safety) → offline DPO family or classic RLHF
2. **Reasoning / verifiable RL** (math, code) → online GRPO, DAPO, REINFORCE++

### Offline preference methods

| Method | Ref model? | Best for |
|--------|------------|----------|
| **DPO** | Yes | Default; 10K–100K pairs |
| **SimPO** | No | Often beats DPO; reference-free |
| **ORPO** | No | Single GPU; SFT + align one pass |
| **IPO** | Yes | Noisy preferences |
| **KTO** | Yes | Thumbs up/down only |
| **CPO / cDPO** | Varies | Joint SFT+align; noisy labels |

### Online RL methods

| Method | Best for | Notes |
|--------|----------|-------|
| **GRPO** | Math, code, RLVR | No critic; DeepSeek-R1 pattern |
| **DAPO / DrGRPO** | SOTA reasoning scale | Fixes length bias |
| **PPO / RLHF** | Subjective multi-objective | 3–4 models in memory |
| **REINFORCE++** | Stable reasoning RL | No critic |
| **RLAIF** | Safety without humans | Judge bias risk |

### Framework selection

| Situation | Pick |
|-----------|------|
| Solo, HF ecosystem | **TRL** + Unsloth |
| Multi-node RLHF, 70B+ | **OpenRLHF** or **verl** |
| Reasoning (AIME-level) | **verl** (DAPO) |
| Teaching / first RLHF | **TRL** smol-course |

### Default pipelines

**Budget chat ($500, 1–2 days):**
```
SFT (1K–10K) → SimPO or ORPO (10K–50K pairs) → optional best-of-n at deploy
```

**Production chat ($5K–$20K):**
```
SFT → Train RM → DPO warm-start → PPO → DPO polish
```

**Reasoning model ($10K–$100K):**
```
SFT (CoT) → GRPO w/ verifiable reward → DrGRPO/DAPO refinement
```

---

## 6. Distillation and compression

### Modern three-stage pattern

```
Mid-training: Off-policy SFT on teacher traces
Post-training: On-policy distillation (GKD)
Deployment: PTQ (AWQ/GPTQ/FP8)
```

### Techniques

| Technique | When |
|-----------|------|
| **Rejection sampling** | Filter R1/o1 traces with verifiers |
| **GKD** | Student learns from own mistakes vs teacher (λ→1 on-policy) |
| **Model merging** (TIES, DARE, SLERP) | Combine specialized fine-tunes |
| **Quantization** | AWQ/GPTQ 4-bit; FP8 on H100 |
| **Test-time compute** | Best-of-N, budget forcing — no training |

**Key insight:** Distillation transfers capability; GRPO optimizes within the base model ceiling.

### Open reasoning datasets

- `open-r1/openr1-220k-math`
- `open-thoughts/OpenThoughts-114k`
- `GAIR/LIMO` (817 samples)
- `simplescaling/s1K` (1,000 samples)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

---

## 7. Synthetic data and reasoning paradigms

### Instruction synthesis

| Method | Role |
|--------|------|
| **Self-Instruct** | Seed → generate → filter → train |
| **Evol-Instruct** | Mutate difficulty/breadth |
| **Magpie** | Auto-generate user queries from chat template |

**2026 trend:** Quality >> quantity for reasoning. LIMO/s1K beat massive noisy sets on benchmarks.

### DeepSeek R1 four-stage pipeline

```
1. Cold-start SFT (10K CoT traces)
2. RLVR with GRPO (math/code verifiable rewards)
3. Rejection sampling → 600K+ traces → SFT
4. RLVR for helpfulness
```

### Indie reasoning recipe ($200–$2K)

1. Start from DeepSeek-R1-Distill-Qwen-7B
2. Generate traces via R1 API; filter with Math-Verify or unit tests
3. SFT with Unsloth
4. Optional GRPO if domain has verifiable rewards
5. Best-of-N at inference

---

## 8. Open-weight base models

**For CPT or clean SFT, prefer `-Base` / `-pt` checkpoints.** Instruct/chat models are fine for lightweight adapters (tone, format) when you accept their baked-in alignment.

### License tiers

**Tier 1 — minimal friction (Apache 2.0 / MIT):**
- Qwen 2.5 / 3 ([Apache 2.0](https://huggingface.co/Qwen/Qwen3-8B-Base))
- Mistral Small 3.x
- OLMo 2 / 3
- Gemma 4 ([Apache 2.0](https://ai.google.dev/gemma/apache_2) — Gemma 1–3 used custom ToS)
- DeepSeek-R1 (MIT)

**Tier 2 — commercial with conditions:**
- Llama 3.x / 4 ([Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE): attribution, AUP, 700M MAU cap)
- Gemma 1–3 (custom ToS)
- DeepSeek-V3 (Model Agreement)

### Recommendations by scenario

| Scenario | Base model |
|----------|------------|
| First fine-tune | `Qwen/Qwen3-8B-Base` |
| Production SaaS (legal simplicity) | Qwen3-14B or Mistral-Small-3.1-24B-Base |
| Multimodal | `google/gemma-4-12b` or `google/gemma-3-12b-pt` |
| Research / full transparency | `allenai/Olmo-3-1125-32B` |
| Reasoning on budget | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| On-device | `google/gemma-4-E2B` or `Qwen/Qwen3-1.7B-Base` |

### VRAM guide

| Size | Full FT | LoRA 16-bit | QLoRA 4-bit |
|------|---------|-------------|-------------|
| 7B | 67–120 GB | 15–28 GB | 5–9 GB |
| 14B | 125 GB | 28 GB | 9–17 GB |
| 70B | 672 GB | 146 GB | 46–88 GB |

---

## 9. Data pipelines

### End-to-end pretraining workflow

```
Collect (Common Crawl, GitHub, Wikipedia, books)
  → Extract (trafilatura on WARC, not WET)
  → Language ID + heuristic filters
  → Model-based quality classifiers
  → Dedup (per-snapshot MinHash — NOT global)
  → PII scrub + toxicity filter
  → Benchmark decontamination (10–13 gram)
  → Tokenize + source mixing → train
```

### Major open corpora

| Dataset | Tokens | Notes |
|---------|--------|-------|
| FineWeb | 15T | datatrove; ablation methodology |
| FineWeb-Edu | 1.3T | Educational classifier subset |
| DCLM-Baseline | 2.6T used | Model-based filtering key |
| Dolma 3 | Multi-T | Fully documented (Ai2) |
| Nemotron-CC | 6.3T | Synthetic recovery of filtered content |
| SlimPajama | 627B | Heavily deduplicated |

### Tools

| Tool | Use case |
|------|----------|
| **datatrove** | CPU-scale pretraining pipelines |
| **NeMo Curator** | GPU trillion-token curation |
| **distilabel** | SFT/DPO/preference generation |

### Instruction formats

- **Alpaca:** `instruction` / `input` / `output`
- **ShareGPT:** `conversations` with `from`/`value`
- **ChatML-style:** `messages` with `role`/`content` — store in this shape, then render with each model's official template via `tokenizer.apply_chat_template()`

Convert ShareGPT → normalized `messages` before applying chat templates.

---

## 10. Architecture (from scratch)

### 2026 default stack

| Component | Choice |
|-----------|--------|
| Architecture | Decoder-only Transformer |
| Attention | **GQA** (grouped query attention) |
| Position | **RoPE** + **YaRN** at inference |
| Norm / FFN | **RMSNorm** + **SwiGLU** |
| Vocab | 64K–128K byte-level BPE |
| Context | Pretrain 4K–8K → extend to 32K–128K |

### When to pick alternatives

| Need | Choice |
|------|--------|
| Inference cost at scale | GQA or MLA (DeepSeek) |
| Capability / lower active FLOPs | MoE (Mixtral / DeepSeek style) |
| Very long context | SWA + global layers, or Mamba hybrids |
| Learning / prototyping | Dense MHA or GQA, nanoGPT |

### Reference training compute

| Model | GPU-hours | Est. cloud cost |
|-------|-----------|-----------------|
| nanoGPT 124M | 300–500 (1×4090) | ~$200–500 |
| TinyLlama 1.1B | ~34,500 A100-hrs | ~$50–70K |
| OLMo 3 7B | ~234K H100-hrs | ~$470K ([source](https://arxiv.org/abs/2512.13961)) |
| Llama 3 8B | ~1.3M H100-hrs | ~$2.6M |

**Token budget:** Plan 100–2,000+ tokens/param (not Chinchilla's ~20).

---

## 11. Tokenizers

First irreversible design decision for from-scratch builds.

### When custom tokenizer is needed

- Low-resource languages
- Domain jargon (medical, legal, code-heavy)
- Multilingual with poor fertility on English-centric tokenizers

### Algorithm choice

| Case | Recommendation |
|------|----------------|
| English/code LLM | Byte-level BPE (Llama/GPT style) |
| Multilingual | SentencePiece Unigram or byte-level BPE on mixed corpus |
| Fine-tuning existing model | Keep original tokenizer |

### Vocabulary sizing

| Model scale | Vocab size |
|-------------|------------|
| < 1B | 8K–16K |
| 1–8B | 32K–64K |
| 8B+ | 64K–128K |
| Multilingual | 64K–256K |

**Rule of thumb:** Target 3–4 bytes/token for English; tie embeddings (`tie_word_embeddings = True`).

**Tools:** HuggingFace `tokenizers`, SentencePiece, tiktoken (inference).

---

## 12. Continued pretraining

### CPT vs SFT vs RAG

| Method | Changes | Data |
|--------|---------|------|
| **RAG** | Inference access | Indexed documents |
| **CPT** | What model knows | Raw text, 1B–200B+ tokens |
| **SFT** | How model behaves | Instruction pairs |

**Default pipeline for domain assistant:**
```
Base → CPT (optional) → SFT → DPO/RLHF → RAG at inference
```

### Forgetting mitigation

1. Mix 5–20% general replay data (FineWeb, Dolma)
2. LR 5–10× lower than original pretrain
3. Re-warm + re-decay schedule (do not continue from near-zero LR)
4. Model merge with base at α=0.85–0.95 after CPT
5. LoRA-CPT first; full CPT only if insufficient

### Cost envelope (spot H100)

| Run | Tokens | Est. total |
|-----|--------|------------|
| LoRA CPT 7B | 5B | $500–$3K |
| Full CPT 7B | 30B | $10–$30K |
| Full CPT 70B | 100B+ | $200K+ |

---

## 13. Frameworks and tooling

### User-facing frameworks

| Framework | Best for |
|-----------|----------|
| **Unsloth** | 1 GPU, fastest iteration |
| **LLaMA-Factory** | GUI, 100+ models, zero-code |
| **Axolotl** | YAML, GRPO, MoE, QAT |
| **TRL + PEFT** | Custom pipelines, HF native |
| **torchtitan** | Pretrain 8B–405B |
| **OLMo-core** | Fully open reproducibility |
| **verl** | Frontier GRPO/DAPO |
| **OpenRLHF** | Multi-node RLHF + vLLM |

### Architecture layers

```
User-facing:  LLaMA-Factory | Axolotl | Unsloth | torchtitan | NeMo
RL:           TRL | verl | OpenRLHF
Model layer:  Transformers | PEFT | torchtune
Distributed:  FSDP2 | DeepSpeed | Megatron
Orchestration: Ray | Accelerate | vLLM | SGLang
```

### 2026 trends

- GRPO is the default for verifiable reasoning RL; PPO still used for subjective multi-objective RLHF
- FSDP2 preferred for new PyTorch stacks
- [TGI is in maintenance mode](https://github.com/huggingface/text-generation-inference) — prefer vLLM or SGLang for new GPU deployments
- FP8/MXFP8 on Blackwell/H200

---

## 14. Hardware and cloud

### Consumer GPU tiers

| VRAM | Capability |
|------|------------|
| 8 GB | QLoRA 1B–3B; tight 7B |
| 16 GB | QLoRA 7B–8B |
| 24 GB (4090) | QLoRA 14B; LoRA 7B — best price/performance |
| 48 GB | QLoRA 70B |
| Apple 64GB+ unified | MLX; 70B inference |

**Free cloud:** Kaggle ~30 GPU hrs/week; Colab (variable).

### Cloud fine-tuning (7B LoRA, ~15M tokens)

*Compute-only, June 2026. Excludes data prep and engineering.*

| Provider | ~Cost |
|----------|-------|
| RunPod / Modal DIY | $2–4 |
| HF Jobs + Unsloth | $3–6 |
| Together / Fireworks managed | $7–15 |

**70B QLoRA:** $45–90 managed; $67–160 DIY.

### HF Jobs example

```bash
hf jobs uv run <script.py> \
  --flavor a10g-small \
  --timeout 4h \
  --dataset your-org/dataset \
  --output-repo your-org/finetuned-model
```

---

## 15. Multimodal extensions

Same pattern: pick base → curate data → LoRA/QLoRA → eval → deploy.

| Modality | Base | Trainer |
|----------|------|---------|
| Vision-language | Qwen2.5-VL, LLaVA, SmolVLM | TRL SFTTrainer + processor |
| Audio | Whisper large-v3-turbo | Seq2SeqTrainer |
| Code | Qwen2.5-Coder, DeepSeek-Coder | TRL SFTTrainer |
| Embeddings | BGE-M3, e5 | sentence-transformers |

**Highest RAG ROI extension:** domain embedding model before VLM or audio work.

---

## 16. Evaluation

### Four dimensions of "good"

1. **Intrinsic** — val loss, perplexity (pretraining)
2. **Knowledge** — MMLU-Pro, GPQA, BBH
3. **Instruction** — IFEval, MT-Bench, AlpacaEval
4. **Downstream + safety** — custom evals, HarmBench

### Eval tiers

| Interval | Eval |
|----------|------|
| Every 500 steps | Val loss, PPL |
| Every 5K steps | hellaswag, arc_easy |
| Every 25K steps | MMLU-Pro, IFEval, HumanEval |
| Milestones | Full Open LLM Leaderboard v2 |
| Release | Custom eval + HarmBench + CoDeC contamination + human review |

### Quick eval command

```bash
# After SFT — pin lm-eval version in CI
lm_eval --model hf \
  --model_args pretrained=your-org/finetuned-model,dtype=bfloat16 \
  --tasks ifeval,hellaswag \
  --batch_size auto
```

### Tools

- **lm-evaluation-harness** — public benchmarks
- **RAGAS / DeepEval** — RAG quality
- **AlpacaEval 2.0** — fast SFT iteration
- **promptfoo** — safety in CI/CD

### Common failure modes

| Symptom | Likely cause |
|---------|--------------|
| PPL great, benchmarks flat | Undertrained or wrong eval setup |
| Benchmarks jump at one checkpoint | Training data contamination |
| AlpacaEval high, MT-Bench low | Overfit to simple instructions |

---

## 17. Deployment

### Pipeline

```
Merge LoRA (optional) → Quantize (AWQ/FP8) → vLLM/SGLang → OpenAI-compatible API
```

### Engine selection

| Target | Stack |
|--------|-------|
| GPU production | **vLLM** (default) or **SGLang** |
| Many LoRA adapters | vLLM multi-LoRA or **LoRAX** |
| Edge / local | Merge → GGUF → **Ollama** or **llama.cpp** |
| Max NVIDIA perf | TensorRT-LLM / NVIDIA NIM |
| Fastest to prod | HuggingFace Inference Endpoints |

### Quantization

| Method | When |
|--------|------|
| FP8 | H100/L40S production default |
| AWQ | 4-bit GPU, memory constrained |
| GPTQ | Legacy ecosystem |
| GGUF Q4_K_M | Edge / CPU |

### vLLM multi-LoRA

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules sql-lora=./adapters/sql \
  --max-loras 4
```

---

## 18. Legal and licensing

**Not legal advice.** Review with counsel before commercial launch.

### Safe commercial pattern

- Base: **Apache 2.0** ([Qwen3](https://huggingface.co/Qwen/Qwen3-8B-Base), Mistral, OLMo, [Gemma 4](https://ai.google.dev/gemma/apache_2))
- Data: owned or explicitly licensed
- Ship: model card + evals + privacy policy

### Llama-specific

- Commercial OK under 700M MAU ([Llama 4 license](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE))
- Attribution: "Built with Meta Llama" (or "Built with Llama" per model generation)
- Naming prefix required when using Llama materials to improve a **distributed** derivative model
- Cannot use outputs to improve non-Llama LLMs (anti-distillation clause)

### EU AI Act (GPAI providers, since Aug 2025)

If you release a general-purpose foundation model to the EU, align with the [GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai):

- Copyright compliance policy
- Public training data summary (EU template)
- Technical documentation (Annex XI) — partial exemption for open-source release
- Systemic-risk models (frontier scale) face additional obligations; most fine-tuned derivatives do not

### GDPR

- Map personal data in training/fine-tuning pipelines
- DPIA if training on personal data
- Scrub PII before training; prefer prevention over "unlearning"

### Data risk tiers

| Low risk | Medium | High |
|----------|--------|------|
| Wikipedia, CC, government data, synthetic from open models | License-filtered GitHub, Gutenberg | Scraped paywalled content, shadow libraries |

---

## 19. Decision trees and recipes

### Master decision tree

```
Need Q&A on YOUR documents?
  → RAG

Need custom tone/format?
  → LoRA/QLoRA

Need full custom assistant?
  → Full SFT + DPO

Need domain knowledge IN weights?
  → CPT → SFT (try RAG + SFT first)

Need reasoning (math/code)?
  → Distill from R1 → SFT → optional GRPO

Learn / edge?
  → 1B from scratch (nanoGPT/nanochat)

Compete with Mistral/Llama?
  → 7B scratch ($500K+, 12 months)

Frontier foundation?
  → 70B+ ($10M+, 24 months)
```

### Recipe A: Solo dev (1–2 weeks, <$500)

1. Base: `Qwen/Qwen3-8B-Base` + Unsloth QLoRA on 1K–5K ChatML-style examples:

```json
{"messages": [
  {"role": "user", "content": "Summarize this ticket: ..."},
  {"role": "assistant", "content": "..."}
]}
```

2. Eval: 50 golden prompts + `lm_eval --tasks ifeval`
3. Deploy: `vllm serve your-org/adapter-merged` or Ollama
4. Add RAG for factual grounding (Path 0 layer on top)

### Recipe B: Startup MVP (1 month, $5K–$20K)

1. RAG on proprietary docs
2. LoRA for format/behavior
3. DPO with 5K pairs if needed
4. Custom golden-set eval + IFEval

### Recipe C: Domain expert (3–6 months, $50K–$300K)

1. CPT on 50B–500B domain tokens (10% replay)
2. SFT on 50K–500K instructions
3. SimPO or GRPO (if verifiable)
4. vLLM + multi-LoRA for task variants

### Recipe D: Reasoning on budget ($200–$2K)

1. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
2. R1 API traces + rejection sampling
3. Unsloth SFT
4. Optional GRPO with unit tests
5. Best-of-N at inference

### Recipe E: Learn the stack ($100–$5K)

1. Karpathy **nanochat** or **nanoGPT**
2. **lit-gpt** 1B on SlimPajama subset
3. Study **OLMo-core** + Dolma 3 for full transparency

### Recommended progression

```
Path 0 (RAG) → Path 1 (LoRA) → Path 2 (SFT) → Path 3 (CPT)
Only if each step fails your eval suite.
Paths 4–6 for learning, edge, sovereignty, or research labs.
```

---

## 20. Essential links

### Learn

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [OLMo-core](https://github.com/allenai/OLMo-core)
- [Orkhon inscriptions](https://en.wikipedia.org/wiki/Orkhon_inscriptions) — namesake of the Orkhon project (oldest known Turkic writing)

### Fine-tune

- [Unsloth](https://github.com/unslothai/unsloth)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- [TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)

### Pretrain

- [torchtitan](https://github.com/pytorch/torchtitan)
- [datatrove](https://github.com/huggingface/datatrove)
- [Dolma 3](https://huggingface.co/datasets/allenai/dolma)

### RL / alignment

- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [open-r1](https://github.com/huggingface/open-r1)

### Data synthesis

- [distilabel](https://github.com/argilla-io/distilabel)
- [OpenThoughts](https://huggingface.co/open-thoughts)

### Eval

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard)
- [HarmBench](https://www.harmbench.org/)

### Deploy

- [vLLM](https://github.com/vllm-project/vllm)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor)
- [HF Jobs](https://huggingface.co/docs/hub/en/jobs)

### Sources cited in this guide

- [Epoch AI](https://epoch.ai) — compute cost trends
- [OLMo 3 paper](https://arxiv.org/abs/2512.13961) — training compute
- [Gemma 4 Apache 2.0](https://ai.google.dev/gemma/apache_2) — license
- [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)
- [EU GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)
- [TGI maintenance notice](https://github.com/huggingface/text-generation-inference)

### Cost references

- [Epoch AI](https://epoch.ai)
- [Together AI pricing](https://www.together.ai/pricing)
- [HF Jobs pricing](https://huggingface.co/docs/hub/en/jobs-pricing)

---

## Bottom line

For most builders in 2026:

> **Open base (Qwen3 / Llama 4) → RAG + LoRA/SFT → optional DPO/GRPO → quantize → vLLM**

That path delivers strong product capability at a fraction of pretraining cost. Full pretraining is justified only for proprietary data moats, sovereignty requirements, research transparency (OLMo), or sub-2B edge models.

---

## Glossary

| Term | Meaning |
|------|---------|
| **CPT** | Continued pretraining — further training a base model on domain text |
| **DPO** | Direct Preference Optimization — alignment from chosen/rejected pairs |
| **GKD** | Generalized Knowledge Distillation — on-policy distillation from a teacher |
| **GRPO** | Group Relative Policy Optimization — RL without a critic (DeepSeek-R1 pattern) |
| **GQA** | Grouped Query Attention — fewer KV heads than query heads |
| **LoRA / QLoRA** | Low-rank adapters; QLoRA quantizes base weights to 4-bit |
| **MLA** | Multi-head Latent Attention (DeepSeek-style compressed KV) |
| **RLVR** | Reinforcement Learning with Verifiable Rewards (math, code) |
| **RAG** | Retrieval-Augmented Generation — fetch docs at inference |
| **SFT** | Supervised Fine-Tuning on instruction/response pairs |
| **SimPO** | Simple Preference Optimization — reference-free DPO variant |
| **YaRN** | Yet another RoPE extension — stretch context at inference |

---

*Research compiled June 2026. Re-verify model IDs, licenses, and costs before production use.*
