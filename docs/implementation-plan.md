# Orkhon — Implementation Plan (Complete AI, V1)

This plan turns the decision framework in [`build-your-own-llm-guide.md`](build-your-own-llm-guide.md)
into a concrete, verification-gated implementation target for Orkhon V1.

## Goal

Turn Orkhon from a planning doc into a **compact, auditable, from-scratch LLM stack that actually runs**:
raw text → tokenizer → pretrain → SFT → DPO → eval → export → chat — implemented by hand in PyTorch
for educational clarity, proven by a **Mac/MPS smoke run** before any cloud spend.

This is explicitly **not** a frontier framework. No MoE / MLA / multimodal / GRPO / k8s in V1.

## Target hardware

Apple M5 Pro, 48 GB, MPS. Smoke runs on MPS/CPU in minutes. Scale configs target a single cloud GPU
(A100/H100) via the same code, with DDP/FSDP2 as a later milestone.

## Tech stack

- Python ≥3.11, managed by **uv** (lockfile committed).
- **PyTorch** as the only training framework. Model implemented by hand — no `AutoModelForCausalLM`.
- HF **tokenizers** for byte-level BPE; **safetensors** for weights; **jinja2** for the chat template.
- **pydantic** + YAML for config; **typer** + **rich** for the `orkhon` CLI; **FastAPI**/uvicorn for serving.
- **pytest** for the test suite. `transformers`/`lm-eval`/vLLM are optional, post-export only.

## Device / dtype policy

- CPU path must always work (tests). MPS for smoke. CUDA bf16 for scale.
- Smoke default dtype = `float32` (MPS mixed-precision is fragile). CUDA default = `bfloat16`.

## Repository layout (src-layout)

```
pyproject.toml  uv.lock  Makefile  .python-version
configs/        model/{smoke_6m,small_125m,base_350m,orkhon_1b}.yaml + train/data/eval/infer presets
data/smoke/     pretrain.txt  sft.jsonl  dpo.jsonl   (generated, deterministic)
src/orkhon/
  cli.py                      # typer entrypoint: tokenizer|data|train|eval|chat|serve|export
  config/        schema.py load.py      # typed config, YAML load, CLI override, validation
  utils/         seed.py device.py dtype.py logging.py paths.py hashing.py env.py
  tokenizer/     train.py special_tokens.py chat_template.jinja render.py validate.py
  data/          synth.py normalize.py tokenize.py pack.py dataset.py sft_format.py dpo_format.py collate.py
  model/         config.py rope.py rmsnorm.py attention.py mlp.py block.py transformer.py kv_cache.py generation.py init.py
  train/         engine.py optim.py schedule.py checkpoint.py losses.py metrics.py pretrain.py sft.py dpo.py
  eval/          perplexity.py smoke.py chat_eval.py report.py
  serve/         sampling.py chat_cli.py api.py schemas.py
  export/        to_hf.py model_card.py
tests/           rope, gqa, kv-cache parity, loss-shift, sft-mask, dpo-loss, checkpoint-resume, smoke pipeline
scripts/         smoke_all.sh
```

## Smoke config (`smoke_6m`, ~6–10M params)

vocab 4096 · block 256 · 4 layers · d_model 256 · 4 heads · 2 kv-heads · head_dim 64 · SwiGLU 768 ·
RoPE θ=10000 · tied embeddings · no bias · float32. Trains a few hundred steps on MPS in minutes.

## Scale configs

`small_125m` (12L/768d/4kv, ctx 1024) · `base_350m` (24L/1024d, ctx 2048) · `orkhon_1b` (24L/2048d, ctx 2048).
Single-GPU bf16 + activation checkpointing + grad-accum; FSDP2 for 1B.

## Build order (verification-gated)

- **M0 — Contract:** pyproject + env, config schema, special tokens, chat template, utils, CLI skeleton,
  smoke configs, deterministic smoke datasets. Gate: `orkhon --help` works; invalid model shapes rejected; `uv run pytest -q` runs.
- **M1 — Tokenizer + Data:** byte-level BPE train, normalize/tokenize/pack, SFT/DPO formatters.
  Gate: tokenizer round-trips; special-token IDs stable; SFT mask trains assistant-only & keeps EOS; DPO batches carry prompt/chosen/rejected masks.
- **M2 — Model core:** RoPE, RMSNorm, GQA attention, SwiGLU, block, transformer, KV-cache, generation.
  Gate: shape tests; overfit-one-batch; **cached generation == non-cached** for fixed prompt; param count matches config.
- **M3 — Pretrain smoke:** engine (AdamW, cosine+warmup, grad-accum, clip, AMP), checkpoint/resume, val loss.
  Gate: smoke loss drops; exact resume; no NaNs; checkpoint reloads & generates.
- **M4 — SFT smoke:** assistant-only masking, chat eval, `orkhon chat`. Gate: overfits tiny chat set; uses assistant prompt at inference; never trains user tokens.
- **M5 — DPO smoke:** frozen reference model, completion-only logprobs, margin metric. Gate: chosen−rejected margin rises; reference stays frozen.
- **M6 — Serve:** `orkhon chat` + `orkhon serve` (`/health`, `/v1/models`, `/v1/chat/completions`, SSE streaming). Gate: OpenAI-style request answered; stop tokens & streaming work.
- **M7 — Export + Eval:** HF-style safetensors folder + reload-parity test; perplexity + model card. Gate: reload logits match within tolerance; eval JSON saved.
- **M8 — Scale path (configs + docs):** 125M/350M/1B configs, bf16, activation checkpointing, DDP/FSDP2 stubs. Gate: configs validate; documented run commands.

## Top correctness traps (drive test-first)

1. RoPE off-by-one in cached decode — decode position must be `past_len`, not `0`.
2. Prefill (`T>1`) vs decode (`T==1`) sharing causal-mask code.
3. GQA K/V repeat: compact `n_kv_heads` then expand to `n_heads` — wrong interleave is silent.
4. SFT label mask leaking user/system tokens; must single-shift and **keep EOS** trainable.
5. Chat template duplicating BOS/EOS or dropping the assistant generation prompt.
6. DPO logprobs over prompt tokens instead of completion; reference model not frozen / dropout on.
7. Grad-accum loss mis-division (divide by accumulation steps).
8. Checkpoint resume restoring weights but not optimizer/scheduler/RNG/data cursor.
9. `block_size+1` windowing so inputs/targets shift correctly; EOS between packed docs.
10. Perplexity = `exp(Σ nll / Σ tokens)` excluding `ignore_index` — never over padded tokens.

## Out of scope for V1

MoE, MLA, multimodal, tool use, GRPO/RLVR training, custom vLLM registration, large-scale crawling, k8s.
Tracked as future milestones once the V1 pipeline is green end-to-end.
