# Orkhon — Uygulama Planı (Complete AI, V1)

[English](implementation-plan.md) | [Türkçe](implementation-plan.tr.md)

Bu plan, [`build-your-own-llm-guide.tr.md`](build-your-own-llm-guide.tr.md) içindeki karar çerçevesini Orkhon V1
için somut ve doğrulama kapılı bir uygulama hedefine çevirir.

## Hedef

Orkhon'u plan dokümanından gerçekten çalışan, küçük ve denetlenebilir bir **sıfırdan LLM yığınına** çevirmek:
raw text → tokenizer → pretrain → SFT → DPO → eval → export → chat. Her şey eğitim amaçlı açıklık için PyTorch
ile elle yazılır ve cloud harcaması öncesi **Mac/MPS smoke run** ile kanıtlanır.

Bu özellikle frontier framework değildir. V1 kapsamına MoE, MLA, multimodal, GRPO, k8s veya büyük crawling girmez.

## Hedef donanım

Apple M5 Pro, 48 GB, MPS. Smoke koşuları MPS/CPU üzerinde dakikalar içinde çalışır. Ölçek config'leri aynı kodla
tek cloud GPU'yu (A100/H100) hedefler; DDP/FSDP2 sonraki milestone'dur.

## Tech stack

- Python ≥3.11; ortam yönetimi **uv**.
- Tek training framework: **PyTorch**. Model elle yazılır; `AutoModelForCausalLM` yok.
- Byte-level BPE için HF **tokenizers**, weights için **safetensors**, chat template için **jinja2**.
- Config için **pydantic** + YAML; CLI için **typer** + **rich**; serving için **FastAPI**/uvicorn.
- Test suite için **pytest**. `transformers`/`lm-eval`/vLLM yalnızca opsiyonel ve post-export.

## Device / dtype politikası

- CPU yolu her zaman çalışmalı (testler). Smoke MPS üzerinde. Ölçek CUDA bf16.
- Smoke varsayılan dtype = `float32`; MPS mixed precision kırılgan. CUDA varsayılanı `bfloat16`.

## Repository layout

```text
pyproject.toml  uv.lock  Makefile  .python-version
configs/        model/{smoke_6m,small_125m,base_350m,orkhon_1b}.yaml + train/data/eval/infer preset'leri
data/smoke/     pretrain.txt  sft.jsonl  dpo.jsonl   (deterministik üretilir)
src/orkhon/
  cli.py                      # tokenizer|data|train|eval|chat|serve|export
  config/        schema.py load.py      # typed config, YAML load, CLI override, validation
  tokenizer/     train.py special_tokens.py chat_template.jinja render.py validate.py
  data/          synth.py normalize.py tokenize.py pack.py dataset.py sft_format.py dpo_format.py collate.py
  model/         rope.py rmsnorm.py attention.py mlp.py block.py transformer.py kv_cache.py generation.py
  train/         engine.py optim.py schedule.py checkpoint.py losses.py metrics.py pretrain.py sft.py dpo.py
  eval/          perplexity.py smoke.py chat_eval.py report.py
  serve/         sampling.py chat_cli.py api.py schemas.py
  export/        to_hf.py model_card.py
tests/           rope, gqa, kv-cache parity, loss-shift, sft-mask, dpo-loss, checkpoint-resume, smoke pipeline
scripts/         smoke_all.sh
```

## Smoke config (`smoke_6m`)

vocab 4096 · block 256 · 4 layer · d_model 256 · 4 head · 2 kv-head · head_dim 64 · SwiGLU 768 · RoPE θ=10000 ·
tied embeddings · bias yok · float32. MPS üzerinde birkaç yüz step dakikalar içinde eğitilir.

## Scale config'leri

`small_125m` (12L/768d/4kv, ctx 1024) · `base_350m` (24L/1024d, ctx 2048) · `orkhon_1b` (24L/2048d, ctx 2048).
Single-GPU bf16 + activation checkpointing + grad-accum; 1B için FSDP2.

## Build order (verification-gated)

- **M0 — Contract:** pyproject/env, config schema, special tokens, chat template, utils, CLI skeleton,
  smoke config'leri, deterministik smoke dataset'leri. Gate: `orkhon --help`, invalid shape rejection,
  `uv run pytest -q`.
- **M1 — Tokenizer + Data:** byte-level BPE, normalize/tokenize/pack, SFT/DPO formatlayıcıları.
  Gate: round-trip, stable special id'ler, assistant-only SFT mask ve DPO mask'leri.
- **M2 — Model core:** RoPE, RMSNorm, GQA, SwiGLU, block, transformer, KV-cache, generation.
  Gate: shape testleri, one-batch overfit, cached generation == non-cached, param count doğru.
- **M3 — Pretrain smoke:** AdamW, cosine+warmup, grad-accum, clip, AMP, checkpoint/resume, val loss.
  Gate: loss düşer, exact resume, NaN yok, checkpoint reload ve generation çalışır.
- **M4 — SFT smoke:** assistant-only masking, chat eval, `orkhon chat`.
  Gate: küçük chat setini overfit eder; inference assistant prompt kullanır.
- **M5 — DPO smoke:** frozen reference model, completion-only logprobs, margin metric.
  Gate: chosen−rejected margin artar; reference frozen kalır.
- **M6 — Serve:** `orkhon chat` + `orkhon serve` (`/health`, `/v1/models`, `/v1/chat/completions`, SSE).
  Gate: OpenAI-style request cevaplanır; stop tokens ve streaming çalışır.
- **M7 — Export + Eval:** HF-style safetensors + reload parity; perplexity + model card.
  Gate: reload logits tolerans içinde eşleşir; eval JSON kaydedilir.
- **M8 — Scale path:** 125M/350M/1B config'leri, bf16, activation checkpointing, DDP/FSDP2 stub'ları.
  Gate: config'ler validate olur; run komutları dokümante edilir.

## Test-first doğruluk tuzakları

1. Cached decode RoPE off-by-one: decode pozisyonu `past_len` olmalı, `0` değil.
2. Prefill (`T>1`) ve decode (`T==1`) aynı causal-mask kodunu paylaşmalı.
3. GQA K/V repeat: önce compact `n_kv_heads`, sonra `n_heads`; yanlış interleave sessiz bozar.
4. SFT label mask user/system token'larını sızdırmamalı; tek shift ve trainable EOS gerekir.
5. Chat template BOS/EOS'u çoğaltmamalı veya assistant generation prompt'u düşürmemeli.
6. DPO prompt token'ları değil completion logprobs kullanmalı; reference frozen ve dropout kapalı olmalı.
7. Grad-accum loss bölümü doğru yapılmalı.
8. Resume yalnız weights değil optimizer/scheduler/RNG/data cursor da restore etmeli.
9. `block_size+1` windowing input/target shift'i doğru yapmalı; packed doc'lar arasında EOS olmalı.
10. Perplexity `ignore_index` hariç token'lar üzerinden hesaplanmalı; padding üstünden değil.

## V1 dışında

MoE, MLA, multimodal, tool use, GRPO/RLVR training, custom vLLM registration, büyük ölçekli crawling ve k8s.
V1 hattı yeşil olduktan sonra gelecek milestone olarak izlenir.
