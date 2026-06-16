# Orkhon Roadmap — Scaling From 51M / 162M Tokens to a Genuinely Capable Multi-Billion-Token Model

> **Status:** authoritative, merged from 7 expert reviews (data, model/scaling-laws, training infra, post-training, eval, long-context, cost). Grounded in `docs/build-your-own-llm-guide.md`, `docs/scaling.md`, `docs/implementation-plan.md`, and the live code under `src/orkhon/` + `configs/`. Param counts below are **verified** against `ModelConfig.estimate_params`.

---

## 1. TL;DR + the ultimate ladder

Orkhon is a correct, adversarially-verified, hand-written decoder-only stack (GQA, RoPE, RMSNorm, SwiGLU, KV-cache; pretrain + SFT + DPO; DDP/FSDP2 wired). It has **never trained past 162.8M tokens** and now has smoke benchmark reports, but those are not yet decision-grade scale gates. The single principle that drives this whole roadmap:

> **"Millions / billions of tokens" means `tokens ≈ 100–200 × params`** for any model you intend to *serve* (the guide's overtraining regime, not Chinchilla's 20×). That is what turns a 51M toy into a model with above-chance HellaSwag, and a 1B into a genuinely useful base.

The token budget — not corpus supply (FineWeb-Edu alone is 1.3T) — is the planning variable; **GPU-hours and pipeline throughput are the real bottleneck above ~70B tokens.**

### The ultimate ladder (one row per stage; verified param counts; H100 economics)

| Rung | Model (config) | Total / non-emb params | Train tokens (≈tok/param) | Hardware | tok/s/GPU | Wall-clock | $ (spot ≈50%) | Expected capability |
|---|---|---|---|---|---|---|---|---|
| **R0 now** | istemi `tiny_50m` | 46.9M / 36.4M | 0.16B (≈3×) | M5 Pro (MPS) | ~19k | (done) | — | pipeline-proof only; ~chance on benches |
| **R1** | `tiny_50m` full | 46.9M / 36.4M | **10B (200×)** | 1× A100 spot | ~45k | ~2.6 d | **~$60** | HellaSwag 30–34%; proves overtraining works |
| **R2** | `small_125m` | 100.7M / 75.5M | **25B (200×)** | 8× H100 spot | ~45k | ~19 hr | **~$190** | first real signal; ARC-e > chance; pipeline-validation run |
| **R3 (MVP)** | `base_350m` | 304M / 270M | **70B (200×)** | 8× H100 spot | ~18k | ~5.6 d | **~$1.35k** | SmolLM2-360M tier; HellaSwag 48–55%; **first model worth a card** |
| **R4** | `orkhon_1b` | 1.149B / 1.082B | **200B (200×)** | 16× H100 (2 nodes) FSDP2 | ~7k | ~18 d | **~$10–17k** | genuinely useful base; HS 60–66%, MMLU 30–36%; +SFT → IFEval 35–50% |
| R4-lite | `orkhon_1b` | 1.149B | 60B (≈55×) | 16× H100 spot | ~7k | ~6 d | ~$3k | the solo-affordable 1B |
| **R5** | `orkhon_3b` (new) | 3.12B / 2.82B | **600B (200×)** | 32× H100/B200 FSDP2 | ~2.5k | ~36 d | **~$83k** | requires data mixing + μP; stretch/funded |
| **R6 (frontier-lite)** | `orkhon_7b` (new) | 6.07B / 5.67B | **1.0–1.4T (150–200×)** | 64× H100 | ~1.1k | ~68 d | **~$315–630k** | brackets OLMo-3-7B (~234k H100-hr / ~$470k); moonshot only |

Costs reconcile with guide references: R6 brackets OLMo-3-7B; R4 brackets TinyLlama-1.1B economics. Orkhon will sit at the **expensive end** of each bracket until the MFU/checkpointing gaps in §6 close.

**Decision rule baked into the ladder:** climb to **R3 (350M)** as the solo deliverable; **R4 (1B)** is the funded-team sweet spot; **R5/R6** are justified only by a data moat or sovereignty requirement — the guide's own bottom line is that fine-tuning an open base beats this for product use.

---

## 2. Where Orkhon is now (honest current state)

**What exists and is good:** correct hand-written transformer (GQA/RoPE/RMSNorm/SwiGLU/KV-cache), byte-level BPE, pretrain + assistant-only SFT + textbook DPO sharing one canonical next-token shift (`train/losses.py`), DDP/FSDP2 wired (`train/distributed.py`, single-proc byte-identical), FineWeb-Edu streaming, exact-parity HF import of Llama-arch models, a model zoo, native perplexity eval, exact resume (model+opt+RNG+step).

**Models so far:** bumin 4M (synthetic), tonyukuk 22M (TinyStories, ppl 4.8), istemi 51M (FineWeb-Edu, 162.8M tokens, ppl 46.5), kashgari = imported SmolLM2-135M, bengü 57M EN/TR base, and bengü-göktürk transliteration SFT.

**The honest gaps that block scale (all verified in code):**

- **Data dies at ~300–500M tokens.** `data/tokenize.py::prepare_pretrain` accumulates two in-RAM Python `list[int]` then `tofile()`s a *single* `train.bin`. `scripts/train_cloud.sh` caps ingestion at `MAX_DOCS=2000000` (~1.5–2B tokens). No sharding, dedup, langID, quality classifier, or decontamination. The pipeline implements only `extract(min_chars) → tokenize` of the guide's 8-stage curation.
- **Benchmarks are implemented but not yet decision-grade.** `orkhon bench` can score built-in, HellaSwag, ARC-Easy, GSM8K, and MBPP-style tasks, but the model zoo still needs durable JSON reports before capability claims or paid scale decisions. Perplexity and samples are not enough.
- **Training loop is single-process-clean but not fully scale-hardened.** Step-keyed sampling now exists for flat/sharded packed datasets and pretraining uses it, but there is still no source-mix reader and `checkpoint.py` is rank-0-only full-`state_dict` `torch.save` (stalls/breaks under FSDP at 1B+). `train.compile` flag exists but is dead; AdamW has no `fused`; no MFU logging, no signal handler.
- **No post-training beyond DPO**, synthetic data only, no GRPO/rejection-sampling, single-sequence `generate()`.
- **Context capped at 2048**, RoPE single-theta hard-capped at `block_size` (`apply_rotary` index_select crashes past it); no YaRN/NTK, no sliding-window, unbounded KV-cache.

---

## 3. The seven dimensions (tight, de-duplicated)

### 3a. Data — from 162M tokens to multi-billion / trillion-token pretraining

The data ladder (bytes/token ≈ 4.40 on the live 16k shard; a 32k tokenizer shrinks token counts ~5–8%):

| Rung | Model | Train tokens | Raw text | Corpus mix (headline) |
|---|---|---|---|---|
| R2 | 125M overtrain | 25B | ~110GB | FineWeb-Edu **sample-100BT** (single source) |
| R3 | 350M | 70B | ~310GB | FineWeb-Edu 100BT + **Wikipedia + Stack-v2 code (10%)** |
| R4 | 1B | 200B | ~900GB | web 70% / code 15% / math 8% / wiki+books 7% |
| R4-ml | 1B multilingual | 300B | ~1.4TB | + **Turkish/multilingual 15%** (HPLT/CulturaX) |
| R5 | 3B | 600B | ~2.8TB | FineWeb 350BT-class web + full code/math/multiling |
| R6 | 7B | 1.0–1.4T | ~5–7TB | FineWeb-full (15T pool) sampled + Nemotron-CC synthetic |

**Corpora (HF ids):** `HuggingFaceFW/fineweb-edu` (0.50 of R4 mix), `mlfoundations/dclm-baseline-1.0` / `HuggingFaceFW/fineweb` (0.20), `bigcode/the-stack-v2-dedup` license-filtered (0.15), `HuggingFaceTB/finemath` + `open-web-math/open-web-math` (0.08), `wikimedia/wikipedia` (0.04), Gutenberg/PG-19 (0.03), `HPLT/HPLT2.0` + `uonlp/CulturaX` Turkish (0.15 @ R4-ml). Weights are **epoch-budget fractions**: web is downsampled; small high-value sources (code/math/books) repeat up to ~4 epochs (web ≤1), safe per Muennighoff repetition findings.

**The four pipeline gaps, built in order** (each detailed in §5): (1) sharded/parallel/resumable tokenize — *the* blocker above 500M tokens; (2) multi-shard reader + source-mix sampler; (3) curation layer (langID + DCLM-style quality classifier + per-snapshot MinHash dedup + 13-gram decontam); (4) tokenizer 16k→32k (R3)→48k multilingual (R4-ml). The tokenizer choice is **irreversible** (guide §11) and forces a fresh pretrain — decide at the R3→R4 boundary, before committing GPU-hours.

### 3b. Model + scaling laws — the capacity ladder (51M → 7B)

**Token-per-param table (verified `estimate_params`):**

| Rung | Total params | 20× (Chinchilla) | 100× | 200× (default for served) | 500× ("strong base") |
|---|---:|---:|---:|---:|---:|
| 50M | 46.9M | 0.9B | 4.7B | 9.4B | 23B |
| 125M | 100.7M | 2.0B | 10.1B | 20.1B | 50B |
| 350M | 304M | 6.1B | 30.4B | 60.8B | 152B |
| 1B | 1.149B | 23B | 115B | 230B | 575B |
| 3B | 3.12B | 62B | 312B | 624B | 1.56T |
| 7B | 6.07B | 121B | 607B | 1.21T | 3.0T |

> **Naming reconciliation:** `small_125m.yaml` actually estimates **100.7M total** (the "125M" label is loose; non-emb ≈ 75.5M). Keep the token budget keyed to the *real* count, or bump `d_model 768→896` / `n_layers 12→14` to genuinely hit 125M.

**Two new verified configs** (Llama-2/3 shape, head_dim 128, GQA 8 KV, vocab 49152, `rope_theta 500000`, untied head):
- `configs/model/orkhon_3b.yaml` → L28 · d3072 · h24(kv8) · inter8192 · ctx4096 → **3.12B / 2.82B non-emb**. FSDP2 required.
- `configs/model/orkhon_7b.yaml` → L32 · d4096 · h32(kv8) · inter11008 · ctx4096 → **6.07B / 5.67B non-emb**. FSDP2 + activation checkpointing.

**Architecture stabilizers, by tier** (none exist today; each maps to a real file):

- **Tier 1 (before any run >350M; cheap, high-impact):** (1) **document-masked attention** — `pack.py` slices windows across `<eos>` so packed sequences attend across unrelated docs; emit segment-ids and AND a same-document mask into `attention.py::_build_mask`. (2) **z-loss** `z_coef·logsumexp(logits)²` (~1e-4) in `losses.py` — #1 cure for bf16 logit drift. (3) **QK-norm** (RMSNorm on q,k before RoPE, fp32) in `attention.py` — the single best >1B bf16 stabilizer. (4) **loss-spike guard** in `engine.py` — branch on the already-computed `grad_norm` to skip NaN/Inf/spike steps.
- **Tier 2 (before 3B):** **μP / width-aware init** — `init.py` hardcodes `init_std=0.02` for every size, so LR tuned at 125M is wrong at 3B+. Either cheap Llama path (`init_std = 0.02·√(d_ref/d_model)`) or full μP (transfer LR zero-shot from a 50M sweep). **LR width rule without μP:** `lr ∝ 1/√d_model` — from a tuned 6e-4@d768: 350M→5.2e-4, 1B→3.7e-4, 3B→3.0e-4, 7B→2.6e-4 (both current configs hardcode a too-hot flat 6e-4).
- **Tier 3 (3B+, optional):** MoE — defer to a `v2` rung after the dense ladder works; README lists it out-of-scope for v1.
- **Weight tying:** keep `tie=true` ≤1B (embeddings are 25% of 125M); flip to `tie=false` at 3B/7B (embedding fraction <5%, untied head buys quality) — a pure config change (`optim.py::_is_embedding_param` already dedupes tied weights).

Global batch stays in the **0.5M–4M token** band; `rope_theta 10000→500000` at ctx≥4096; vocab 32768→49152 above 1B.

### 3c. Training infrastructure — single-GPU → multi-node FSDP2

The architecture and single-process loop are production-quality; the scale delta is concentrated in **four files** + two new modules. Per-GPU MFU targets (MFU = 6·N·tok/s ÷ peak; A100 312TF, H100 990TF, B200 ≈2250TF):

| Stage | Model / tokens | Hardware | tok/s/GPU | MFU target | GPU-hr | $ spot |
|---|---|---|---|---|---|---|
| S0 prove | 50M / 10B | 1× A100 | 45k | ~30% | 62 | ~$60 |
| S1 base | 125M / 25B | 8× H100 | 50k | ~40% | 70 | ~$350 |
| S2 small | 350M / 70B | 8× H100 | 20k | ~42% | 970 | ~$2.4k |
| **S3 headline** | 1B / 200B | 16× H100 (2 nodes) FSDP2 | 8k | ~40% | 6.9k | ~$17k |
| S3-fast | 1B / 200B | 32× B200 | 22k | ~38% | 2.5k | ~$25k |

**The cliff is the node boundary (S3+):** cross-node all-reduce/all-gather erodes scaling. The 1B/200B run is *un-runnable today* because a single spot preemption in 18 days restarts from the last rank-0 `.pt` with a re-randomized data stream.

The cheap engine wins (no new hardware): **`torch.compile`** (`train.compile` flag is dead → +15–30% tok/s; `unwrap_model` must peel `_orig_mod`), **fused AdamW** (+3–8%), **activation checkpointing** (~40% activation memory for ~25% compute — required for 1B to fit), and **MFU logging** (you cannot tune what you don't measure).

The two correctness prerequisites for any run longer than one `ckpt_interval`: **(a) step-keyed resumable dataloader** — make sampling a pure fn of `(global_step, rank, seed)` via `np.random.default_rng((seed,step,rank))` and persist a cursor in the checkpoint; **(b) DCP sharded async checkpoint** — replace rank-0 full-`state_dict` `torch.save` with `torch.distributed.checkpoint` (each rank writes its own shard + `async_save`), keeping a consolidated `.pt` only at run-end for export/serve.

Then **spot survival** (SIGTERM handler that saves with cursor; `train_cloud.sh` restart-until-done loop; c10d rendezvous `--rdzv_backend=c10d --max_restarts=10` replacing `--standalone`) and **monitoring** (`train/monitor.py`: wandb + TB behind a `MetricsSink` protocol, loss-EMA/grad-norm spike alerts, per-rank straggler step-time). Pre-tokenize ~200B FineWeb-Edu **once** to sharded `.bin` on Lustre/object storage; never re-tokenize per run.

### 3d. Post-training & alignment — mid-training → SFT → preference → RL

The existing SFT (assistant-only masking) + DPO (frozen ref, completion-only logprobs) stack is correct and is the foundation. The modern recipe is a **four-phase ladder**, sized for the 350M→1B models that are the realistic target:

- **Phase 0 — Mid-training** (the missing pre-SFT stage): 10–50B tokens of CoT/code/reasoning text (`open-thoughts/OpenThoughts-114k`, `open-r1/openr1-220k-math`) + 10–20% FineWeb-Edu replay. **Reuses `pretrain.py` unchanged** — just a new `configs/train/midtrain_*.yaml`. 1B/30B tok ≈ ~$1k spot.
- **Phase 1 — SFT on real data:** `allenai/tulu-3-sft-mixture` (~939K), `OpenHermes-2.5`, `SmolTalk`; start 200K–500K examples. `sft.py` works as-is; gaps are **data conversion**, **multi-turn packing/truncation** (`dataset.py` truncates mid-conversation and pads to batch-max, wasting 30–50% of FLOPs), and a **`tool` role**. 1B/~1B tok ≈ ~$350.
- **Phase 2 — Preference:** `allenai/ultrafeedback_binarized` (~61K), Tülu-3 pref mix, HelpSteer2. Add **SimPO/ORPO** as pluggable objectives in `dpo.py` (reference-free → drops a full model of memory at 1B). ~$150.
- **Phase 3 — RL (GRPO/RLVR, the big new build):** the DeepSeek-R1 four-stage pipeline. GRPO (no critic) maps onto the existing `micro_batch_loss` closure; the frozen-reference pattern from `dpo.py` is the KL-control template. Needs: `train/grpo.py`, `train/rewards.py` (MathVerify `\boxed{}` sympy-compare, code unit-test sandbox, format/length), **batched `generate_batch`** (current `generate()` is single-sequence — the most invasive change; short-term loop it, right answer is batched KV-cache or a vLLM rollout split), and `token_logprobs()`/`grpo_loss()` in `losses.py`. 1B GRPO ≈ ~$500–1.5k (generation-dominated).

**Sequencing insight (guide §6):** *distillation transfers capability; GRPO only optimizes within the base ceiling.* Build **rejection sampling** (`generate_batch` + reward filter + re-run `sft.py`) *before* full GRPO — it captures most of the reasoning gain at a fraction of the complexity. A `<|tool|>` token must be **appended at index 8** (never reorder ids 0–7 — load-bearing for every checkpoint).

### 3e. Evaluation & quality gates — the eval ladder

The intrinsic layer (token-weighted PPL, smoke gate, golden-set chat) is correct; the repo now also has native benchmark adapters and generative pass@k plumbing. What is missing is the official scoreboard run across the zoo and promotion gates that consume those JSON reports. Expected benchmark scores (acc_norm) per rung:

| Rung | Model / tokens | HellaSwag | ARC-e | MMLU | GSM8K | Gate |
|---|---|---|---|---|---|---|
| R1 | 51M / 10B | 30–34% | 40–45% | chance | 0% | "above chance on HS" |
| R2 | 125M / 25B | 36–42% | 48–55% | 25–27% | <2% | gate on HS + ARC-e only |
| R3 | 350M / 70B | 48–55% | 60–66% | 32–36% | 2–4% | SmolLM2-360M tier |
| R4 | 1B / 200B | 60–66% | 70–76% | 30–36% | 5–12% | genuinely useful base |
| R4+SFT | 1B + SFT/DPO | (held) | (held) | (held) | 15–25% | IFEval 35–50% headline |

**Capability floor:** MMLU/GSM8K/HumanEval are **noise below ~350M** — gate only on HellaSwag + ARC-e until R3 (matches guide's per-interval tiers). **Free bridge validation:** run the harness against imported **kashgari (SmolLM2-135M)** first — if it reproduces published HellaSwag ~42%, the bridge is correct, *zero training cost*.

Builds: keep the native `orkhon bench` path as the source of truth (`eval/benchmarks.py`, `eval/loglikelihood.py`, `eval/generative.py`) and add durable scoreboard reports for the current zoo before cloud spend; tiered in-training bench (HellaSwag+ARC-e every ~5k steps, rank-0, `--limit 1000` ≈ 2–4 GPU-min, <2% overhead); `eval/domains.py` (hash-pinned multi-domain held-out PPL); `eval/decontam.py` (13-gram vs benchmark test sets — **run pre-train, not discovered later as an unexplained jump**); and **regression gates on zoo promotion** (`registry.quality_gate()` + `configs/eval/gates.yaml` blocking `register_model` on NaN/contamination/below-floor; `build_index` must surface HellaSwag/ARC columns, not just ppl).

### 3f. Long context (the orthogonal axis) — 2K → 32K → 128K → 1M

Orkhon has the right *primitives* (RoPE, GQA) but **none of the extension layer**. Two composing levers, staged: pretrain cheap at 4K–8K, then a *short* length-extension CPT with rescaled RoPE, optionally pushed further with YaRN at inference. KV-cache is O(T) memory, attention O(T²) compute (128K prefill ≈ 4000× the attention FLOPs of 2K), so:

| Model | Comfortable dense | With SWA/chunked | Aspirational |
|---|---|---|---|
| tiny_50m | 32K | 128K–256K | 1M (window) |
| base_350m | 32K | 128K | 256K (window) |
| orkhon_1b | 16K–32K | 128K | 128K (window) |

- **L0 (no GPU):** decouple the RoPE table from `block_size`; add `rope_scaling` (linear PI / NTK / YaRN with `mscale = 0.1·ln(factor)+1`) + `max_position_embeddings` to `config.py`/`rope.py`/`transformer.py`. Unblocks evaluating a 2K model at 8K–16K.
- **L1 (no GPU):** document-masked packing — emit `doc_id.bin`, block-diagonal mask + per-doc RoPE reset (shares the Tier-1 doc-mask work in §3b).
- **L2 (~$150, 1 H100):** length-extension CPT to 32K — `rope_theta 10000→~500000`, ~1–5B tokens, 5–20% short-context replay, LR 5–10× lower; add an SDPA flash/causal fast path (the dense `[B,1,T,T]` bool mask is impossible at 32K = ~1B booleans/head).
- **L3 (~$50–100):** 128K via sliding-window + global-layer hybrid + sliding-window KV eviction (current `kv_cache.py` only grows via `torch.cat`).
- **L4 (research, tiny size):** 1M only at ≤125M with paged KV-cache + chunked prefill; or export and serve 128K+ under vLLM (guide §17).

### 3g. Cost — the budget reality

Method (`scaling.md` §3): `GPU-hr = tokens ÷ (tok/s/GPU × 3600)`; `$ ≈ GPU-hr × $/GPU-hr` (**GPU-count-invariant** — more GPUs buy wall-clock, not money); `wall-clock = GPU-hr ÷ N`. Spot = 40–60% off and **safe because of Orkhon's exact resume** (`engine.py::maybe_resume` loses at most `ckpt_interval` steps) — this is Orkhon's single biggest cost lever.

The two realistic paths: **(a) Solo dev** — prove at 125M (~$190), ship at **350M / 70B / ~$1,350 spot** and stop; 1B-full ($10–17k, ~18–41 GPU-days) is where solo stops being fun (1B-lite ~$3k is the stretch). **(b) Funded team** — **1B is the sweet spot** (~$10–17k compute + ~$5–15k overhead); 7B (~$315–630k + 8–30 engineers) is Path-5 territory justified only by a data moat. **Rent, on spot** — buying an 8×H100 node (~$320k) only pays off above ~5,300 GPU-days of continuous >60% utilization. **Do not rent B200 until FP8 lands** — its ~2.5× edge is FP8-gated (`utils/dtype.py` is bf16/fp32 only), so at $4.5–6/hr it is a worse deal than H100 spot for Orkhon today.

Storage is cheap (1B = 400GB `.bin`; 7B = 2TB) but the **single-file architecture is the blocker**. Checkpoints at 12 B/param: 1B = 12GB, 7B = 84GB (a single `torch.save` of 84GB stalls training → DCP needed).

---

## 4. The single ultimate ladder, condensed

```
R0 (now) istemi 51M / 162.8M tok ─────────── pipeline proven, smoke benchmark reports only
   │  build: shard pipeline + lm-eval bridge + step-keyed cursor
R1   tiny_50m / 10B  / 1×A100   / ~$60   / 2.6d ── HS 30-34%; overtraining proven
   │  build: source-mix reader + 32k tokenizer + Tier-1 stabilizers + DCP ckpt
R2   small_125m / 25B / 8×H100  / ~$190  / 19h  ── pipeline-validation on cloud, ARC-e>chance
   │  build: activation-ckpt toggle + curation (dedup/decontam) + bench-in-loop
R3   base_350m / 70B / 8×H100   / ~$1.35k/ 5.6d ── ★ MVP: SmolLM2-360M tier, first model card
   │  build: FSDP2 multi-node + spot survival + μP + post-training (SFT real data)
R4   orkhon_1b / 200B / 16×H100 / ~$17k  / 18d  ── ★ useful base; +SFT/DPO → IFEval 35-50%
   │  build: 48k multilingual tokenizer + GRPO/rejection-sampling + long-ctx CPT
R5   orkhon_3b / 600B / 32×GPU  / ~$83k  / 36d  ── data mixing + μP required; funded
R6   orkhon_7b / 1-1.4T/ 64×H100/ ~$315k / 68d  ── frontier-lite; moonshot, data-moat only
```

---

## 5. Consolidated code gaps (prioritized, by file — this is what makes the plan actionable)

Ordered so each unlocks the next rung. **P0** blocks R1–R3; **P1** blocks R4; **P2** is R5/R6/long-context.

### P0 — blocks everything above ~500M tokens (must land before R1/R3)

1. **`data/shard.py` (NEW)** — streaming, parallel (`multiprocessing`), resumable tokenizer with `manifest.json` (`{shard_id, n_tokens, sha256, status, dtype}`); ~256M-token / ~512MB shards; never holds the corpus in RAM. *Replaces the in-RAM list in `data/tokenize.py::prepare_pretrain`.* **The single highest-leverage change.**
2. **`scripts/train_cloud.sh`** — remove the `MAX_DOCS=2000000` cap (~1.5–2B tokens, 20× short of R2, 100× short of R4); pair with sharded ingestion.
3. **`data/mixed.py::MixedShardDataset` (NEW)** + **`data/pack.py`** — multi-shard memmap reader with weighted source sampling, preserving the `(x,y)` contract and rank-seeding. *`pack.py` and `pretrain.py` both assume one `.bin`.*
4. **`config/schema.py::DataConfig`** — add `sources: list[{dir, weight}]` (+ `val_dir`); keep `prepared_dir` as back-compat shorthand.
5. **`data/pack.py::get_batch` / `data/shard.py::get_batch`** — **implemented for pretraining**: sampling can be a pure fn of `(seed, step, rank)`, and pretraining threads a gradient-accumulation microstep so microbatches do not repeat. Still needs source-mix integration.
6. **`train/checkpoint.py` + `engine.py::_save`** — DCP sharded `async_save` path (each rank writes its shard), atomic `.tmp`→`os.replace`, retain last-k, persist the dataloader cursor; keep a consolidated `.pt` for export/serve. *Today: rank-0 full-`state_dict` `torch.save` (84GB blocking save at 7B; broken under FSDP).*
7. **Benchmark scoreboard** — `orkhon bench` already exists; run it across `kashgari`/`istemi`/`bengü`/`bengü-göktürk` where local artifacts exist, write JSON reports, and only then decide whether an `lm-eval` bridge is still needed.
8. **`engine.py`** — apply the dead `train.compile` flag (`torch.compile(mode="max-autotune")`, peel `_orig_mod` in `unwrap_model`); add `fused=(device.type=="cuda")` to `optim.py` AdamW; add `model_flops_per_token`/`mfu` to `metrics.py` and log it.
9. **Tier-1 stabilizers:** doc-masked attention (`pack.py` segment-ids + `attention.py::_build_mask`), z-loss (`losses.py` + `engine.py`), QK-norm (`attention.py`, fp32), spike guard (`engine.py` — branch on existing `grad_norm`). Add optional fields to `ModelConfig` (`qk_norm`, `z_loss_coef`, `doc_masked_attention`) and `OptimConfig` (`z_loss_coef`, `spike_grad_mult`), all defaulting to current behavior.
10. **`model/config.py` + `model/block.py`** — activation-checkpointing toggle (`checkpoint: bool`, `torch.utils.checkpoint(use_reentrant=False)`). Required for 350M/1B to fit a usable micro-batch.
11. **Tokenizer + configs:** retrain at **32k** on web+code (`configs/tokenizer/`); reconcile the **16384-vs-32768 vocab conflict** (`fineweb.yaml` says 16384, `small_125m.yaml`/`train_cloud.sh` use 32768) before sizing any `.bin` or budget; add `configs/model/orkhon_3b.yaml` + `orkhon_7b.yaml`; reconcile the `small_125m` mislabel.

### P1 — blocks R4 (1B) and post-training

12. **`data/curate.py` (NEW)** — langID (fastText), DCLM-style quality classifier for broad web, MinHash-LSH per-snapshot dedup (datasketch, 128 perms, J=0.8). **`data/decontam.py` (NEW)** — 13-gram bloom filter vs MMLU/HellaSwag/ARC/GSM8K/HumanEval; run **before** the first billion-param run.
13. **Spot survival:** SIGTERM/SIGUSR1 handler in `engine.py` (save+cursor+barrier+exit); `train_cloud.sh` restart-until-done loop + c10d rendezvous (`--rdzv_backend=c10d --max_restarts=10` replacing `--standalone`). **`train/monitor.py` (NEW)** — `MetricsSink` protocol (wandb+TB), loss-spike/grad-norm alerts, per-rank straggler step-time.
14. **μP / width-aware init** — `init.py` (`init_std` scaling) + `optim.py` (per-group LR multipliers) + `ModelConfig` (`mup`, `mup_base_d_model`); replace flat `lr:6e-4` with width-scaled LRs.
15. **Post-training:** `eval/chat_eval.py` (add `\boxed{}`/numeric extractor — shared by GRPO reward and eval); SimPO/ORPO in `dpo.py`; `<|tool|>` appended at index 8 in `special_tokens.py` + `render.py`; conversation-aware packing in `dataset.py`; data converters (`data/instruct_convert.py`, `data/preference_convert.py`); `train/rejection_sample.py`, then `train/grpo.py` + `train/rewards.py` + `generate_batch` in `generation.py` + `token_logprobs`/`grpo_loss` in `losses.py` + `GRPOConfig`.
16. **Bench-in-training:** generalize `engine.py` `eval_fn` beyond val_loss; tiered `bench_interval`/`bench_full_interval` (rank-0, `--limit`); add keys to the 50M/125M configs. `registry.quality_gate()` + `configs/eval/gates.yaml`; extend `build_index` to surface benchmark columns.

### P2 — R5/R6 and long context

17. **`utils/dtype.py`** — FP8/MXFP8 autocast (gates B200 cost-effectiveness).
18. **`data/tokenize.py` / shard dtype** — `uint16` (≤32k vocab) → `uint32` (48k–64k multilingual); make dtype a manifest field. 48k multilingual tokenizer at the R3→R4 boundary.
19. **Long-context:** `rope.py`/`config.py`/`transformer.py` YaRN/NTK + `max_position_embeddings` (L0); `doc_id.bin` (L1); SDPA flash fast path + 32K CPT config (L2); `sliding_window`/`global_attn_layers` + KV eviction (L3); paged KV-cache + chunked prefill + remove `block_size` decode stops in `generation.py`/`serve/api.py` + needle-in-haystack eval (L4).
20. **MoE** (`mlp.py`/`block.py`/loss/FSDP wrap) — deferred to a `v2` rung after the dense ladder works.

---

## 6. Phased execution plan

### NEXT (days) — prove the pipeline cheaply, build the spine

**Goal: a validated multi-GPU + spot-resume pipeline and a working benchmark scoreboard, for ~$200.**

1. **Zero-training benchmark validation.** Use the native `orkhon bench` path first, and only add `lm-eval` if the native adapters are insufficient:
   ```bash
   orkhon bench --checkpoint models/kashgari-*/checkpoint --tasks hellaswag,arc_easy --num-fewshot 0
   ```
   If `kashgari` weights/tokenizer are present, confirm it roughly reproduces SmolLM2-135M's published HellaSwag tier. If not, record the missing artifact and run `istemi`, `bengü`, and `bengü-göktürk` to lock an honest S0 baseline and JSON report schema.
2. **Land the remaining P0 data + checkpoint spine** (gaps 1–6): `MixedShardDataset` + `DataConfig.sources`, remove `MAX_DOCS`, DCP async checkpoint, and a forced-resume proof over a sharded/source-mix run. Step-keyed flat/sharded sampling already has local tests.
3. **Land the cheap engine wins** (gap 8): `torch.compile`, fused AdamW, MFU logging.
4. **Re-prepare FineWeb-Edu sample-100BT** into ~100 shards on the M5 Pro overnight via `data/shard.py`.
5. **R1 prove-run:**
   ```bash
   NUM_GPUS=1 CONFIG=configs/train/pretrain_50m.yaml bash scripts/train_cloud.sh   # ~$60 spot, validates end-to-end
   ```

**Decision point:** if R1 reaches HellaSwag >30% and resume is byte-exact, proceed to spend cloud $ on R2/R3. If MFU <30% on the first cloud run, tune knobs before scaling.

### NEAR (weeks) — the MVP base worth shipping

**Goal: a 350M / 70B-token base with a model card and honest benchmarks, ~$1.35k spot.**

6. Tier-1 stabilizers + activation-checkpoint toggle + 32k tokenizer (gaps 9–11); reconcile the vocab conflict and the `small_125m` mislabel.
7. **R2 validation run** (cloud pipeline shakedown):
   ```bash
   NUM_GPUS=8 bash scripts/train_cloud.sh   # small_125m / 25B / ~$190 / ~19h
   ```
8. Curation + decontamination (gap 12) over the FineWeb-Edu shards **before** R3.
9. Wire bench-in-training (HellaSwag+ARC-e fast tier at 5k steps) + `domains.py` held-out PPL.
10. **R3 MVP run:**
    ```bash
    NUM_GPUS=8 CONFIG=configs/train/pretrain_350m.yaml bash scripts/train_cloud.sh  # 70B tokens, ~5.6d, ~$1.35k spot
    ```
    Then register: `orkhon register --kind base ... --eval-prepared ...` with a model card + HellaSwag/ARC numbers, gated by `registry.quality_gate()`.

**Decision point:** R3 is the **solo ceiling** — a genuinely usable base for one good GPU-week. Stop here unless funded. Spend on R4 only after FSDP2 spot survival + DCP are dry-run-tested with a forced 2-node preemption.

### MID (months) — the 1B headline run and capability

**Goal: a 1B / 200B-token base + instruction-following model.**

11. Spot survival + monitoring + μP (gaps 13–14); 48k multilingual tokenizer at the R3→R4 boundary (irreversible — decide now).
12. Pre-tokenize ~200B FineWeb-Edu **once** to sharded `.bin` on Lustre/object storage.
13. **R4 headline run** (16× H100, 2 nodes, FSDP2, c10d rendezvous, restart loop, wandb alerts): 200B tokens, ~18 days, ~$17k spot.
14. **Post-training ladder on the 1B base** (gap 15), re-running the eval suite as a gate after each stage: mid-train (reuse `pretrain.py`) → SFT (Tülu-3/OpenHermes, real data) → DPO/SimPO → rejection-sampling → GRPO (start with the slow loop-`generate()` path on a smoke GSM8K set to prove reward rises and KL stays bounded, then invest in batched/vLLM rollouts).
15. **Long context (parallel track):** L0 + L1 (no GPU) anytime; one 32K length-extension CPT (L2, ~$150) on the 350M/1B base to ship a long-context variant.

**Decision points / when to STOP scaling:**
- **Stop at R3** if solo/budget-constrained — the guide's bottom line is that fine-tuning an open base beats from-scratch for product use.
- **Proceed to R5/R6 only** with (a) FP8 implemented (gap 17, or B200 is a worse deal than H100 spot), (b) a single-source token budget exhausted forcing real data mixing, and (c) a concrete data-moat or sovereignty justification for $83k–$630k. R6 (7B) brackets OLMo-3-7B's ~$470k / 234k H100-hr and the 8–30-engineer overhead the guide flags — it is a moonshot, not a default.
