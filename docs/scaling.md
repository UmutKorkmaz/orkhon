# Scaling Orkhon: GPUs, token budgets, wall-clock, and cost

This document is the practical guide to taking Orkhon from a laptop-sized run to a
real cloud pretraining job. It covers how big a model to train, how many tokens to
feed it, how long that takes on different GPUs, roughly what it costs, and the exact
commands for 1-8 GPUs with both DDP and FSDP.

> **These are estimates.** Throughput depends on sequence length, batch size,
> attention kernel (SDPA/FlashAttention), bf16 vs fp32, interconnect (NVLink vs
> PCIe), data-loader speed, and how well the kernels are tuned. Treat the numbers
> below as order-of-magnitude planning figures, not benchmarks. Measure on your
> actual box (the engine logs `tokens_per_sec` every `log_interval` steps).

---

## 1. Model sizes in this repo

| Config | Params | block | vocab | Where it runs |
|---|---|---|---|---|
| `configs/model/smoke_6m.yaml`  | ~6M   | 256  | 4096  | CPU/CI smoke |
| `configs/model/tiny_24m.yaml`  | ~24M  | 512  | 8192  | Apple Silicon (MPS) |
| `configs/model/tiny_50m.yaml`  | ~51M  | 512  | 16384 | One local device, overnight |
| `configs/model/small_125m.yaml`| ~125M | 1024 | 32768 | One cloud GPU (A100/H100/4090) |
| `configs/model/base_350m.yaml` | ~350M | 2048 | 32768 | One big GPU (bf16 + ckpt) |
| `configs/model/orkhon_1b.yaml` | ~1B   | 2048 | 32768 | Multi-GPU (FSDP) |

The scale path this document targets is **50M local -> 125M cloud**:
`pretrain_50m.yaml` and `pretrain_125m_cloud.yaml`.

---

## 2. How many tokens? (the budget question)

Two regimes:

- **Chinchilla-optimal (~20 tokens/param).** Minimizes loss for a *fixed compute
  budget*. For 125M params that's ~2.5B tokens. Cheapest way to a given loss if you
  only train once and never run inference.
- **Overtraining (100-1000x params).** Trains *past* the compute-optimal point to
  get a stronger model *per parameter*. The model is smaller and cheaper to serve,
  at the cost of more training compute. This is what modern small base models
  (and `pretrain_125m_cloud.yaml`, ~200 tok/param ≈ 24.6B tokens) do, because the
  model gets used for inference far more than it gets trained.

| Model | Chinchilla (20x) | Overtrain (100x) | Overtrain (200x) |
|---|---|---|---|
| 50M  | 1.0B   | 5B   | 10B   |
| 125M | 2.5B   | 12.5B| 25B   |
| 350M | 7B     | 35B  | 70B   |
| 1B   | 20B    | 100B | 200B  |

**Rule of thumb:** if the model will be served a lot, overtrain it. If you just want
the lowest loss for the least compute, stay near 20x. FineWeb-Edu is large enough
(1.3T tokens) that token count is never the bottleneck — GPU-hours are.

`tokens/step = batch_size * grad_accum_steps * seq_len * world_size`. Pick
`max_steps` so `tokens/step * max_steps` hits your target budget. When you add GPUs,
**lower `grad_accum_steps` proportionally** to keep the global batch (and thus the
LR schedule) constant.

---

## 3. Throughput, wall-clock, and cost

Per-GPU throughput assumes bf16, SDPA attention, and the sequence lengths in the
configs. "tok/s/GPU" is realistic-but-rough sustained training throughput.

| Model | GPU | tok/s/GPU | $/GPU-hr* |
|---|---|---|---|
| 50M  | A100 40GB | ~45k  | ~$1.5 |
| 50M  | H100 80GB | ~85k  | ~$2.5 |
| 125M | A100 40GB | ~22k  | ~$1.5 |
| 125M | H100 80GB | ~45k  | ~$2.5 |
| 350M | A100 80GB | ~9k   | ~$1.8 |
| 350M | H100 80GB | ~18k  | ~$2.5 |
| 1B   | H100 80GB | ~7k   | ~$2.5 |

\* On-demand cloud GPU prices vary 2-3x by provider/region and drop sharply on
spot/community instances. Use these only to size a budget.

**Wall-clock** for a token budget `T` on `N` GPUs (near-linear DDP scaling up to a
few nodes): `hours ≈ T / (tok_per_s_per_gpu * N * 3600)`. **Cost** ≈
`hours * N * $/GPU-hr` (note `N` cancels in cost — more GPUs finish faster for the
same total $, ignoring efficiency losses).

### Worked examples

**50M, 10B tokens (200x overtrain) on H100s (~85k tok/s/GPU):**

| GPUs | Wall-clock | Approx cost |
|---|---|---|
| 1 | ~33 hr | ~$82  |
| 4 | ~8.2 hr| ~$82  |
| 8 | ~4.1 hr| ~$82  |

**125M, 25B tokens (200x overtrain) on A100 40GB (~22k tok/s/GPU):**

| GPUs | Wall-clock | Approx cost |
|---|---|---|
| 1 | ~316 hr (13 d) | ~$473 |
| 4 | ~79 hr  | ~$473 |
| 8 | ~39 hr  | ~$473 |

**125M, 25B tokens on H100 80GB (~45k tok/s/GPU):**

| GPUs | Wall-clock | Approx cost |
|---|---|---|
| 1 | ~154 hr (6.4 d)| ~$386 |
| 4 | ~39 hr  | ~$386 |
| 8 | ~19 hr  | ~$386 |

The cost is roughly GPU-count-invariant; more GPUs buy *time*, not *money*. Scaling
efficiency erodes past ~8 GPUs / one node (cross-node all-reduce over the network),
so real multi-node cost is somewhat higher than these single-node estimates.

A practical first cloud run: **125M on 8x H100 for ~$400 and under a day.** Or, to
prove the pipeline cheaply: **50M on 1x A100 for a few dollars overnight.**

---

## 4. DDP vs FSDP — which to use

| | DDP | FSDP |
|---|---|---|
| What it shards | nothing (full model on every GPU) | params + grads + optimizer state |
| Memory/GPU | ~1x model | ~1/N model |
| Comms | all-reduce grads each step | all-gather params + reduce-scatter grads |
| Use when | model + optimizer fit on one GPU | model is too big for one GPU |
| In this repo | 50M, 125M, 350M | 1B (or 350M on small GPUs) |

For 125M, **DDP is the right default** — the model and its AdamW state fit
comfortably in 40-80GB, and DDP has lower communication overhead than FSDP. Reach
for FSDP only when a single GPU can't hold the model + optimizer (≈1B+ params, or
when you want very long sequences). The wrap strategy is chosen by the
`ORKHON_DISTRIBUTED` env var (`ddp` | `fsdp` | `none`), read by the training engine;
no config-file change is required.

---

## 5. Exact commands (1-8 GPUs)

### Turnkey (recommended)

```bash
# 125M on 8 GPUs with DDP, end to end (install -> data -> tokenizer -> train):
NUM_GPUS=8 bash scripts/train_cloud.sh

# 50M on a single GPU:
NUM_GPUS=1 CONFIG=configs/train/pretrain_50m.yaml bash scripts/train_cloud.sh

# 125M on 8 GPUs with FSDP sharding:
NUM_GPUS=8 DIST_MODE=fsdp bash scripts/train_cloud.sh
```

### Manual torchrun (single node)

`torchrun --standalone` sets `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `MASTER_*` for
each process; the engine reads them, inits the process group (NCCL on CUDA), wraps
the model, shards the data by a rank-seeded sampler, and logs/checkpoints only on
rank 0. **`WORLD_SIZE==1` keeps the exact single-process code path.**

```bash
# Single GPU (no torchrun needed):
python -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 1 GPU via torchrun (identical behavior, just exercises the launcher):
ORKHON_DISTRIBUTED=ddp torchrun --standalone --nproc_per_node=1 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 4 GPUs, DDP:
ORKHON_DISTRIBUTED=ddp torchrun --standalone --nproc_per_node=4 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 8 GPUs, DDP:
ORKHON_DISTRIBUTED=ddp torchrun --standalone --nproc_per_node=8 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 8 GPUs, FSDP (per-transformer-block sharding):
ORKHON_DISTRIBUTED=fsdp torchrun --standalone --nproc_per_node=8 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml
```

### Multi-node torchrun

```bash
# On every node (set NODE_RANK 0..N-1; MASTER_ADDR = rank-0 host):
ORKHON_DISTRIBUTED=ddp torchrun \
  --nnodes=2 --node_rank=$NODE_RANK \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml
```

### Keeping the global batch constant as you add GPUs

The global batch (and therefore the LR schedule) is
`batch_size * grad_accum_steps * world_size`. `pretrain_125m_cloud.yaml` uses
`grad_accum_steps: 20` tuned for **one** GPU. To hold the same global batch:

| GPUs | Set `--set train.grad_accum_steps=` | Global batch (tokens) |
|---|---|---|
| 1 | 20 (default) | ~0.49M |
| 2 | 10 | ~0.49M |
| 4 | 5  | ~0.49M |
| 8 | 2 or 3 | ~0.49M-0.74M |

Example: `... -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml --set train.grad_accum_steps=5`
for a 4-GPU run that matches the single-GPU recipe.

---

## 6. Operational notes

- **Checkpoints/metrics**: written only by rank 0 to the config's `out_dir`
  (`runs/fineweb_125m`, `runs/fineweb_50m`). Saved state is always the *unwrapped*
  model, so checkpoints load in single-process `eval` / `chat` / `export` paths.
- **Resume**: re-run the same command; the engine restores model + optimizer + RNG
  + step from `out_dir/ckpt_last.pt`. The LR is recomputed from the step, so resume
  is exact regardless of GPU count.
- **Logged loss / throughput**: loss is all-reduced to the global mean across ranks;
  `tokens_per_sec` is the *aggregate* cluster throughput (sum over ranks).
- **bf16**: `dtype: auto` selects bf16 on CUDA and fp32 elsewhere. bf16 needs Ampere
  (A100/RTX 30xx) or newer; on older GPUs set `--set train.dtype=float16` and add
  loss scaling, or stick to fp32.
- **OOM on 125M?** Lower `train.batch_size` and raise `grad_accum_steps` to keep the
  global batch; or switch to `ORKHON_DISTRIBUTED=fsdp`.
