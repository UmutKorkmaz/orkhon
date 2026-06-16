# Orkhon'u Ölçeklemek: GPU, token bütçesi, süre ve maliyet

[English](scaling.md) | [Türkçe](scaling.tr.md)

Bu belge Orkhon'u laptop boyutlu bir koşudan gerçek cloud pretraining işine taşımak için pratik rehberdir:
hangi boyutta model eğitilecek, kaç token okutulacak, farklı GPU'larda ne kadar sürer, yaklaşık maliyet nedir
ve 1-8 GPU için DDP/FSDP komutları nelerdir.

> **Bunlar tahmindir.** Throughput; sequence length, batch size, attention kernel, bf16/fp32, interconnect,
> data-loader hızı ve kernel tuning'e bağlıdır. Aşağıdaki sayıları benchmark değil, bütçe büyüklüğü olarak
> okuyun. Kendi makinenizde ölçün; engine her `log_interval` step'te `tokens_per_sec` yazar.

---

## 1. Bu repodaki model boyutları

| Config | Parametre | block | vocab | Nerede çalışır |
|---|---:|---:|---:|---|
| `configs/model/smoke_6m.yaml` | ~6M | 256 | 4096 | CPU/CI smoke |
| `configs/model/tiny_24m.yaml` | ~24M | 512 | 8192 | Apple Silicon (MPS) |
| `configs/model/tiny_50m.yaml` | ~51M | 512 | 16384 | tek local device |
| `configs/model/small_125m.yaml` | ~125M | 1024 | 32768 | tek cloud GPU |
| `configs/model/base_350m.yaml` | ~350M | 2048 | 32768 | büyük GPU, bf16 + ckpt |
| `configs/model/orkhon_1b.yaml` | ~1B | 2048 | 32768 | multi-GPU (FSDP) |

Bu dokümanın hedeflediği scale path **50M local → 125M cloud** hattıdır:
`pretrain_50m.yaml` ve `pretrain_125m_cloud.yaml`.

---

## 2. Kaç token?

İki rejim vardır:

- **Chinchilla-optimal (~20 token/param).** Sabit compute bütçesinde loss'u minimize eder.
- **Overtraining (100-1000× param).** Parametre başına daha güçlü model için compute-optimal noktanın ötesinde
  eğitir. Serve edilecek küçük base modeller için modern pratik budur.

| Model | Chinchilla (20×) | Overtrain (100×) | Overtrain (200×) |
|---|---:|---:|---:|
| 50M | 1.0B | 5B | 10B |
| 125M | 2.5B | 12.5B | 25B |
| 350M | 7B | 35B | 70B |
| 1B | 20B | 100B | 200B |

Kural: model çok serve edilecekse overtrain edin; yalnız en ucuz loss gerekiyorsa 20× civarında kalın.
FineWeb-Edu yeterince büyüktür; darboğaz token arzı değil GPU-saatidir.

`tokens/step = batch_size * grad_accum_steps * seq_len * world_size`. GPU eklerken global batch ve LR schedule
sabit kalsın diye `grad_accum_steps` değerini orantılı azaltın.

---

## 3. Throughput, süre ve maliyet

| Model | GPU | tok/s/GPU | $/GPU-saat |
|---|---|---:|---:|
| 50M | A100 40GB | ~45k | ~$1.5 |
| 50M | H100 80GB | ~85k | ~$2.5 |
| 125M | A100 40GB | ~22k | ~$1.5 |
| 125M | H100 80GB | ~45k | ~$2.5 |
| 350M | A100 80GB | ~9k | ~$1.8 |
| 350M | H100 80GB | ~18k | ~$2.5 |
| 1B | H100 80GB | ~7k | ~$2.5 |

Token bütçesi `T` ve GPU sayısı `N` için yaklaşık formül:

```text
hours ≈ T / (tok_per_s_per_gpu * N * 3600)
cost  ≈ hours * N * $/GPU-hr
```

GPU sayısı maliyette yaklaşık sadeleşir; daha çok GPU çoğunlukla zaman alır, para değil. Node sınırı aşıldığında
network scaling verimi düşer.

### Örnekler

**50M, 10B token, H100 (~85k tok/s/GPU):**

| GPU | Süre | Yaklaşık maliyet |
|---:|---:|---:|
| 1 | ~33 saat | ~$82 |
| 4 | ~8.2 saat | ~$82 |
| 8 | ~4.1 saat | ~$82 |

**125M, 25B token, H100 (~45k tok/s/GPU):**

| GPU | Süre | Yaklaşık maliyet |
|---:|---:|---:|
| 1 | ~154 saat | ~$386 |
| 4 | ~39 saat | ~$386 |
| 8 | ~19 saat | ~$386 |

İlk pratik cloud run: **8×H100 üzerinde 125M, yaklaşık $400 ve bir günden az**. Hattı ucuz kanıtlamak için:
**1×A100 üzerinde 50M, gece boyunca**.

---

## 4. DDP mi FSDP mi?

| | DDP | FSDP |
|---|---|---|
| Ne shard eder | hiçbir şey; model her GPU'da tamdır | params + grads + optimizer state |
| GPU başı bellek | ~1× model | ~1/N model |
| İletişim | her step gradient all-reduce | param all-gather + grad reduce-scatter |
| Ne zaman | model + optimizer tek GPU'ya sığıyorsa | tek GPU'ya sığmıyorsa |
| Bu repoda | 50M, 125M, 350M | 1B veya küçük GPU'da 350M |

125M için **varsayılan DDP** olmalıdır. FSDP'ye yalnız model+optimizer tek GPU'ya sığmadığında geçin.
Wrap stratejisi `ORKHON_DISTRIBUTED` env var'ı ile seçilir: `ddp` | `fsdp` | `none`.

---

## 5. Komutlar

### Turnkey

```bash
# 8 GPU DDP ile 125M, uçtan uca:
NUM_GPUS=8 bash scripts/train_cloud.sh

# Tek GPU 50M:
NUM_GPUS=1 CONFIG=configs/train/pretrain_50m.yaml bash scripts/train_cloud.sh

# 8 GPU FSDP:
NUM_GPUS=8 DIST_MODE=fsdp bash scripts/train_cloud.sh
```

### Manuel torchrun

```bash
# Tek GPU:
python -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 4 GPU DDP:
ORKHON_DISTRIBUTED=ddp torchrun --standalone --nproc_per_node=4 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml

# 8 GPU FSDP:
ORKHON_DISTRIBUTED=fsdp torchrun --standalone --nproc_per_node=8 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml
```

### Multi-node

```bash
# Her node üzerinde çalıştırın; NODE_RANK 0..N-1, MASTER_ADDR rank-0 host:
ORKHON_DISTRIBUTED=ddp torchrun \
  --nnodes=2 --node_rank=$NODE_RANK \
  --nproc_per_node=8 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml
```

### Global batch sabit tutma

`pretrain_125m_cloud.yaml` tek GPU için `grad_accum_steps: 20` kullanır. Aynı global batch için:

| GPU | `--set train.grad_accum_steps=` | Global batch |
|---:|---:|---:|
| 1 | 20 | ~0.49M token |
| 2 | 10 | ~0.49M token |
| 4 | 5 | ~0.49M token |
| 8 | 2 veya 3 | ~0.49M-0.74M token |

---

## 6. Operasyon notları

- **Checkpoint/metrics:** yalnız rank 0 tarafından config'in `out_dir` dizinine yazılır.
- **Resume:** aynı komutu yeniden çalıştırın; engine model + optimizer + RNG + step'i `ckpt_last.pt` üzerinden
  restore eder.
- **Loss / throughput:** loss rank'ler arası global mean'e all-reduce edilir; `tokens_per_sec` cluster toplamıdır.
- **bf16:** `dtype: auto` CUDA'da bf16, diğerlerinde fp32 seçer. Eski GPU'da fp32 veya loss scaling gerekir.
- **OOM:** `train.batch_size` düşürüp `grad_accum_steps` artırın veya `ORKHON_DISTRIBUTED=fsdp` kullanın.
