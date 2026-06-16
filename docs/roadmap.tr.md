# Orkhon Roadmap — 51M / 162.8M Token'dan Çok Milyar Token'lı Kullanılabilir Modele

[English](roadmap.md) | [Türkçe](roadmap.tr.md)

> **Durum:** data, model/scaling laws, training infra, post-training, eval, long-context ve cost incelemelerinin
> birleşik sonucu. [`build-your-own-llm-guide.tr.md`](build-your-own-llm-guide.tr.md),
> [`scaling.tr.md`](scaling.tr.md), [`implementation-plan.tr.md`](implementation-plan.tr.md) ve canlı
> `src/orkhon/` + `configs/` koduna dayanır.

---

## 1. TL;DR ve nihai merdiven

Orkhon doğru, adversarial olarak doğrulanmış, elle yazılmış decoder-only stack'tir: GQA, RoPE, RMSNorm, SwiGLU,
KV-cache, pretrain + SFT + DPO, DDP/FSDP2. Bugüne kadar 162.8M token ötesine geçmedi; smoke benchmark raporları
var, fakat bunlar henüz scale harcamasını yönetecek decision-grade gate değildir.

Ana kural:

> Serve edilecek model için `tokens ≈ 100-200 × params` hedeflenir. 51M modelin toy olmaktan çıkması için 10B
> token, 1B modelin gerçekten işe yarar base olması için 200B token gerekir.

### Nihai ladder

| Rung | Model | Token | Donanım | Süre | Spot maliyet | Beklenen kabiliyet |
|---|---|---:|---|---:|---:|---|
| **R0 şimdi** | `istemi` 51M | 0.162B | M5 Pro | bitti | — | pipeline proof, smoke benchmark |
| **R1** | 50M full | 10B | 1× A100 | ~2.6g | ~$60 | overtraining çalışıyor mu? |
| **R2** | 125M | 25B | 8× H100 | ~19s | ~$190 | ilk gerçek sinyal |
| **R3 MVP** | 350M `tengri` | 70B | 8× H100 | ~5.6g | ~$1.35k | paylaşmaya değer ilk model |
| **R4** | 1B `otuken` | 200B | 16× H100 | ~18g | ~$10-17k | gerçekten işe yarar base |
| **R5** | 3B | 600B | 32× GPU | ~36g | ~$83k | fonlu stretch |
| **R6** | 7B | 1.0-1.4T | 64× H100 | ~68g | ~$315k+ | moonshot |

Karar kuralı: solo deliverable olarak **R3**; fonlu ekip için **R4**; R5/R6 yalnız veri moat veya sovereignty
gerektiğinde.

---

## 2. Orkhon şu anda nerede?

**İyi olan:** doğru hand-written transformer, byte-level BPE, pretrain + assistant-only SFT + DPO, DDP/FSDP2,
FineWeb-Edu streaming, Llama-arch HF import parity, model zoo, native perplexity eval, exact resume.

**Modeller:** bumin 4M, tonyukuk 22M, istemi 51M (162.8M FineWeb-Edu token, ppl 46.5), kashgari import edilmiş
SmolLM2-135M, bengü 57M EN/TR base, bengü-göktürk transliterasyon SFT.

**Scale'i bloklayan gap'ler:**

- Data pipeline tek `.bin` ve RAM listelerine dayanıyordu; sharding geldi ama source-mix ve scale proof tamamlanmalı.
- Benchmark harness var; model zoo için kalıcı JSON raporları karar kapısı olmalı.
- Training loop single-process temiz; FSDP-scale için DCP checkpoint, spot survival, source-mix resume ve MFU logging gerekir.
- Post-training SFT/DPO ötesine geçti fakat GRPO/RLVR sadece smoke seviyesinde.
- Context 2048; long-context için RoPE scaling, SDPA fast path ve KV eviction gerekir.

---

## 3. Yedi eksen

### 3a. Data

| Rung | Model | Token | Raw text | Mix |
|---|---|---:|---:|---|
| R2 | 125M | 25B | ~110GB | FineWeb-Edu sample |
| R3 | 350M | 70B | ~310GB | FineWeb-Edu + Wikipedia + code |
| R4 | 1B | 200B | ~900GB | web/code/math/wiki/books |
| R4-ml | 1B | 300B | ~1.4TB | + Turkish/multilingual |
| R5 | 3B | 600B | ~2.8TB | full data mix |
| R6 | 7B | 1.0-1.4T | ~5-7TB | FineWeb-full + synthetic |

Önce sharded tokenize, mixed reader, curation/dedup/decontam ve tokenizer freeze çözülür.

### 3b. Model + scaling laws

| Rung | Parametre | 20× | 100× | 200× |
|---|---:|---:|---:|---:|
| 50M | 46.9M | 0.9B | 4.7B | 9.4B |
| 125M | 100.7M | 2.0B | 10.1B | 20.1B |
| 350M | 304M | 6.1B | 30.4B | 60.8B |
| 1B | 1.149B | 23B | 115B | 230B |
| 3B | 3.12B | 62B | 312B | 624B |
| 7B | 6.07B | 121B | 607B | 1.21T |

Tier-1 stabilizer'lar: document-masked attention, z-loss, QK-norm, loss-spike guard. 3B öncesi μP/width-aware
init gerekir. Dense ladder çalışmadan MoE ertelenir.

### 3c. Training infra

Ölçek farkı birkaç dosyada yoğunlaşır: `data/shard.py`, mixed reader, DCP checkpoint, spot survival, `torch.compile`,
fused AdamW, activation checkpointing, MFU logging, source-mix aware deterministic resume.

### 3d. Post-training

Modern sıra:

1. Mid-training: CoT/code/reasoning + FineWeb replay.
2. SFT: gerçek instruction data.
3. Preference: DPO/SimPO/ORPO.
4. Rejection sampling → GRPO/RLVR: reward functions, `generate_batch`, KL control.

Distillation, küçük modeller için GRPO'dan önce daha ucuz ve daha etkilidir.

### 3e. Evaluation

Perplexity ve sample yetmez. Native `orkhon bench` source of truth olmalı; HellaSwag/ARC JSON raporları scale
harcamasından önce üretilmelidir. MMLU/GSM8K/HumanEval 350M altı gürültülüdür; R3 öncesi gate HellaSwag + ARC.

### 3f. Long context

RoPE scaling, document-masked packing, SDPA flash path, sliding-window attention, KV eviction ve needle-in-haystack
eval staged gelir. 32K R3 sonrası ucuz bir length-extension CPT ile mümkündür.

### 3g. Cost

`GPU-hr = tokens ÷ (tok/s/GPU × 3600)`. GPU sayısı maliyeti değil süreyi azaltır. Solo yol: R2 proof, R3 ship,
dur. Fonlu yol: R4 1B sweet spot. R5/R6 için FP8, data moat ve ekip gerekir.

---

## 4. Tek merdiven

```text
R0  istemi 51M / 162.8M tok ─ pipeline proven, smoke benchmark only
R1  tiny_50m / 10B   / 1×A100   / ~$60
R2  small_125m / 25B / 8×H100   / ~$190
R3  base_350m / 70B  / 8×H100   / ~$1.35k  ★ MVP
R4  orkhon_1b / 200B / 16×H100  / ~$17k    ★ useful base
R5  orkhon_3b / 600B / 32×GPU   / ~$83k
R6  orkhon_7b / 1-1.4T / 64×H100 / ~$315k+
```

---

## 5. Kod gap'leri

### P0 — R1/R3 öncesi

1. `data/shard.py`: streaming, parallel, resumable tokenizer + manifest.
2. `scripts/train_cloud.sh`: `MAX_DOCS` cap kaldırma.
3. `MixedShardDataset` + source-mix sampler.
4. `DataConfig.sources`.
5. Step-keyed sampling + resume proof.
6. DCP sharded async checkpoint.
7. Benchmark scoreboard JSON.
8. `torch.compile`, fused AdamW, MFU logging.
9. Tier-1 stabilizer'lar.
10. Activation checkpoint toggle.
11. Tokenizer/config reconciliation.

### P1 — R4 ve post-training

12. `data/curate.py`, `data/decontam.py`.
13. Spot survival + monitoring.
14. μP / width-aware init.
15. Post-training: SimPO/ORPO, tool token, conversation packing, rejection sampling, GRPO.
16. Bench-in-training + registry quality gate.

### P2 — R5/R6 ve long context

17. FP8/MXFP8.
18. 48k-64k tokenizer için shard dtype planı.
19. YaRN/NTK, SDPA, sliding-window, paged KV.
20. MoE, dense ladder çalıştıktan sonra.

---

## 6. Faz planı

### NEXT

Amaç: ~$200 altında benchmark scoreboard + multi-GPU/spot-resume proof.

1. `orkhon bench` ile zero-training validation.
2. P0 data + checkpoint spine.
3. Engine wins: compile, fused AdamW, MFU.
4. FineWeb-Edu sample shard'ları.
5. R1 prove-run.

### NEAR

Amaç: 350M / 70B-token base, model card ve dürüst benchmark.

6. Tier-1 stabilizer + activation checkpoint + tokenizer reconciliation.
7. R2 validation run.
8. Curation + decontamination.
9. Bench-in-training.
10. R3 MVP run ve gated `orkhon register`.

### MID

Amaç: 1B / 200B-token base + instruction-following model.

11. Spot survival, monitoring, μP, 48k multilingual tokenizer freeze.
12. 200B shard'ı bir kez tokenize et.
13. R4 headline run.
14. Mid-train → SFT → DPO/SimPO → rejection sampling → GRPO.
15. Long-context track.

**Durma noktaları:** solo ise R3'te durmak mantıklıdır. R5/R6 yalnız FP8, veri moat ve fonlu ekip varsa.
