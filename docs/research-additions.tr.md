# Orkhon'a Başka Ne Eklenmeli — Araştırma Sentezi

[English](research-additions.md) | [Türkçe](research-additions.tr.md)

> Mevcut planların ötesindeki eklemeler için öncelikli liste:
> [`roadmap.tr.md`](roadmap.tr.md) ölçek ekseni,
> [`capability-roadmap.tr.md`](capability-roadmap.tr.md) C0-C8,
> [`turkic-languages.tr.md`](turkic-languages.tr.md) Türkçe + Göktürk hattı.
> Amaç, yüksek kaldıraçlı mühendislik işlerini cazip ama erken model hırslarından ayırmaktır.

Lens: M5 Pro üzerinde çalışan solo builder + ara sıra ucuz cloud GPU. Aşağıdakilerin çoğu laptop üzerinde $0
ile inşa/test edilir; training maliyeti zaten bütçelenmiş R-ladder run'larının içine biner.

## TL;DR

İncelenen yönlerin ortak sonucu: en yüksek değer yeni model hırsları değil, scale planının zaten ihtiyaç duyduğu
**altyapı ve ölçüm** katmanlarıdır: ops spine, data curation, eval. Bunlara birkaç ucuz kaldıraç eklenir:
`generate_batch`, distillation, Turkish leaderboard.

Türkçe/Turkic kimlik; Turkish leaderboard, pan-Turkic mix, Göktürk transliterasyon ve teknik raporla görünür olur.

---

## Tier 1 — en yüksek değer, sıradaki işler

| # | Ek | Neden üstte | Efor | Nereye oturur |
|---|---|---|---:|---|
| **1** | **Ops spine** — monitoring, CI, NaN/spike guard, provenance, step-keyed cursor, DCP async checkpoint | Her multi-day cloud run'ın önkoşulu | ~8-9g | roadmap P0 / capability C0 |
| **2** | **Data flywheel** — sharded tokenize, MinHash dedup, 13-gram decontam, quality classifier | 500M token sonrası ana blokaj | ~8-12g | roadmap P0 |
| **3** | **Eval + Turkish leaderboard** | Benchmark'sız uçuş kördür; Turkish leaderboard community/trust kazandırır | ~5-7g | roadmap P0 |
| **4** | **`generate_batch()` + serving** | GRPO, batched eval ve serving'in ortak primitive'i; repetition penalty mevcut loop'ları iyileştirir | ~8g | C5/C7/eval |
| **5** | **White-box distillation** | Küçük modeller için RL'den çok daha ucuz capability transferi | ~4-6g | train stage |
| **6** | **Open-source & trust layer** | LICENSE/NOTICE, model card, CO2, contamination, governance, R3 tech report | ~6-9g | C0 |
| **7** | **Code + math verticals** | R3 token bütçesine binebilir; Turkish coding/math tutor somut ürün yüzeyi | ~14g | R3 |

En underrated ek: **`generate_batch()`**. Serving optimizasyonu gibi görünür, fakat GRPO, batched inference ve
hızlı eval'i aynı anda açar.

---

## Tier 2 — iyi ama daha sonra

- **Reasoning / GRPO-RLVR:** rejection sampling şimdi, gerçek GRPO R4 (`otuken`) sonrası.
- **Long context:** RoPE table decouple + YaRN/NTK önce; 32K CPT R3 sonrası.
- **Pan-Turkic:** R3/R4 mix içinde küçük pay; dedicated run ancak transfer proof sonrası.
- **Speech/audio:** Whisper-tr ASR → chat → TTS serve-time tool; native audio sonra.
- **Safety/alignment:** Turkish refusal/red-team SFT+DPO ve eval.
- **Embeddings/retrieval:** BGE-M3/e5 reuse; embedding model from scratch yok.
- **Domain vertical:** tek bir dikey seç; örn. Turkish-law RAG assistant.
- **Product/UX:** Gradio HF Space + OpenAI-compatible SDK shim.
- **Funding:** grant/credit planı ve run-cost logger.
- **Model merging:** specialist fine-tune'ları SLERP/TIES ile birleştirme; ucuz ama geç.

## Tier 3 — şimdilik atla

- **Internal MoE:** v1 kapsamı dışında; dense R4/R5 ve FP8 sonrası yeniden bakılır.
- **Mamba/SSM hybrids:** model rewrite + Llama parity kaybı; SWA+YaRN+RAG çoğu ihtiyacı karşılar.
- **From-scratch embedding model / generic frontier chase:** SOTA'ya karşı anlamsız; reuse edin.

---

## İki mevcut eksene eşleme

- **Scale'i açanlar:** ops spine + data flywheel + eval = roadmap P0.
- **Force-multiplier'lar:** `generate_batch` ve distillation küçük modelleri yukarı taşır.
- **Kimlik/diferansiyasyon:** Turkish leaderboard, pan-Turkic, Göktürk transliterasyon ve teknik rapor Orkhon'u
  başka bir küçük Llama clone'u olmaktan çıkarır.

*Sonuç: önce ölçüm ve altyapı; yeni model hırsları yalnız mevcut scale gate'e biniyorsa veya Turkic kimliği
doğrudan açıyorsa.*
