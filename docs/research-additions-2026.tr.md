# Orkhon Araştırma Ekleri (Haziran 2026)

[English](research-additions-2026.md) | [Türkçe](research-additions-2026.tr.md)

> **Amaç:** Halihazırda belgelenmiş
> [`roadmap.tr.md`](roadmap.tr.md) (R0→R6 ölçek),
> [`capability-roadmap.tr.md`](capability-roadmap.tr.md) (C0→C8 tools/RAG/agents/vision) ve
> [`turkic-languages.tr.md`](turkic-languages.tr.md) (Türkçe + Göktürk/`bengü`) ötesindeki yönleri taramak.
> 2026 odaklı araştırma yönlerinden sentezlendi. Hedef kitle: Apple Silicon/MPS üstünde solo builder + ara sıra
> ucuz cloud GPU.

---

## Executive summary

Orkhon'un mevcut dokümanları solo proje için alışılmadık derecede tamamdır. Scale spine, capability spine ve
Turkic identity toplamda generic "build your own LLM" roadmap'inin çoğunu kapsar.

Eksik kalanlar üç kovaya ayrılır:

1. **Packaging & trust:** GGUF/MLX export, Turkish eval leaderboard, provenance/watermark hooks, eğitim GTM.
2. **Ucuz capability multiplier:** distillation/rejection sampling, audio-as-tool, legal RAG vertical,
   Turkish safety SFT, data flywheel tooling.
3. **Positioning:** "SmolLM2 ama Türkçe" değil; **auditable Turkic heritage lab**. Göktürk savunulabilir moat'tır.

En yüksek ROI eklerin çoğu MPS üzerinde $0 veya küçük cloud bütçesiyle yapılır; R4 beklemez.

---

## Zaten belgelenmiş olanlar

| İş | Nerede | Neden tekrarlandı |
|---|---|---|
| Sharded tokenize + mixed reader + step-keyed dataloader | [`roadmap.tr.md`](roadmap.tr.md) | Her yön buna dayanıyor |
| DCP async checkpoints + spot survival | [`roadmap.tr.md`](roadmap.tr.md) | 2 günden uzun paid run için şart |
| Eval bridge / native benchmark scoreboard | [`roadmap.tr.md`](roadmap.tr.md), [`eval.tr.md`](eval.tr.md) | Benchmark yoksa karar yok |
| 48k tokenizer freeze | [`turkic-languages.tr.md`](turkic-languages.tr.md), [`capability-roadmap.tr.md`](capability-roadmap.tr.md) | R3 öncesi tek geri dönüşsüz event |
| C0-C5 serve layer | [`capability-roadmap.tr.md`](capability-roadmap.tr.md) | Saat başına en yüksek kaldıraç |
| `bengü` bilingual track | [`turkic-languages.tr.md`](turkic-languages.tr.md) | Türkçe kimlik ana fark |

---

## Yön bulguları

| # | Yön | Kısa karar |
|---|---|---|
| 1 | Audio / speech | ASR + TTS tool olarak yüksek değer; native audio sonra. |
| 2 | Reasoning & RLVR | Rejection sampling şimdi; GRPO R4 sonrası. |
| 3 | Distillation | R3'te çok değerli; GRPO'dan ucuz. |
| 4 | Quantization & edge | GGUF/MLX export R2/R3'te şart; core quant kernel yazma. |
| 5 | Pan-Turkic | Türkçe-first kal; transfer proof sonrası genişlet. |
| 6 | Safety/alignment | Public demo öncesi Turkish safety eval ve refusal layer. |
| 7 | Public leaderboard | P0 extension; R3 gate için opsiyonel değil. |
| 8 | Code/math specialist | R3 mix içinde explicit deliverable. |
| 9 | Embeddings/RAG | BGE-M3/e5 reuse; embedding from scratch yok. |
| 10 | Long context/memory | Memory = RAG; YaRN/NTK R3 sonrası. |
| 11 | MoE/efficiency | Serve optimizasyonları; MoE dense R4 sonrası. |
| 12 | Product/UX | HF Space + OpenAI-compatible client yüzeyi. |
| 13 | Open-source/community/grants | Güven, funding ve model adoption için gerekli. |
| 14 | Cultural-heritage NLP | Göktürk/Ottoman/epics; lisans ve kaynaklı veri şart. |
| 15 | Data flywheel | Synthetic, dedup, decontam; scale spine'ın parçası. |
| 16 | Domain verticals | Tek dikey seç; hukuk/eğitim/gov RAG mantıklı. |
| 17 | Structured output | Grammar decoding/tool schema için düşük maliyetli kalite. |
| 18 | Model merging | Specialist fine-tune birleştirme; geç ve ucuz. |
| 19 | MLX training adjunct | Apple Silicon için inference ve küçük adapter faydalı. |
| 20 | Watermarking/provenance | Model card ve release trust için eklenmeli. |
| 21 | Teaching/courseware GTM | Orkhon'un öğrenme stack'i olarak değeri var. |
| 22 | Roadmap gap'leri | P0/P1 zaten doğru; priority execution gerekiyor. |

---

## Öncelikli ekler

### Tier A — Top 7

1. Data flywheel + provenance.
2. Benchmark scoreboard + Turkish leaderboard.
3. Tokenizer freeze + fertility report.
4. `generate_batch()` + repetition penalty.
5. Distillation/rejection sampling.
6. GGUF/MLX export + public demo.
7. Trust layer: model cards, LICENSE/NOTICE, contamination notes, R3 tech report.

### Tier B — Sonra

Audio-as-tool, Turkish safety layer, legal RAG vertical, pan-Turkic eval, long-context 32K, structured output,
model merging.

### Tier C — Şimdilik out of scope

Internal MoE, native video, frontier chase, embedding model from scratch, public code execution.

---

## 90 günlük öneri

1. **İlk 2 hafta:** docs/status sync, benchmark JSON, tokenizer fertility gate, Turkish docs, release hardening.
2. **Hafta 3-6:** sharded/source-mix proof, DCP/resume, leaderboard scaffold, `generate_batch`.
3. **Hafta 7-10:** R1/R2 validation run, GGUF/MLX export, HF Space.
4. **Hafta 11-13:** R3 readiness, Turkish mix, post-training data, trust/report layer.

## Competitive positioning

Orkhon'un güçlü cümlesi "en büyük Türkçe model" değildir. Daha savunulabilir konum:

> Sıfırdan yazılmış, denetlenebilir, Türkçe/Turkic veri hattı ve Orhun yazıtı demosu olan açık küçük-model lab'ı.

Bu pozisyon ölçülebilir benchmark, kaynaklı Göktürk claim'i ve temiz release artefact'leri olmadan çalışmaz.

## Metadata

Bu Türkçe sürüm public okuyucu için yoğunlaştırılmıştır; ayrıntılı direction matrix İngilizce kaynakta kalır.
