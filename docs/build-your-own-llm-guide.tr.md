# Kendi LLM'ini Kur — Karar Rehberi (2026)

[English](build-your-own-llm-guide.md) | [Türkçe](build-your-own-llm-guide.tr.md)

Bu rehber "LLM yapmak" isteğini doğru teknik yola bağlar. Çoğu ürün için cevap sıfırdan pretraining değildir;
RAG, LoRA veya açık base üstüne fine-tune daha doğru olur. Orkhon'un amacı ürün için en ucuz yol değil,
eğitimsel olarak denetlenebilir, Türkçe/Turkic kimliği olan bir stack kurmaktır.

## İçindekiler

1. Nereden başlamalı
2. "Kendi LLM'in" ne demek?
3. Yedi yol
4. Fine-tuning yöntemleri
5. Alignment ve post-training
6. Distillation ve compression
7. Synthetic data ve reasoning
8. Open-weight base modeller
9. Data pipeline
10. Architecture
11. Tokenizer
12. Continued pretraining
13. Framework/tooling
14. Hardware/cloud
15. Multimodal
16. Evaluation
17. Deployment
18. Legal/licensing
19. Karar ağaçları ve tarifler
20. Linkler ve sözlük

---

## Başlangıç

### 30 saniyelik seçim

| İhtiyaç | Varsayılan yol |
|---|---|
| Belgelerle doğru cevap | Prompt + RAG |
| Belirli stil/format | LoRA / QLoRA |
| Domain dili ve davranış | SFT |
| Yeni corpus'tan bilgi emdirme | Continued pretraining |
| Sistemi öğrenmek / bağımsız stack | Küçük modeli sıfırdan eğit |
| Egemenlik veya veri moat | Orta/büyük modeli sıfırdan eğit |

### Fine-tune öncesi minimum eval

- Held-out perplexity veya domain QA.
- HellaSwag / ARC tarzı genel akıl yürütme.
- Format takip eval'i.
- Contamination kontrolü.
- Güvenlik ve red-team seti.

---

## 1. "Kendi LLM'in" ne demek?

Bu ifade en az yedi farklı işi kapsar:

1. Prompt/RAG ile mevcut modeli kullanmak.
2. Adapter fine-tune.
3. Full SFT.
4. Continued pretraining.
5. Küçük modeli sıfırdan eğitmek.
6. 7B civarı base modeli sıfırdan eğitmek.
7. 70B+ frontier ölçeğine girmek.

Orkhon bilinçli olarak 5. yoldan başlar: stack'i öğrenmek, denetlemek ve küçük modellerde gerçek hatları görmek.

## 2. Modern training stack

Modern stack yalnız model dosyası değildir:

- tokenizer,
- data curation/dedup/decontam,
- model architecture,
- pretraining engine,
- SFT/preference/RL,
- eval harness,
- serving/export,
- model cards, lisans ve provenance.

Bu parçaların biri zayıfsa "model iyi mi?" sorusuna cevap verilemez.

## 3. Yedi yol

| Yol | Ne zaman | Maliyet | Risk |
|---|---|---:|---|
| Prompt + RAG | bilgi güncel/kaynaklı olmalı | düşük | retrieval kalitesi |
| LoRA/QLoRA | stil/format/adaptasyon | düşük | base sınırı |
| Full SFT | davranış derin değişecek | orta | forgetting |
| CPT | yeni domain bilgisi | orta-yüksek | veri hijyeni |
| Küçük from-scratch | öğrenme/sovereignty | orta | benchmark zayıf kalabilir |
| 7B from-scratch | veri moat/fon | çok yüksek | operasyon |
| 70B+ | frontier lab | devasa | ekip/altyapı |

## 4. Fine-tuning yöntemleri

- **LoRA/QLoRA:** en hızlı ve en ucuz; çoğu ürün için varsayılan.
- **Full SFT:** daha pahalı ama behavior transferi daha güçlü.
- **DPO/SimPO/ORPO:** preference alignment; SimPO/ORPO reference model yükünü azaltabilir.
- **RLVR/GRPO:** verifiable reward varsa güçlü; generation ve reward pipeline ister.

2026 varsayılanı: önce SFT, sonra preference; reasoning için önce rejection sampling/distillation, sonra GRPO.

## 5. Alignment ve post-training

İki ana paradigma vardır:

- **Offline preference:** DPO/IPO/SimPO/ORPO; dataset hazırdır.
- **Online RL:** model örnek üretir, reward hesaplanır, policy güncellenir.

Orkhon için güvenli sıra: SFT → DPO/SimPO → rejection sampling → GRPO/RLVR. GRPO küçük base'in ceiling'ini
aşmaz; distillation çoğu zaman daha ucuz capability transferidir.

## 6. Distillation ve compression

Modern pattern:

1. Güçlü teacher'dan yüksek kaliteli trace üret.
2. Student'ı SFT/distill ile eğit.
3. Quantization/export ile deployment'a indir.

Distillation küçük modeller için özellikle mantıklıdır; RL'den önce denenmelidir.

## 7. Synthetic data ve reasoning

Instruction synthesis, tool-call trace generation, FIM code data ve math verifier data küçük ekipler için
kaldıraçtır. Kural: sentetik veri deterministik, izlenebilir ve eval ile denetlenebilir olmalı.

## 8. Open-weight base modeller

Ürün için çoğu zaman en iyi yol açık base seçip fine-tune etmektir. License tiers:

- permissive / Apache-style,
- custom commercial terms,
- research-only veya non-commercial.

Orkhon'un `kashgari` import'u bu köprüyü kanıtlar: Llama-architecture open base, Orkhon Transformer'ına exact
logit parity ile yüklenebilir.

## 9. Data pipeline

Pretraining workflow:

1. source seçimi,
2. license/provenance,
3. extraction,
4. quality filter,
5. language ID,
6. dedup,
7. decontamination,
8. sharded tokenize/pack.

Tek `.bin` dosyası ve RAM listeleri ölçek darboğazıdır; Orkhon için sharded pipeline release gate'tir.

## 10. Architecture

2026 küçük/base default stack:

- decoder-only Transformer,
- RoPE,
- RMSNorm,
- SwiGLU,
- GQA,
- tied embeddings küçük modellerde,
- bf16 CUDA, fp32 CPU/MPS smoke,
- KV-cache generation.

Alternatifler (MoE, SSM, MLA) ancak dense baseline çalıştıktan sonra düşünülmelidir.

## 11. Tokenizer

Tokenizer geri dönüşsüz karardır. Özel token id'leri, vocab boyutu ve merge table pretraining checkpoint'lerine
kilitlenir. Multilingual proje için fertility ölçmeden tokenizer dondurulmaz.

Orkhon kararı: pre-R3 48k multilingual aday, `uint16`, EN + code + Turkish + rune seed corpus,
`<|tool|>` id 8, `<image>` id 9. Gate: Türkçe fertility iyileşmeli, Eski Türkçe rune hedefi geçmeli,
İngilizce gerilememeli.

## 12. Continued pretraining

CPT, SFT ve RAG farklı işlerdir:

- RAG bilgiyi dışarıda tutar ve kaynak gösterir.
- SFT davranış/format öğretir.
- CPT modelin weights'ine domain dağılımı işler.

CPT forgetting riskini English replay, düşük LR ve held-out eval ile yönetir.

## 13. Framework ve tooling

Kullanıcıya dönük framework'ler hızlı ürün çıkarır; Orkhon eğitimsel/denetlenebilir core için kendi modelini
yazar. Dış araçlar post-export veya eval köprüsü olarak kullanılabilir: `transformers`, `lm-eval`, vLLM,
llama.cpp, MLX.

## 14. Hardware ve cloud

Consumer GPU prototip için iyidir. Gerçek pretraining GPU-saat hesabıdır. Spot GPU, exact resume varsa en büyük
maliyet kaldıraçlarından biridir. Daha çok GPU çoğu zaman aynı paraya daha hızlı sonuç verir; node sınırı ve
network verimi hesaba katılmalıdır.

## 15. Multimodal

Perception önce tool olarak başlar: image/audio için mevcut VLM/ASR/TTS çağrılır. Native multimodal model,
encoder + projector + instruction train gerektirir ve scale base hazır olmadan anlamlı değildir.

## 16. Evaluation

"İyi" dört boyuttur:

- intrinsic loss/perplexity,
- benchmark accuracy,
- instruction/format following,
- safety/reliability.

Orkhon için minimum: HellaSwag/ARC smoke, held-out PPL, GSM8K/MBPP pass@k smoke, Turkish eval ve contamination
kontrolü.

## 17. Deployment

Pipeline: checkpoint → HF export → quantization → serving engine → API/demo. Küçük modeller için MLX/llama.cpp,
server için vLLM veya OpenAI-compatible FastAPI düşünülebilir. Quantization deployment kararıdır; training
core'u karmaşıklaştırmamalıdır.

## 18. Legal ve licensing

Ticari kullanımda dataset/model license, attribution, non-commercial kısıtları, GDPR ve EU AI Act gibi kurallar
yeniden kontrol edilmelidir. Göktürk kaynakları için CC BY-NC-SA riski özellikle önemlidir; model card ve data
provenance bunu açık yazmalıdır.

## 19. Tarifler

### Solo dev

RAG + LoRA ile başla; öğrenmek için küçük from-scratch; Orkhon için R3'te dur.

### Startup MVP

Açık base + RAG + SFT; from-scratch yalnız veri moat varsa.

### Domain expert

CPT + SFT + eval + compliance.

### Reasoning on budget

Verifier + rejection sampling + distillation; GRPO'yu en sona bırak.

### Stack'i öğrenmek

Tokenizer, tiny pretrain, SFT, DPO, eval, export. Orkhon tam olarak bu hattı görünür kılar.

## 20. Temel link kümeleri

- Learn: transformer, tokenizer, scaling laws, eval.
- Fine-tune: PEFT, TRL, Tülu/OpenHermes tarzı karışımlar.
- Pretrain: FineWeb, DCLM, HPLT, CulturaX.
- RL/alignment: DPO, SimPO, GRPO, math/code verifier.
- Eval: lm-eval, HellaSwag, ARC, MMLU, GSM8K, MBPP, Turkish eval sets.
- Deploy: HF, vLLM, llama.cpp, MLX.

## Son söz

Sıfırdan pretraining çoğu ürün için yanlış başlangıçtır; ama stack'i öğrenmek, denetlenebilir model üretmek ve
Türkçe/Turkic kimliği olan bağımsız bir araştırma hattı kurmak için doğru projedir. Orkhon bu yüzden küçükten
başlar, ölçer, sonra büyür.

## Sözlük

- **BPE:** byte-pair encoding tokenizer.
- **CPT:** continued pretraining.
- **DPO:** direct preference optimization.
- **Fertility:** dil başına bytes/token veya tokens/rune verimliliği.
- **GQA:** grouped-query attention.
- **GRPO/RLVR:** verifiable reward ile RL.
- **RAG:** retrieval-augmented generation.
- **SFT:** supervised fine-tuning.
