# Orkhon

[English](README.md) | [Türkçe](README.tr.md)

![Orkhon kahraman görseli: bir yazıt taşı modern bir sinir ağına bağlanıyor](docs/assets/orkhon-hero.png)

**Orkhon**, ham metinden kendi makinenizde eğitip konuşabileceğiniz sohbet kabiliyetli bir dil modeline kadar
uzanan, uçtan uca ve denetlenebilir, **sıfırdan yazılmış bir LLM yığınıdır**. Tokenizer → ön-eğitim → SFT → DPO
→ eval → servisleme → export hattı, anlaşılabilirlik için PyTorch ile elle yazılmıştır. Tüm hattın çalıştığı,
küçük bir modeli yaklaşık bir dakikada uçtan uca eğiten **Mac/MPS smoke run** ile kanıtlanmıştır.

İsim, 8. yüzyıldaki [**Orhun Yazıtları**](https://tr.wikipedia.org/wiki/Orhun_Yaz%C4%B1tlar%C4%B1)ndan gelir:
Bilge Kağan döneminde dikilen, bilinen en eski Türkçe yazılı metinler. Projenin fikri bilerek doğrudandır:
bir dilin ilk yazılı sözlerinden, dil yazabilen bir sisteme.

> Frontier framework değil. Orkhon; typed config, deterministik seed, kaldığı yerden devam eden eğitim,
> eval kapıları, bağımsız incelemeyle doğrulanmış model çekirdeği, testler, chat CLI ve OpenAI uyumlu API
> içeren küçük, okunabilir bir referans uygulamadır.

## İçindekiler

Elle yazılmış **decoder-only Transformer**: GQA, RoPE, RMSNorm, SwiGLU, KV-cache ve etrafındaki tüm araçlar.

| Aşama | Ne yapar |
|-------|----------|
| **Tokenizer** | Sabit special-token id'leri ve chat şablonu olan byte-level BPE |
| **Data** | Sentetik smoke corpus, normalize → tokenize → pack (`.bin` shard), SFT/DPO formatlayıcıları |
| **Pretrain** | AdamW, cosine+warmup LR, grad-accum, grad-clip, AMP, checkpoint/resume |
| **Post-training** | SFT, DPO, GRPO/RLVR smoke hattı, distillation |
| **Eval** | Perplexity, çoktan seçmeli benchmark'lar, generative pass@k |
| **Tools/RAG/Agent** | Calculator/read-file/retrieve araçları, yerel RAG index'i, sınırlı agent döngüsü |
| **Serve** | `orkhon chat`, `orkhon agent`, `orkhon serve` (OpenAI uyumlu chat + `/v1/agent/run`) |
| **Export** | Reload-parity kontrolü olan HuggingFace tarzı `safetensors` klasörü |

## Hızlı başlangıç

```bash
uv sync --extra dev            # ortamı kurar (PyTorch, tokenizers, ...)
bash scripts/smoke_all.sh      # küçük modelle tüm hat, Apple Silicon'da ~1 dk
```

`smoke_all.sh` tüm yığını uçtan uca çalıştırır: synth data → tokenizer eğitimi → shard hazırlama →
pretrain (600 step) → SFT (200) → DPO → eval → tek chat turu → HF export. Bittiğinde sohbet edebilirsiniz:

```bash
uv run orkhon chat --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke
```

### Gerçek smoke sonucu (Apple M5 Pro, MPS, float32)

Yaklaşık 4M parametreli, sıfırdan eğitilmiş küçük model chat formatını öğrenir ve temiz cevap verir:

```text
you> What is 2 plus 2?
bot> Answer: 4.
you> What is 9 plus 1?
bot> Answer: 10.
```

Pretrain en iyi val loss ≈ 1.17 · SFT loss ≈ 0.29 · perplexity ≈ 6.8 · MPS üzerinde ~48k token/sn.
Smoke corpus küçük ve şablonludur; bu yüzden model öncelikle *formatı* ve durmayı öğrenir. 4M'lik oyuncak
modelden eğitim aralığı dışındaki aritmetiği genellemesi beklenmez; gerçek kabiliyet için config'i büyütün.

## Gerçek metinle gerçek model eğitimi (TinyStories)

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) üzerinde eğitilmiş ~22M model akıcı,
tutarlı İngilizce kısa hikayeler yazar. Apple Silicon'da (MPS) yaklaşık 43 dakika sürer, cloud gerekmez:

```bash
orkhon data download --split train --out data/tinystories/train.txt --max-stories 250000
orkhon tokenizer train --config configs/tokenizer/tinystories.yaml      # 8K-vocab BPE
orkhon data prepare    --config configs/data/tinystories.yaml           # -> 47.4M train tokens
orkhon train pretrain  --config configs/train/pretrain_tinystories.yaml # 4000 steps, ~19.3k tok/s
orkhon generate --checkpoint runs/tinystories --tokenizer artifacts/tokenizer/tinystories \
  -p "One day, a little girl named Mia found a magic paintbrush."
```

Sonuç (val loss **1.55**, held-out perplexity **5.3**):

> One day, a little girl named Mia found a magic paintbrush. She was very excited... Mia's friend, Tom, came
> over to play. He saw the paint and said, *"Wow, Mia! Your paint is so pretty!"* Mia smiled and said, *"Thank
> you, Tom! I love to paint too!"* They both used the magic paint to make their art fun and pretty. They had a
> great day painting together.

## TinyStories sonrası: gerçek veri, açık base modeller ve ölçek

Aynı yığın dört yönde büyür. Laptop'ta çalışan parçalar hazırdır; cloud tarafı da kabloludur ve testlidir.

- **Gerçek ve çeşitli web metni.** `orkhon data download --dataset fineweb-edu --out data/fineweb/train.txt --max-docs N`
  komutu [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) veya herhangi bir HF text
  dataset'ini corpus formatına akıtır. **51M** model (`tiny_50m`) MPS üzerinde eğitilir; TinyStories'in kapalı
  kelime dünyasının dışına çıkar.
- **Instruction tuning.** `orkhon data instruct --corpus ...` SFT + DPO instruction verisi üretir. Story base,
  instruction-following modele dönüşür (`orkhon train sft/dpo` → `orkhon chat "Write a short story about ..."`).
- **Açık base yükleme.** `orkhon import-hf --repo HuggingFaceTB/SmolLM2-135M --out runs/base`, herhangi bir
  **Llama mimarili** modeli (SmolLM2, Llama-3.2, TinyLlama, ...) Orkhon'un elle yazılmış Transformer'ına yükler
  ve `transformers` ile **exact logit parity** sağlar. Sonra Orkhon araçlarıyla fine-tune/serve edebilirsiniz.
- **Multi-GPU + cloud.** DDP/FSDP2 bağlıdır; single-process yol byte-identical kalır. `scripts/train_cloud.sh`
  ve [`docs/scaling.tr.md`](docs/scaling.tr.md), A100/H100 üzerinde 125M-1B için çalıştırılabilir komutlar ve maliyet
  notları verir:
  `torchrun --nproc_per_node=$N -m orkhon train pretrain --config configs/train/pretrain_125m_cloud.yaml`.

## CLI

```bash
orkhon data download   --dataset fineweb-edu --out data/fineweb/train.txt --max-docs 150000  # corpus indir
orkhon data download   --dataset wikipedia-tr --out data/turkish/train.txt --max-docs 150000  # Türkçe
orkhon tokenizer train --config configs/tokenizer/smoke.yaml
orkhon data prepare    --config configs/data/smoke.yaml [--sharded]
orkhon train pretrain  --config configs/train/pretrain_smoke.yaml [--set train.max_steps=2000]
orkhon train sft       --config configs/train/sft_smoke.yaml
orkhon train dpo       --config configs/train/dpo_smoke.yaml
orkhon train grpo      --config configs/train/grpo_copy_digit_stable.yaml   # doğrulanabilir reward ile RL
orkhon rag ingest      README.md docs --out data/rag/repo                    # RAG index oluştur
orkhon rag search      "What is C3?" --index data/rag/repo
orkhon bench           --checkpoint MODEL --tokenizer TOK --task builtin --json-out report.json
orkhon bench           --checkpoint MODEL --tokenizer TOK --task gsm8k --samples 4 --pass-at 4 --temperature 0.8
orkhon generate        --checkpoint MODEL --tokenizer TOK -p "Once upon a time"
orkhon chat            --checkpoint MODEL --tokenizer TOK [--tools calculator] [--rag-index IDX]
orkhon agent           --checkpoint MODEL --tokenizer TOK --tools calculator --rag-index IDX --max-steps 5
orkhon serve           --checkpoint MODEL --tokenizer TOK --tools calculator --rag-index IDX --port 8000
orkhon register        --checkpoint runs/tinystories --tokenizer TOK --kind base --lineage "..."
orkhon export hf       --checkpoint MODEL --out exports/orkhon --tokenizer TOK
orkhon import-hf       --repo HuggingFaceTB/SmolLM2-135M --out runs/smollm2  # açık base yükle
```

`generate` raw text devam ettirir (base model); `chat` chat şablonunu kullanır (SFT/instruct checkpoint).
Her config değeri komut satırında `--set dotted.key=value` ile değiştirilebilir.

## Model adları ve model zoo

![Orkhon model soyağacı: eğitim sinyalleriyle bağlanan, giderek büyüyen yazıt taşları](docs/assets/model-lineage.png)

Eğitilen her model `models/` altında isimli, tarihli ve kendi kendine yeten bir klasöre arşivlenir. Kod adları
rastgele release etiketleri değildir; teknik büyüme merdivenini Türk tarihinden gelen bir hatla anlatır.

- **Temiz mevcut hat:** `bumin-mini`, `tonyuk`, `tegin`, `istem`, `kashgar`, `bunghu`, `tangri`.
- **Birlesik kural:** normal public modellerin tamami Ingilizce, Turkce ve Kokturk/Eski Turkce transliterasyon bilmeli; ayri bir `-gokturk` urun dali yoktur.
- **Siradaki olcek basamagi:** `qaghan`, sonra `otuken`, `balasagun`, `kutadgu`.
- **Frontier rezervi:** `oguz`, `manas`, `tarkan`, `ergenekon`, `atilla`, `timur`, `korkut`.

Tam isim hikayesi [`docs/lineage.tr.md`](docs/lineage.tr.md) içinde.

```bash
orkhon register --checkpoint runs/tinystories --tokenizer artifacts/tokenizer/tinystories \
  --kind base --mode complete --lineage "22M TinyStories base" --eval-prepared data/prepared/tinystories
orkhon registry        # index'i yeniden üretir ve yazdırır
```

Her `models/<name>-<YYYYMMDD>/` klasörü **weights** (`checkpoint/`, `run.sh` ile çalışır), `tokenizer/`,
üretilmiş **outputs** (`samples.txt`), `eval.json`, üretimi yapan `src/` + config snapshot'ı
(`code_snapshot.tgz`), `model_card.md` ve `manifest.json` içerir. `next_name`, bir sonraki boş kod adını verir.

| İsim | Kaynak isim | Nedir | Metrik / durum |
|------|-------------|-------|----------------|
| `bumin-mini` | Bumin Kagan, Birinci Gokturk Kaganligi'nin kurucusu | 4M kompakt birlesik asistan | backport hedefi |
| `tonyuk` | Tonyukuk, Orhun donemi stratejisti ve yazit sahibi | hikaye base'inden 22M birlesik asistan | backport hedefi |
| `tegin` | Kul Tigin, Orhun yazitlarinda anilan Gokturk prensi | eski story-instruct modelinden 22M birlesik asistan | backport hedefi |
| `istem` | Istemi Kagan, batiya acilan kurucu ortak | FineWeb-Edu base'inden 51M birlesik asistan | backport hedefi |
| `kashgar` | Mahmud el-Kashgari, ilk buyuk Turk dili sozlugunun yazari | imported/open-base slot | weights bekliyor |
| `bunghu` | *Bengu Tas*, ebedi yazit tasi icin ASCII urun yazimi | bilingual hattan 57M EN/TR/Kokturk asistan | backport hedefi |
| `tangri` | Tengri/Tangri, gok olcegi basamagi | mixed base'den 100M EN/TR/Kokturk asistan | egitim/eval hedefi |

## Model boyutları

| Config | Parametre | Katman · d_model · head (kv) | Context | Hedef |
|--------|-----------|------------------------------|---------|-------|
| `smoke_6m`  | ~4-6M | 4 · 256 · 4 (2)  | 256  | Mac/CPU/MPS smoke run |
| `tiny_24m`  | ~22M  | 6 · 512 · 8 (4)  | 512  | **MPS üzerinde gerçek metin** (TinyStories) |
| `tiny_50m`  | ~51M  | 8 · 640 · 10 (5) | 512  | MPS üzerinde gerçek web metni (FineWeb-Edu) |
| `small_125m`| ~125M | 12 · 768 · 12 (4) | 1024 | Tek GPU (4090/A100) |
| `base_350m` | ~350M | 24 · 1024 · 16 (4) | 2048 | Ciddi tek GPU koşusu |
| `orkhon_1b` | ~1B   | 24 · 2048 · 16 (4) | 2048 | Multi-GPU / FSDP2 |

Tüm boyutlar aynı kod yolunu kullanır. Smoke varsayılanı MPS/CPU üzerinde float32, ölçek config'leri CUDA'da bf16'dır.

## Dizin yapısı

```text
src/orkhon/
  config/      typed config + --set override destekli YAML loader
  tokenizer/   byte-level BPE, sabit specials, chat şablonu, fertility gate
  data/        download, normalize, tokenize, shard, pack, synth, Old Turkic araçları
  model/       RoPE · RMSNorm · GQA attention · SwiGLU · KV-cache · generation
  train/       pretrain · SFT · DPO · GRPO · distillation · checkpoint/resume
  eval/        perplexity · loglikelihood benchmark · generative pass@k · reports
  rag/         ingest · chunk · embed · store · retrieve
  serve/       chat CLI · tool loop · bounded agent · OpenAI-compatible API
  export/      HuggingFace safetensors export + reload parity
configs/   model/ + train/ + tokenizer/ + data/ preset'leri
models/    isimli zoo arşivleri (weights, tokenizer, samples, cards, manifests)
reports/   benchmark ve tokenizer-fertility JSON raporları
tests/     260 test (model core, data, eval, tools, RAG, agent, export, resume)
docs/      roadmaps, lineage, eval, Turkic-language plan, scaling guide
```

## Test ve doğrulama

```bash
UV_CACHE_DIR=/tmp/uv-cache-bilge uv run pytest -q
make smoke                      # tam uçtan uca hat
```

Doğruluk açısından kritik model çekirdeği, klasik LLM hatalarına karşı bağımsız reviewer'lar tarafından
adversarial olarak incelendi: RoPE cached-decode offset, GQA repeat, SFT label masking, DPO objective,
grad-accum, exact resume ve perplexity masking. Bu inceleme cached decoding içinde gerçek bir RoPE position
bug'ı buldu ve düzeltti; artık `tests/test_kv_cache_rope_regression.py` ile korunuyor.

## Karar rehberi

Tasarım [`docs/build-your-own-llm-guide.tr.md`](docs/build-your-own-llm-guide.tr.md) üzerine kuruludur: RAG'den
frontier pretraining'e kadar yolları karşılaştıran 2026 karar rehberi. Uygulama planı
[`docs/implementation-plan.tr.md`](docs/implementation-plan.tr.md) içindedir. İki roadmap geleceği ayırır:
[`docs/roadmap.tr.md`](docs/roadmap.tr.md) **ölçek ekseni**dir (51M'den 7B'ye R0→R6 merdiveni, kod boşlukları,
GPU ekonomisi); [`docs/capability-roadmap.tr.md`](docs/capability-roadmap.tr.md) **kabiliyet ekseni**dir (C0→C8:
tools, RAG, web search, agents, native vision). [`docs/turkic-languages.tr.md`](docs/turkic-languages.tr.md),
**Türkçe + Göktürk/Eski Türkçe** hattını anlatır: Orhun yazıtlarının adını taşıyan modelin onları okumayı
öğrenmesi (`bengü` dalı). Ek araştırma maddeleri [`docs/research-additions.tr.md`](docs/research-additions.tr.md)
içinde sıralanmıştır.

## Durum

- ✅ Uçtan uca hat Apple Silicon (MPS) ve CPU üzerinde çalışır, tutarlı modeller eğitir.
- ✅ **260 test geçiyor**; model core adversarial olarak doğrulandı; tool/RAG/agent/HTTP katmanları güvenlik incelemesinden geçti.
- ✅ Tam eğitim merdiveni: pretrain → SFT → DPO → **GRPO/RLVR** (öğrendiği kanıtlandı) → distillation.
- ✅ Agent zemini: tools (calculator/read_file) + **RAG** (ingest/retrieve) + bounded agent loop'u (CLI + HTTP `/v1/agent/run`).
- ✅ Çift eval: loglikelihood çoktan seçmeli benchmark (HellaSwag/ARC) **ve generative pass@k** (GSM8K/MBPP sandbox).
- ✅ Türkçe + Göktürk/Eski Türkçe scaffolding: Türkçe data hattı + Eski Türkçe transliterasyon ve demo fine-tune.
- 🚧 Göktürk hedefi kaynaklı transliterasyon/çeviri/RAG'dir; serbest, akıcı dil üretimi iddiası değildir.
- 🚧 Ölçek config'leri (125M-1B) sağlanmış ve doğrulanmıştır; çalıştırmak GPU/bütçe gerektirir.
- 🚧 v1 kapsamı dışı: MoE/MLA, native multimodal (vision), multi-agent orchestration.
- ⚠️ Not: mevcut checkpoint'ler `<|tool|>` token'ından önce üretildi; tool use, tool-trained checkpoint'e kadar loop/prompt tabanlıdır (C6).

## Lisans

Apache-2.0. Karar rehberi araştırma sentezidir, hukuki tavsiye değildir; production kullanımdan önce model
ID'lerini, lisansları ve maliyetleri yeniden doğrulayın.
