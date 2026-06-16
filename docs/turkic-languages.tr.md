# Orkhon'un Ana Dili — Türkçe ve Göktürk

[English](turkic-languages.md) | [Türkçe](turkic-languages.tr.md)

> Orkhon'a **modern Türkçe** ve projenin adını aldığı Orhun Yazıtları'nın **Göktürk / Eski Türkçe** dilini
> öğretme planı. Tematik olarak en doğru kabiliyet: en eski Türkçe yazının adını taşıyan sistemin o yazıyı
> okumayı öğrenmesi.
>
> **Kesin ifade:** "Göktürk kabiliyeti" script handling, transliterasyon, kaynaklı çeviri/glossing ve attested
> inscription'lar üzerinde scholarly RAG demektir. Akıcı serbest Eski Türkçe üretimi demek değildir.

---

## 1. İki tamamen farklı problem

- **Modern Türkçe = gerçek multilingual training.** Veri bol; bağlayıcı kısıt *token bütçesi* ve *tokenizer'dır*.
  R3 (`tengri`, 350M) için gerçekçi sonuç: Türkçe yüzey biçimi ve morfolojide akıcı, fakat derin bilgi açısından
  sınırlı model. Bunu model card'da açık söyleyin.
- **Göktürk / Eski Türkçe = ultra-low-resource transliterasyon/çeviri kabiliyeti + imza demo.** Hayatta kalan
  primary corpus birkaç yüz satır mertebesindedir; bir dili pretrain etmek için 4-6 büyüklük mertebesi eksiktir.
  Akıcı Old Turkic generator olamaz. Fakat **script ⇄ transliterasyon ⇄ modern Türkçe/İngilizce** asistanı olabilir.

İkisi yalnızca tek sert bağımlılığı paylaşır: geri dönüşsüz tokenizer freeze.

---

## 2. Geri dönüşsüz karar — pre-R3 tokenizer freeze

Byte-level BPE merge table geri dönüşsüzdür. Yeni special token veya değişen merge table, o tokenizer ile
pretrain edilmiş checkpoint'leri geçersiz kılar. Bu yüzden üç ayrı bayrak aslında tek olaydır:

1. [`roadmap.tr.md`](roadmap.tr.md): 16k→32k→48k tokenizer planı.
2. [`capability-roadmap.tr.md`](capability-roadmap.tr.md): `<|tool|>` id 8, `<image>` id 9.
3. Bu doküman: multilingual vocab + Türkçe + Eski Türkçe script coverage.

**Bunların hepsi R3 (`tengri`, 350M) pretrain başlamadan önce tek commit'te dondurulmalı.**

| Karar | Seçim | Neden |
|---|---|---|
| **Vocab size** | **48,000** | Fertility ana hedef; maliyet bir kez ödenir. `otuken` (R4, 1B) `tengri` tokenizer'ını miras alacağı için 1B için boyutlandırılır. |
| **dtype** | **uint16** | `data/tokenize.py` vocab < 65536 için uint16 kullanır; 48k sığar. |
| **Special tokens** | `<\|tool\|>` @ 8, `<image>` @ 9 | append-only; 0-7 id'leri asla yeniden sıralanmaz. |
| **Freeze corpus** | EN + code + TR + Old-Turkic synthetic seed | Tokenizer'ın eğitim corpus'u fertility'yi belirler; TR/code/rune over-sample edilir. |

Canlı 16k tokenizer ölçümü:

| Dil | Şimdi | 48k multilingual hedef |
|---|---:|---:|
| Türkçe | ~1.65 bytes/tok | ≥ 3.5 bytes/tok |
| İngilizce | ~4.4 bytes/tok | regression yok |
| Eski Türkçe runes | ~4.1 tokens/rune | ≤ 1.5 tokens/rune |

`tokenizer/fertility.py` bu hedefleri R3 öncesi go/no-go gate olarak denetler.

---

## 3. Modern Türkçe

### Veri

| Kaynak | Boyut | Kullanım |
|---|---:|---|
| `uonlp/CulturaX` (`tr`) | 64.3B token, 94.2M doc | ana Türkçe web havuzu |
| `HPLT/HPLT2.0_cleaned` (`tur_Latn`) | 51.7B word, 390B char | ikinci ana havuz |
| `oscar-corpus/OSCAR-2301` (`tr`) | 8.29B word | dedup/overlap kontrolü |
| Turkish Wikipedia / news / books | küçük ama temiz | held-out eval + kaliteli replay |
| `malhajar/OpenOrca-tr`, `TFLai/Turkish-Alpaca` | instruction seed | SFT, sıkı quality filter ile |

Dedup/quality/license filtresi sonrası 50-150B kullanılabilir Türkçe token mümkündür. Arz darboğaz değildir;
token bütçesi darboğazdır.

### Mix

| Rung | Türkçe payı |
|---|---:|
| R1 50M | English-only baseline + fertility eval |
| R2 125M | opsiyonel %4-8 Türkçe proof |
| **R3 350M (`tengri`)** | **%5-8 Türkçe** |
| R4 1B (`otuken`) | Türkçe headline ise %8-12 |

Türkçe validation kaynak bazında ayrılmalıdır: `hplt_tr`, `culturax_tr`, `wiki_tr`, `news_tr`, `sft_tr`.

### SFT + eval

Türkçe SFT R3 sonrası gelir. Türkçe pretraining'e girdiğinde blocking eval'ler: kaynak bazlı held-out PPL,
**TR-MMLU**, **TurkBench**, **TurBLiMP**.

---

## 4. Göktürk / Eski Türkçe — imza kabiliyet

### Gerçek

Çekirdek yazıt corpus'u birkaç yüz satırdır. Uppsala *Database of Turkic Runiform Inscriptions* metadata,
transliterasyon ve çeviri sağlar; bu kadar. Bu yüzden Göktürk pretraining target değildir. `tengri` üstünde
küçük bir SFT fine-tune'dur:

```text
Old Turkic script (𐰇𐰏...)  ⇄  Latin transliteration  ⇄  modern Turkish / English gloss
```

### Ulaşılabilir kabiliyetler

- **Script handling:** U+10C00-U+10C4F detect/render; tokenizer bugün lossless temsil eder.
- **Transliterasyon:** Old Turkic script ⇄ Latin.
- **Çeviri/gloss:** attested satırlar → modern Türkçe/İngilizce; kaynak, güven ve belirsizlik alanıyla.
- **Scholarly RAG:** yazıt sayfaları ve notları üzerinde retrieve.
- **Refusal mode:** serbest Göktürk composition reddedilir; model fluent speaker değildir.

### Veri

`data/old_turkic/inscriptions.jsonl` alanları: `source`, `inscription`, `line_id`, `old_turkic_unicode`,
`transliteration_raw`, `transliteration_normalized`, `modern_turkish`, `english`, `notes`, `license`,
`confidence`. Raw satırları 10-50× büyütmek için deterministik script↔translit sentetik veri kullanılır.

### İmza demo

> Bilge Kağan yazıtından bir satır yapıştırın (rune veya transliterasyon). Orkhon normalize transliterasyonu,
> modern Türkçeyi, İngilizceyi, kaynak satırı ve güven skorunu döndürür. Orhun Yazıtları'nın adını taşıyan
> modelin Orhun Yazıtları'nı okuması.

### Model card'a konacak dürüst sınır

> *Old Turkic script'i, transliterasyonu ve attested inscription satırlarının çevirisini işler. Bu, scholarly
> transliteration/translation assistant'tır; akıcı Göktürk konuşuru değildir. Böyle bir training data yoktur.*

### Lisans

Uppsala runiform database **CC BY-NC-SA**. Araştırma/non-commercial için uygun; ticari weights için review veya
izin gerekir. Corpus içinde satır bazlı `license` tutulmalıdır.

---

## 5. İsimli model — `bengü` / *Bengü Taş*

Türkçe + Eski Türkçe uzmanı kendi zoo kod adını alır:

> **`bengü`** — ***Bengü Taş***, yani yazıt anıtlarının kendisi, "ebedi taş". Ebedi taşları okuyan model,
> onların adıyla anılır.

Bu, scale branch'in yanında language branch olarak soyağacına katılır.

---

## 6. Merdivenlerle ilişkisi

| Zaman | Bağlam | Turkic çalışma |
|---|---|---|
| Pre-R3 | `tengri` öncesi freeze | **48k multilingual tokenizer** + TR/rune seed + `<\|tool\|>`/`<image>` |
| R2 125M | proof run | opsiyonel %4-8 Türkçe; fertility eval |
| **R3 350M `tengri`** | MVP base | %5-8 Türkçe mix |
| post-R3 | C6 SFT | Türkçe SFT; sonra `bengü` Old-Turkic fine-tune + RAG demo |
| R4 1B `otuken` | headline base | Türkçe headline ise %8-12 |

Göktürk R3 ve C3 RAG'a bağlıdır. Türkçe yalnız tokenizer freeze'e bağlıdır.

---

## 7. Veri kaynakları ve sıradaki iş

**Veri:** CulturaX-tr · HPLT2-tr · OSCAR-2301-tr · Turkish Wikipedia/Wikisource · OpenOrca-tr · Turkish-Alpaca ·
TR-MMLU/TurkBench/TurBLiMP · Uppsala Runiform DB + Türk Bitig · akademik edisyonlar (Tekin, Erdal, Clauson;
referans olarak, scrape edilmez).

**Yeni / dokunulan dosyalar:**

```text
src/orkhon/tokenizer/special_tokens.py
src/orkhon/tokenizer/fertility.py
configs/tokenizer/multilingual_48k.yaml
src/orkhon/data/hf_text.py
configs/data/fineweb_tr_mix.yaml
configs/train/{pretrain_350m_tr_mix,sft_tr,sft_old_turkic}.yaml
src/orkhon/data/old_turkic/
src/orkhon/eval/domains.py · configs/eval/{tr,old_turkic}.yaml
src/orkhon/registry.py
```

**Sıra:**

1. Tokenizer freeze kararını şimdi ver: 48k, TR + rune-seed corpus, `<|tool|>`/`<image>` token'ları.
2. Türkçe dataset descriptor'larını ekle; küçük dilim indir; eski-yeni fertility ölç.
3. R3 (`tengri`) mix'ine %5-8 Türkçe kat.
4. `data/old_turkic/` paralel corpus + sentetik augmentation kur.
5. R3 sonrası: Türkçe SFT, sonra **`bengü`** Old-Turkic fine-tune ve demo.

---

*Orhun Yazıtları'nın adını taşıyan model, onları okumayı öğreniyor. `bengü` — ebedi taş.*
