# Eski Türkçe (Göktürk) verisi

[English](README.md) | [Türkçe](README.tr.md)

Orkhon için gerçekçi Göktürk kabiliyeti **transliterasyon / kaynaklı çeviri asistanlığıdır**; akıcı konuşan bir
Eski Türkçe modeli değildir. Orhun + Yenisey + Tonyukuk yazıtlarından oluşan attested corpus yalnızca birkaç yüz
satırdır. Ayrıntı için [`docs/turkic-languages.tr.md`](../../docs/turkic-languages.tr.md).

## Burada ne var

- `translit_sft.jsonl` — sentetik **rune → Latin transliterasyon** SFT örnekleri. Üretici:
  `orkhon.data.old_turkic.synth.make_translit_sft`. Rune↔Latin eşlemesi Unicode Old Turkic block
  U+10C00-U+10C4F üzerinden doğru kabul edilir; her örnek deterministik ve correct-by-construction'dır.
  Bu veri scholarship gerektirmeden *script* öğretir.

## Bilerek burada olmayan şey

- **Çeviri** çiftleri (Eski Türkçe → modern Türkçe / İngilizce). Bunlar gerçek akademik attestation ve lisans ister:
  - [Uppsala runiform database](https://www.runiform.lingfil.uu.se/) (CC BY-NC-SA; ticari weights için inceleme gerekir),
  - kritik edisyonlar: Talât Tekin, Marcel Erdal, Clauson sözlüğü.

Bu veri `inscriptions.jsonl` içine şu alanlarla eklenmelidir: `inscription`, `line_id`, `old_turkic_unicode`,
`transliteration`, `modern_turkish`, `english`, `notes`, `license`, `confidence`. Model asla uydurulmuş anlamlar
üzerinde eğitilmez.

## Sonraki adım

`bengü` Türkçe base eğitimi bittikten sonra `translit_sft.jsonl` ile SFT edilir. Bu, Eski Türkçe runeleri okuyan
bir model verir. Çeviri daha sonraki adımdır ve yalnızca kaynaklı veriyle açılır.
