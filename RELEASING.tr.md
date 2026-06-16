# Orkhon Release Süreci

[English](RELEASING.md) | [Türkçe](RELEASING.tr.md)

Bu kontrol listesi, release gününün mühendislik değil hesap/auth ve doğrulama işi olmasını sağlar.

## 1. Release öncesi yerel doğrulama

```bash
uv sync --extra dev          # temiz ortam
uv run pytest -m "not slow"  # tüm hızlı testler yeşil
bash scripts/smoke_all.sh    # Apple Silicon'da tam hat, ~1 dk
```

Kontrol edin:

- `git status` temiz; yanlışlıkla `runs/`, `exports/`, `*.pt`, `models/*/checkpoint/` yok.
- Model zoo index'i güncel: `models/registry.md` / `models/registry.tr.md` için `orkhon registry`.
- README "Durum" test sayısı ve özellik tablosu gerçek repo durumuyla uyumlu.

## 2. Doküman drift kontrolü

- README feature tablosu gerçekten ship edilen kabiliyeti anlatıyor: eğitim merdiveni, tools, RAG, agent, eval.
- [`docs/agents.tr.md`](docs/agents.tr.md), [`docs/eval.tr.md`](docs/eval.tr.md) ve
  [`docs/capability-roadmap.tr.md`](docs/capability-roadmap.tr.md) artık var olan şeyleri "out of scope"
  diye göstermiyor.

## 3. Model kartları

- Her yayınlanan checkpoint için HF kartı üretin:
  `orkhon export hf --checkpoint ... --model-name ...` YAML front-matter ve sınırlamalar içeren `README.md`
  yazar.
- `tool_token_trained` alanını dürüst işaretleyin. Mevcut checkpoint'ler `<|tool|>` öncesinden gelir; native
  tool-calling iddiası yoktur.

## 4. Public demo (opsiyonel)

- `spaces/orkhon-demo/` Hugging Face Space'e deploy edilir (Gradio SDK).
- `ORKHON_CHECKPOINT`/`ORKHON_TOKENIZER` ve opsiyonel `ORKHON_TOOLS`/`ORKHON_RAG_INDEX` ayarlanır.
- Container isolation, network kapatma ve düşük yetkili runtime olmadan public code execution eklemeyin.

## 5. Tag + publish

1. `main` GitHub'a push edilir.
2. `git tag v0.X.0 && git push --tags`.
3. Lisans uygunsa her zoo modeli HF Hub'a (`orkhon publish`, bağlandığında) ve dataset'ler yayınlanır.
4. Duyuru model-card eval tablosuyla yapılır.

## Rollback

Tag'i revert edin. HF model repoları versioned olduğu için sorunlu revision silinerek geri alınır.
