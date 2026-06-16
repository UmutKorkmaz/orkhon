# Orkhon'a Katkı

[English](CONTRIBUTING.md) | [Türkçe](CONTRIBUTING.tr.md)

Orkhon'a ilginiz için teşekkürler. Bu repo sıfırdan yazılmış, elde okunabilir bir LLM yığınıdır; katkılar da
aynı ölçüde küçük, ölçülebilir ve doğrulanabilir olmalıdır.

## Kurulum

```bash
uv sync --extra dev          # çekirdek + test bağımlılıkları
uv sync --extra hub          # + gerçek veri ingest'i ve HF import için datasets/transformers
uv sync --extra demo         # + Space demosu için gradio (opsiyonel)
```

## Geliştirme

- **Testler:** `uv run pytest -m "not slow"` hızlı seti çalıştırır. Uzun uçtan uca koşular `@pytest.mark.slow`
  ile işaretlenmelidir.
- **Tam smoke hattı:** `bash scripts/smoke_all.sh` Apple Silicon üzerinde yaklaşık bir dakika sürer.
- **PR başına tek değişiklik.** İnceleme istemeden önce ilgili testleri çalıştırın.

## Kod stili

- Dosyalar küçük ve odaklı kalmalı; tip ipuçları ve docstring kullanın.
- Yeni eğitim aşamaları ortak engine yardımcılarını yeniden kullanmalı: `build_optimizer`, `lr_at`,
  `save_checkpoint`, `maybe_resume`.
- Yeni eval görevleri `orkhon bench` içine `eval/benchmarks.py` (çoktan seçmeli) veya
  `eval/generative_tasks.py` (pass@k) üzerinden bağlanmalı.
- Mevcut kalıpları izleyin; model çekirdeği bağımlılık açısından hafif kalır, `AutoModel` eklenmez.

## Artifact politikası

**Asla commit'lemeyin:** `runs/`, `exports/`, `.venv/`, `*.pt`, `*.safetensors`,
`models/*/checkpoint/`. `.gitignore` bunları dışlar. Model zoo yalnızca metadata'yı
(kartlar, manifestler, sample'lar, kod snapshot'ları) commit'ler; ağırlıkları değil. Push öncesi
`git status` çalıştırın.

## Veri ve lisans politikası

- FineWeb, TinyStories, Wikipedia-tr, GSM8K ve MBPP gibi public dataset'ler `hub` extra'sı üzerinden akar.
  Yeniden dağıtılan her artifact için ilgili dataset lisansını kontrol edin.
- **Göktürk / Eski Türkçe** yazıt kaynakları (örneğin Uppsala runiform veritabanı) çoğu zaman
  **CC BY-NC-SA** lisanslıdır. İzin almadan bunlarla eğitilmiş ticari weights yayınlamayın. Ayrıntı:
  [`docs/turkic-languages.tr.md`](docs/turkic-languages.tr.md).
- Üretilmiş/sentetik izler (tool-call SFT, FIM, transliterasyon) deterministik ve yeniden üretilebilir olmalıdır.

## Güvenlik notları

- `read_file` aracı açıkça verilen root'lar ister; cwd varsayılan root değildir.
- Bu sürümde tool olarak kod çalıştırma yoktur; MBPP sandbox yalnızca **trusted-local** kullanım içindir.
- HTTP agent, dış mesaj içeriğini escape eder ve inbound `role="tool"` kabul etmez.
