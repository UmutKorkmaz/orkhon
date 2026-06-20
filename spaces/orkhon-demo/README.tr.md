# Orkhon model ailesi demosu

[English](README.md) | [Turkce](README.tr.md)

Orkhon model ailesi icin Gradio Space. Uygulama
`scripts/prepare_hf_family.py` ile hazirlanan HF model repolarini yukler:

- `orkhon-tangri`
- `orkhon-bunghu`
- `orkhon-tegin`
- `orkhon-tonyuk`
- `orkhon-istem`
- `orkhon-bumin-mini`

`tangri` varsayilan demodur. Normal public uyelerin tamami birlesik asistandir:
Ingilizce, Turkce ve Kokturk/Eski Turkce rune -> Latin transliterasyon ayni
model hattindadir; ayri bir `-gokturk` dali yoktur.

## Space ayarlari

| Ayar | Deger |
| --- | --- |
| SDK | Gradio |
| Donanim | Ilk demo icin CPU basic yeterlidir; gecikme kotuyse yukseltin |
| App file | `app.py` |

## Ortam

| Degisken | Zorunlu | Anlam |
| --- | --- | --- |
| `ORKHON_HF_OWNER` | hayir | Model repolari icin HF kullanici/org; varsayilan `korkmazumut` |
| `ORKHON_DEVICE` | hayir | Varsayilan `cpu`; GPU donanimda `cuda` kullanilabilir |
| `ORKHON_FAMILY_LOCAL_ROOT` | sadece yerel | Hazirlanan export kok dizini, orn. `exports/huggingface` |

## Yerel test

Once aile exportlarini hazirlayin:

```bash
uv run python scripts/prepare_hf_family.py --owner korkmazumut
```

Sonra Space'i yerelde bu exportlar uzerinden calistirin:

```bash
ORKHON_FAMILY_LOCAL_ROOT=exports/huggingface \
  uv run --extra demo python spaces/orkhon-demo/app.py
```

## Public guvenlik notu

Public Space uzerinde kod calistirma araclarini acmayin. Bu demo yalnizca export
edilmis Orkhon agirliklariyla text generation ve deterministik transliterasyon
calistirir.
