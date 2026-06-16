# Orkhon model zoo

[English](registry.md) | [Türkçe](registry.tr.md)

İsimli, tarihli ve kendi kendine yeten model arşivleri. Her klasörde yerelde saklanıyorsa weights, tokenizer,
sample'lar, eval, kod snapshot'ı ve `run.sh` bulunur. Import edilmiş modeller yalnızca kaynak repo ve yeniden
import komutu tutabilir. Kod adlarının hikayesi [`docs/lineage.tr.md`](../docs/lineage.tr.md) içinde anlatılır.

| Tarih | İsim | Tür | Parametre | Metrik | Klasör |
|------|------|-----|-----------|--------|--------|
| 20260615 | **bengu** | base | 57M | ppl 47.102 | `bengu-20260615/` |
| 20260615 | **bengu-gokturk** | instruct | 57M |  | `bengu-gokturk-20260615/` |
| 20260614 | **bumin** | instruct | 4M | ppl 5.128 | `bumin-20260614/` |
| 20260614 | **istemi** | base | 51M | ppl 46.506 | `istemi-20260614/` |
| 20260614 | **kashgari** | imported | 135M |  | `kashgari-20260614/` |
| 20260614 | **kultigin** | instruct | 22M |  | `kultigin-20260614/` |
| 20260614 | **tonyukuk** | base | 22M | ppl 4.769 | `tonyukuk-20260614/` |
