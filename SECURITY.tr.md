# Güvenlik Politikası

[English](SECURITY.md) | [Türkçe](SECURITY.tr.md)

## Desteklenen sürümler

Güvenlik düzeltmeleri yalnızca en güncel `main` branch'i için desteklenir.

## Zafiyet bildirme

Güvenlik problemleri için public issue açmayın. GitHub Security Advisory yapılandırılana kadar maintainer'a
özel olarak bildirin. Lütfen minimal repro, etkilenen dosya ve etki açıklamasını ekleyin.

## Threat model ve sınırlar

Orkhon **local-first araştırma/eğitim yığınıdır**; hardened multi-tenant servis değildir.

| Yüzey | Durum | Şunlar için güvenmeyin |
|---|---|---|
| **MBPP sandbox** (`eval/code_sandbox.py`) | **Trusted-local only.** Basit `os`/`socket`/`open` çağrılarını import allowlist + rlimit ile engeller, fakat object-graph hileleriyle aşılabilir. | Güvenilmeyen model üretimi kod. Bunun için gerçek container (nsjail/firejail/seccomp), network kapalı, read-only FS ve düşük yetkili kullanıcı gerekir. |
| **`read_file` tool** | Açıkça verilen `--file-root` içine hapsedilir; cwd varsayılan değildir. | Verilmeyen root dışındaki dosyaları okuma. |
| **`/v1/agent/run` HTTP** | Dış mesaj içeriğini escape eder; `<\|end\|>`/`<\|system\|>` injection'ını nötralize eder; inbound `role="tool"` reddedilir. | Auth + reverse proxy olmadan public internete açma. Varsayılan gibi `127.0.0.1`'e bind edin. |
| **RAG retrieved text** | Güvenilmeyen veri olarak ele alınır ve yeniden encode edilmeden escape edilir. | Kötü niyetli indexed doküman yine veridir; gelecekteki code-execution tool'larını güvensiz index'lerden ayrı tutun. |

## Deployment için sert kurallar

1. `orkhon serve` internet'e doğrudan ve auth olmadan açılmaz.
2. Gerçek container isolation olmadan public code execution eklenmez.
3. Her tool/RAG index içeriği adversarial input kabul edilir.
