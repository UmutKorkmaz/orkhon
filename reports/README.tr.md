# Orkhon Raporları

[English](README.md) | [Türkçe](README.tr.md)

Yeni ölçek harcaması yapılmadan önce üretilen kalıcı yerel raporlar.

## Benchmark scoreboard

Mevcut raporlar smoke baseline'dır; headline benchmark iddiası değildir.

| Model | Builtin acc / acc_norm | HellaSwag limit-20 acc / acc_norm | ARC-Easy limit-20 acc / acc_norm |
|---|---:|---:|---:|
| `istemi` | 0.375 / 0.500 | 0.350 / 0.300 | 0.300 / 0.200 |
| `bengü` | 0.625 / 0.625 | 0.300 / 0.350 | 0.300 / 0.300 |
| `bengü-göktürk` | 0.500 / 0.250 | 0.250 / 0.300 | 0.200 / 0.250 |
| `kashgari` | bekliyor | bekliyor | bekliyor |

`kashgari` bekliyor çünkü model-zoo arşivinde yerel checkpoint/tokenizer değil, metadata ve yeniden import ipucu
var. Skorlamadan önce yeniden import edilmelidir.

Hub tabanlı koşular auth olmadan çalıştırıldı. Bunları 500 örnekli veya tam benchmark raporuna terfi ettirmeden
önce `HF_TOKEN` kullanın.

## Tokenizer fertility

| Tokenizer | İngilizce bytes/token | Türkçe bytes/token | Eski Türkçe tokens/rune | Gate |
|---|---:|---:|---:|---|
| `fineweb` | 4.250 | 1.904 | 4.208 | fail |
| `bilingual` vs `fineweb` | 3.825 | 4.788 | 4.042 | fail |

Bilingual tokenizer Türkçe fertility sorununu düzeltir, fakat rune hedefini hâlâ kaçırır ve FineWeb tokenizer'a
göre İngilizceyi geriletir. Pre-R3 48k aday, sıkı fertility gate'i geçene kadar blokludur.
