# Orkhon Reports

Durable local reports produced before any further scale spend.

## Benchmark Scoreboard

Current reports are smoke baselines, not headline benchmark claims:

| Model | Builtin acc / acc_norm | HellaSwag limit-20 acc / acc_norm | ARC-Easy limit-20 acc / acc_norm |
|---|---:|---:|---:|
| `istemi` | 0.375 / 0.500 | 0.350 / 0.300 | 0.300 / 0.200 |
| `bengü` | 0.625 / 0.625 | 0.300 / 0.350 | 0.300 / 0.300 |
| `bengü-göktürk` | 0.500 / 0.250 | 0.250 / 0.300 | 0.200 / 0.250 |
| `kashgari` | pending | pending | pending |

`kashgari` is pending because the model-zoo archive stores metadata and a re-import
hint, not local checkpoint/tokenizer files. Re-import it before scoring.

The Hub-backed runs were executed unauthenticated; use `HF_TOKEN` before promoting
these to 500-example or full benchmark reports.

## Tokenizer Fertility

| Tokenizer | English bytes/token | Turkish bytes/token | Old Turkic tokens/rune | Gate |
|---|---:|---:|---:|---|
| `fineweb` | 4.250 | 1.904 | 4.208 | fail |
| `bilingual` vs `fineweb` | 3.825 | 4.788 | 4.042 | fail |

The bilingual tokenizer fixes Turkish fertility but still fails the rune target and
regresses English versus the FineWeb tokenizer. The pre-R3 48k candidate remains
blocked until it passes the strict fertility gate.
