# Evaluation

[English](eval.md) | [Türkçe](eval.tr.md)

Orkhon'un iki eval yüzeyi vardır; ikisi de `orkhon bench` altındadır.

## 1. Loglikelihood çoktan seçmeli

HellaSwag/ARC/MMLU tarzı görevleri standart yöntemle skorlar: modelin `log P(choice | context)` değerini
hesaplar ve en yüksek seçeneği seçer. `acc` ve byte-length-normalized `acc_norm` raporlanır.

```bash
orkhon bench --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke --task builtin
orkhon bench --checkpoint MODEL --tokenizer TOK --task hellaswag --limit 500
orkhon bench --checkpoint MODEL --tokenizer TOK --task arc_easy  --limit 500
```

Görevler: `builtin` (offline EN+TR common-sense), `hellaswag`, `arc_easy`. Adapter'lar
`eval/benchmarks.py` içindedir; engine `eval/loglikelihood.py` dosyasındadır ve `lm-eval` bağımlılığı yoktur.

## 2. Generative pass@k

Her problem için `n` serbest completion örnekler ve her birini deterministik grader ile değerlendirir.
`pass@1`, `pass@k` ve `mean_reward` raporlanır.

```bash
# GSM8K (math) — verifiable math_reward ile skorlanır
orkhon bench --checkpoint MODEL --tokenizer TOK --task gsm8k --samples 4 --pass-at 4 \
  --temperature 0.8 --max-new-tokens 256

# MBPP (code) — sandbox içinde run + assert testleriyle skorlanır
orkhon bench --checkpoint MODEL --tokenizer TOK --task mbpp --samples 3 --pass-at 3 \
  --temperature 0.8 --max-new-tokens 256
```

- `--samples N --pass-at K`: pass@k çeşitli sampling ister; `--temperature > 0` verin. CLI, `samples>1`
  ve greedy sampling birlikte kullanılırsa uyarır ve sıcaklığı yükseltir.
- `--fixture`: smoke/CI için küçük offline fixture kullanır.
- MBPP sandbox (`eval/code_sandbox.py`) **trusted-local only** kabul edilir; bkz.
  [`SECURITY.tr.md`](../SECURITY.tr.md).

## Held-out perplexity

```bash
orkhon eval --checkpoint MODEL --tokenizer TOK --prepared data/prepared/CORPUS --max-batches 50
```

## Yeniden üretilebilirlik

Her `orkhon bench` koşusu metrics tablosu basar; rapor için JSON çıktısını dosyaya yönlendirin. `--fixture`
koşuları deterministik ve offline yapar. Decontamination (`orkhon data curate --ngram 13`) benchmark metnini
training corpus dışında tutar.
