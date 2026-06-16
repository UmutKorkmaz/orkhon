# Evaluation

Orkhon has two eval surfaces, both under `orkhon bench`.

## 1. Loglikelihood multiple-choice (intrinsic + benchmarks)

Scores HellaSwag/ARC/MMLU-style tasks the standard way: the model's `log P(choice | context)`,
pick the max. Reports `acc` and `acc_norm` (byte-length-normalized).

```bash
orkhon bench --checkpoint runs/sft_smoke --tokenizer artifacts/tokenizer/smoke --task builtin
orkhon bench --checkpoint MODEL --tokenizer TOK --task hellaswag --limit 500
orkhon bench --checkpoint MODEL --tokenizer TOK --task arc_easy  --limit 500
```

Tasks: `builtin` (EN+TR common-sense, offline), `hellaswag`, `arc_easy`. Adapters in
`eval/benchmarks.py`; the engine is `eval/loglikelihood.py` (no `lm-eval` dependency).

## 2. Generative pass@k (reasoning + code)

Samples `n` free-form completions per problem and grades each with a deterministic grader. Reports
`pass@1`, `pass@k`, `mean_reward`.

```bash
# GSM8K (math) — graded by the verifiable math_reward (boxed/trailing-number equality)
orkhon bench --checkpoint MODEL --tokenizer TOK --task gsm8k --samples 4 --pass-at 4 \
  --temperature 0.8 --max-new-tokens 256

# MBPP (code) — graded by the sandbox (run + assert tests)
orkhon bench --checkpoint MODEL --tokenizer TOK --task mbpp --samples 3 --pass-at 3 \
  --temperature 0.8 --max-new-tokens 256
```

- `--samples N --pass-at K`: pass@k needs **diverse** sampling, so set `--temperature > 0`
  (the CLI warns + auto-bumps when `samples>1` with greedy).
- `--fixture`: use the tiny offline fixture (no dataset download) for smoke/CI.
- The MBPP sandbox (`eval/code_sandbox.py`) is **trusted-local only** — see [`SECURITY.md`](../SECURITY.md).

## Held-out perplexity

```bash
orkhon eval --checkpoint MODEL --tokenizer TOK --prepared data/prepared/CORPUS --max-batches 50
```

## Reproducibility

Every `orkhon bench` run prints a metrics table; pipe the JSON form to a file for reports. The
`--fixture` flag makes runs deterministic and offline. Decontamination (`orkhon data curate --ngram 13`)
keeps benchmark text out of the training corpus.
