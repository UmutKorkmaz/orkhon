# Contributing to Orkhon

Thanks for your interest in Orkhon — a from-scratch, hand-written LLM stack.

## Setup

```bash
uv sync --extra dev          # core + test deps
uv sync --extra hub          # + datasets/transformers for real-data ingestion & HF import
uv sync --extra demo         # + gradio for the Space demo (optional)
```

## Develop

- **Tests:** `uv run pytest -m "not slow"` (fast, ~2s). Mark long end-to-end runs `@pytest.mark.slow`.
- **Full pipeline smoke:** `bash scripts/smoke_all.sh` (~1 min on Apple Silicon).
- **One change per PR.** Run the suite before requesting review.

## Code style

- Small, focused files (<400 lines). Type hints + docstrings.
- New training stages reuse the shared engine helpers (`build_optimizer`, `lr_at`, `save_checkpoint`, `maybe_resume`).
- New eval tasks plug into `orkhon bench` via `eval/benchmarks.py` (MC) or `eval/generative_tasks.py` (pass@k).
- Follow the existing patterns; the model core stays dependency-light (no `AutoModel`).

## Artifact policy

**Never commit** `runs/`, `exports/`, `.venv/`, `*.pt`, `*.safetensors`, or `models/*/checkpoint/`.
The `.gitignore` already excludes these. The model zoo commits only metadata
(cards/manifests/samples/code snapshots), never weights. Run `git status` before pushing.

## Data & license policy

- Public datasets (FineWeb, TinyStories, Wikipedia-tr, GSM8K, MBPP) stream via the `hub` extra —
  respect each dataset's license for any redistributed artifact.
- **Göktürk / Old Turkic** inscription sources (e.g. the Uppsala runiform database) are often
  **CC BY-NC-SA** — do not ship commercial weights trained on them without permission. See
  [`docs/turkic-languages.md`](docs/turkic-languages.md).
- Generated/synthetic traces (tool-call SFT, FIM, transliteration) are deterministic + reproducible.

## Safety notes for contributors

- `read_file` tool requires explicit roots (never the cwd by default).
- Code execution as a tool is not implemented in this release; the MBPP sandbox is **trusted-local only**.
- The HTTP agent escapes all external message content and rejects inbound `role="tool"`.
