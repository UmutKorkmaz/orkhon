# Releasing Orkhon

A checklist so push-day is account/auth work, not engineering.

## 1. Pre-release verification (local)

```bash
uv sync --extra dev          # clean env
uv run pytest -m "not slow"  # all green
bash scripts/smoke_all.sh    # full pipeline ~1 min on Apple Silicon
```

Confirm:
- `git status` is clean (no stray `runs/`, `exports/`, `*.pt`, `models/*/checkpoint/`).
- The model zoo `models/registry.md` index is current (`orkhon registry`).
- README "Status" test count + feature table match reality (no stale claims).

## 2. Docs drift check

- README feature table reflects shipped capability (training ladder, tools, RAG, agent, eval).
- `docs/agents.md`, `docs/eval.md`, `docs/capability-roadmap.md` don't claim "out of scope" for
  things that now exist.

## 3. Model cards

- Each released checkpoint gets an HF card: `orkhon export hf --checkpoint ... --model-name ...`
  writes `README.md` with YAML front-matter + limitations.
- Mark `tool_token_trained` honestly (current checkpoints predate `<|tool|>` → not native tool-calling).

## 4. Public demo (optional)

- `spaces/orkhon-demo/` deploys to a Hugging Face Space (Gradio SDK).
- Set `ORKHON_CHECKPOINT`/`ORKHON_TOKENIZER` (+ optional `ORKHON_TOOLS`/`ORKHON_RAG_INDEX`).
- **Never** add public code execution without container isolation, no-network policy, and a low-privilege runtime.

## 5. Tag + publish (once GitHub auth is available)

1. Push `main` to GitHub.
2. `git tag v0.X.0 && git push --tags`.
3. Publish each zoo model to the HF Hub (`orkhon publish`, when wired) + datasets where licensed.
4. Announce with the model-card eval table.

## Rollback

Revert the tag; HF model repos are versioned, so unpublish by deleting the offending revision.
