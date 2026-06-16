#!/usr/bin/env bash
# End-to-end smoke test of the full Orkhon pipeline on the tiny smoke configs.
#
# Runs: synth data -> train tokenizer -> prepare data -> pretrain -> sft -> dpo
#       -> eval -> one chat turn -> export hf
#
# Must be run from the repo root. Exits nonzero on any failure (set -euo pipefail).
set -euo pipefail

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Use the project venv python explicitly; never bare `python`.
PY="${ROOT_DIR}/.venv/bin/python"
ORKHON="${PY} -m orkhon"

TOKENIZER_DIR="artifacts/tokenizer/smoke"
PRETRAIN_DIR="runs/pretrain_smoke"
SFT_DIR="runs/sft_smoke"
DPO_DIR="runs/dpo_smoke"
EXPORT_DIR="exports/orkhon_smoke"

# Faithful end-to-end run on the full smoke configs (~1 min total on Apple
# Silicon): pretrain 600 + SFT 200 + DPO 6 steps. This produces a model that
# actually answers in the trained "Answer: N." format, so the chat step is
# coherent rather than noise. For an ultra-fast plumbing-only check, append e.g.
#   --set train.max_steps=20   to each stage below.
PRETRAIN_OVERRIDES=""
SFT_OVERRIDES=""
DPO_OVERRIDES=""

banner() { echo; echo "==================== $* ===================="; echo; }

# Fresh, reproducible run: clear prior outputs so a stage never auto-resumes a
# stale (e.g. config-changed) checkpoint from its out_dir.
banner "0/9 clean prior smoke outputs"
rm -rf "${PRETRAIN_DIR}" "${SFT_DIR}" "${DPO_DIR}" "${EXPORT_DIR}"

banner "1/9 synth smoke datasets"
${ORKHON} data synth

banner "2/9 train tokenizer"
${ORKHON} tokenizer train --config configs/tokenizer/smoke.yaml

banner "3/9 prepare pretraining shards"
${ORKHON} data prepare --config configs/data/smoke.yaml

banner "4/9 pretrain"
${ORKHON} train pretrain --config configs/train/pretrain_smoke.yaml ${PRETRAIN_OVERRIDES}

banner "5/9 supervised fine-tuning"
${ORKHON} train sft --config configs/train/sft_smoke.yaml ${SFT_OVERRIDES}

banner "6/9 direct preference optimization"
${ORKHON} train dpo --config configs/train/dpo_smoke.yaml ${DPO_OVERRIDES}

banner "7/9 evaluate perplexity"
${ORKHON} eval --checkpoint "${SFT_DIR}" --tokenizer "${TOKENIZER_DIR}" \
  --prepared data/prepared/smoke --max-batches 4 --batch-size 4 --seq-len 64

banner "8/9 one chat turn"
# Feed a single user turn then EOF so the REPL produces one reply and exits.
printf 'What is 2 plus 2?\n/exit\n' | ${ORKHON} chat \
  --checkpoint "${SFT_DIR}" --tokenizer "${TOKENIZER_DIR}" --max-new-tokens 16

banner "9/9 export to HuggingFace format"
${ORKHON} export hf --checkpoint "${DPO_DIR}" --out "${EXPORT_DIR}" --tokenizer "${TOKENIZER_DIR}"

banner "SMOKE PIPELINE COMPLETE"
echo "tokenizer : ${TOKENIZER_DIR}"
echo "pretrain  : ${PRETRAIN_DIR}"
echo "sft       : ${SFT_DIR}"
echo "dpo       : ${DPO_DIR}"
echo "export    : ${EXPORT_DIR}"
