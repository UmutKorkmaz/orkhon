#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Turnkey FineWeb-Edu pretraining on a fresh cloud GPU box.
#
# From a clean Ubuntu + NVIDIA-driver instance to a training 125M model:
#   1. install uv (fast Python package manager)
#   2. sync the project with the `hub` extra (datasets + huggingface-hub)
#   3. download FineWeb-Edu  -> data/fineweb/corpus.txt
#   4. train a byte-level BPE tokenizer -> artifacts/tokenizer/fineweb
#   5. pack the corpus into train.bin / val.bin -> data/prepared/fineweb
#   6. launch multi-GPU pretraining with torchrun (DDP by default)
#
# Usage:
#   NUM_GPUS=8 bash scripts/train_cloud.sh                 # 125M, 8 GPUs, DDP
#   NUM_GPUS=1 CONFIG=configs/train/pretrain_50m.yaml \
#     MODEL=tiny_50m bash scripts/train_cloud.sh           # 50M, 1 GPU
#   NUM_GPUS=8 DIST_MODE=fsdp bash scripts/train_cloud.sh  # FSDP sharding
#
# Re-running is safe: each stage skips work whose output already exists.
# -----------------------------------------------------------------------------
set -euo pipefail

# --- Tunables (override via environment) ------------------------------------
NUM_GPUS="${NUM_GPUS:-1}"                                   # GPUs to use
CONFIG="${CONFIG:-configs/train/pretrain_125m_cloud.yaml}"  # train YAML
DIST_MODE="${DIST_MODE:-ddp}"                               # ddp | fsdp | none
DATASET="${DATASET:-fineweb-edu}"                           # corpus to fetch
MAX_DOCS="${MAX_DOCS:-2000000}"                             # cap on docs fetched
VOCAB_SIZE="${VOCAB_SIZE:-32768}"                           # tokenizer vocab
VAL_FRACTION="${VAL_FRACTION:-0.005}"                       # held-out fraction

CORPUS="data/fineweb/corpus.txt"
TOKENIZER_DIR="artifacts/tokenizer/fineweb"
PREPARED_DIR="data/prepared/fineweb"

# Resolve repo root from this script's location so paths work from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- 1. install uv ----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"

# --- 2. sync the project (with real-data + hub extras) ----------------------
log "Syncing dependencies (uv sync --extra hub)"
uv sync --extra hub

# Run everything else through the project venv.
PY="uv run python"

# --- 3. download FineWeb-Edu ------------------------------------------------
if [[ -s "$CORPUS" ]]; then
  log "Corpus already present at $CORPUS (skipping download)"
else
  log "Downloading $DATASET -> $CORPUS"
  mkdir -p "$(dirname "$CORPUS")"
  $PY -m orkhon data download \
    --dataset "$DATASET" \
    --split train \
    --out "$CORPUS" \
    --max-stories "$MAX_DOCS"
fi

# --- 4. train the tokenizer -------------------------------------------------
if [[ -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
  log "Tokenizer already present at $TOKENIZER_DIR (skipping)"
else
  log "Training tokenizer (vocab_size=$VOCAB_SIZE) -> $TOKENIZER_DIR"
  $PY -m orkhon tokenizer train \
    --corpus "$CORPUS" \
    --out "$TOKENIZER_DIR" \
    --vocab-size "$VOCAB_SIZE"
fi

# --- 5. pack the corpus -----------------------------------------------------
if [[ -f "$PREPARED_DIR/train.bin" ]]; then
  log "Packed shards already present at $PREPARED_DIR (skipping)"
else
  log "Packing corpus -> $PREPARED_DIR"
  $PY -m orkhon data prepare \
    --corpus "$CORPUS" \
    --tokenizer "$TOKENIZER_DIR" \
    --out "$PREPARED_DIR" \
    --val-fraction "$VAL_FRACTION"
fi

# --- 6. launch training -----------------------------------------------------
# ORKHON_DISTRIBUTED selects the wrap strategy read by the engine (ddp|fsdp|none).
# torchrun --standalone runs a single-node group of NUM_GPUS processes; the engine
# inits the process group, wraps the model, shards the data by rank, and only the
# main process logs / checkpoints. WORLD_SIZE==1 keeps the single-process path.
export ORKHON_DISTRIBUTED="$DIST_MODE"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  log "Launching $DIST_MODE training on $NUM_GPUS GPUs ($CONFIG)"
  uv run torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m orkhon train pretrain --config "$CONFIG"
else
  log "Launching single-GPU training ($CONFIG)"
  $PY -m orkhon train pretrain --config "$CONFIG"
fi

log "Done. Checkpoints + metrics are under the run's out_dir (see $CONFIG)."
