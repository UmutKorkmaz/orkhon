"""Pydantic config schemas shared by the CLI and all training stages.

Field names here are a contract: the CLI, trainers, and YAML files must agree.
The architecture lives in :class:`orkhon.model.config.ModelConfig` and is referenced
from stage configs via ``arch_path`` (a path to a model YAML), to keep one source
of truth for shape. We avoid the ``model_`` field prefix because pydantic v2
reserves that namespace.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimConfig(_Base):
    lr: float = 3e-4
    min_lr_ratio: float = 0.1  # final LR = lr * min_lr_ratio
    weight_decay: float = 0.1  # applied to matmul weights only, never norms/bias/embeds
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 50
    schedule: str = Field(default="cosine")  # cosine | linear | constant


class TrainConfig(_Base):
    device: str = "auto"  # auto | cpu | mps | cuda | cuda:N
    dtype: str = "auto"  # auto -> bf16 on cuda, fp32 otherwise
    seed: int = 1337
    batch_size: int = 8  # micro-batch (per optimizer step = batch_size * grad_accum_steps)
    grad_accum_steps: int = 1
    max_steps: int = 1000  # optimizer steps
    seq_len: int = 256
    log_interval: int = 10
    eval_interval: int = 100
    eval_iters: int = 50
    ckpt_interval: int = 250
    out_dir: str = "runs/smoke"
    compile: bool = False
    num_workers: int = 0
    monitor: str = ""  # "" | "wandb" | "tensorboard" | "all" (else read ORKHON_MONITOR)


class DataConfig(_Base):
    # Pretraining: directory containing packed train.bin / val.bin (+ meta.json).
    prepared_dir: str | None = None
    # SFT / DPO: jsonl paths (messages / preference pairs).
    sft_path: str | None = None
    dpo_path: str | None = None
    val_sft_path: str | None = None
    val_dpo_path: str | None = None
    # RLVR/GRPO: jsonl of {"prompt": "...", "gold": "..."} (math) or
    # {"prompt": [...], "tests": [...]} (code).
    rlvr_path: str | None = None
    val_rlvr_path: str | None = None


class TokenizerConfig(_Base):
    # Directory holding tokenizer.json (+ special_tokens_map / chat_template).
    dir: str = "artifacts/tokenizer/smoke"


class PretrainConfig(_Base):
    arch_path: str  # path to a model YAML (ModelConfig fields)
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig = TrainConfig()


class SFTConfig(_Base):
    arch_path: str
    init_from: str  # checkpoint dir of the pretrained model to fine-tune
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig(lr=1e-4, warmup_steps=10)
    train: TrainConfig = TrainConfig(out_dir="runs/sft")


class DPOConfig(_Base):
    arch_path: str
    init_from: str  # checkpoint dir of the SFT model (also used as frozen reference)
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig(lr=5e-6, warmup_steps=10)
    train: TrainConfig = TrainConfig(out_dir="runs/dpo")
    beta: float = 0.1
    ref_from: str | None = None  # defaults to init_from when None


class GRPOConfig(_Base):
    arch_path: str
    init_from: str              # SFT checkpoint; the trainable policy starts here
    ref_from: str | None = None  # frozen reference; defaults to init_from
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig(lr=1e-6, warmup_steps=10)
    train: TrainConfig = TrainConfig(out_dir="runs/grpo")
    group_size: int = 4         # G completions per prompt
    beta: float = 0.02          # KL-to-reference weight
    kl_stop: float | None = None  # early-stop training if sampled KL exceeds this
    advantage_eps: float = 1e-6
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    reward_kind: str = "auto"   # auto | math | code
    code_timeout: float = 2.0
