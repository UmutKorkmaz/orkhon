"""Training engine and the three stage trainers (pretrain, SFT, DPO).

Public surface mirrors the integration contract::

    from orkhon.train import pretrain, sft, dpo
    pretrain.run(cfg)   # -> {final_train_loss, best_val_loss, steps}
    sft.run(cfg)        # -> summary
    dpo.run(cfg)        # -> summary incl. final_margin

Lower-level building blocks (optimizer groups, LR schedule, losses, checkpointing,
the shared loop) are also exported for reuse and testing.
"""

from orkhon.train import dpo, grpo, pretrain, sft
from orkhon.train.rewards import code_reward, extract_boxed, math_equal, math_reward
from orkhon.train.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from orkhon.train.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_main_process,
    rank,
    resolve_wrap_mode,
    world_size,
    wrap_model,
)
from orkhon.train.engine import run_training_loop
from orkhon.train.losses import (
    dpo_loss,
    lm_cross_entropy,
    sequence_logprob,
)
from orkhon.train.optim import build_optimizer
from orkhon.train.schedule import build_scheduler, lr_at

__all__ = [
    # stage runners
    "pretrain",
    "sft",
    "grpo",
    "dpo",
    # building blocks
    "build_optimizer",
    "build_scheduler",
    "lr_at",
    "lm_cross_entropy",
    "sequence_logprob",
    "dpo_loss",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "run_training_loop",
    # distributed
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "world_size",
    "rank",
    "barrier",
    "all_reduce_mean",
    "wrap_model",
    "resolve_wrap_mode",
]
