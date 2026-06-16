"""Direct Preference Optimization stage.

``run`` trains a *policy* (loaded from ``cfg.init_from``) against a FROZEN
*reference* (loaded from ``cfg.ref_from`` or ``cfg.init_from``). For each
preference pair we compute completion-only sequence log-probabilities for the
chosen and rejected responses under both models, then apply :func:`dpo_loss`.

Correctness points (all easy to get wrong):

* The reference is a separate model in ``eval`` mode with ``requires_grad=False``
  and dropout disabled; it never receives gradients and is never updated.
* Chosen and rejected share an identical prompt encoding (guaranteed upstream by
  :class:`orkhon.data.DPODataset`), and logprobs are summed over completion
  positions only (prompt positions are ``IGNORE_INDEX`` -> excluded).
* Reference logprobs are computed under ``torch.no_grad`` so the frozen model adds
  no autograd overhead.

Logged metrics include the reward margin (chosen - rejected) and preference
accuracy (fraction where the chosen reward exceeds the rejected reward).
"""

from __future__ import annotations

import random

import torch

from orkhon.config.schema import DPOConfig
from orkhon.data.dataset import DPODataset
from orkhon.tokenizer.tokenizer import load_tokenizer
from orkhon.train.checkpoint import load_model_from_checkpoint
from orkhon.train.engine import prepare_run, run_training_loop
from orkhon.train.losses import dpo_loss, sequence_logprob
from orkhon.train.optim import build_optimizer
from orkhon.utils.logging import JsonlMetrics, get_logger

_logger = get_logger("orkhon.train.dpo")


def _freeze(model: torch.nn.Module) -> None:
    """Put a model in eval mode and disable gradients (frozen reference)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _iter_batches(dataset: DPODataset, batch_size: int, rng: random.Random):
    """Yield collated preference batches forever (random sampling)."""
    n = len(dataset)
    while True:
        idx = [rng.randrange(n) for _ in range(min(batch_size, n))]
        yield dataset.collate([dataset[i] for i in idx])


def _side_logps(
    model: torch.nn.Module,
    batch: dict,
    side: str,
    device: torch.device,
    max_ctx: int,
) -> torch.Tensor:
    """Completion-only summed logprobs for ``side`` (chosen|rejected).

    Sequences are right-truncated to ``max_ctx`` so they fit within the model's
    block_size (RoPE position table). Chosen and rejected are truncated with the
    same cap, preserving the shared prompt prefix.
    """
    input_ids = batch[f"{side}_input_ids"][:, :max_ctx].to(device)
    labels = batch[f"{side}_labels"][:, :max_ctx].to(device)
    attn = batch[f"{side}_attention_mask"][:, :max_ctx].to(device)
    logits, _ = model(input_ids, attention_mask=attn)
    return sequence_logprob(logits, labels)


def run(cfg: DPOConfig) -> dict:
    """Run DPO and return ``{final_train_loss, best_val_loss, steps, final_margin}``."""
    if cfg.data.dpo_path is None:
        raise ValueError("dpo requires cfg.data.dpo_path (preference jsonl)")

    device, dtype = prepare_run(cfg.train)

    # Policy: trainable, loaded from init_from.
    policy, model_cfg = load_model_from_checkpoint(cfg.init_from, device=device)
    policy.train()
    optimizer = build_optimizer(policy, cfg.optim)

    # Reference: frozen copy from ref_from (default init_from).
    ref_from = cfg.ref_from or cfg.init_from
    reference, _ = load_model_from_checkpoint(ref_from, device=device)
    _freeze(reference)

    tokenizer = load_tokenizer(cfg.tokenizer.dir)
    train_ds = DPODataset(cfg.data.dpo_path, tokenizer)
    val_ds = (
        DPODataset(cfg.data.val_dpo_path, tokenizer)
        if cfg.data.val_dpo_path
        else None
    )

    rng = random.Random(cfg.train.seed)
    train_batches = _iter_batches(train_ds, cfg.train.batch_size, rng)
    beta = cfg.beta
    # Cap context to the model's position table (block_size).
    max_ctx = min(cfg.train.seq_len, model_cfg.block_size)

    # Side channel so we can surface the final reward margin in the summary.
    state = {"last_margin": float("nan")}
    margin_log = JsonlMetrics(_margin_path(cfg.train.out_dir))

    def micro_batch_loss(step: int) -> tuple[torch.Tensor, int]:
        batch = next(train_batches)

        policy_chosen = _side_logps(policy, batch, "chosen", device, max_ctx)
        policy_rejected = _side_logps(policy, batch, "rejected", device, max_ctx)

        with torch.no_grad():
            ref_chosen = _side_logps(reference, batch, "chosen", device, max_ctx)
            ref_rejected = _side_logps(reference, batch, "rejected", device, max_ctx)

        loss, metrics = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
        )
        state["last_margin"] = float(metrics["reward_margin"].item())
        margin_log.log(
            step,
            dpo_loss=float(loss.detach().item()),
            reward_margin=state["last_margin"],
            reward_accuracy=float(metrics["reward_accuracy"].item()),
        )
        n_tokens = int(batch["chosen_attention_mask"].sum().item())
        return loss, n_tokens

    @torch.no_grad()
    def eval_fn() -> dict | None:
        if val_ds is None:
            return None
        was_training = policy.training
        policy.eval()
        val_rng = random.Random(cfg.train.seed + 1)
        batches = _iter_batches(val_ds, cfg.train.batch_size, val_rng)
        losses, margins, accs = [], [], []
        for _ in range(max(1, cfg.train.eval_iters)):
            batch = next(batches)
            pc = _side_logps(policy, batch, "chosen", device, max_ctx)
            pr = _side_logps(policy, batch, "rejected", device, max_ctx)
            rc = _side_logps(reference, batch, "chosen", device, max_ctx)
            rr = _side_logps(reference, batch, "rejected", device, max_ctx)
            loss, m = dpo_loss(pc, pr, rc, rr, beta)
            losses.append(float(loss.item()))
            margins.append(float(m["reward_margin"].item()))
            accs.append(float(m["reward_accuracy"].item()))
        if was_training:
            policy.train()
        return {
            "val_loss": sum(losses) / len(losses),
            "val_reward_margin": sum(margins) / len(margins),
            "val_reward_accuracy": sum(accs) / len(accs),
        }

    _logger.info(
        "dpo: %d pairs, policy=%s reference=%s beta=%.3f",
        len(train_ds), cfg.init_from, ref_from, beta,
    )

    try:
        result = run_training_loop(
            model=policy,
            optimizer=optimizer,
            micro_batch_loss=micro_batch_loss,
            optim_cfg=cfg.optim,
            train_cfg=cfg.train,
            model_cfg=model_cfg,
            device=device,
            dtype=dtype,
            eval_fn=eval_fn,
        )
    finally:
        margin_log.close()

    summary = result.as_dict()
    summary["final_margin"] = state["last_margin"]
    return summary


def _margin_path(out_dir: str) -> str:
    from pathlib import Path

    return str(Path(out_dir) / "dpo_margins.jsonl")
