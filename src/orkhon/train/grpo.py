"""GRPO / RLVR — verifiable-reward RL via group-relative advantages.

For each prompt, sample ``G`` completions with the policy, score each with an exact
verifiable reward (:mod:`orkhon.train.rewards`), and optimize a per-token policy-
gradient loss with a KL anchor to a frozen reference (the SFT init). No critic:
advantages are group-relative, ``A_i = (r_i - mean) / std``. The KL uses the Schulman
estimator (``exp(r) - r - 1``, ``r = log p_ref - log p_policy``) which is non-negative
and zero when policy == reference.

GRPO has a coupled per-step structure (rollout → reward → advantage → update) that
does not fit :func:`~orkhon.train.engine.run_training_loop`, so it owns a dedicated
loop while reusing the same checkpoint / optimizer / schedule / monitor / NaN-guard
helpers.

The completion-only masking matches the rest of the stack: logits at position
``t-1`` score the token at ``t``; only completion-token positions are supervised.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from orkhon.config.load import load_model_config
from orkhon.config.schema import GRPOConfig, OptimConfig, TrainConfig
from orkhon.model import generate
from orkhon.tokenizer import load_tokenizer
from orkhon.tokenizer.render import encode_for_inference
from orkhon.train.checkpoint import load_checkpoint, load_model_from_checkpoint, save_checkpoint
from orkhon.train.optim import build_optimizer
from orkhon.train.rewards import code_reward, copy_digit_reward, math_reward
from orkhon.train.schedule import lr_at
from orkhon.train.metrics import grad_global_norm
from orkhon.utils.dtype import autocast_ctx, resolve_dtype
from orkhon.utils.device import resolve_device
from orkhon.utils.logging import get_logger
from orkhon.utils.seed import get_rng_state, set_seed

_logger = get_logger("orkhon.train.grpo")
IGNORE = -100
_SPIKE_MULT = 8.0


# ---------------------------------------------------------------------------
# The loss (independently testable).
# ---------------------------------------------------------------------------

def grpo_loss(
    policy: nn.Module,
    reference: nn.Module,
    input_ids: torch.Tensor,        # [B, T] prompt+completion, left-padded
    attention_mask: torch.Tensor,   # [B, T] bool, True = real token
    completion_mask: torch.Tensor,  # [B, T] bool, True = completion token (target)
    advantages: torch.Tensor,       # [B]
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Mean GRPO loss over supervised completion tokens + a metrics dict.

    Returns ``(loss, {"kl": float, "ratio": float})`` where ``kl`` is the mean
    Schulman KL over completion tokens.
    """
    logits_p, _ = policy(input_ids, attention_mask=attention_mask)
    with torch.no_grad():
        logits_r, _ = reference(input_ids, attention_mask=attention_mask)
    logp_p = F.log_softmax(logits_p[:, :-1, :].float(), dim=-1)
    logp_r = F.log_softmax(logits_r[:, :-1, :].float(), dim=-1)

    labels = input_ids[:, 1:]                      # [B, T-1]
    mask = (completion_mask[:, 1:] & attention_mask[:, 1:]).float()  # [B, T-1]

    tok_logp_p = logp_p.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    tok_logp_r = logp_r.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    r = tok_logp_r - tok_logp_p                      # log p_ref - log p_policy
    kl_token = torch.exp(r) - r - 1.0                # Schulman, >= 0

    adv = advantages.unsqueeze(-1)                   # [B, 1] broadcast
    per_token = -adv * tok_logp_p + beta * kl_token  # [B, T-1]

    # GRPO reduction: per-completion token mean FIRST (1/T), then mean over
    # completions — a flat token mean would overweight longer completions.
    row_denom = mask.sum(1).clamp_min(1.0)           # [B]
    row_loss = (per_token * mask).sum(1) / row_denom
    row_kl = (kl_token * mask).sum(1) / row_denom
    loss = row_loss.mean()
    kl = float(row_kl.mean().item())
    return loss, {"kl": kl}


# ---------------------------------------------------------------------------
# Data + rollout.
# ---------------------------------------------------------------------------

def _load_rlvr(path: str) -> list[dict]:
    rows = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _score(output: str, ex: dict, kind: str, timeout: float) -> float:
    if kind == "copy_digit":
        return copy_digit_reward(output, ex.get("gold", ""))
    if kind == "code" or (kind == "auto" and "tests" in ex):
        return code_reward(output, tests=ex.get("tests", []), timeout=timeout)
    return math_reward(output, ex.get("gold", ""))


def _group_advantages(rewards: list[list[float]], eps: float) -> list[list[float]]:
    out = []
    for group in rewards:
        mu = sum(group) / len(group)
        var = sum((r - mu) ** 2 for r in group) / len(group)
        sigma = math.sqrt(var)
        if sigma < eps:
            out.append([0.0] * len(group))
        else:
            out.append([(r - mu) / sigma for r in group])
    return out


def _prompt_ids(ex: dict, tok) -> list[int]:
    p = ex["prompt"]
    if isinstance(p, str):
        # plain completion prompt (math): treat as raw text with a fresh-doc eos.
        return [tok.special.eos] + tok.encode(p)
    return encode_for_inference(p, tok.encode, tok.special)  # chat-style messages


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------

def _freeze(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def run(cfg: GRPOConfig) -> dict:
    if cfg.data.rlvr_path is None:
        raise ValueError("grpo requires cfg.data.rlvr_path (RLVR jsonl)")
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    set_seed(cfg.train.seed)

    model_cfg = load_model_config(cfg.arch_path)
    policy, _ = load_model_from_checkpoint(cfg.init_from, device=device)
    policy.train()
    ref, _ = load_model_from_checkpoint(cfg.ref_from or cfg.init_from, device=device)
    _freeze(ref)
    tok = load_tokenizer(cfg.tokenizer.dir)
    optimizer = build_optimizer(policy, cfg.optim)

    rows = _load_rlvr(cfg.data.rlvr_path)
    rng = random.Random(cfg.train.seed)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_ctx = min(cfg.train.seq_len, model_cfg.block_size)

    # Exact resume (model + optimizer + RNG + step) so a long GRPO run survives restarts.
    from orkhon.train.engine import maybe_resume
    start_step = maybe_resume(out_dir, policy, optimizer, map_location=device)

    first_window: list[float] = []
    last_window: list[float] = []
    final_kl = float("nan")

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_at(step, cfg.optim, cfg.train.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch_rows = rng.sample(rows, min(cfg.train.batch_size, len(rows)))
        t0 = time.perf_counter()
        seqs: list[torch.Tensor] = []
        cmasks: list[torch.Tensor] = []
        amasks: list[torch.Tensor] = []
        advs: list[float] = []
        rewards_flat: list[float] = []

        for ex in batch_rows:
            # Keep the prompt SUFFIX (most recent context); budget room for completions.
            prompt_budget = max(1, max_ctx - cfg.max_new_tokens)
            pids = _prompt_ids(ex, tok)[-prompt_budget:]
            comps, rws = [], []
            for _ in range(cfg.group_size):
                c = generate(policy, pids, cfg.max_new_tokens, temperature=cfg.temperature,
                             top_k=cfg.top_k, top_p=cfg.top_p,
                             repetition_penalty=cfg.repetition_penalty,
                             eos_ids=(tok.special.end,), device=device)
                comps.append(c)
                text = tok.decode(c, skip_special=True)
                rws.append(_score(text, ex, cfg.reward_kind, cfg.code_timeout))
            rewards_flat.extend(rws)
            advs_group = _group_advantages([rws], cfg.advantage_eps)[0]
            for c, a in zip(comps, advs_group):
                full = pids + c
                full_mask = [False] * len(pids) + [True] * len(c)
                # Truncate BOTH with the SAME suffix so the mask stays aligned to
                # the retained tokens (front-truncating only `full` would misalign).
                if len(full) > max_ctx:
                    full = full[-max_ctx:]
                    full_mask = full_mask[-max_ctx:]
                T = len(full)
                seqs.append(torch.tensor(full, dtype=torch.long, device=device))
                cmasks.append(torch.tensor(full_mask, dtype=torch.bool, device=device))
                amasks.append(torch.ones(T, dtype=torch.bool, device=device))
                advs.append(a)

        # Left-pad the batch to a common length.
        maxlen = max(s.shape[0] for s in seqs)
        pad = tok.special.pad
        B = len(seqs)
        input_ids = torch.full((B, maxlen), pad, dtype=torch.long, device=device)
        amask = torch.zeros((B, maxlen), dtype=torch.bool, device=device)
        cmask = torch.zeros((B, maxlen), dtype=torch.bool, device=device)
        for i, (s, cm, am) in enumerate(zip(seqs, cmasks, amasks)):
            L = s.shape[0]
            input_ids[i, maxlen - L:] = s
            amask[i, maxlen - L:] = am
            cmask[i, maxlen - L:] = cm

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, dtype):
            loss, m = grpo_loss(policy, ref, input_ids, amask, cmask,
                                torch.tensor(advs, device=device, dtype=torch.float32),
                                beta=cfg.beta)
        if not (math.isfinite(float(loss.item())) and math.isfinite(m["kl"])):
            _logger.warning("step %d skipped (non-finite loss/kl)", step)
            continue
        # KL early-stop safety guard: stop training (skip this update) if the
        # policy has drifted too far from the reference. A "clip KL" would hide
        # the warning signal; stopping is the honest response.
        if cfg.kl_stop is not None and m["kl"] > cfg.kl_stop:
            _logger.warning("step %d early-stopped: kl %.3f > kl_stop %.3f",
                            step, m["kl"], cfg.kl_stop)
            final_kl = m["kl"]
            break
        loss.backward()
        gnorm = grad_global_norm(policy)
        if cfg.optim.grad_clip and cfg.optim.grad_clip > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        final_kl = m["kl"]

        rmean = sum(rewards_flat) / max(len(rewards_flat), 1)
        # first_window = the EARLIEST ~20 steps (baseline); last_window = rolling tail.
        if len(first_window) < 20:
            first_window.append(rmean)
        last_window.append(rmean)
        if len(last_window) > 20:
            last_window.pop(0)

        if step % max(cfg.train.log_interval, 1) == 0:
            _logger.info("step %d/%d loss %.4f reward %.3f kl %.3f gnorm %.2f",
                         step, cfg.train.max_steps, float(loss.item()), rmean, m["kl"], gnorm)

        if cfg.train.ckpt_interval and (step + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(out_dir, policy, optimizer, None, step + 1, get_rng_state(),
                            model_cfg, cfg.train, tag="last")

    fw = sum(first_window[:20]) / max(len(first_window[:20]), 1)
    lw = sum(last_window) / max(len(last_window), 1)
    return {"steps": cfg.train.max_steps, "final_train_loss": float(loss.item()),
            "initial_mean_reward": fw, "final_mean_reward": lw, "reward_delta": lw - fw,
            "final_kl": final_kl}
