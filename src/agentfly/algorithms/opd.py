"""On-policy distillation (OPD) for agent training batches.

A teacher model scores the student's own sampled tokens; the per-token log-ratio
``log p_teacher − log p_student`` becomes the advantage (see SkillsScrape ``docs/opd_design.md``).

This module holds:

- the trainer's **teacher step**, :func:`compute_teacher_log_probs`, run when
  ``algorithm.teacher.endpoints_file`` is set (like the reference-policy and critic steps);
- the **OPD advantage estimator** ``opd`` (both batch layouts), configured by ``algorithm.opd``;
- :func:`compute_opd_metrics`, which the trainer logs for ``adv_estimator=opd`` (like GDPO's
  per-component metrics).

Position convention (matches the trainer's ``old_log_probs`` and shifted ``action_mask``): rows are
right-padded, and position ``t`` refers to token ``t+1``. For a row with ``n`` valid tokens,
``teacher_log_probs[i, t] = log p_teacher(input_ids[i, t+1] | input_ids[i, :t+1])`` for
``t < n − 1``, and 0 at ``t = n − 1`` and on padding.

Advantage, per position ``t`` (per-token reward, no future sum)::

    A_t = action_mask_t · w_t · clamp(teacher_log_probs_t − old_log_probs_t, −clamp, clamp)
          [+ task_reward_coef · A_GRPO_t]

``w_t = think_weight`` on thinking positions (a turn's tokens up to and including its first
``</think>``; generation prompts end with ``<think>\\n``, so every turn starts in thinking), else 1.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .common import batch_field, config_value
from .registry import register_estimator
from .teacher import TeacherClient

# "</think>" in the Qwen3.5 / Qwen3.8 tokenizer (same token ids; checked 2026-09-16). Hardcoded for
# now; other model families need their own id.
THINK_END_TOKEN_ID = 248069

OPD_DEFAULTS = {"clamp": 5.0, "think_weight": 1.0, "task_reward_coef": 0.0}

_CLIENTS: Dict[Tuple[Any, ...], TeacherClient] = {}


def teacher_client(teacher_config: Any) -> TeacherClient:
    """The process-wide :class:`TeacherClient` for these settings (created on first use)."""
    endpoints_file = config_value(teacher_config, "endpoints_file", None)
    if not endpoints_file:
        raise ValueError("algorithm.teacher.endpoints_file is required for the teacher step")
    key = (
        str(endpoints_file),
        config_value(teacher_config, "expect_model", None),
        config_value(teacher_config, "expect_revision", None),
        config_value(teacher_config, "concurrency", None),
        float(config_value(teacher_config, "max_wait_s", 1800.0)),
    )
    client = _CLIENTS.get(key)
    if client is None:
        client = TeacherClient(
            key[0], expect_model=key[1], expect_revision=key[2], concurrency=key[3], max_wait_s=key[4]
        )
        _CLIENTS[key] = client
    return client


def compute_teacher_log_probs(batch: Any, teacher_config: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Teacher log-probabilities for every row of a training batch, in ``old_log_probs`` layout.

    Args:
        batch: a ``DataProto`` (anything with ``batch["input_ids"]`` and ``batch["attention_mask"]``,
            dense ``[B, L]`` tensors).
        teacher_config: the ``algorithm.teacher`` block.

    Returns:
        ``(teacher_log_probs, metrics)``: a float32 ``[B, L]`` CPU tensor and ``teacher/*`` metrics.
    """
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    if input_ids.is_nested or attention_mask.is_nested:
        raise ValueError("teacher step expects dense (padded) input_ids and attention_mask")
    if input_ids.shape != attention_mask.shape or input_ids.dim() != 2:
        raise ValueError(f"input_ids {tuple(input_ids.shape)} and attention_mask "
                         f"{tuple(attention_mask.shape)} must be the same [B, L] shape")

    mask = attention_mask.to(torch.bool).cpu()
    lengths = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1]).unsqueeze(0)
    if not torch.equal(mask, positions < lengths.unsqueeze(1)):
        raise ValueError(
            "teacher step expects right-padded rows (attention_mask ones then zeros); "
            "the alignment with old_log_probs assumes that layout"
        )

    ids = input_ids.cpu()
    sequences = [ids[i, : int(n)].tolist() for i, n in enumerate(lengths)]
    logprobs, stats = teacher_client(teacher_config).score_blocking(sequences)

    out = torch.zeros(ids.shape, dtype=torch.float32)
    for i, (lp, n) in enumerate(zip(logprobs, lengths.tolist())):
        if n > 1:
            out[i, : n - 1] = torch.from_numpy(lp[1:n])
    metrics = {
        "teacher/score_seconds": stats.seconds,
        "teacher/tokens": float(stats.tokens),
        "teacher/rows": float(stats.sequences),
        "teacher/unique_rows": float(stats.unique),
        "teacher/requests": float(stats.requests),
        "teacher/retries": float(stats.retries),
    }
    return out, metrics


# ---- OPD advantage -----------------------------------------------------------------------------


def opd_options(config: Any) -> Dict[str, float]:
    """``algorithm.opd`` options with defaults."""
    opd_config = config_value(config, "opd", None) or {}
    return {key: float(config_value(opd_config, key, default)) for key, default in OPD_DEFAULTS.items()}


def think_position_mask(
    input_ids: torch.Tensor, action_mask: torch.Tensor, think_end_token_id: int = THINK_END_TOKEN_ID
) -> torch.Tensor:
    """``[B, L]`` bool: trained positions whose token is part of the turn's thinking.

    Position ``t`` refers to token ``t+1``. A turn is one contiguous run of ``action_mask``; its
    thinking is every token up to and including the turn's first ``</think>``. A turn cut off
    before ``</think>`` is thinking throughout.
    """
    gen = action_mask > 0
    next_ids = torch.zeros_like(input_ids)
    next_ids[:, :-1] = input_ids[:, 1:]
    is_end = ((next_ids == think_end_token_id) & gen).long()
    ends_before = torch.cumsum(is_end, dim=1) - is_end            # ends strictly before t, row-wide
    previous = torch.zeros_like(gen)
    previous[:, 1:] = gen[:, :-1]
    turn_start = gen & ~previous
    # ends_before is non-decreasing, so the running max over turn starts is its value at this turn's start.
    at_turn_start = torch.cummax(
        torch.where(turn_start, ends_before, torch.full_like(ends_before, -1)), dim=1
    ).values
    return gen & (ends_before == at_turn_start)


def compute_opd_advantage(data: Any, config: Any, layout: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """OPD advantages (and returns, equal to them) for an AgentFly batch of either layout."""

    def field(name):
        return batch_field(data, name, "opd", layout)

    options = opd_options(config)
    action_mask = field("action_mask").to(torch.float32)
    teacher = field("teacher_log_probs").to(torch.float32)
    old = field("old_log_probs").to(torch.float32)
    clamp = options["clamp"]

    ratio = torch.clamp(teacher - old, -clamp, clamp)
    weights = torch.ones_like(ratio)
    if options["think_weight"] != 1.0:
        think = think_position_mask(field("input_ids"), action_mask)
        weights = torch.where(think, torch.full_like(ratio, options["think_weight"]), weights)
    advantages = action_mask * weights * ratio

    if options["task_reward_coef"] != 0.0:
        from ..verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage  # heavy; only when used

        grpo, _ = compute_grpo_outcome_advantage(
            token_level_rewards=field("token_level_rewards"),
            response_mask=action_mask,
            index=field("uid"),
            norm_adv_by_std_in_grpo=bool(config_value(config, "norm_adv_by_std_in_grpo", True)),
        )
        advantages = advantages + options["task_reward_coef"] * grpo.to(advantages.dtype) * action_mask
    return advantages, advantages


@register_estimator("opd", layout="per_segment")
def opd_per_segment(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return compute_opd_advantage(data, config, "per_segment")


@register_estimator("opd", layout="per_step")
def opd_per_step(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return compute_opd_advantage(data, config, "per_step")


def compute_opd_metrics(batch: Any, config: Any) -> Dict[str, float]:
    """``opd/*`` metrics for a batch after the advantage step (trainer logs them for ``opd``).

    ``kl_k1`` is ``old_log_probs − teacher_log_probs`` on trained positions: a single-sample estimate
    of the per-token reverse KL from the student (which sampled the tokens) to the teacher.
    """
    options = opd_options(config)
    mask = batch.batch["action_mask"] > 0
    if not mask.any():
        return {}
    teacher = batch.batch["teacher_log_probs"].to(torch.float32)
    old = batch.batch["old_log_probs"].to(torch.float32)
    k1 = old - teacher
    think = think_position_mask(batch.batch["input_ids"], batch.batch["action_mask"])
    non_think = mask & ~think

    def mean(values, where):
        return float(values[where].mean()) if where.any() else 0.0

    metrics = {
        "opd/kl_k1_mean": mean(k1, mask),
        "opd/kl_k1_think": mean(k1, think),
        "opd/kl_k1_non_think": mean(k1, non_think),
        "opd/think_token_frac": float(think.sum() / mask.sum()),
        "opd/teacher_prefers_frac": mean((teacher > old).to(torch.float32), mask),
        "opd/adv_clamp_frac": mean((k1.abs() > options["clamp"]).to(torch.float32), mask),
    }
    if "advantages" in batch.batch.keys():
        metrics["opd/adv_abs_mean"] = mean(batch.batch["advantages"].to(torch.float32).abs(), mask)
    return metrics
