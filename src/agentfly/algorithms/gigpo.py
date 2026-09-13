"""GiGPO (Group-in-Group Policy Optimization, NeurIPS 2025) for both agent batch layouts.

GiGPO is critic-free and adds two group-relative signals:

- **episode advantage** — each row's outcome reward normalised within its prompt group
  (``uid``), broadcast to the row's policy tokens;
- **step advantage** — each step's discounted return normalised within its *step group*:
  steps of the same prompt group that acted from the same anchor observation.

``advantages = episode_adv + step_advantage_w * step_adv``; ``step_advantage_w=0`` is GRPO.
Both layouts use verl-agent's grouping and normalisation (:mod:`.common`), the
``gigpo_mode`` setting (``mean_norm`` or ``mean_std_norm``) and a post-discount
invalid-action penalty (verl-agent's ``apply_invalid_action_penalty``):

- ``per_step`` batches are exactly verl-agent's shape (one row per step), so this is
  verl-agent's pipeline, restricted to active (non-padding) rows.
- ``per_segment`` batches hold several turns per row; the turns are flattened into steps,
  normalised with the same functions, and each step advantage is written onto its turn's
  token span.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .common import (
    batch_field,
    build_step_group,
    compute_step_discounted_returns,
    compute_turn_ids,
    config_value,
    episode_norm_reward,
    object_array,
    print_group_diagnostics,
    remove_std_for,
    step_norm_reward,
    terminal_token_mask,
)
from .registry import register_estimator


def gigpo_params(config: Any) -> Dict[str, Any]:
    """GiGPO settings from the trainer's algorithm config (defaults = ``AlgoConfig``'s)."""
    return {
        "gamma": float(config_value(config, "gamma", 1.0)),
        "step_advantage_w": float(config_value(config, "step_advantage_w", 1.0)),
        "mode": config_value(config, "gigpo_mode", "mean_std_norm"),
        "invalid_action_penalty_coef": float(config_value(config, "invalid_action_penalty_coef", 0.1)),
    }


# ------------------------------------------------------------------------------------ #
# per_step: one row per environment step (StepRollout)                                  #
# ------------------------------------------------------------------------------------ #
def compute_gigpo_per_step_advantage(
    token_level_rewards: torch.Tensor,             # [B, L] outcome broadcast to every step row
    response_mask: torch.Tensor,                   # [B, L]
    step_env_rewards: np.ndarray,                  # [B] raw per-step env reward
    anchor_obs: np.ndarray,                        # [B] state the action was taken from
    uid: np.ndarray,                               # [B] prompt group
    traj_uid: np.ndarray,                          # [B] trajectory
    is_action_valid: Optional[np.ndarray] = None,  # [B] 1 valid / 0 invalid
    active_masks: Optional[np.ndarray] = None,     # [B] 1 real row / 0 divisor padding
    gamma: float = 0.95,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    invalid_action_penalty_coef: float = 0.1,
    epsilon: float = 1e-6,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GiGPO on a ``per_step`` batch, matching verl-agent's trainer + ``core_gigpo`` flow.

    1. restrict to active rows (divisor padding repeats the last row and must not count);
    2. discounted step returns per ``traj_uid``;
    3. invalid-action penalty after discounting, on both the step return and the episode
       reward on the row's last response token;
    4. episode advantage over ``uid`` rows + step advantage over anchor step groups.

    Returns ``(advantages, returns)`` of shape ``[B, L]`` (critic-free, identical tensors),
    zero on inactive rows.
    """
    B, L = token_level_rewards.shape
    device, dtype = token_level_rewards.device, token_level_rewards.dtype
    remove_std = remove_std_for(mode)

    active = np.ones(B, dtype=bool) if active_masks is None else np.asarray(active_masks).astype(bool)
    idx = np.where(active)[0]
    if len(idx) == 0:
        zeros = torch.zeros(B, L, device=device, dtype=dtype)
        return zeros, zeros

    tlr = token_level_rewards[idx].clone()
    rmask = response_mask[idx]
    sub_uid = np.asarray(uid)[idx]
    sub_traj = np.asarray(traj_uid)[idx]
    sub_anchor = np.asarray(anchor_obs, dtype=object)[idx]
    sub_rewards = np.asarray(step_env_rewards, dtype=np.float32)[idx]

    returns = compute_step_discounted_returns(sub_rewards, sub_traj, gamma)
    if is_action_valid is not None and invalid_action_penalty_coef:
        invalid = 1.0 - np.asarray(is_action_valid, dtype=np.float32)[idx]
        penalty = invalid_action_penalty_coef * invalid
        returns = returns - penalty
        tlr = tlr - torch.tensor(penalty, device=device, dtype=dtype).unsqueeze(-1) * terminal_token_mask(rmask)
    step_returns = torch.tensor(returns, device=device, dtype=dtype)

    episode_adv = episode_norm_reward(tlr, rmask, sub_uid, sub_traj, epsilon, remove_std)
    step_group_uids = build_step_group(sub_anchor, sub_uid, enable_similarity, similarity_thresh)
    step_adv = step_norm_reward(step_returns, rmask, step_group_uids, epsilon, remove_std)
    sub_adv = episode_adv + step_advantage_w * step_adv

    n_traj = len(np.unique(sub_traj))
    outcomes = tlr.sum(dim=-1)
    print_group_diagnostics(
        "GiGPO per_step", step_group_uids, episode_adv, step_adv, rmask, step_advantage_w,
        extra=(
            f"rows={len(idx)} trajs={n_traj} groups={len(np.unique(sub_uid))} "
            f"avg_steps/traj={len(idx) / max(1, n_traj):.1f} "
            f"success={float((outcomes > 0).float().mean()):.3f} "
            f"step_return_mean={float(step_returns.mean()):.3f}"
        ),
    )

    advantages = torch.zeros(B, L, device=device, dtype=dtype)
    advantages[torch.tensor(idx, device=device)] = sub_adv.to(dtype)
    return advantages, advantages


@register_estimator("gigpo", layout="per_step")
def gigpo_per_step(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def field(name, optional=False):
        return batch_field(data, name, "gigpo", "per_step", optional)

    return compute_gigpo_per_step_advantage(
        token_level_rewards=field("token_level_rewards"),
        response_mask=field("action_mask"),
        step_env_rewards=field("step_env_reward"),
        anchor_obs=field("anchor_obs"),
        uid=field("uid"),
        traj_uid=field("traj_uid"),
        is_action_valid=field("is_action_valid", optional=True),
        active_masks=field("active_masks", optional=True),
        **gigpo_params(config),
    )


# ------------------------------------------------------------------------------------ #
# per_segment: several turns per row (ChainRollout)                                     #
# ------------------------------------------------------------------------------------ #
def compute_gigpo_per_segment_advantage(
    token_level_rewards: torch.Tensor,   # [B, L] outcome reward on the row's reward token
    response_mask: torch.Tensor,         # [B, L] 1 on policy tokens; contiguous runs = turns
    uid: np.ndarray,                     # [B] prompt group
    step_observations: np.ndarray,       # [B] object array: per-turn anchor lists
    step_rewards: np.ndarray,            # [B] object array: per-turn raw env reward lists
    step_invalids: Optional[np.ndarray] = None,  # [B] object array: per-turn 0/1 invalid flags
    gamma: float = 1.0,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    invalid_action_penalty_coef: float = 0.1,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GiGPO on a ``per_segment`` batch with verl-agent's normalisation.

    The episode advantage normalises each row's outcome within ``uid``. For the step
    advantage every turn becomes one step: its discounted return (minus the invalid-action
    penalty, after discounting), its anchor and its ``uid``. Steps are grouped and normalised
    exactly as ``per_step`` rows are, then written onto their turn's tokens. Turns beyond a
    row's step lists get no step advantage.

    Returns ``(advantages, returns)`` of shape ``[B, L]`` (critic-free, identical tensors).
    """
    B, L = token_level_rewards.shape
    device, dtype = token_level_rewards.device, token_level_rewards.dtype
    remove_std = remove_std_for(mode)
    uid = np.asarray(uid, dtype=object)

    episode_adv = episode_norm_reward(
        token_level_rewards.clone(), response_mask, uid, np.arange(B), epsilon, remove_std
    )

    # Flatten every row's turns into steps (row order, then turn order).
    flat_rewards, flat_invalid, flat_anchors, flat_uids, flat_positions = [], [], [], [], []
    for b in range(B):
        rewards = [0.0 if r is None else float(r) for r in (step_rewards[b] or [])]
        anchors = list(step_observations[b] or [])
        if len(anchors) < len(rewards):
            raise ValueError(
                f"Row {b} has {len(rewards)} step rewards but only {len(anchors)} step observations."
            )
        invalids = list(step_invalids[b] or []) if step_invalids is not None else []
        for t, reward in enumerate(rewards):
            flat_rewards.append(reward)
            flat_invalid.append(float(invalids[t]) if t < len(invalids) and invalids[t] else 0.0)
            flat_anchors.append(anchors[t])
            flat_uids.append(uid[b])
            flat_positions.append((b, t))

    step_advantages = torch.zeros(B, L, device=device, dtype=dtype)
    step_group_uids = np.empty(0, dtype=object)
    if flat_rewards:
        rows = np.array([b for b, _ in flat_positions])
        flat_returns = compute_step_discounted_returns(np.asarray(flat_rewards, dtype=np.float32), rows, gamma)
        if invalid_action_penalty_coef:
            flat_returns = flat_returns - invalid_action_penalty_coef * np.asarray(flat_invalid, dtype=np.float32)
        step_group_uids = build_step_group(object_array(flat_anchors), object_array(flat_uids))
        flat_adv = step_norm_reward(
            torch.tensor(flat_returns, device=device, dtype=dtype),
            torch.ones(len(flat_returns), 1, device=device, dtype=dtype),
            step_group_uids, epsilon, remove_std,
        )[:, 0]
        turn_ids = compute_turn_ids(response_mask)
        for k, (b, t) in enumerate(flat_positions):
            step_advantages[b][turn_ids[b] == t] = flat_adv[k]
    step_advantages = step_advantages * response_mask

    print_group_diagnostics(
        "GiGPO per_segment", step_group_uids, episode_adv, step_advantages, response_mask,
        step_advantage_w, extra=f"rows={B} groups={len(np.unique(uid))}",
    )

    advantages = episode_adv + step_advantage_w * step_advantages
    return advantages, advantages


@register_estimator("gigpo", layout="per_segment")
def gigpo_per_segment(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def field(name, optional=False):
        return batch_field(data, name, "gigpo", "per_segment", optional)

    return compute_gigpo_per_segment_advantage(
        token_level_rewards=field("token_level_rewards"),
        response_mask=field("action_mask"),
        uid=field("uid"),
        step_observations=field("step_observations"),
        step_rewards=field("step_rewards"),
        step_invalids=field("step_invalids", optional=True),
        **gigpo_params(config),
    )
