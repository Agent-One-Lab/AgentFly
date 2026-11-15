"""RAO's credit for recursive agents (Recursive Agent Optimization, as in SERA's platoon:
``train/areal/workflows/step_wise.py`` and ``train/areal/rl.py``), for ``per_segment`` batches.

Each row is trained on its own session's reward (``session_reward``: a subagent's own outcome, the
main agent's task outcome) minus a leave-one-out baseline: the average main-agent outcome of the
other episodes of its task (group). The outcome is what every row of an episode carries in
``token_level_rewards``.

With ``depth_weighting`` (``algorithm.estimator_kwargs.depth_weighting``, default on), each depth of
the agent tree gets the same total weight: a row's advantage is scaled by 1 / (sessions at its depth
in the batch), and all weights are rescaled so their mean over policy tokens is 1 (SERA rescales
over steps), which keeps the gradient size of an unweighted batch. A group with one episode gets
advantage 0, as in verl's GRPO.
"""

from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .common import batch_field, config_value, episode_outcomes
from .registry import register_estimator, register_metrics


def depth_weights(
    response_mask: torch.Tensor,
    session_depth: np.ndarray,
    session_id: np.ndarray,
    uid: np.ndarray,
    episode: np.ndarray,
) -> torch.Tensor:
    """``[B]``: 1 / (sessions at the row's depth in the batch), rescaled to mean 1 over policy tokens.
    Padding rows (no policy tokens) count no session."""
    tokens = response_mask.sum(dim=-1).float()
    # A session can span several rows (context folding, padding copies): count each once.
    sessions = {
        (uid[i], episode[i], int(session_id[i])): int(session_depth[i]) for i in range(len(tokens)) if tokens[i] > 0
    }
    count = Counter(sessions.values())
    raw = torch.tensor(
        [1.0 / count[int(d)] if count[int(d)] else 0.0 for d in session_depth],
        dtype=tokens.dtype,
        device=tokens.device,
    )
    total = (tokens * raw).sum()
    return raw * (tokens.sum() / total) if total > 0 else raw


def compute_rao_advantage(
    token_level_rewards: torch.Tensor,  # [B, L] the episode outcome on the row's reward token
    response_mask: torch.Tensor,  # [B, L] 1 on policy tokens
    session_reward: np.ndarray,  # [B] the row's own session reward
    session_depth: np.ndarray,  # [B] 0 for the main agent, 1 for its subagents, ...
    session_id: np.ndarray,  # [B]
    uid: np.ndarray,  # [B] group (task) of each row
    episode: np.ndarray,  # [B] episode of each row
    depth_weighting: bool = True,
) -> torch.Tensor:
    """Advantages ``[B, L]``: ``session_reward - baseline`` on each row's policy tokens, the baseline
    being the mean outcome of the group's other episodes; scaled by :func:`depth_weights` if asked."""
    outcome, group_episodes = episode_outcomes(token_level_rewards, uid, episode)
    baseline: Dict[Any, Any] = {}
    for keys in group_episodes.values():
        total = sum(float(outcome[key]) for key in keys)
        for key in keys:
            baseline[key] = (total - float(outcome[key])) / (len(keys) - 1) if len(keys) > 1 else None
    advantages = torch.zeros_like(token_level_rewards[:, 0])
    for i in range(advantages.shape[0]):
        b = baseline[(uid[i], episode[i])]
        advantages[i] = 0.0 if b is None else float(session_reward[i]) - b
    if depth_weighting:
        advantages = advantages * depth_weights(response_mask, session_depth, session_id, uid, episode)
    return advantages.unsqueeze(-1) * response_mask


@register_estimator("rao", layout="per_segment")
def rao_per_segment(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def field(name):
        return batch_field(data, name, "rao", "per_segment")

    estimator_kwargs = config_value(config, "estimator_kwargs", None) or {}
    advantages = compute_rao_advantage(
        token_level_rewards=field("token_level_rewards"),
        response_mask=field("action_mask"),
        session_reward=field("session_reward"),
        session_depth=field("session_depth"),
        session_id=field("session_id"),
        uid=field("uid"),
        episode=field("batch_idx"),
        depth_weighting=bool(estimator_kwargs.get("depth_weighting", True)),
    )
    return advantages, advantages


@register_metrics("rao")
def rao_metrics(batch: Any, config: Any) -> Dict[str, float]:
    """Per depth ``d`` of the agent tree: ``rao/depth{d}/sessions``, ``rao/depth{d}/reward`` (mean session
    reward) and ``rao/depth{d}/advantage`` (mean advantage over policy tokens)."""

    def field(name):
        return batch_field(batch, name, "rao", "per_segment")

    mask = field("action_mask").float()
    advantages = field("advantages")
    session_reward, session_depth = field("session_reward"), field("session_depth")
    uid, episode, session_id = field("uid"), field("batch_idx"), field("session_id")
    real = [i for i in range(mask.shape[0]) if mask[i].sum() > 0]
    first_row = {}
    for i in real:
        first_row.setdefault((uid[i], episode[i], int(session_id[i])), i)
    metrics: Dict[str, float] = {}
    for d in sorted({int(session_depth[i]) for i in real}):
        sessions = [i for i in first_row.values() if int(session_depth[i]) == d]
        rows = [i for i in real if int(session_depth[i]) == d]
        tokens = mask[rows].sum()
        metrics[f"rao/depth{d}/sessions"] = float(len(sessions))
        metrics[f"rao/depth{d}/reward"] = float(np.mean([float(session_reward[i]) for i in sessions]))
        metrics[f"rao/depth{d}/advantage"] = float((advantages[rows] * mask[rows]).sum() / tokens)
    return metrics
