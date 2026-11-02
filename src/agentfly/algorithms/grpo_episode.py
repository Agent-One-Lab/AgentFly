"""GRPO with episode-level group statistics, for ``per_segment`` batches.

A ``per_segment`` batch can hold several rows per episode: context folding splits a conversation
into segments, subagent sessions add their own rows, and the converter pads with copies of the
last row. verl's GRPO takes the group mean and std over rows, so an episode with more rows counts
more. Here each episode (``batch_idx``) counts once within its group (``uid``); every row of an
episode then gets that episode's normalized outcome, broadcast over its policy tokens.

``algorithm.estimator_kwargs.filter_zero_variance_groups`` (default off; as in SERA's training): rows of groups whose
episodes all got the same reward leave the loss: their policy-token mask (``action_mask``) is set to 0, so
they add nothing to the policy loss, its averaging or the entropy term. Their advantage is 0 anyway.
"""

from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .common import batch_field, config_value
from .registry import register_estimator


def compute_grpo_episode_advantage(
    token_level_rewards: torch.Tensor,  # [B, L] the episode outcome on the row's reward token
    response_mask: torch.Tensor,  # [B, L] 1 on policy tokens
    uid: np.ndarray,  # [B] group (task) of each row
    episode: np.ndarray,  # [B] episode of each row (rows of one episode share it)
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advantages ``[B, L]``: ``(score - mean) / (std + epsilon)`` (or ``score - mean``), with mean
    and std over the group's episodes, one score per episode. A group with one episode gets 0, as
    in verl's GRPO. Returns ``(advantages, returns)``; they are equal, as in verl."""
    scores = token_level_rewards.sum(dim=-1)
    episode_score: Dict[Any, torch.Tensor] = {}
    group_episodes: Dict[Any, list] = defaultdict(list)
    for i in range(scores.shape[0]):
        key = (uid[i], episode[i])
        if key not in episode_score:  # every row of an episode carries the same outcome
            episode_score[key] = scores[i]
            group_episodes[uid[i]].append(scores[i])
    stats = {}  # mean and std of groups with more than one episode
    for group, values in group_episodes.items():
        if len(values) > 1:
            stacked = torch.stack(values)
            stats[group] = (stacked.mean(), stacked.std())
    advantages = torch.zeros_like(scores)
    for i in range(scores.shape[0]):
        if uid[i] not in stats:
            continue  # one episode in the group: no comparison, advantage 0
        mean, std = stats[uid[i]]
        centered = episode_score[(uid[i], episode[i])] - mean
        advantages[i] = centered / (std + epsilon) if norm_adv_by_std_in_grpo else centered
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def zero_variance_rows(token_level_rewards: torch.Tensor, uid: np.ndarray, episode: np.ndarray) -> torch.Tensor:
    """``[B]`` bool: rows whose group's episodes all got the same reward (including one-episode groups)."""
    scores = token_level_rewards.sum(dim=-1)
    group_scores: Dict[Any, Dict[Any, float]] = defaultdict(dict)
    for i in range(scores.shape[0]):
        group_scores[uid[i]].setdefault(episode[i], float(scores[i]))
    constant = {g: len(set(v.values())) <= 1 for g, v in group_scores.items()}
    return torch.tensor([constant[uid[i]] for i in range(scores.shape[0])], dtype=torch.bool)


@register_estimator("grpo_episode", layout="per_segment")
def grpo_episode_per_segment(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def field(name):
        return batch_field(data, name, "grpo_episode", "per_segment")

    advantages, returns = compute_grpo_episode_advantage(
        token_level_rewards=field("token_level_rewards"),
        response_mask=field("action_mask"),
        uid=field("uid"),
        episode=field("batch_idx"),
        norm_adv_by_std_in_grpo=bool(config_value(config, "norm_adv_by_std_in_grpo", True)),
    )
    estimator_kwargs = config_value(config, "estimator_kwargs", None) or {}
    if bool(estimator_kwargs.get("filter_zero_variance_groups", False)):
        rows = zero_variance_rows(field("token_level_rewards"), field("uid"), field("batch_idx"))
        data.batch["action_mask"][rows] = 0  # in place: the actor's loss and entropy read this mask
    return advantages, returns
