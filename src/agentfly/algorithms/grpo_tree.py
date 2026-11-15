"""GRPO over trees of episodes, for ``per_segment`` batches with branched rollouts.

A rollout may branch an episode at a turn of one session (``Rollout.plan_branches``): the environment is
restored to its snapshot at that turn, the turn's response is replayed, and a second leaf is generated from
there. The two leaves share a prefix (the branching session's turns up to and including the branch turn)
and differ afterwards. The converter marks each row with its ``tree_id`` (shared by the leaves of one
tree) and, on the branching session's rows, ``branch_boundary``: the token offset where the shared prefix
ends (``-1`` on every other row). The branch leaf's copy of the prefix is masked out of ``action_mask``
by the converter, so the prefix is trained once, through the source leaf's row.

Credit (``R`` an episode's outcome, ``V`` the mean outcome of its tree's leaves, ``b``/``s`` the mean and
sample std of the group's tree values, each tree counted once):

* shared prefix tokens: ``(V - b) / s`` -- the prefix is judged by what the tree achieved on average;
* leaf-specific tokens of a branched tree: ``(R - b) / s + lambda * (R - V) / s`` -- the outcome term plus
  the within-tree comparison, which is ``+-(R - R') / 2`` for two leaves;
* every other row (unbranched trees, sessions outside the branch point): ``(R - b) / s``.

With ``lambda = 0`` and no branched trees this is exactly :mod:`.grpo_episode`. With ``norm_adv_by_std_in_grpo``
off the divisions by ``s`` are dropped. A group with a single tree gets advantage 0 on every row, as in verl's
GRPO (the within-tree term is kept, so a lone branched tree still ranks its leaves).

``algorithm.estimator_kwargs.branch_lambda`` sets ``lambda`` (default 1.0).
"""

from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .common import batch_field, config_value, episode_outcomes
from .registry import register_estimator


def tree_values(episode_score: Dict[Any, torch.Tensor], uid: np.ndarray, episode: np.ndarray, tree_id: np.ndarray):
    """``(value, group_trees)``: ``value[(uid, tree)]`` is the mean outcome of the tree's episodes (each episode
    counted once), ``group_trees[uid]`` lists the group's tree keys in batch order."""
    tree_episodes: Dict[Any, list] = defaultdict(list)
    group_trees: Dict[Any, list] = defaultdict(list)
    seen = set()
    for i in range(len(uid)):
        key = (uid[i], episode[i])
        if key in seen:
            continue
        seen.add(key)
        tree_key = (uid[i], tree_id[i])
        if tree_key not in tree_episodes:
            group_trees[uid[i]].append(tree_key)
        tree_episodes[tree_key].append(episode_score[key])
    value = {k: torch.stack(v).mean() for k, v in tree_episodes.items()}
    return value, dict(group_trees)


def compute_grpo_tree_advantage(
    token_level_rewards: torch.Tensor,  # [B, L] the episode outcome on the row's reward token
    response_mask: torch.Tensor,  # [B, L] 1 on policy tokens
    uid: np.ndarray,  # [B] group (task) of each row
    episode: np.ndarray,  # [B] episode of each row (rows of one episode share it)
    tree_id: np.ndarray,  # [B] tree of each row's episode (the leaves of one tree share it)
    branch_boundary: np.ndarray,  # [B] end of the shared prefix on the branching session's rows, else -1
    branch_lambda: float = 1.0,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advantages ``[B, L]`` as described in the module docstring. Returns ``(advantages, returns)``; they
    are equal, as in verl."""
    episode_score, _ = episode_outcomes(token_level_rewards, uid, episode)
    value, group_trees = tree_values(episode_score, uid, episode, tree_id)
    stats = {}  # mean and std of groups with more than one tree
    for group, keys in group_trees.items():
        if len(keys) > 1:
            stacked = torch.stack([value[key] for key in keys])
            stats[group] = (stacked.mean(), stacked.std())

    B, L = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)
    positions = torch.arange(L, device=token_level_rewards.device)
    for i in range(B):
        R = episode_score[(uid[i], episode[i])]
        V = value[(uid[i], tree_id[i])]
        if uid[i] in stats:
            mean, std = stats[uid[i]]
            group_term, prefix_term = R - mean, V - mean
        else:  # one tree in the group: no group comparison, only the within-tree term
            std = torch.tensor(1.0)
            group_term = prefix_term = torch.zeros(())
        scale = (std + epsilon) if norm_adv_by_std_in_grpo else 1.0
        leaf_adv = (group_term + branch_lambda * (R - V)) / scale
        boundary = int(branch_boundary[i])
        if boundary >= 0:
            prefix_adv = prefix_term / scale
            in_prefix = positions < boundary
            advantages[i] = torch.where(in_prefix, prefix_adv, leaf_adv)
        else:
            advantages[i] = leaf_adv
    advantages = advantages * response_mask
    return advantages, advantages


@register_estimator("grpo_tree", layout="per_segment")
def grpo_tree_per_segment(data: Any, config: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    def field(name):
        return batch_field(data, name, "grpo_tree", "per_segment")

    estimator_kwargs = config_value(config, "estimator_kwargs", None) or {}
    return compute_grpo_tree_advantage(
        token_level_rewards=field("token_level_rewards"),
        response_mask=field("action_mask"),
        uid=field("uid"),
        episode=field("batch_idx"),
        tree_id=field("tree_id"),
        branch_boundary=field("branch_boundary"),
        branch_lambda=float(estimator_kwargs.get("branch_lambda", 1.0)),
        norm_adv_by_std_in_grpo=bool(config_value(config, "norm_adv_by_std_in_grpo", True)),
    )
