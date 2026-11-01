"""grpo_episode: GRPO whose group mean and std count each episode once, however many rows
(segments, subagent sessions, padding copies) it has in a per_segment batch."""

import numpy as np
import pytest
import torch

from agentfly.algorithms import get_estimator
from agentfly.algorithms.grpo_episode import compute_grpo_episode_advantage


def rows(scores):
    """Token-level rewards [B, 3] with each row's score on its last token, and a full mask."""
    rewards = torch.zeros(len(scores), 3)
    rewards[:, -1] = torch.tensor(scores, dtype=torch.float32)
    return rewards, torch.ones(len(scores), 3)


def test_each_episode_counts_once():
    # Episode 0 (score 1) has 1 row; episode 1 (score 0) has 3 rows.
    rewards, mask = rows([1.0, 0.0, 0.0, 0.0])
    uid = np.array(["G"] * 4, dtype=object)
    episode = np.array([0, 1, 1, 1])
    adv, ret = compute_grpo_episode_advantage(rewards, mask, uid, episode)
    mean, std = 0.5, torch.tensor([1.0, 0.0]).std().item()  # over 2 episodes, not 4 rows
    expected = [(1 - mean) / (std + 1e-6)] + [(0 - mean) / (std + 1e-6)] * 3
    torch.testing.assert_close(adv[:, 0], torch.tensor(expected))
    assert torch.equal(adv, ret)
    assert (adv == adv[:, :1]).all()  # broadcast over the row's policy tokens


def test_padding_copies_change_nothing():
    rewards, mask = rows([1.0, 0.0])
    uid = np.array(["G", "G"], dtype=object)
    plain, _ = compute_grpo_episode_advantage(rewards, mask, uid, np.array([0, 1]))
    rewards_p, mask_p = rows([1.0, 0.0, 0.0, 0.0])  # the converter repeats the last row as padding
    padded, _ = compute_grpo_episode_advantage(
        rewards_p, mask_p, np.array(["G"] * 4, dtype=object), np.array([0, 1, 1, 1])
    )
    torch.testing.assert_close(padded[:2], plain)


def test_one_episode_group_gets_zero_and_groups_are_separate():
    rewards, mask = rows([1.0, 1.0, 0.0, 3.0])
    uid = np.array(["A", "A", "B", "B"], dtype=object)
    episode = np.array([0, 0, 1, 2])  # group A: one episode (two segments); group B: two episodes
    adv, _ = compute_grpo_episode_advantage(rewards, mask, uid, episode, norm_adv_by_std_in_grpo=False)
    assert adv[:, 0].tolist() == pytest.approx([0.0, 0.0, -1.5, 1.5])


def test_policy_token_mask_is_applied():
    rewards, _ = rows([1.0, 0.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    adv, _ = compute_grpo_episode_advantage(rewards, mask, np.array(["G", "G"], dtype=object), np.array([0, 1]))
    assert adv[0, 2] == 0 and adv[1, 0] == 0


def test_registered_for_per_segment_only():
    assert get_estimator("grpo_episode", "per_segment").__name__ == "grpo_episode_per_segment"
    with pytest.raises(Exception, match="per_segment"):
        get_estimator("grpo_episode", "per_step")
