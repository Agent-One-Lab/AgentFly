"""grpo_tree: GRPO whose group statistics are over trees of branched episodes; the shared prefix of a tree
gets the tree's mean outcome, each leaf's own tokens get the outcome term plus the within-tree comparison."""

import numpy as np
import torch

from agentfly.algorithms import get_estimator
from agentfly.algorithms.grpo_episode import compute_grpo_episode_advantage
from agentfly.algorithms.grpo_tree import compute_grpo_tree_advantage

EPS = 1e-6


def rows(scores, length=4):
    rewards = torch.zeros(len(scores), length)
    rewards[:, -1] = torch.tensor(scores, dtype=torch.float32)
    return rewards, torch.ones(len(scores), length)


def obj(values):
    return np.array(values, dtype=object)


def test_unbranched_groups_match_grpo_episode():
    rewards, mask = rows([1.0, 0.0, 0.0, 1.0])
    uid, episode = obj(["G"] * 4), np.array([0, 1, 1, 2])
    tree, boundary = obj(["t0", "t1", "t1", "t2"]), np.full(4, -1)
    tree_adv, _ = compute_grpo_tree_advantage(rewards, mask, uid, episode, tree, boundary, branch_lambda=1.0)
    ep_adv, _ = compute_grpo_episode_advantage(rewards, mask, uid, episode)
    torch.testing.assert_close(tree_adv, ep_adv)


def test_branched_tree_prefix_and_leaves():
    # Group of 3 trees: t0 unbranched (R=0), t1 branched into episodes 1 (R=1) and 2 (R=0), t2 unbranched (R=1).
    # Episode 1 row is the branching session's row in the source leaf (prefix = first 2 tokens); episode 2 row
    # is the branch leaf's row (its prefix copy is masked by the converter: mask 0 on the first 2 tokens).
    rewards, mask = rows([0.0, 1.0, 0.0, 1.0])
    mask[2, :2] = 0
    uid, episode = obj(["G"] * 4), np.array([0, 1, 2, 3])
    tree, boundary = obj(["t0", "t1", "t1", "t2"]), np.array([-1, 2, 2, -1])
    adv, ret = compute_grpo_tree_advantage(rewards, mask, uid, episode, tree, boundary, branch_lambda=1.0)
    assert torch.equal(adv, ret)
    values = torch.tensor([0.0, 0.5, 1.0])  # tree values, each tree once
    b, s = values.mean(), values.std()
    scale = s + EPS
    # unbranched rows: R - b
    torch.testing.assert_close(adv[0], torch.full((4,), ((0.0 - b) / scale).item()))
    torch.testing.assert_close(adv[3], torch.full((4,), ((1.0 - b) / scale).item()))
    # source leaf: prefix gets V - b, its own tokens get (R - b) + (R - V)
    V = 0.5
    torch.testing.assert_close(adv[1, :2], torch.full((2,), ((V - b) / scale).item()))
    torch.testing.assert_close(adv[1, 2:], torch.full((2,), ((1.0 - b + (1.0 - V)) / scale).item()))
    # branch leaf: masked prefix, own tokens get the opposite sibling term
    assert torch.equal(adv[2, :2], torch.zeros(2))
    torch.testing.assert_close(adv[2, 2:], torch.full((2,), ((0.0 - b + (0.0 - V)) / scale).item()))
    # the sibling terms are opposite: (R1 - V) = -(R2 - V)
    own1 = adv[1, 2] * scale - (1.0 - b)
    own2 = adv[2, 2] * scale - (0.0 - b)
    torch.testing.assert_close(own1, -own2)


def test_lambda_zero_keeps_group_term_only():
    rewards, mask = rows([0.0, 1.0, 0.0, 1.0])
    uid, episode = obj(["G"] * 4), np.array([0, 1, 2, 3])
    tree, boundary = obj(["t0", "t1", "t1", "t2"]), np.array([-1, 2, 2, -1])
    adv, _ = compute_grpo_tree_advantage(rewards, mask, uid, episode, tree, boundary, branch_lambda=0.0)
    b, s = torch.tensor([0.0, 0.5, 1.0]).mean(), torch.tensor([0.0, 0.5, 1.0]).std()
    torch.testing.assert_close(adv[1, 2:], torch.full((2,), ((1.0 - b) / (s + EPS)).item()))
    torch.testing.assert_close(adv[1, :2], torch.full((2,), ((0.5 - b) / (s + EPS)).item()))


def test_single_tree_group_keeps_only_within_tree_term():
    rewards, mask = rows([1.0, 0.0])
    uid, episode = obj(["G", "G"]), np.array([0, 1])
    tree, boundary = obj(["t", "t"]), np.array([2, 2])
    adv, _ = compute_grpo_tree_advantage(rewards, mask, uid, episode, tree, boundary, branch_lambda=1.0)
    torch.testing.assert_close(adv[0, :2], torch.zeros(2))  # prefix: V - V
    torch.testing.assert_close(adv[0, 2:], torch.full((2,), 0.5 / (1.0 + EPS)))
    torch.testing.assert_close(adv[1, 2:], torch.full((2,), -0.5 / (1.0 + EPS)))


def test_padding_copies_change_nothing():
    rewards, mask = rows([0.0, 1.0, 0.0, 1.0, 1.0])  # last row repeats episode 3
    uid, episode = obj(["G"] * 5), np.array([0, 1, 2, 3, 3])
    tree, boundary = obj(["t0", "t1", "t1", "t2", "t2"]), np.array([-1, 2, 2, -1, -1])
    padded, _ = compute_grpo_tree_advantage(rewards, mask, uid, episode, tree, boundary)
    plain, _ = compute_grpo_tree_advantage(rewards[:4], mask[:4], uid[:4], episode[:4], tree[:4], boundary[:4])
    torch.testing.assert_close(padded[:4], plain)
    torch.testing.assert_close(padded[4], padded[3])


def test_registered_for_per_segment():
    class Data:
        def __init__(self):
            rewards, mask = rows([0.0, 1.0, 0.0, 1.0])
            self.batch = {"token_level_rewards": rewards, "action_mask": mask}
            self.non_tensor_batch = {
                "uid": obj(["G"] * 4), "batch_idx": np.array([0, 1, 2, 3]),
                "tree_id": obj(["t0", "t1", "t1", "t2"]), "branch_boundary": np.array([-1, 2, 2, -1]),
            }

    fn = get_estimator("grpo_tree", "per_segment")
    adv, ret = fn(Data(), {"estimator_kwargs": {"branch_lambda": 1.0}, "norm_adv_by_std_in_grpo": True})
    assert adv.shape == (4, 4) and torch.equal(adv, ret)


def test_metrics_measure_leaf_disagreement():
    # Group G: t0 unbranched (R=0), t1 branched with leaves 1.0 and 0.0 (split), t2 branched with leaves 0.9 and 0.8
    # (differ, no split). Group H: one tree, branched, both leaves 0 (tie). Episode 1 has two rows (two sessions).
    import types

    from agentfly.algorithms import compute_estimator_metrics

    scores = [0.0, 1.0, 1.0, 0.0, 0.9, 0.8, 0.0, 0.0]
    rewards, mask = rows(scores)
    uid = obj(["G"] * 6 + ["H"] * 2)
    episode = np.array([0, 1, 1, 2, 3, 4, 0, 1])
    tree = obj(["t0", "t1", "t1", "t1", "t2", "t2", "h", "h"])
    data = types.SimpleNamespace(
        batch={"token_level_rewards": rewards, "action_mask": mask},
        non_tensor_batch={"uid": uid, "batch_idx": episode, "tree_id": tree},
    )
    m = compute_estimator_metrics("grpo_tree", data, {"estimator_kwargs": {"branch_lambda": 2.0}})
    assert m["grpo_tree/branched_trees"] == 0.75  # t1, t2, h of 4 trees
    assert m["grpo_tree/leaves_differ"] == 2 / 3
    assert m["grpo_tree/leaves_split"] == 1 / 3
    assert abs(m["grpo_tree/leaf_gap"] - (1.0 + 0.1 + 0.0) / 3) < 1e-6
    # within: lambda |R - V| -> t1: 2*0.5 twice, t2: 2*0.05 twice, h: 0 twice
    assert abs(m["grpo_tree/within_term"] - (1.0 + 1.0 + 0.1 + 0.1) / 6) < 1e-6
    # group: G's b = mean(0, 0.5, 0.85) = 0.45; H has one tree -> 0
    b = 0.45
    expected = (abs(1.0 - b) + abs(0.0 - b) + abs(0.9 - b) + abs(0.8 - b)) / 6
    assert abs(m["grpo_tree/group_term"] - expected) < 1e-6


def test_metrics_without_branching():
    import types

    from agentfly.algorithms import compute_estimator_metrics

    rewards, mask = rows([1.0, 0.0])
    data = types.SimpleNamespace(
        batch={"token_level_rewards": rewards, "action_mask": mask},
        non_tensor_batch={"uid": obj(["G", "G"]), "batch_idx": np.array([0, 1]), "tree_id": obj(["a", "b"])},
    )
    assert compute_estimator_metrics("grpo_tree", data, {}) == {"grpo_tree/branched_trees": 0.0}
