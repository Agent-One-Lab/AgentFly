"""rao: each row is trained on its own session's reward minus a leave-one-out baseline of the task's
other main-agent outcomes, optionally weighted so each depth of the agent tree counts equally."""

import types

import numpy as np
import pytest
import torch

from agentfly.algorithms import compute_estimator_metrics, get_estimator
from agentfly.algorithms.rao import compute_rao_advantage, depth_weights, rao_per_segment


def batch(outcomes, session_rewards, depths, session_ids, uid, episode, tokens=None):
    """Rows of a per_segment batch: each row's episode outcome on its last token; ``tokens[i]`` policy tokens."""
    n = len(outcomes)
    tokens = tokens or [3] * n
    width = max(tokens + [1])
    rewards = torch.zeros(n, width)
    rewards[:, -1] = torch.tensor(outcomes, dtype=torch.float32)
    mask = torch.zeros(n, width)
    for i, t in enumerate(tokens):
        mask[i, :t] = 1
    return dict(
        token_level_rewards=rewards,
        response_mask=mask,
        session_reward=np.array(session_rewards, dtype=np.float32),
        session_depth=np.array(depths, dtype=np.int32),
        session_id=np.array(session_ids, dtype=np.int32),
        uid=np.array(uid, dtype=object),
        episode=np.array(episode),
    )


# Three episodes of one task: main agents score 1, 0, 0; episode 1 has a subagent that scored 1.
EXAMPLE = dict(
    outcomes=[1.0, 0.0, 0.0, 0.0],
    session_rewards=[1.0, 0.0, 1.0, 0.0],
    depths=[0, 0, 1, 0],
    session_ids=[0, 0, 1, 0],
    uid=["G"] * 4,
    episode=[0, 1, 1, 2],
)


def test_leave_one_out_baseline_and_own_session_reward():
    adv = compute_rao_advantage(**batch(**EXAMPLE), depth_weighting=False)
    # Baselines: episode 0 -> (0 + 0) / 2 = 0; episodes 1, 2 -> (1 + 0) / 2 = 0.5.
    assert adv[:, 0].tolist() == pytest.approx([1.0, -0.5, 0.5, -0.5])
    assert (adv == adv[:, :1]).all()  # broadcast over the row's policy tokens


def test_depth_weights_give_each_depth_equal_weight_and_mean_one_per_token():
    rows = batch(**EXAMPLE)
    w = depth_weights(rows["response_mask"], rows["session_depth"], rows["session_id"], rows["uid"], rows["episode"])
    # 3 main-agent sessions and 1 subagent session: raw weights 1/3 and 1, every row has 3 tokens.
    raw = torch.tensor([1 / 3, 1 / 3, 1.0, 1 / 3])
    torch.testing.assert_close(w, raw * (4 / raw.sum()))
    assert (w * 3).sum().item() == pytest.approx(12.0)  # mean 1 over the 12 policy tokens
    adv = compute_rao_advantage(**rows)
    plain = compute_rao_advantage(**rows, depth_weighting=False)
    torch.testing.assert_close(adv[:, 0], plain[:, 0] * w)


def test_token_rescaling_counts_policy_tokens():
    rows = batch(**{**EXAMPLE, "tokens": [1, 1, 6, 1]})
    w = depth_weights(rows["response_mask"], rows["session_depth"], rows["session_id"], rows["uid"], rows["episode"])
    tokens = torch.tensor([1.0, 1.0, 6.0, 1.0])
    assert (w * tokens).sum().item() == pytest.approx(tokens.sum().item())
    assert w[2] / w[0] == pytest.approx(3.0)  # still 1 vs 1/3 per session


def test_padding_copies_change_nothing():
    plain = compute_rao_advantage(**batch(**EXAMPLE))
    # The converter repeats the last row as padding, with no policy tokens.
    padded_rows = {key: value + value[-1:] for key, value in EXAMPLE.items()}
    padded = compute_rao_advantage(**batch(**padded_rows, tokens=[3, 3, 3, 3, 0]))
    torch.testing.assert_close(padded[:4], plain)
    assert padded[4].abs().sum() == 0


def test_a_session_split_over_rows_counts_once():
    # Episode 1's subagent has two rows (context folding): still one session at depth 1.
    split = {key: value[:3] + value[2:] for key, value in EXAMPLE.items()}
    rows = batch(**split)
    w = depth_weights(rows["response_mask"], rows["session_depth"], rows["session_id"], rows["uid"], rows["episode"])
    assert w[2] == w[3] and w[2] / w[0] == pytest.approx(3.0)


def test_one_episode_group_gets_zero_and_groups_are_separate():
    rows = batch(
        outcomes=[1.0, 1.0, 0.0, 1.0],
        session_rewards=[1.0, 0.0, 0.0, 1.0],
        depths=[0, 1, 0, 0],
        session_ids=[0, 1, 0, 0],
        uid=["A", "A", "B", "B"],  # A: one episode (main agent and a subagent); B: two episodes
        episode=[0, 0, 1, 2],
    )
    adv = compute_rao_advantage(**rows, depth_weighting=False)
    assert adv[:, 0].tolist() == pytest.approx([0.0, 0.0, -1.0, 1.0])


def data_proto(rows):
    tensors = {"token_level_rewards": rows["token_level_rewards"], "action_mask": rows["response_mask"]}
    non_tensor = {
        "session_reward": rows["session_reward"],
        "session_depth": rows["session_depth"],
        "session_id": rows["session_id"],
        "uid": rows["uid"],
        "batch_idx": rows["episode"],
    }
    return types.SimpleNamespace(batch=tensors, non_tensor_batch=non_tensor)


def test_registered_for_per_segment_with_depth_weighting_on_by_default():
    assert get_estimator("rao", "per_segment") is rao_per_segment
    with pytest.raises(Exception, match="per_segment"):
        get_estimator("rao", "per_step")
    rows = batch(**EXAMPLE)
    adv, ret = rao_per_segment(data_proto(rows), {})
    torch.testing.assert_close(adv, compute_rao_advantage(**rows))
    assert torch.equal(adv, ret)
    off, _ = rao_per_segment(data_proto(rows), {"estimator_kwargs": {"depth_weighting": False}})
    torch.testing.assert_close(off, compute_rao_advantage(**rows, depth_weighting=False))


def test_metrics_per_depth():
    rows = batch(**EXAMPLE)
    data = data_proto(rows)
    data.batch["advantages"] = compute_rao_advantage(**rows, depth_weighting=False)
    metrics = compute_estimator_metrics("rao", data, {})
    assert metrics["rao/depth0/sessions"] == 3 and metrics["rao/depth1/sessions"] == 1
    assert metrics["rao/depth0/reward"] == pytest.approx(1 / 3) and metrics["rao/depth1/reward"] == 1.0
    assert metrics["rao/depth0/advantage"] == pytest.approx(0.0) and metrics["rao/depth1/advantage"] == 0.5
