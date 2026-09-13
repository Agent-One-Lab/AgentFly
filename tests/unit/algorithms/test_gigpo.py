"""GiGPO estimators in ``agentfly.algorithms`` for both batch layouts.

``per_step`` is checked bit-for-bit against verl-agent's own ``gigpo/core_gigpo.py``;
``per_segment`` must normalise exactly like verl-agent too (sample std, ``gigpo_mode``) and
give each turn the same step advantage the ``per_step`` layout gives that step.
"""

import sys
import types

import numpy as np
import pytest
import torch

from agentfly.algorithms import common
from agentfly.algorithms.gigpo import (
    compute_gigpo_per_segment_advantage as per_segment,
    compute_gigpo_per_step_advantage as per_step,
)


def _obj(values):
    return common.object_array(values)


# ------------------------------------------------------------------------------------ #
# per_step vs verl-agent's real code                                                    #
# ------------------------------------------------------------------------------------ #
_VERL_AGENT = "/mnt/weka/home/renxi.wang/verl-agent"


@pytest.fixture(scope="module")
def verl_agent():
    stub = types.ModuleType("verl")
    stub.DataProto = object
    sys.modules.setdefault("verl", stub)
    if _VERL_AGENT not in sys.path:
        sys.path.insert(0, _VERL_AGENT)
    return pytest.importorskip("gigpo.core_gigpo")


def _synthetic_steps():
    """2 tasks x 2 trajectories, a few steps each, L=3; shared anchors, sparse rewards, invalids."""
    L = 3
    rows = [
        ("gA", "gA-0", "s0", 0.0, 1.0, 10.0), ("gA", "gA-0", "s1", 0.0, 0.0, 10.0), ("gA", "gA-0", "s2", 1.0, 1.0, 10.0),
        ("gA", "gA-1", "s0", 0.0, 1.0, 0.0), ("gA", "gA-1", "s1", 0.0, 1.0, 0.0),
        ("gB", "gB-0", "t0", 0.0, 1.0, 10.0), ("gB", "gB-0", "t1", 1.0, 1.0, 10.0),
        ("gB", "gB-1", "t0", 0.0, 0.0, 0.0), ("gB", "gB-1", "t1", 0.0, 1.0, 0.0),
    ]
    B = len(rows)
    response_mask = torch.zeros(B, L)
    response_mask[:, 1:] = 1.0
    tlr = torch.zeros(B, L)
    tlr[:, -1] = torch.tensor([r[5] for r in rows])
    return dict(
        tlr=tlr, response_mask=response_mask, B=B, L=L,
        uid=np.array([r[0] for r in rows], dtype=object),
        traj=np.array([r[1] for r in rows], dtype=object),
        anchor=np.array([r[2] for r in rows], dtype=object),
        step_r=np.array([r[3] for r in rows], dtype=np.float32),
        valid=np.array([r[4] for r in rows], dtype=np.float32),
    )


def _verl_agent_reference(V, d, gamma, coef, w, mode):
    B = d["B"]
    returns = np.zeros(B, dtype=np.float32)
    for u in np.unique(d["traj"]):
        idx = np.where(d["traj"] == u)[0]
        run = 0.0
        for t in reversed(range(len(idx))):
            run = d["step_r"][idx[t]] + gamma * run
            returns[idx[t]] = run
    invalid = 1.0 - d["valid"]
    returns = returns - coef * invalid
    tlr = d["tlr"].clone()
    tlr[torch.arange(B), -1] -= torch.tensor(coef * invalid)
    adv, _ = V.compute_gigpo_outcome_advantage(
        token_level_rewards=tlr, step_rewards=torch.tensor(returns), response_mask=d["response_mask"],
        anchor_obs=d["anchor"], index=d["uid"], traj_index=d["traj"], step_advantage_w=w, mode=mode,
    )
    return adv


@pytest.mark.parametrize("mode", ["mean_std_norm", "mean_norm"])
def test_per_step_matches_verl_agent(verl_agent, mode):
    d = _synthetic_steps()
    adv, ret = per_step(
        token_level_rewards=d["tlr"], response_mask=d["response_mask"], step_env_rewards=d["step_r"],
        anchor_obs=d["anchor"], uid=d["uid"], traj_uid=d["traj"], is_action_valid=d["valid"],
        gamma=0.95, step_advantage_w=1.0, mode=mode, invalid_action_penalty_coef=0.1,
    )
    ref = _verl_agent_reference(verl_agent, d, 0.95, 0.1, 1.0, mode)
    assert torch.allclose(adv, ref, atol=1e-5), f"{mode}: max diff {(adv - ref).abs().max()}"
    assert torch.equal(adv, ret)


def test_common_normalisers_are_verl_agent_verbatim(verl_agent):
    """The shared normalisers give verl-agent's numbers on the same inputs."""
    V = verl_agent
    scores = torch.tensor([[0.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 4.0], [0.0, 7.0]])
    mask = torch.ones(5, 2)
    index = np.array(["a", "a", "a", "b", "c"], dtype=object)
    traj = np.arange(5)
    for remove_std in (True, False):
        assert torch.equal(
            common.episode_norm_reward(scores.clone(), mask, index, traj, 1e-6, remove_std),
            V.episode_norm_reward(scores.clone(), mask, index, traj, 1e-6, remove_std),
        )
        step = torch.tensor([1.0, 0.0, 2.0, 5.0, 3.0])
        assert torch.equal(
            common.step_norm_reward(step, mask, index, 1e-6, remove_std),
            V.step_norm_reward(step, mask, index, 1e-6, remove_std),
        )


def test_per_step_padding_rows_are_ignored():
    d = _synthetic_steps()
    B, L, pad = d["B"], d["L"], 2
    base, _ = per_step(
        token_level_rewards=d["tlr"], response_mask=d["response_mask"], step_env_rewards=d["step_r"],
        anchor_obs=d["anchor"], uid=d["uid"], traj_uid=d["traj"], is_action_valid=d["valid"],
        gamma=0.95, mode="mean_std_norm",
    )
    repeat = lambda a: np.concatenate([a, np.array([a[-1]] * pad, dtype=a.dtype)])  # noqa: E731
    padded, _ = per_step(
        token_level_rewards=torch.cat([d["tlr"], torch.zeros(pad, L)]),
        response_mask=torch.cat([d["response_mask"], torch.zeros(pad, L)]),
        step_env_rewards=np.concatenate([d["step_r"], np.zeros(pad, np.float32)]),
        anchor_obs=repeat(d["anchor"]), uid=repeat(d["uid"]), traj_uid=repeat(d["traj"]),
        is_action_valid=np.concatenate([d["valid"], np.ones(pad, np.float32)]),
        active_masks=np.array([1.0] * B + [0.0] * pad, dtype=np.float32),
        gamma=0.95, mode="mean_std_norm",
    )
    assert torch.all(padded[B:] == 0)
    assert torch.allclose(padded[:B], base, atol=1e-6)


def test_discounted_returns():
    out = common.compute_step_discounted_returns(
        np.array([0.0, 0.0, 1.0, 0.0, 5.0], dtype=np.float32), np.array(["a", "a", "a", "b", "b"], dtype=object), 0.5
    )
    assert np.allclose(out, [0.25, 0.5, 1.0, 2.5, 5.0])


# ------------------------------------------------------------------------------------ #
# turn spans                                                                            #
# ------------------------------------------------------------------------------------ #
def test_turn_ids():
    assert common.compute_turn_ids(torch.tensor([[1, 1, 0, 1, 0, 1, 1]], dtype=torch.float32)).tolist()[0] == [0, 0, -1, 1, -1, 2, 2]
    ti = common.compute_turn_ids(torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)).tolist()
    assert ti == [[-1, -1, 0, 0], [0, -1, 1, -1], [-1, -1, -1, -1]]


def test_terminal_token_mask():
    m = common.terminal_token_mask(torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
    assert m.tolist() == [[0, 1, 0], [0, 0, 0], [0, 0, 1]]


# ------------------------------------------------------------------------------------ #
# per_segment                                                                           #
# ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("mode,expected", [
    ("mean_norm", [7.5, -2.5, -2.5, -2.5]),          # subtract the group mean only
    ("mean_std_norm", [1.5, -0.5, -0.5, -0.5]),      # ÷ sample std (5.0), like verl-agent
])
def test_per_segment_honours_gigpo_mode(mode, expected):
    tlr = torch.zeros(4, 2)
    tlr[:, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    am = torch.zeros(4, 2)
    am[:, 0] = 1.0
    adv, _ = per_segment(tlr, am, np.array(["g"] * 4, dtype=object), _obj([["A"]] * 4), _obj([[0.0]] * 4),
                         gamma=1.0, step_advantage_w=1.0, mode=mode)
    assert np.allclose(adv[:, 0].numpy(), expected, atol=1e-4)


def test_per_segment_episode_plus_step_hand_computed():
    """outcomes [1, 0] -> episode ±0.7071 (sample std). Anchor A returns [1, 0] -> ±0.7071;
    B and C are single-step groups -> 0 (their own mean)."""
    tlr = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    am = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    adv, ret = per_segment(tlr, am, np.array(["g", "g"], dtype=object),
                           _obj([["A", "B"], ["A", "C"]]), _obj([[0.0, 1.0], [0.0, 0.0]]),
                           gamma=1.0, step_advantage_w=1.0, mode="mean_std_norm")
    r = 2 ** -0.5
    assert np.allclose(adv.numpy(), [[2 * r, 0, r, 0], [-2 * r, 0, -r, 0]], atol=1e-4)
    assert torch.equal(adv, ret)


def test_per_segment_step_advantage_equals_per_step():
    """Same steps in both layouts (zero outcomes, so the episode term is 0): every turn gets the
    step advantage its per_step row gets."""
    trajs = {  # traj -> (uid, [(anchor, step_reward), ...])
        "g1-a": ("g1", [("s0", 0.0), ("s1", 1.0), ("s2", 0.0)]),
        "g1-b": ("g1", [("s0", 0.0), ("s1", 0.0)]),
        "g2-a": ("g2", [("t0", 2.0), ("t1", 0.0)]),
        "g2-b": ("g2", [("t0", 0.0), ("t1", 3.0)]),
    }
    # per_step batch: one row per step, L=1
    rows = [(u, traj, a, r) for traj, (u, steps) in trajs.items() for a, r in steps]
    n = len(rows)
    step_adv, _ = per_step(
        token_level_rewards=torch.zeros(n, 1), response_mask=torch.ones(n, 1),
        step_env_rewards=np.array([r[3] for r in rows], dtype=np.float32),
        anchor_obs=np.array([r[2] for r in rows], dtype=object),
        uid=np.array([r[0] for r in rows], dtype=object), traj_uid=np.array([r[1] for r in rows], dtype=object),
        gamma=0.9, mode="mean_std_norm", invalid_action_penalty_coef=0.0,
    )
    # per_segment batch: one row per trajectory, turn t on token 2t
    L = 2 * max(len(s) for _, s in trajs.values())
    am = torch.zeros(len(trajs), L)
    for b, (_, steps) in enumerate(trajs.values()):
        am[b, 0:2 * len(steps):2] = 1.0
    seg_adv, _ = per_segment(
        torch.zeros(len(trajs), L), am, np.array([u for u, _ in trajs.values()], dtype=object),
        _obj([[a for a, _ in s] for _, s in trajs.values()]), _obj([[r for _, r in s] for _, s in trajs.values()]),
        gamma=0.9, step_advantage_w=1.0, mode="mean_std_norm", invalid_action_penalty_coef=0.0,
    )
    k = 0
    for b, (_, steps) in enumerate(trajs.values()):
        for t in range(len(steps)):
            assert abs(seg_adv[b, 2 * t].item() - step_adv[k, 0].item()) < 1e-5, (b, t)
            k += 1


def test_per_segment_grouping_scoped_by_uid():
    tlr = torch.zeros(4, 2)
    am = torch.zeros(4, 2)
    am[:, 0] = 1
    adv, _ = per_segment(tlr, am, np.array(["g1", "g1", "g2", "g2"], dtype=object), _obj([["A"]] * 4),
                         _obj([[1.0], [0.0], [9.0], [0.0]]), gamma=1.0, step_advantage_w=1.0, mode="mean_norm")
    assert np.allclose(adv[:, 0].numpy(), [0.5, -0.5, 4.5, -4.5], atol=1e-4)


def test_per_segment_weight_gamma_none_and_mask():
    tlr = torch.zeros(2, 4)
    am = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    uid = np.array(["g", "g"], dtype=object)
    obs, sr = _obj([["X", "Y"], ["X", "Y"]]), _obj([[0.0, 1.0], [0.0, 0.0]])
    a1, _ = per_segment(tlr, am, uid, obs, sr, gamma=1.0, step_advantage_w=1.0, mode="mean_norm")
    a2, _ = per_segment(tlr, am, uid, obs, sr, gamma=0.5, step_advantage_w=2.0, mode="mean_norm")
    assert abs(a1[0, 0].item() - 0.5) < 1e-5            # X returns [1, 0]
    assert abs(a2[0, 0].item() - 2 * 0.25) < 1e-5       # gamma .5 -> [.5, 0], weight 2
    assert a1[:, 1].abs().sum() == 0 and a1[:, 3].abs().sum() == 0  # non-policy tokens stay 0
    a3, _ = per_segment(tlr, am, uid, obs, _obj([[None, None], [None, None]]), gamma=1.0, mode="mean_norm")
    assert a3.abs().sum() == 0                          # None rewards count as 0


def test_per_segment_invalid_penalty_after_discount():
    tlr = torch.zeros(2, 2)
    am = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    adv, _ = per_segment(tlr, am, np.array(["g", "g"], dtype=object), _obj([["A"], ["A"]]), _obj([[0.0], [0.0]]),
                         step_invalids=_obj([[1.0], [0.0]]), gamma=1.0, mode="mean_norm", invalid_action_penalty_coef=0.1)
    assert np.allclose(adv[:, 0].numpy(), [-0.05, 0.05], atol=1e-6)   # returns [-.1, 0] - mean


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="normalisation mode"):
        per_segment(torch.zeros(1, 1), torch.ones(1, 1), np.array(["g"], dtype=object), _obj([["A"]]), _obj([[0.0]]),
                    mode="std_only")
