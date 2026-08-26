"""Unit tests for the multi-turn GiGPO advantage estimator (core_gigpo)."""
import numpy as np
import torch

from agentfly.verl.trainer.ppo.core_gigpo import (
    compute_turn_ids,
    compute_gigpo_multiturn_advantage as gigpo,
)
from agentfly.verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _obj(list_of_lists):
    """Pack a list of per-turn lists into a 1D object array (no 2D collapse)."""
    a = np.empty(len(list_of_lists), dtype=object)
    a[:] = list_of_lists
    return a


# ----------------------------- turn_ids ------------------------------------

def test_turn_ids_basic():
    # runs of 1s are turns; 0->1 starts a turn; non-action tokens are -1
    am = torch.tensor([[1, 1, 0, 1, 0, 1, 1]], dtype=torch.float32)
    assert compute_turn_ids(am).tolist()[0] == [0, 0, -1, 1, -1, 2, 2]


def test_turn_ids_edges():
    am = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)
    ti = compute_turn_ids(am).tolist()
    assert ti[0] == [-1, -1, 0, 0]      # turn only at the end
    assert ti[1] == [0, -1, 1, -1]      # two single-token turns
    assert ti[2] == [-1, -1, -1, -1]    # no turns


# --------------------------- full advantage --------------------------------

def test_episode_plus_step():
    """Hand-computed 2-trajectory / 2-turn group.

    outcomes [1, 0] -> episode adv (std) [+1, -1].
    anchors traj0 [A,B], traj1 [A,C]; step rewards [0,1] / [0,0]; gamma=1.
    returns [1,1] / [0,0]. Anchor A group [1,0] -> traj0-t0 +1, traj1-t0 -1;
    B,C singletons -> 0. Combined = episode + step:
        traj0: token0 (turn0/A) 1+1=2 ; token2 (turn1/B) 1+0=1
        traj1: token0 (turn0/A) -1-1=-2; token2 (turn1/C) -1+0=-1
    """
    tlr = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    am = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    adv, ret = gigpo(
        tlr, am, np.array(["g", "g"], dtype=object),
        _obj([["A", "B"], ["A", "C"]]), _obj([[0.0, 1.0], [0.0, 0.0]]),
        gamma=1.0, step_advantage_w=1.0, norm_by_std=True,
    )
    assert np.allclose(adv.numpy(), [[2, 0, 1, 0], [-2, 0, -1, 0]], atol=1e-4)
    assert torch.equal(adv, ret)  # critic-free: returns == advantages


def test_reduces_to_grpo_meanonly():
    """With no step signal, GiGPO's episode part == GRPO (mean-only avoids std/singleton
    convention differences)."""
    B, L = 3, 2
    tlr = torch.zeros(B, L)
    am = torch.zeros(B, L)
    am[:, 0] = 1
    tlr[0, 0], tlr[1, 0], tlr[2, 0] = 1.0, 2.0, 3.0
    uid = np.array(["g", "g", "g"], dtype=object)
    g_adv, _ = gigpo(tlr, am, uid, _obj([["A"], ["A"], ["A"]]), _obj([[0.0], [0.0], [0.0]]),
                     gamma=1.0, step_advantage_w=1.0, norm_by_std=False)
    grpo_adv, _ = compute_grpo_outcome_advantage(tlr, am, uid, norm_adv_by_std_in_grpo=False)
    assert np.allclose(g_adv.numpy(), grpo_adv.numpy(), atol=1e-5)
    assert np.allclose(g_adv[:, 0].numpy(), [-1, 0, 1], atol=1e-5)  # [1,2,3]-mean(2)


def test_singleton_anchors_give_zero_step():
    """Nonzero step rewards but all-distinct anchors -> every step group singleton -> step
    advantage 0; only the episode advantage remains."""
    tlr = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    am = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uid = np.array(["g", "g"], dtype=object)
    adv, _ = gigpo(tlr, am, uid, _obj([["X"], ["Y"]]), _obj([[5.0], [3.0]]),
                   gamma=1.0, step_advantage_w=1.0, norm_by_std=True)
    # episode [1,0] (std) -> [+1,-1]; step 0
    assert np.allclose(adv[:, 0].numpy(), [1, -1], atol=1e-4)


def test_grouping_scoped_by_uid():
    """Identical anchor 'A' in different prompt groups must NOT cluster together."""
    tlr = torch.zeros(4, 2)
    am = torch.zeros(4, 2)
    am[:, 0] = 1
    uid = np.array(["g1", "g1", "g2", "g2"], dtype=object)
    adv, _ = gigpo(tlr, am, uid, _obj([["A"], ["A"], ["A"], ["A"]]),
                   _obj([[1.0], [0.0], [9.0], [0.0]]),
                   gamma=1.0, step_advantage_w=1.0, norm_by_std=False)
    # g1 A-group mean .5 -> [.5,-.5]; g2 A-group mean 4.5 -> [4.5,-4.5]; episode 0
    assert np.allclose(adv[:, 0].numpy(), [0.5, -0.5, 4.5, -4.5], atol=1e-4)


def test_step_advantage_weight_scales():
    tlr = torch.zeros(2, 2)
    am = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uid = np.array(["g", "g"], dtype=object)
    obs, sr = _obj([["A"], ["A"]]), _obj([[1.0], [0.0]])
    a1, _ = gigpo(tlr, am, uid, obs, sr, gamma=1.0, step_advantage_w=1.0, norm_by_std=False)
    a2, _ = gigpo(tlr, am, uid, obs, sr, gamma=1.0, step_advantage_w=2.0, norm_by_std=False)
    assert np.allclose(a1[:, 0].numpy(), [0.5, -0.5], atol=1e-4)   # A mean .5
    assert np.allclose(a2[:, 0].numpy(), [1.0, -1.0], atol=1e-4)   # scaled x2


def test_gamma_discounting():
    """gamma changes the discounted step return-to-go (isolated: all outcomes 0)."""
    tlr = torch.zeros(2, 4)
    am = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    uid = np.array(["g", "g"], dtype=object)
    obs = _obj([["X", "Y"], ["X", "Y"]])
    sr = _obj([[0.0, 1.0], [0.0, 0.0]])
    a1, _ = gigpo(tlr, am, uid, obs, sr, gamma=1.0, step_advantage_w=1.0, norm_by_std=False)
    a2, _ = gigpo(tlr, am, uid, obs, sr, gamma=0.5, step_advantage_w=1.0, norm_by_std=False)
    # traj0 turn0 is token0. gamma=1: X-returns [1,0] mean .5 -> .5. gamma=.5: [.5,0] mean .25 -> .25
    assert abs(a1[0, 0].item() - 0.5) < 1e-4
    assert abs(a2[0, 0].item() - 0.25) < 1e-4


def test_none_step_rewards_treated_as_zero():
    tlr = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    am = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uid = np.array(["g", "g"], dtype=object)
    adv, _ = gigpo(tlr, am, uid, _obj([["A"], ["A"]]), _obj([[None], [None]]),
                   gamma=1.0, step_advantage_w=1.0, norm_by_std=True)
    # None -> 0 returns -> step 0; episode [1,0] (std) -> [+1,-1]
    assert np.allclose(adv[:, 0].numpy(), [1, -1], atol=1e-4)


def test_advantages_only_on_policy_tokens():
    """Non-policy tokens (mask 0) must stay 0 in both episode and step parts."""
    tlr = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    am = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    uid = np.array(["g"], dtype=object)
    adv, _ = gigpo(tlr, am, uid, _obj([["A", "B"]]), _obj([[0.0, 1.0]]),
                   gamma=1.0, step_advantage_w=1.0, norm_by_std=False)
    assert adv[0, 1].item() == 0.0 and adv[0, 3].item() == 0.0  # masked positions
