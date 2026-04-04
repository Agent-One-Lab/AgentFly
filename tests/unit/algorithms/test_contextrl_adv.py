import torch
import numpy as np
from agentfly.verl.trainer.ppo.core_algos import compute_contextrl_advantage_return


def test_single_segment_no_summary():
    """
    Test 1: Single segment, no summarization.
    Should behave EXACTLY like standard GAE. No high-level advantage
    since there's only one segment (no summarization happened).

    Layout (L=8):
      pos: 0  1  2  3  4  5  6  7
      mask: 0  0  1  1  1  0  1  1
      rew:  0  0  0  0  0  0  0  1.0
      val:  0  0 .5 .4 .3  0 .2 .1

    gamma=1, lam=1. V^high_next=0 (single segment).
    GAE with gamma=lam=1 reduces to: adv_t = G_t - V_t (Monte Carlo).
    Return-to-go is 1.0 from any position (gamma=1, only reward at pos 7).
    
    Expected:
      pos 7: adv = 1.0 - 0.1 = 0.9
      pos 6: adv = 1.0 - 0.2 = 0.8
      pos 4: adv = 1.0 - 0.3 = 0.7
      pos 3: adv = 1.0 - 0.4 = 0.6
      pos 2: adv = 1.0 - 0.5 = 0.5
    """
    print("=" * 60)
    print("Test 1: Single segment, no summarization")
    print("=" * 60)

    B, L = 1, 8
    response_mask = torch.zeros(B, L)
    response_mask[0, [2, 3, 4, 6, 7]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[0, 7] = 1.0

    values = torch.zeros(B, L)
    values[0, 2] = 0.5
    values[0, 3] = 0.4
    values[0, 4] = 0.3
    values[0, 6] = 0.2
    values[0, 7] = 0.1

    index = np.array(["groupA"])
    segment_index = np.array([0])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=1.0, lam=1.0
    )

    print("Response positions: [2, 3, 4, 6, 7]")
    print("Values:            [0.5, 0.4, 0.3, 0.2, 0.1]")
    print(f"V^high_next = 0 (single segment, no high-level advantage)")
    print()
    for p in [2, 3, 4, 6, 7]:
        print(f"  pos {p}: adv={adv[0, p].item():.4f}, ret={ret[0, p].item():.4f}")
    print()
    print("Expected (pure low-level GAE, gamma=lam=1 -> Monte Carlo):")
    print("  pos 2: 0.5, pos 3: 0.6, pos 4: 0.7, pos 6: 0.8, pos 7: 0.9")
    _check("Test 1", [
        (adv[0, 2].item(), 0.5),
        (adv[0, 3].item(), 0.6),
        (adv[0, 4].item(), 0.7),
        (adv[0, 6].item(), 0.8),
        (adv[0, 7].item(), 0.9),
    ])
    print()


def test_two_segments_sparse_reward():
    """
    Test 2: Two segments in one rollout, reward only at the end.

    Segment 0 (b=0): response at [1, 2, 3], rewards all 0
      values: [0, 0.3, 0.2, 0.1, 0, 0]
    Segment 1 (b=1): response at [1, 2, 3], reward=1.0 at pos 3
      values: [0, 0.6, 0.4, 0.2, 0, 0]

    V^high(seg0) = 0.3, V^high(seg1) = 0.6, V^high_next(seg1) = 0.

    Low-level GAE for segment 1 (bootstraps to 0):
      pos 3: delta = 1.0 + 0 - 0.2 = 0.8,   gae = 0.8
      pos 2: delta = 0 + 0.2 - 0.4 = -0.2,   gae = -0.2 + 0.8 = 0.6
      pos 1: delta = 0 + 0.4 - 0.6 = -0.2,   gae = -0.2 + 0.6 = 0.4

    Low-level GAE for segment 0 (bootstraps to V^high(seg1) = 0.6):
      pos 3: delta = 0 + 0.6 - 0.1 = 0.5,   gae = 0.5
      pos 2: delta = 0 + 0.1 - 0.2 = -0.1,   gae = -0.1 + 0.5 = 0.4
      pos 1: delta = 0 + 0.2 - 0.3 = -0.1,   gae = -0.1 + 0.4 = 0.3

    High-level (seg 0 is non-final, seg 1 is final):
      Segment 1 (final): delta_high = 1.0 + 1*0 - 0.6 = 0.4
      Segment 0 (non-final): delta_high = 0 + 1*0.6 - 0.3 = 0.3
        gae_high = 0.3 + 1*1*0.4 = 0.7
      Applied to summary tokens of seg 0 (last contiguous block = [3]):
        Actually [1,2,3] are all contiguous, so summary = [1,2,3].
        All get +0.7.
      
    Wait - [1,2,3] is one contiguous block, so _find_last_contiguous_block
    returns all of them. This means ALL tokens in seg 0 get the high-level
    advantage. Let me note this for the expected values.

    Seg 0 final: pos 1: 0.3+0.7=1.0, pos 2: 0.4+0.7=1.1, pos 3: 0.5+0.7=1.2
    Seg 1 final: pos 1: 0.4, pos 2: 0.6, pos 3: 0.8
    """
    print("=" * 60)
    print("Test 2: Two segments, sparse reward only at end")
    print("=" * 60)

    B, L = 2, 6
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 3]] = 1
    response_mask[1, [1, 2, 3]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[1, 3] = 1.0

    values = torch.zeros(B, L)
    values[0, 1] = 0.3
    values[0, 2] = 0.2
    values[0, 3] = 0.1
    values[1, 1] = 0.6
    values[1, 2] = 0.4
    values[1, 3] = 0.2

    index = np.array(["rollout_0", "rollout_0"])
    segment_index = np.array([0, 1])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=1.0, lam=1.0
    )

    print("Segment 0 (non-final, no reward, bootstraps to V^high(seg1)=0.6):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[0, p].item():.4f}, ret={ret[0, p].item():.4f}")

    print("Segment 1 (final, reward=1.0 at pos 3, bootstraps to 0):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[1, p].item():.4f}, ret={ret[1, p].item():.4f}")

    print()
    print("Expected low-level only for seg 1: [0.4, 0.6, 0.8]")
    print("Expected low-level for seg 0: [0.3, 0.4, 0.5]")
    print("Expected high-level for seg 0: 0.7 added to summary tokens")
    print("Note: [1,2,3] is one contiguous block -> all are 'summary tokens'")
    print("Expected seg 0 final: [0.3+0.7, 0.4+0.7, 0.5+0.7] = [1.0, 1.1, 1.2]")

    _check("Test 2 seg1", [
        (adv[1, 1].item(), 0.4),
        (adv[1, 2].item(), 0.6),
        (adv[1, 3].item(), 0.8),
    ])
    _check("Test 2 seg0", [
        (adv[0, 1].item(), 1.0),
        (adv[0, 2].item(), 1.1),
        (adv[0, 3].item(), 1.2),
    ])
    print()


def test_two_segments_high_level_on_summary_only():
    """
    Test 3: Verify high-level advantage is added to summary tokens only.

    Segment 0: response at [1, 2, 5, 6]
      action tokens: [1, 2] (gap at 3,4)
      summary tokens: [5, 6] (last contiguous block)

    Segment 1: response at [1, 2, 5, 6]
      action tokens: [1, 2]
      final answer: [5, 6]

    With a gap, _find_last_contiguous_block correctly returns [5,6] for
    seg 0. So ONLY pos 5,6 in seg 0 get the high-level advantage.
    Seg 1 is final, so no high-level advantage.
    """
    print("=" * 60)
    print("Test 3: High-level advantage on summary tokens only")
    print("=" * 60)

    B, L = 2, 8
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 5, 6]] = 1
    response_mask[1, [1, 2, 5, 6]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[1, 6] = 1.0

    values = torch.tensor([
        [0, 0.3, 0.25, 0, 0, 0.15, 0.1, 0],  # seg 0
        [0, 0.6, 0.5,  0, 0, 0.3,  0.2, 0],   # seg 1
    ], dtype=torch.float32)

    index = np.array(["r1", "r1"])
    segment_index = np.array([0, 1])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=1.0, lam=1.0
    )

    print("Segment 0: actions at [1,2], summary at [5,6]")
    print("Segment 1 (final): actions at [1,2], answer at [5,6]")
    print()
    print("Segment 0 advantages:")
    for p in [1, 2, 5, 6]:
        label = "SUMMARY" if p >= 5 else "action"
        print(f"  pos {p} ({label}): adv={adv[0, p].item():.4f}")

    print()
    print("Segment 1 advantages (final, no high-level added):")
    for p in [1, 2, 5, 6]:
        label = "answer" if p >= 5 else "action"
        print(f"  pos {p} ({label}): adv={adv[1, p].item():.4f}")

    print()
    print("Key check: In seg 0, summary tokens [5,6] should have LARGER")
    print("advantages than action tokens [1,2] due to high-level addition.")
    print("In seg 1 (final), all are pure low-level GAE.")
    print()


def test_two_rollouts_independent():
    """
    Test 4: Two independent single-segment rollouts.
    No high-level advantage (single segments). Pure GAE.

    Rollout A: reward=1.0, values=[0.5, 0.3, 0.1]
    Rollout B: reward=0.0, values=[0.5, 0.3, 0.1]

    gamma=1, lam=1 -> Monte Carlo:
      Rollout A: return=1.0, adv = [1.0-0.5, 1.0-0.3, 1.0-0.1] = [0.5, 0.7, 0.9]
      Rollout B: return=0.0, adv = [0.0-0.5, 0.0-0.3, 0.0-0.1] = [-0.5, -0.3, -0.1]
    """
    print("=" * 60)
    print("Test 4: Two independent rollouts")
    print("=" * 60)

    B, L = 2, 5
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 3]] = 1
    response_mask[1, [1, 2, 3]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[0, 3] = 1.0
    token_level_rewards[1, 3] = 0.0

    values = torch.zeros(B, L)
    values[0, [1, 2, 3]] = torch.tensor([0.5, 0.3, 0.1])
    values[1, [1, 2, 3]] = torch.tensor([0.5, 0.3, 0.1])

    index = np.array(["rolloutA", "rolloutB"])
    segment_index = np.array([0, 0])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=1.0, lam=1.0
    )

    print("Rollout A (reward=1.0):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[0, p].item():.4f}")
    print("Rollout B (reward=0.0):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[1, p].item():.4f}")

    print()
    _check("Test 4 rollout A", [
        (adv[0, 1].item(), 0.5),
        (adv[0, 2].item(), 0.7),
        (adv[0, 3].item(), 0.9),
    ])
    _check("Test 4 rollout B", [
        (adv[1, 1].item(), -0.5),
        (adv[1, 2].item(), -0.3),
        (adv[1, 3].item(), -0.1),
    ])
    print()


def test_three_segments_long_rollout():
    """
    Test 5: Three segments in one rollout.

    Seg 0: response at [1,2,3], rewards all 0, values=[0.2, 0.15, 0.1]
    Seg 1: response at [1,2,3], rewards all 0, values=[0.4, 0.3, 0.2]
    Seg 2: response at [1,2,3], reward=1.0 at pos 3, values=[0.7, 0.5, 0.3]

    V^high: seg0=0.2, seg1=0.4, seg2=0.7

    Low-level GAE (gamma=lam=1):
      Seg 2 (bootstraps to 0): G_t=1.0 from all pos
        pos 3: 1.0-0.3=0.7, pos 2: 1.0-0.5=0.5, pos 1: 1.0-0.7=0.3
      Seg 1 (bootstraps to V^high(seg2)=0.7): G_t=0.7 from all pos
        pos 3: 0.7-0.2=0.5, pos 2: 0.7-0.3=0.4, pos 1: 0.7-0.4=0.3
      Seg 0 (bootstraps to V^high(seg1)=0.4): G_t=0.4 from all pos
        pos 3: 0.4-0.1=0.3, pos 2: 0.4-0.15=0.25, pos 1: 0.4-0.2=0.2

    High-level GAE (gamma=lam=1):
      Seg 2 (final): delta = 1.0 + 0 - 0.7 = 0.3, gae = 0.3
      Seg 1 (non-final): delta = 0 + 0.7 - 0.4 = 0.3, gae = 0.3 + 0.3 = 0.6
      Seg 0 (non-final): delta = 0 + 0.4 - 0.2 = 0.2, gae = 0.2 + 0.6 = 0.8

    High-level applied to summary tokens of non-final segments:
      Seg 0 (all contiguous [1,2,3]): all get +0.8
      Seg 1 (all contiguous [1,2,3]): all get +0.6
      Seg 2 (final): no high-level

    Final advantages:
      Seg 0: [0.2+0.8, 0.25+0.8, 0.3+0.8] = [1.0, 1.05, 1.1]
      Seg 1: [0.3+0.6, 0.4+0.6, 0.5+0.6] = [0.9, 1.0, 1.1]
      Seg 2: [0.3, 0.5, 0.7]
    """
    print("=" * 60)
    print("Test 5: Three segments, reward propagation chain")
    print("=" * 60)

    B, L = 3, 5
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 3]] = 1
    response_mask[1, [1, 2, 3]] = 1
    response_mask[2, [1, 2, 3]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[2, 3] = 1.0

    values = torch.zeros(B, L)
    values[0, [1, 2, 3]] = torch.tensor([0.2, 0.15, 0.1])
    values[1, [1, 2, 3]] = torch.tensor([0.4, 0.3, 0.2])
    values[2, [1, 2, 3]] = torch.tensor([0.7, 0.5, 0.3])

    index = np.array(["long_rollout", "long_rollout", "long_rollout"])
    segment_index = np.array([0, 1, 2])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=1.0, lam=1.0
    )

    for seg in range(3):
        label = "(final)" if seg == 2 else "(non-final)"
        print(f"Segment {seg} {label}:")
        for p in [1, 2, 3]:
            print(f"  pos {p}: adv={adv[seg, p].item():.4f}, ret={ret[seg, p].item():.4f}")
        print()

    print("Expected:")
    print("  Seg 0: [1.0, 1.05, 1.1] (low=[0.2,0.25,0.3] + high=0.8)")
    print("  Seg 1: [0.9, 1.0, 1.1]  (low=[0.3,0.4,0.5] + high=0.6)")
    print("  Seg 2: [0.3, 0.5, 0.7]  (low only, final segment)")

    _check("Test 5 seg0", [
        (adv[0, 1].item(), 1.0),
        (adv[0, 2].item(), 1.05),
        (adv[0, 3].item(), 1.1),
    ])
    _check("Test 5 seg1", [
        (adv[1, 1].item(), 0.9),
        (adv[1, 2].item(), 1.0),
        (adv[1, 3].item(), 1.1),
    ])
    _check("Test 5 seg2", [
        (adv[2, 1].item(), 0.3),
        (adv[2, 2].item(), 0.5),
        (adv[2, 3].item(), 0.7),
    ])
    print()


def test_discount_factor():
    """
    Test 6: Verify discount factor works correctly.
    gamma=0.5, lam=1.0, all values=0.

    Low-level GAE:
      Seg 1 (final, bootstraps to 0):
        pos 3 (i=2): delta=1.0, gae=1.0
        pos 2 (i=1): delta=0+0.5*0-0=0, gae=0+0.5*1*1.0=0.5
        pos 1 (i=0): delta=0+0.5*0-0=0, gae=0+0.5*1*0.5=0.25
      Seg 0 (non-final, bootstraps to V^high(seg1)=0):
        All values=0, V^high_next=0, no reward -> all low-level adv=0

    High-level GAE:
      Macro quantities:
        Seg 0: r_macro=0, gamma_macro=0.5^3=0.125
        Seg 1: r_macro = 0.5^0*0 + 0.5^1*0 + 0.5^2*1.0 = 0.25
                (reward at 3rd response token gets discounted by gamma^2)
                gamma_macro=0.5^3=0.125
      TD residuals (all V^high=0):
        Seg 1 (k=1): delta = 0.25 + 0.125*0 - 0 = 0.25, gae=0.25
        Seg 0 (k=0): delta = 0 + 0.125*0 - 0 = 0
                      gae = 0 + 0.125*1.0*0.25 = 0.03125

    Final:
      Seg 1: [0.25, 0.5, 1.0] (low-level only, final segment)
      Seg 0: [0.03125, 0.03125, 0.03125] (low=0, high=0.03125)
    """
    print("=" * 60)
    print("Test 6: Discount factor effect (gamma=0.5)")
    print("=" * 60)

    B, L = 2, 5
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 3]] = 1
    response_mask[1, [1, 2, 3]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[1, 3] = 1.0

    values = torch.zeros(B, L)

    index = np.array(["r0", "r0"])
    segment_index = np.array([0, 1])

    adv, ret = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.5, lam=1.0
    )

    print("gamma=0.5, lam=1.0, all values=0")
    print()
    print("Segment 1 (final, low-level only):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[1, p].item():.4f}")

    print()
    print("Segment 0 (non-final, low=0, high=0.03125):")
    for p in [1, 2, 3]:
        print(f"  pos {p}: adv={adv[0, p].item():.6f}")
    print()
    print("Note: Seg 1 r_macro=0.25 (not 1.0) because the reward at the")
    print("3rd response token (i=2) is discounted by gamma^2 = 0.25.")
    print("This propagates through high-level GAE as 0.125 * 0.25 = 0.03125.")

    _check("Test 6 seg1", [
        (adv[1, 1].item(), 0.25),
        (adv[1, 2].item(), 0.5),
        (adv[1, 3].item(), 1.0),
    ])
    _check("Test 6 seg0", [
        (adv[0, 1].item(), 0.03125),
        (adv[0, 2].item(), 0.03125),
        (adv[0, 3].item(), 0.03125),
    ])
    print()


def _check(name, pairs, tol=1e-3):
    passed = True
    for actual, expected in pairs:
        if abs(actual - expected) > tol:
            print(f"  FAIL {name}: got {actual:.4f}, expected {expected:.4f}")
            passed = False
    if passed:
        print(f"  PASS {name}")


def test_prevent_penalize_context_failed_rollout():
    """
    Test 7: prevent_penalize_context with a FAILED rollout (reward=0).
    penalize_context_max_segments=0 means full protection for all segments.

    Without flag: summary tokens get negative low-level + negative high-level.
    With flag: summary tokens get negative low-level + ZERO high-level.
    """
    print("=" * 60)
    print("Test 7: prevent_penalize_context on failed rollout")
    print("=" * 60)

    B, L = 2, 8
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 5, 6]] = 1
    response_mask[1, [1, 2, 5, 6]] = 1

    token_level_rewards = torch.zeros(B, L)

    values = torch.tensor([
        [0, 0.5, 0.4, 0, 0, 0.3, 0.2, 0],
        [0, 0.6, 0.5, 0, 0, 0.3, 0.2, 0],
    ], dtype=torch.float32)

    index = np.array(["r_fail", "r_fail"])
    segment_index = np.array([0, 1])

    adv_off, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=False,
    )

    adv_on, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=True,
        penalize_context_max_segments=0,
    )

    print("Segment 0 (non-final), flag OFF:")
    for p in [1, 2, 5, 6]:
        label = "SUMMARY" if p >= 5 else "action"
        print(f"  pos {p} ({label}): adv={adv_off[0, p].item():.4f}")

    print()
    print("Segment 0 (non-final), flag ON (max_segments=0, full protection):")
    for p in [1, 2, 5, 6]:
        label = "SUMMARY" if p >= 5 else "action"
        print(f"  pos {p} ({label}): adv={adv_on[0, p].item():.4f}")

    print()

    action_same = all(
        abs(adv_off[0, p].item() - adv_on[0, p].item()) < 1e-6
        for p in [1, 2]
    )
    summary_less_negative = all(
        adv_on[0, p].item() >= adv_off[0, p].item() - 1e-6
        for p in [5, 6]
    )
    final_same = all(
        abs(adv_off[1, p].item() - adv_on[1, p].item()) < 1e-6
        for p in [1, 2, 5, 6]
    )

    print(f"  Action tokens unchanged: {action_same}")
    print(f"  Summary tokens less negative with flag: {summary_less_negative}")
    print(f"  Final segment unchanged: {final_same}")
    print()


def test_prevent_penalize_context_successful_rollout():
    """
    Test 8: prevent_penalize_context with a SUCCESSFUL rollout (reward=1).

    When the rollout succeeds, the high-level advantage is positive.
    The flag should have NO effect — positive advantages pass through.

    Same layout as Test 7 but with reward=1.0 at seg 1 pos 6.
    """
    print("=" * 60)
    print("Test 8: prevent_penalize_context on successful rollout")
    print("=" * 60)

    B, L = 2, 8
    response_mask = torch.zeros(B, L)
    response_mask[0, [1, 2, 5, 6]] = 1
    response_mask[1, [1, 2, 5, 6]] = 1

    token_level_rewards = torch.zeros(B, L)
    token_level_rewards[1, 6] = 10.0  # successful rollout

    values = torch.tensor([
        [0, 0.5, 0.4, 0, 0, 0.3, 0.2, 0],  # seg 0
        [0, 0.6, 0.5, 0, 0, 0.3, 0.2, 0],   # seg 1
    ], dtype=torch.float32)

    index = np.array(["r_succ", "r_succ"])
    segment_index = np.array([0, 1])

    # --- Without flag ---
    adv_off, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=False,
    )

    # --- With flag ---
    adv_on, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=True,
    )

    print("Successful rollout (reward=10.0). Flag should have NO effect.")
    print()

    all_same = True
    for seg in range(2):
        seg_label = "non-final" if seg == 0 else "final"
        print(f"Segment {seg} ({seg_label}):")
        for p in [1, 2, 5, 6]:
            a_off = adv_off[seg, p].item()
            a_on = adv_on[seg, p].item()
            match = "OK" if abs(a_off - a_on) < 1e-6 else "MISMATCH"
            print(f"  pos {p}: off={a_off:.4f}, on={a_on:.4f} [{match}]")
            if abs(a_off - a_on) >= 1e-6:
                all_same = False
        print()

    print(f"Key check: All advantages identical with/without flag: {all_same}")
    print("(Positive high-level advantages pass through unclamped)")
    print()


def test_dynamic_penalization_schedule():
    """
    Test 9: Dynamic penalization with penalize_context_max_segments=2.

    4 segments (seg 0,1,2 are non-final, seg 3 is final), all failed (reward=0).
    Critic predicts positive values, so high-level advantages are negative.

    Schedule with max_segments=2:
      Seg 0 (k=0): penalty_ratio = 0/2 = 0.0 -> fully clamped (0% of negative adv)
      Seg 1 (k=1): penalty_ratio = 1/2 = 0.5 -> half penalty (50% of negative adv)
      Seg 2 (k=2): penalty_ratio = 2/2 = 1.0 -> full penalty (100% of negative adv)
      Seg 3: final, no high-level advantage

    We compare against flag=OFF to verify the scaling.
    """
    print("=" * 60)
    print("Test 9: Dynamic penalization schedule (max_segments=2)")
    print("=" * 60)

    B, L = 4, 6
    response_mask = torch.zeros(B, L)
    for b in range(4):
        response_mask[b, [1, 2, 4, 5]] = 1  # action=[1,2], summary=[4,5]

    token_level_rewards = torch.zeros(B, L)  # all failed

    values = torch.zeros(B, L)
    values[0, [1, 2, 4, 5]] = torch.tensor([0.3, 0.25, 0.15, 0.1])
    values[1, [1, 2, 4, 5]] = torch.tensor([0.4, 0.35, 0.2, 0.15])
    values[2, [1, 2, 4, 5]] = torch.tensor([0.5, 0.4, 0.25, 0.2])
    values[3, [1, 2, 4, 5]] = torch.tensor([0.6, 0.5, 0.3, 0.2])

    index = np.array(["r0", "r0", "r0", "r0"])
    segment_index = np.array([0, 1, 2, 3])

    # Without flag
    adv_off, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=False,
    )

    # With dynamic schedule
    adv_on, _ = compute_contextrl_advantage_return(
        token_level_rewards, values, response_mask,
        index, segment_index, gamma=0.99, lam=0.95,
        prevent_penalize_context=True,
        penalize_context_max_segments=2,
    )

    print("Schedule: seg0 -> 0% penalty, seg1 -> 50%, seg2 -> 100%")
    print()

    for seg in range(3):  # non-final segments only
        ratio = min(seg / 2.0, 1.0)
        print(f"Segment {seg} (penalty_ratio={ratio:.1f}):")

        # Show summary token advantages
        for p in [4, 5]:
            a_off = adv_off[seg, p].item()
            a_on = adv_on[seg, p].item()
            # The high-level component is: a_on_total - a_low = adv_high * ratio
            # where a_low is the same in both cases
            print(f"  pos {p}: off={a_off:.4f}, on={a_on:.4f}")
        print()

    # Verify the scaling relationship
    print("Verification:")

    # For each non-final segment, compute the high-level component
    # high_level_off = adv_off[summary] - adv_off[action_extrapolated]
    # But simpler: diff = adv_on - adv_off at summary positions
    # This diff = adv_high * (ratio - 1) when adv_high < 0
    # So diff = -adv_high * (1 - ratio)

    for seg in range(3):
        ratio = min(seg / 2.0, 1.0)
        # Action tokens should be identical
        action_same = all(
            abs(adv_off[seg, p].item() - adv_on[seg, p].item()) < 1e-6
            for p in [1, 2]
        )
        print(f"  Seg {seg}: action tokens unchanged: {action_same}")

        # At summary tokens, the diff tells us how much was clamped
        diff_4 = adv_on[seg, 4].item() - adv_off[seg, 4].item()
        diff_5 = adv_on[seg, 5].item() - adv_off[seg, 5].item()

        if seg == 0:
            # ratio=0: full clamp, diff should equal |high_level_adv|
            print(f"  Seg 0: full clamp, diff at summary = [{diff_4:.4f}, {diff_5:.4f}] (should be positive)")
        elif seg == 1:
            # ratio=0.5: half penalty
            print(f"  Seg 1: half penalty, diff at summary = [{diff_4:.4f}, {diff_5:.4f}]")
        elif seg == 2:
            # ratio=1.0: no clamp, diff should be ~0
            print(f"  Seg 2: no clamp, diff at summary = [{diff_4:.4f}, {diff_5:.4f}] (should be ~0)")
            full_penalty = all(abs(d) < 1e-6 for d in [diff_4, diff_5])
            print(f"    Full penalty restored: {full_penalty}")

    print()

    # Final segment should be identical
    final_same = all(
        abs(adv_off[3, p].item() - adv_on[3, p].item()) < 1e-6
        for p in [1, 2, 4, 5]
    )
    print(f"  Final segment unchanged: {final_same}")
    print()
