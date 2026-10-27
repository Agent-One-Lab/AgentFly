"""The [Rollout] console summary line, built from the same values sent as metrics."""

from agentfly.agents.rollout.loop.metrics import group_rewards, pass_rate_distribution, rollout_summary
from agentfly.agents.types import Trajectory


def _traj(group, reward, finish, turns, tools, seconds, **metadata):
    return Trajectory(
        segments=[],
        reward=reward,
        finish_reason=finish,
        num_turns=turns,
        tool_call_counts=tools,
        rollout_time_sec=seconds,
        group_id=group,
        metadata=metadata,
    )


def test_summary_line_fields():
    trajs = [
        _traj("A", 1.0, "terminal", 3, {"calculator": 2, "answer": 1}, 4.0, capped_turns=1),
        _traj("A", 0.1, "max_turns", 5, {"calculator": 4}, 6.0),
        _traj("B", 0.0, "no_tool_calls", 1, {}, 2.0),
        _traj("B", 0.0, "no_tool_calls", 1, {}, 2.0),
    ]
    distribution = pass_rate_distribution(group_rewards(trajs))
    line = rollout_summary(global_step=7, trajectories=trajs, distribution=distribution)

    assert line.startswith("[Rollout] step 7: trajs=4")
    assert "reward_mean=0.275 reward>0=2/4" in line
    assert "turns=2.5" in line
    assert "tool_calls=1.8/traj {calculator: 6, answer: 1}" in line
    assert "groups(2 tasks x 2, pass=reward>=1): all_pass=0.00 all_fail=0.50 mixed=0.50" in line
    assert "finish={no_tool_calls: 2, terminal: 1, max_turns: 1}" in line
    assert "capped_turns=1" in line
    assert "no_tool_call_turns" not in line and "nudges" not in line  # shown only when non-zero
    assert "time(s) avg=3.5 max=6.0" in line


def test_summary_without_groups_or_rewards():
    line = rollout_summary(global_step=None, trajectories=[_traj("A", None, "terminal", 2, None, None)])
    assert line == "[Rollout] step None: trajs=1 turns=2.0 | finish={terminal: 1} capped_turns=0"
