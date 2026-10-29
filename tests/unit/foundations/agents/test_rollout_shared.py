"""The trajectory lifecycle pieces on the base ``Rollout`` that every strategy shares."""

import asyncio
import types

import pytest

from agentfly.agents.rollout import base as rollout_base
from agentfly.agents.rollout.base import EpisodeSpec, Rollout, unwrap_reward
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.types import Segment
from agentfly.rewards.types import RewardResult


def tasks(n):
    return [
        {"messages": [{"role": "user", "content": f"Q{i}"}], "task_id": f"t{i}", "answer": i}
        for i in range(n)
    ]


# ---- running many trajectories ----


def test_validate_run_args_is_shared():
    for bad in [(0, 1, None), (1, 0, None), (1, 1, 0)]:
        with pytest.raises(AssertionError):
            Rollout.validate_run_args(*bad)
    Rollout.validate_run_args(1, 1, None)
    assert "validate_run_args" not in vars(ChainRollout)
    assert "validate_run_args" not in vars(StepRollout)


def test_make_specs_order_ids_and_info():
    specs = Rollout.make_specs(tasks(2), group_size=3)
    assert [(s.group_idx, s.episode_idx) for s in specs] == [(g, c) for g in range(2) for c in range(3)]
    assert len({s.episode_id for s in specs}) == 6
    assert len({s.group_id for s in specs[:3]}) == 1
    assert specs[0].group_id != specs[3].group_id
    info = specs[4].info
    assert info["task_id"] == "t1" and info["answer"] == 1
    assert (info["episode_id"], info["group_id"], info["group_idx"], info["episode_idx"]) == (
        specs[4].episode_id, specs[4].group_id, 1, 1,
    )
    info["scratch"] = 1
    assert "scratch" not in specs[4].info  # a fresh dict every time


@pytest.mark.asyncio
async def test_run_episodes_merges_events_limits_concurrency_and_counts_progress(monkeypatch):
    updates = []
    monkeypatch.setattr(
        rollout_base, "progress_bar",
        lambda total: types.SimpleNamespace(update=updates.append, close=lambda: updates.append("closed")),
    )
    specs = Rollout.make_specs(tasks(5), group_size=1)
    in_flight = peak = 0

    async def run_episode(spec):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        yield ("start", spec.group_idx)
        await asyncio.sleep(0.01)
        in_flight -= 1
        yield ("end", spec.group_idx)

    events = [ev async for ev in Rollout.run_episodes(specs, run_episode, max_concurrent_episodes=2)]
    assert peak <= 2
    assert sorted(events) == sorted([(kind, i) for i in range(5) for kind in ("start", "end")])
    assert updates == [1] * 5 + ["closed"]


@pytest.mark.asyncio
async def test_run_episodes_accepts_coroutines_and_raises_first_error_after_all_finish(monkeypatch):
    monkeypatch.setattr(rollout_base, "progress_bar", lambda total: None)
    specs = Rollout.make_specs(tasks(3), group_size=1)
    finished = []

    async def run_episode(spec):
        await asyncio.sleep(0.01 * (3 - spec.group_idx))
        finished.append(spec.group_idx)
        if spec.group_idx in (0, 2):
            raise ValueError(f"fail {spec.group_idx}")

    with pytest.raises(ValueError, match="fail 0"):
        async for _ in Rollout.run_episodes(specs, run_episode, show_progress=False):
            pass
    assert sorted(finished) == [0, 1, 2]


# ---- one trajectory's lifecycle ----


class FakeContext:
    def __init__(self):
        self.calls = []

    async def release_resource(self, **kwargs):
        self.calls.append(("release", kwargs))

    async def end_resource(self, **kwargs):
        self.calls.append(("end", kwargs))


@pytest.mark.asyncio
async def test_release_resources_releases_global_then_ends_rollout():
    context = FakeContext()
    await Rollout.release_resources(context)
    assert context.calls == [("release", {"scope": "global"}), ("end", {"scope": "rollout"})]


@pytest.mark.asyncio
async def test_run_first_step_hook_logs_failures(caplog):
    context, step = object(), object()
    seen = []

    async def hook(c, s):
        seen.append((c, s))

    await Rollout.run_first_step_hook(types.SimpleNamespace(prepare_first_step=hook), context, step, "c1")
    assert seen == [(context, step)]
    await Rollout.run_first_step_hook(types.SimpleNamespace(), context, step, "c1")  # no hook

    async def failing(c, s):
        raise RuntimeError("boom")

    await Rollout.run_first_step_hook(types.SimpleNamespace(prepare_first_step=failing), context, step, "c1")
    assert "prepare_first_step hook failed for episode c1: boom" in caplog.text


@pytest.mark.asyncio
async def test_compute_reward_without_reward_fn_returns_none_and_leaves_context(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("no reward function is configured")

    monkeypatch.setattr(rollout_base, "calculate_reward", unexpected)
    context = types.SimpleNamespace()
    agent = types.SimpleNamespace(reward_fn=None)
    reward = await Rollout.compute_reward(
        agent, context, episode_id="c1", trajectory=[], final_messages=[], reward_args={},
    )
    assert reward is None
    assert vars(context) == {}


@pytest.mark.asyncio
async def test_compute_reward_passes_trajectory_final_response_and_task_args(monkeypatch):
    calls = []

    async def fake_calculate(reward_fn, **kwargs):
        calls.append((reward_fn, kwargs))
        return RewardResult(reward=1.0)

    monkeypatch.setattr(rollout_base, "calculate_reward", fake_calculate)
    final_messages = [{"role": "assistant", "content": "done"}]
    agent = types.SimpleNamespace(reward_fn="fn", extract_final_response=lambda messages: messages[-1]["content"])
    context = types.SimpleNamespace()
    trajectory = [[{"role": "user", "content": "Q"}], final_messages]
    reward = await Rollout.compute_reward(
        agent, context, episode_id="c1", trajectory=trajectory, final_messages=final_messages,
        reward_args={"answer": 3, "final_response": "x", "trajectory": "x", "id": "x"},
    )
    assert reward.reward == 1.0
    (fn, kwargs), = calls
    assert fn == "fn"
    assert kwargs == {
        "final_response": "done", "answer": 3, "trajectory": trajectory, "id": "c1", "context": context,
    }
    assert context.trajectory is trajectory and context.final_response == "done"


@pytest.mark.parametrize("raw,expected", [
    (RewardResult(reward=0.5, metrics={"acc": 1.0}), (0.5, {"acc": 1.0})),
    ({"reward": 0.25, "extra": 2}, (0.25, {"extra": 2})),
    (1, (1.0, {})),
    (None, (None, {})),
])
def test_unwrap_reward(raw, expected):
    assert unwrap_reward(raw) == expected


def test_make_trajectory_merges_tool_metrics_last_and_keeps_task_metadata():
    spec = Rollout.make_specs(tasks(1), group_size=1)[0]
    info = spec.info | {"finish_reason": "ignored", "capped_turns": 2}
    trajectory = StepRollout().make_trajectory(
        info=info,
        segments=[Segment(messages=[])],
        reward=RewardResult(reward=0.5, metrics={"score": 1, "tool/act/x": 999}),
        tool_metrics={"tool/act/x": 2.0},
        finish_reason="max_turns",
        rollout_time_sec=1.5,
        num_turns=3,
        tool_call_counts={"act": 3},
        steps=[],
    )
    assert trajectory.reward == 0.5
    assert trajectory.metrics == {"score": 1, "tool/act/x": 2.0}
    assert (trajectory.episode_id, trajectory.group_id, trajectory.group_idx, trajectory.episode_idx) == (
        spec.episode_id, spec.group_id, 0, 0,
    )
    assert trajectory.finish_reason == "max_turns"
    assert trajectory.metadata == {"task_id": "t0", "answer": 0, "capped_turns": 2}


def test_chain_only_machinery_lives_on_chain_rollout():
    for name in (
        "execute_turn_tools", "apply_context_folding", "render_no_tool_call_message",
        "finalize_chain", "build_trajectories",
    ):
        assert not hasattr(Rollout, name), name
        assert hasattr(ChainRollout, name), name
    assert not hasattr(StepRollout(), "max_consecutive_no_tool_calls")
    assert isinstance(EpisodeSpec, type)
