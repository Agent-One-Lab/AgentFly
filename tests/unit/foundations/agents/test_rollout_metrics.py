"""Shared reporting consumes trajectories without chain-specific runtime state."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from agentfly.agents.rollout.loop import metrics
from agentfly.agents.types import Segment, Trajectory
from agentfly.tools.types import ToolResult


def test_chain_logging_preserves_payloads_without_accumulating_state(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    reporter = metrics.RolloutMetrics()

    for index in range(3):
        events.clear()
        trajectory = [[{"role": "assistant", "content": f"answer-{index}"}], []]
        info = {"chain_id": f"chain-{index}", "reward": 0.5}
        before = deepcopy((trajectory, info))

        reporter.record_chain(global_step=7, trajectory=trajectory, info=info)

        assert Counter(event.name for event in events) == {
            "agent/rollout/trajectory": 1,
            "agent/rollout/info": 1,
        }
        assert json.loads(events[0].value) == trajectory
        assert json.loads(events[1].value) == info
        for event in events:
            assert event.sinks == ["jsonl"]
            assert event.kind == "text"
            assert event.x == 7
            assert event.x_name == "agent/rollout/step"
        assert (trajectory, info) == before
        assert not hasattr(reporter, "monitor_info")


def test_step_emits_each_summary_once_with_existing_values(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    trajectory = Trajectory(
        segments=[Segment(messages=[{"role": "assistant", "content": "answer"}]),
                  Segment(messages=[])],
        rollout_time_sec=1.5,
        num_turns=2,
        tool_call_counts={"act": 1, "search": 1},
        reward=0.5,
        metrics={"accuracy": 0.75},
    )
    before = deepcopy(trajectory.model_dump())
    expected_scalars = {
        "agent/rollout/time/min": 1.5,
        "agent/rollout/time/max": 1.5,
        "agent/rollout/time/avg": 1.5,
        "agent/rollout/avg_segments": 2.0,
        "agent/rollout/step": 4,
        "agent/rollout/avg_turns": 2.0,
        "agent/rollout/avg_tool_calls": 2.0,
        "agent/rollout/tool_calls/act": 1.0,
        "agent/rollout/tool_calls/search": 1.0,
        "agent/rollout/reward": 0.5,
        "agent/rollout/accuracy": 0.75,
    }
    expected_texts = {"agent/rollout/sample_trajectory", "agent/rollout/slowest_message"}

    metrics.RolloutMetrics().record_step(global_step=4, trajectories=[trajectory])

    # Check multiplicity before making dictionaries, which would hide duplicates.
    assert Counter(event.name for event in events) == {
        name: 1 for name in expected_scalars.keys() | expected_texts
    }
    assert {event.name: event.value for event in events if event.kind == "scalar"} == expected_scalars
    for event in events:
        assert event.sinks is None
        if event.name in expected_texts:
            assert event.kind == "text"
            assert json.loads(event.value) == before
        # Every summary shares the rollout's axis, reward extras included: an event
        # without one is logged against wandb's own step and drifts from the rest.
        assert event.x == 4
        assert event.x_name == "agent/rollout/step"
    assert trajectory.model_dump() == before


def sampled(rewards_by_task):
    """One batch of trajectories: {task: [reward per sample]} in rollout order."""
    return [
        Trajectory(segments=[], group_id=task, chain_idx=index, reward=reward)
        for task, rewards in rewards_by_task.items()
        for index, reward in enumerate(rewards)
    ]


def group_values(events):
    return {
        event.name.removeprefix("agent/rollout/group/"): event.value
        for event in events
        if event.name.startswith("agent/rollout/group/")
    }


def test_pass_rate_distribution_separates_trainable_tasks_from_saturated_ones(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    # A U-shaped pool: two tasks never solved, one always solved, one mixed. Only the
    # mixed task gives GRPO a non-zero advantage, which mean reward alone would hide.
    trajectories = sampled({
        "always_fails": [0.0, 0.0, 0.0, 0.0],
        "also_fails": [0.0, 0.0, 0.0, 0.0],
        "always_solved": [1.0, 1.0, 1.0, 1.0],
        "mixed": [0.0, 1.0, 1.0, 0.0],
    })

    metrics.RolloutMetrics().record_step(global_step=6, trajectories=trajectories)

    observed = group_values(events)
    assert observed == {
        "num_groups": 4,
        "samples_per_task": 4.0,
        "pass_rate_mean": 0.375,
        "frac_all_fail": 0.5,
        "frac_all_pass": 0.25,
        "frac_mixed": 0.25,
        "frac_zero_advantage": 0.75,
        # Only the mixed task has any spread (std 0.5), averaged over the four tasks.
        "reward_std_mean": 0.125,
        "pass_rate_hist": [0.0, 0.0, 1.0, 0.5],
    }
    hist = next(event for event in events if event.name.endswith("pass_rate_hist"))
    assert hist.kind == "hist"
    assert all(
        event.x == 6 and event.x_name == "agent/rollout/step"
        for event in events
        if event.name.startswith("agent/rollout/group/")
    )
    # The distribution is reported alongside — not instead of — the mean reward.
    assert next(e.value for e in events if e.name == "agent/rollout/reward") == 0.375


def test_pass_rate_distribution_skipped_without_repeated_samples(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    # Validation and num_chains=1 give one sample per task, where every pass rate is 0
    # or 1 by construction and says nothing about difficulty.
    single = sampled({"task_a": [1.0], "task_b": [0.0]})

    metrics.RolloutMetrics().record_step(global_step=1, trajectories=single)

    assert group_values(events) == {}


def test_ungrouped_and_ungraded_rollouts_do_not_form_a_shared_task(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    trajectories = sampled({"task_a": [1.0, 0.0]}) + [
        # No group_id: separate single-sample tasks, never pooled into one.
        Trajectory(segments=[], reward=1.0),
        Trajectory(segments=[], reward=0.0),
        # Ungradable rollouts carry no reward and are not counted as failures.
        Trajectory(segments=[], group_id="task_b", reward=None),
        Trajectory(segments=[], group_id="task_b", reward=None),
    ]

    metrics.RolloutMetrics().record_step(global_step=2, trajectories=trajectories)

    observed = group_values(events)
    assert observed["num_groups"] == 3
    assert observed["samples_per_task"] == 4 / 3
    assert observed["pass_rate_hist"] == [0.5, 1.0, 0.0]


@pytest.mark.parametrize("threshold,expected_pass_rates", [(1.0, [0.0, 0.5]), (0.5, [0.5, 1.0])])
def test_success_threshold_selects_what_counts_as_a_pass(threshold, expected_pass_rates, monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    trajectories = sampled({"partial": [0.2, 0.6], "good": [0.6, 1.0]})

    metrics.RolloutMetrics(success_threshold=threshold).record_step(
        global_step=5, trajectories=trajectories
    )

    observed = group_values(events)
    assert observed["pass_rate_hist"] == expected_pass_rates
    # Threshold-free signals are unaffected: neither task is degenerate here.
    assert observed["frac_zero_advantage"] == 0.0
    assert observed["reward_std_mean"] == pytest.approx(0.2)


def test_identical_continuous_rewards_count_as_zero_advantage():
    distribution = metrics.pass_rate_distribution(
        {"flat": [0.4, 0.4, 0.4], "varied": [0.4, 0.9, 0.4]}, success_threshold=1.0
    )
    # Neither task ever reaches the pass threshold, so the pass-rate view calls both
    # hopeless; only the raw-reward view shows one of them still carries signal.
    assert distribution["frac_all_fail"] == 1.0
    assert distribution["frac_zero_advantage"] == 0.5
    assert distribution["reward_std_mean"] > 0.0


def test_group_rewards_reconstructs_the_samples_of_each_task():
    trajectories = sampled({"task_a": [0.0, 1.0], "task_b": [0.5]})
    assert metrics.group_rewards(trajectories) == {"task_a": [0.0, 1.0], "task_b": [0.5]}
    assert metrics.pass_rate_distribution({}) == {"pass_rates": [], "num_groups": 0}


@pytest.mark.parametrize("results", [[], [None, None]])
def test_empty_tool_metrics(results):
    assert metrics.aggregate_tool_metrics(results) == {}
    assert metrics.count_tool_calls(results) == {}


def test_numeric_metrics_are_averaged_per_tool_and_per_available_key():
    results = [
        None,  # Seed/thought or a generation without a tool result.
        ToolResult(
            name="act", arguments={}, observation="one", step_reward=0.0,
            metrics={"valid": True, "latency": 1, "mixed": "text", "labels": [1, 2]},
        ),
        ToolResult(
            name="act", arguments={}, observation="two", step_reward=1.0,
            metrics={"valid": False, "latency": 3.0, "mixed": 8},
        ),
        ToolResult(
            name="act", arguments={}, observation="three",
            metrics={"latency": None},
        ),
        ToolResult(name="search", arguments={}, observation="four", metrics={"latency": 10}),
        ToolResult(name="", arguments={}, observation="five", metrics={"score": 2}),
    ]
    original = deepcopy(results)
    assert metrics.aggregate_tool_metrics(iter(results)) == {
        "tool/act/valid": 0.5,
        "tool/act/latency": 2.0,
        "tool/act/mixed": 8.0,
        "tool/act/step_reward": 0.5,
        "tool/search/latency": 10.0,
        "tool/tool/score": 2.0,
    }
    assert results == original


def test_metric_and_first_class_step_reward_keep_existing_chain_aggregation():
    result = ToolResult(
        name="act", arguments={}, observation="one", step_reward=1.0,
        metrics={"step_reward": 3.0},
    )
    # Preserve the existing chain rule when both sources supply the same key.
    assert metrics.aggregate_tool_metrics([result]) == {"tool/act/step_reward": 2.0}


def test_tool_call_counts_include_failures_and_preserve_result_names():
    results = [
        None,
        ToolResult(name="act", arguments={}, observation="ok"),
        ToolResult(name="act", arguments={}, observation="bad", status="invalid"),
        ToolResult(name="act", arguments={}, observation="failed", status="error"),
        ToolResult(name="summarize", arguments={}, observation="folded"),
        ToolResult(name="", arguments={}, observation="missing name", status="invalid"),
    ]
    before = deepcopy(results)
    assert metrics.count_tool_calls(iter(results)) == {"act": 3, "summarize": 1, "": 1}
    assert results == before


@pytest.mark.parametrize("counts,expected", [
    ([(2, {"act": 2}), (0, {}), (None, None), (None, {"search": 1})],
     {"avg_turns": 1.0, "avg_tool_calls": 1.0, "tool_calls/act": 2 / 3, "tool_calls/search": 1 / 3}),
    ([(4, None), (None, {"act": 2})],
     {"avg_turns": 4.0, "avg_tool_calls": 2.0, "tool_calls/act": 2.0}),
    ([(None, None)], {}),
    ([(0, {}), (0, {})], {"avg_turns": 0.0, "avg_tool_calls": 0.0}),
])
def test_execution_averages_use_known_counts_not_messages(counts, expected, monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    # These overlapping views and fake internal records must not be used as counters.
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "summarize"}}]},
        {"role": "tool", "content": "old observation"},
    ]
    trajectories = [
        Trajectory(
            segments=[Segment(messages=messages), Segment(messages=messages)],
            num_turns=num_turns, tool_call_counts=tool_counts, steps=[object()],
        )
        for num_turns, tool_counts in counts
    ]
    before = [deepcopy(t.model_dump()) for t in trajectories]
    reporter = metrics.RolloutMetrics()
    # The public fields alone must suffice, including after JSON removes internal steps.
    for batch in (trajectories, [Trajectory.model_validate_json(t.model_dump_json()) for t in trajectories]):
        events.clear()
        reporter.record_step(global_step=3, trajectories=batch)
        assert all(count == 1 for count in Counter(event.name for event in events).values())
        observed = {
            event.name.removeprefix("agent/rollout/"): event.value for event in events
            if event.name in ("agent/rollout/avg_turns", "agent/rollout/avg_tool_calls")
            or event.name.startswith("agent/rollout/tool_calls/")
        }
        assert observed == expected
        assert next(event.value for event in events if event.name == "agent/rollout/avg_segments") == 2.0
        for event in events:
            if event.name.removeprefix("agent/rollout/") in expected:
                assert event.x == 3
                assert event.x_name == "agent/rollout/step"
    assert [t.model_dump() for t in trajectories] == before


@pytest.mark.parametrize("durations,expected_stats,slowest_index", [
    ([0.0, None, 3.0, 6.0], (0.0, 6.0, 3.0), 3),
    ([None, 0.0], (0.0, 0.0, 0.0), 1),
    ([5.0, 5.0, None], (5.0, 5.0, 5.0), 0),
    ([None, None], None, None),
    ([], None, None),
])
def test_timing_summaries_use_trajectory_durations(
    durations, expected_stats, slowest_index, monkeypatch,
):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    trajectories = [
        Trajectory(segments=[], chain_id=f"chain-{i}", rollout_time_sec=duration, reward=0.5)
        for i, duration in enumerate(durations)
    ]
    before = [trajectory.model_dump() for trajectory in trajectories]

    metrics.RolloutMetrics().record_step(global_step=7, trajectories=trajectories)

    event_counts = Counter(event.name for event in events)
    assert event_counts["agent/rollout/step"] == (1 if trajectories else 0)
    assert all(count == 1 for count in event_counts.values())
    timing_events = [event for event in events if event.name.startswith("agent/rollout/time/")]
    slowest_events = [event for event in events if event.name == "agent/rollout/slowest_message"]
    if expected_stats is None:
        assert timing_events == []
        assert slowest_events == []
    else:
        assert len(timing_events) == 3
        assert {event.name: event.value for event in timing_events} == dict(zip(
            ("agent/rollout/time/min", "agent/rollout/time/max", "agent/rollout/time/avg"),
            expected_stats,
        ))
        assert len(slowest_events) == 1
        assert json.loads(slowest_events[0].value) == before[slowest_index]
        assert all(event.kind == "scalar" for event in timing_events)
        assert slowest_events[0].kind == "text"
        for event in timing_events + slowest_events:
            assert event.x == 7
            assert event.x_name == "agent/rollout/step"
    if not trajectories:
        assert events == []
    else:
        # Missing timing data does not suppress the other existing summaries.
        reward_events = [event for event in events if event.name == "agent/rollout/reward"]
        assert len(reward_events) == 1
        assert reward_events[0].value == 0.5
    assert [trajectory.model_dump() for trajectory in trajectories] == before


@pytest.mark.parametrize("layout", ["folded_chain", "step"])
def test_slowest_log_preserves_complete_trajectory_payload(layout, monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    first_messages = [
        {"role": "user", "content": "first observation"},
        {"role": "assistant", "content": "first action", "token_ids": [11, 2]},
    ]
    next_messages = [{"role": "user", "content": "next observation"}]
    if layout == "step":
        next_messages = first_messages + next_messages + [
            {"role": "assistant", "content": "second action", "token_ids": [12, 2]},
        ]
    # A folded chain can end with a context-only view; preserve it, even when it is empty.
    segments = [Segment(messages=first_messages), Segment(messages=next_messages)]
    if layout == "folded_chain":
        segments.append(Segment(messages=[]))
    step = object()
    slowest = Trajectory(
        segments=segments,
        rollout_time_sec=8.0,
        reward=0.75,
        metrics={"accuracy": 1.0, "output": {"patch": "+fixed"}},
        metadata={"task_id": "task-1", "rollout_time_sec": 999.0},
        runtime_info={"attempt": 2},
        chain_id="slow", group_id="group-1", chain_idx=1, group_idx=0,
        finish_reason="max_turns",
        steps=[step],
    )
    faster = Trajectory(segments=[], rollout_time_sec=1.0, chain_id="fast")
    before = deepcopy(slowest.model_dump())

    metrics.RolloutMetrics().record_step(global_step=4, trajectories=[faster, slowest])

    slowest_events = [event for event in events if event.name == "agent/rollout/slowest_message"]
    assert len(slowest_events) == 1
    payload = json.loads(slowest_events[0].value)
    assert payload == before
    assert "messages" not in payload
    assert "steps" not in payload
    assert payload["rollout_time_sec"] == 8.0
    assert payload["metadata"]["rollout_time_sec"] == 999.0
    assert Trajectory.model_validate(payload).segments == segments
    assert slowest.model_dump() == before
    assert slowest.steps[0] is step


def test_reporting_does_not_reuse_previous_batch_timing(monkeypatch):
    events = []
    monkeypatch.setattr(metrics, "emit", events.append)
    reporter = metrics.RolloutMetrics()
    first = Trajectory(segments=[], rollout_time_sec=12.0, chain_id="first")
    second = Trajectory(segments=[], rollout_time_sec=0.0, chain_id="second")

    for index, trajectory in enumerate((first, second, first)):
        events.clear()
        reporter.record_step(global_step=index, trajectories=[trajectory])
        timing_events = [event for event in events if event.name.startswith("agent/rollout/time/")]
        assert len(timing_events) == 3
        assert all(event.value == trajectory.rollout_time_sec for event in timing_events)
        slowest_events = [event for event in events if event.name == "agent/rollout/slowest_message"]
        assert len(slowest_events) == 1
        assert json.loads(slowest_events[0].value)["chain_id"] == trajectory.chain_id

    events.clear()
    reporter.record_step(global_step=3, trajectories=[Trajectory(segments=[])])
    assert not any(
        event.name.startswith("agent/rollout/time/") or event.name == "agent/rollout/slowest_message"
        for event in events
    )
