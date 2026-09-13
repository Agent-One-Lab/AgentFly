"""Exercise the shared segment protocol without model or environment services."""

import asyncio
import json
import sys
import types
from collections import Counter
from copy import deepcopy

import pytest

from agentfly.agents.rollout import base as rollout_base
from agentfly.agents.rollout.loop import metrics as rollout_metrics
from agentfly.agents.rollout.strategies import chain_rollout, step_rollout
from agentfly.agents.rollout.events import FinishReason
from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout import conversion
from agentfly.agents.specialized.action_agent import ActionAgent
from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.rewards.types import RewardResult
from agentfly.tools.tool_base import InvalidToolCallError
from agentfly.tools.types import ToolResult
from agentfly.utils.trajectories import gather_responses


class FakeContext:
    def __init__(self, *, metadata, **kwargs):
        self.metadata = dict(metadata)
        self.trajectory_format = "flat"

    async def release_resource(self, **kwargs):
        pass

    async def end_resource(self, **kwargs):
        pass


class FakeAgent:
    system_prompt = "SYS"
    tokenizer = None
    processor = None
    max_model_len = None
    _reward_fn = object()

    def __init__(self, tool_names, terminal_after=None):
        self.tool_names = tool_names
        self.terminal_after = terminal_after
        self.tool_calls = 0

    def prompt_tools(self):
        return []

    def maybe_append_context_trigger_user_message(self, step):
        pass

    def extract_final_response(self, messages):
        return "answer"

    async def execute_tool_call(self, context, tool_call, *args):
        self.tool_calls += 1
        name = tool_call["function"]["name"]
        observation = "summary" if name == "summarize" else f"O{self.tool_calls}"
        return ToolResult(
            name=name,
            arguments={},
            observation=observation,
            anchor=f"state{self.tool_calls}",
            step_reward=float(self.tool_calls == self.terminal_after),
            control="end" if self.tool_calls == self.terminal_after else None,
        )


async def fake_generation(agent, chain, current_step, tools, depth, chain_id,
                          generation_config, context):
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"A{depth + 1}"}],
        "token_ids": [101 + depth, 2],
    }
    names = agent.tool_names[depth]
    if names:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": "{}"}}
            for i, name in enumerate(names)
        ]
    return message, 10


async def fake_reward(*args, **kwargs):
    return RewardResult(reward=1.0, metrics={"accuracy": 1.0})


@pytest.fixture
def isolated_runtime(monkeypatch):
    for module in (chain_rollout, step_rollout):
        monkeypatch.setattr(module, "Context", FakeContext)
        monkeypatch.setattr(module, "generate_response", fake_generation)
    for module in (rollout_base, step_rollout):
        monkeypatch.setattr(module, "calculate_reward", fake_reward)
    monkeypatch.setattr(rollout_base, "observation_token_length", lambda *a, **k: 1)
    monkeypatch.setattr(rollout_base, "estimate_chat_prompt_tokens", lambda *a, **k: 10)
    monkeypatch.setattr(chain_rollout, "tqdm", lambda *a, **k: None)
    monkeypatch.setattr(chain_rollout.Monitor, "ensure_started", lambda: None)
    # Exercise real metric aggregation, but do not write to monitor sinks.
    events = []
    monkeypatch.setattr(rollout_metrics, "emit", events.append)
    return events


async def run(strategy, agent):
    return await strategy.run(
        agent=agent,
        messages=[{
            "messages": [{"role": "system", "content": "SYS"},
                         {"role": "user", "content": "O0"}],
            "task_id": "task_1",
        }],
        max_turns=len(agent.tool_names),
        num_chains=1,
    )


def roles(segment):
    return [message["role"] for message in segment.messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_reused_rollout_emits_one_summary_per_batch(strategy_cls, isolated_runtime):
    strategy = strategy_cls()
    per_chain_names = {"agent/rollout/trajectory", "agent/rollout/info"}
    for global_step in (1, 2):
        isolated_runtime.clear()
        result = await strategy.run(
            agent=FakeAgent([["act"]]),
            messages=[{"role": "user", "content": "O0"}],
            max_turns=1,
            num_chains=3,
            max_concurrent_chains=2,
        )

        summaries = [event for event in isolated_runtime if event.name not in per_chain_names]
        counts = Counter(event.name for event in summaries)
        assert counts["agent/rollout/step"] == 1
        assert all(count == 1 for count in counts.values())
        step_event = next(event for event in summaries if event.name == "agent/rollout/step")
        assert step_event.value == global_step
        assert step_event.x == global_step
        assert step_event.x_name == "agent/rollout/step"

        # Chain still logs once per completed trajectory; step keeps its summary-only path.
        for name in per_chain_names:
            events = [event for event in isolated_runtime if event.name == name]
            assert len(events) == (3 if strategy_cls is chain_rollout.ChainRollout else 0)
            assert all(event.sinks == ["jsonl"] for event in events)
        assert not hasattr(strategy, "finished_chains_count")
        assert not hasattr(strategy.metrics, "monitor_info")
        assert len(result) == 3
        assert all(trajectory.num_turns == 1 for trajectory in result)
        assert all(trajectory.tool_call_counts == {"act": 1} for trajectory in result)
        assert result.rewards == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls,expected_roles", [
    (chain_rollout.ChainRollout, [
        ["system", "user", "assistant", "tool", "assistant", "tool"],
    ]),
    (step_rollout.StepRollout, [
        ["system", "user", "assistant"],
        ["system", "user", "assistant", "user", "assistant"],
    ]),
])
async def test_rollouts_return_segments(strategy_cls, expected_roles, isolated_runtime):
    result = await run(strategy_cls(), FakeAgent([["act"], ["act"]], terminal_after=2))
    assert isinstance(result, RunResult)
    assert result.rollout == ("chain" if strategy_cls is chain_rollout.ChainRollout else "step")
    assert len(result) == 1
    trajectory = result[0]
    assert all(isinstance(segment, Segment) for segment in trajectory.segments)
    assert [roles(segment) for segment in trajectory.segments] == expected_roles
    assert trajectory.reward == 1.0
    assert trajectory.metrics["accuracy"] == 1.0
    assistants = [message for segment in trajectory.segments
                  for message in segment.messages if message["role"] == "assistant"]
    assert [101, 2] in [message.get("token_ids") for message in assistants]
    assert [102, 2] in [message.get("token_ids") for message in assistants]
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert restored.rollout == result.rollout
    assert restored[0].segments == trajectory.segments
    assert trajectory.num_turns == restored[0].num_turns == 2
    assert trajectory.tool_call_counts == restored[0].tool_call_counts == {"act": 2}
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_segments"] == len(expected_roles)
    assert averages["agent/rollout/avg_turns"] == 2.0
    assert averages["agent/rollout/avg_tool_calls"] == 2.0
    assert averages["agent/rollout/tool_calls/act"] == 2.0


@pytest.mark.asyncio
async def test_folded_chain_preserves_each_context_view(isolated_runtime):
    strategy = chain_rollout.ChainRollout()
    result = await run(strategy, FakeAgent([["summarize"], ["act"]], terminal_after=2))
    trajectory = result[0]
    assert trajectory.num_segments == 2
    before, after = trajectory.segments
    assert roles(before) == ["system", "user", "assistant"]
    assert roles(after) == ["system", "user", "assistant", "tool"]
    assert before.messages[1]["content"][0]["text"] == "O0"
    assert after.messages[1]["content"][0]["text"] == "O0 summary"
    assert before.messages[-1]["token_ids"] == [101, 2]
    assert after.messages[-2]["token_ids"] == [102, 2]
    chain = strategy.chains[trajectory.chain_id]
    assert [segment.messages for segment in trajectory.segments] == chain.histories
    assert trajectory.num_turns == 2
    assert trajectory.tool_call_counts == {"summarize": 1, "act": 1}
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_turns"] == 2.0
    assert averages["agent/rollout/avg_tool_calls"] == 2.0
    assert averages["agent/rollout/tool_calls/summarize"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chain", "step"])
async def test_run_preserves_final_context_and_conversion_only_selects_training_rows(
    kind, isolated_runtime, monkeypatch,
):
    import torch

    agent = FakeAgent([["summarize"]])
    agent._preprocess_messages = lambda messages: messages
    agent._preprocess_backends = lambda: None
    agent._postprocess_backends = lambda: None
    # Exercise the actual specialized hook through BaseAgent.run, not an
    # identity stand-in. The model and environment are the isolated fixtures.
    agent.postprocess_trajectories = types.MethodType(ActionAgent.postprocess_trajectories, agent)
    result = await BaseAgent.run(
        agent, rollout=kind, max_turns=1, num_chains=1,
        messages=[{
            "messages": [{"role": "system", "content": "SYS"},
                         {"role": "user", "content": "O0"}],
            "task_id": "task_1",
        }],
    )
    assert result.rollout == kind
    assert roles(result[0].segments[0]) == ["system", "user", "assistant"]
    if kind == "chain":
        assert result[0].num_segments == 2
        assert roles(result[0].segments[-1]) == ["system", "user"]
        assert "summary" in result[0].segments[-1].messages[-1]["content"][0]["text"]
    else:
        assert result[0].num_segments == 1
    before = deepcopy(result.model_dump())
    steps = result[0].steps
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        n = len(messages_list)
        reward_mask = torch.zeros(n, 4)
        reward_mask[:, -1] = 1.0
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
            "action_mask": torch.ones(n, 4, dtype=torch.long),
            "reward_mask": reward_mask,
        }

    class FakeDataProto:
        @staticmethod
        def from_single_dict(inputs, meta_info=None):
            return types.SimpleNamespace(inputs=inputs, meta_info=meta_info)

    verl = types.ModuleType("agentfly.verl")
    protocol = types.ModuleType("agentfly.verl.protocol")
    protocol.DataProto = FakeDataProto
    verl.protocol = protocol
    monkeypatch.setitem(sys.modules, "agentfly.verl", verl)
    monkeypatch.setitem(sys.modules, "agentfly.verl.protocol", protocol)
    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    batch = BaseAgent.to_verl_dataproto(agent, result, pad_to_multiple_of=3, log_token_drift=False)
    assert seen == [result[0].segments[0].messages]
    assert batch.meta_info["repeat_times"] == [3]
    assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [1.0] * 3
    assert batch.inputs["action_mask"].sum(dim=-1).tolist() == [4, 0, 0]
    assert result.model_dump() == before
    assert result[0].steps is steps


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [1.0, 0.25, 0.0, -0.5])
@pytest.mark.parametrize("pad_to_multiple_of", [None, 3])
async def test_chain_conversion_broadcasts_folded_rewards_and_preserves_padding(
    monkeypatch, isolated_runtime, reward, pad_to_multiple_of,
):
    import torch

    strategy = chain_rollout.ChainRollout()
    agent = FakeAgent([["summarize"], ["act"]], terminal_after=2)
    result = await run(strategy, agent)
    result[0].reward = reward
    expected = [segment.messages for segment in result[0].segments]
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        n = len(messages_list)
        reward_mask = torch.zeros(n, 4)
        reward_mask[:, -1] = 1.0
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
            "action_mask": torch.ones(n, 4, dtype=torch.long),
            "position_ids": torch.zeros(n, 4, dtype=torch.long),
            "reward_mask": reward_mask,
        }

    class FakeDataProto:
        @staticmethod
        def from_single_dict(inputs, meta_info=None):
            return types.SimpleNamespace(inputs=inputs, meta_info=meta_info)

    verl = types.ModuleType("agentfly.verl")
    protocol = types.ModuleType("agentfly.verl.protocol")
    protocol.DataProto = FakeDataProto
    verl.protocol = protocol
    monkeypatch.setitem(sys.modules, "agentfly.verl", verl)
    monkeypatch.setitem(sys.modules, "agentfly.verl.protocol", protocol)
    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    batch = BaseAgent.to_verl_dataproto(agent, result, pad_to_multiple_of=pad_to_multiple_of)
    row_count = pad_to_multiple_of or 2
    # The training tokenization comes first; the token-drift diagnostic then
    # re-tokenizes a token_ids-stripped copy of the real rows for comparison.
    assert seen[: len(expected)] == expected
    # Each segment gets the full outcome, not R / num_segments or final-only R.
    assert batch.inputs["rm_scores"][:, -1].tolist() == [reward] * row_count
    assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [reward] * row_count
    assert batch.inputs["action_mask"][:2].tolist() == [[1, 1, 1, 1]] * 2
    if pad_to_multiple_of:
        assert batch.inputs["action_mask"][2].tolist() == [0, 0, 0, 0]
    assert batch.inputs["segment_idx"].tolist() == [0] + [1] * (row_count - 1)
    assert batch.inputs["batch_idx"].tolist() == [0] * row_count
    assert batch.inputs["step_rewards"].tolist() == [[0.0, 1.0]] * row_count
    assert batch.inputs["step_observations"].tolist() == [["__init__", "state1"]] * row_count
    assert batch.meta_info["use_agent"] is True
    assert batch.meta_info["repeat_times"] == [row_count]
    # Chain batches now carry the same token-drift diagnostic as step batches.
    assert batch.meta_info["layout"] == "per_segment"
    assert set(batch.meta_info) <= {"use_agent", "layout", "repeat_times", "token_drift", "rollout_stats"}
    assert result[0].reward == reward
    assert result[0].num_segments == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_no_tool_response_is_still_a_segment(strategy_cls, isolated_runtime):
    result = await run(strategy_cls(), FakeAgent([[]]))
    assert result[0].num_segments == 1
    assert roles(result[0].segments[0]) == ["system", "user", "assistant"]
    assert result[0].segments[0].messages[-1]["token_ids"] == [101, 2]
    assert result[0].finish_reason == "no_tool_calls"
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_task_metadata_is_preserved_without_runtime_fields(
    strategy_cls, monkeypatch, isolated_runtime,
):
    agent = FakeAgent([[]])

    async def prepare(context, step):
        # Scratch state and top-level context edits must not replace task metadata.
        context.metadata["runtime_only"] = "scratch"
        context.metadata["answer"] = "runtime answer"

    monkeypatch.setattr(agent, "prepare_first_step", prepare, raising=False)
    task_metadata = {
        "task_id": "task_1", "id": "dataset_1", "answer": "gold",
        "extra": {"tags": ["example"]},
    }
    messages = [{
        "messages": [{"role": "user", "content": "O0"}],
        **deepcopy(task_metadata),
        "chain_id": "old_chain", "group_id": "old_group",
        "chain_idx": -1, "group_idx": -1,
        "num_turns": 999, "tool_call_counts": {"fake": 999},
    }]
    original = deepcopy(messages)
    result = await strategy_cls().run(
        agent=agent, messages=messages, max_turns=1, num_chains=2,
    )
    assert messages == original
    assert len({trajectory.chain_id for trajectory in result}) == 2
    for i, trajectory in enumerate(result):
        assert trajectory.metadata == task_metadata
        assert trajectory.chain_id != "old_chain"
        assert trajectory.group_id != "old_group"
        assert trajectory.chain_idx == i
        assert trajectory.group_idx == 0
        assert trajectory.num_turns == 1
        assert trajectory.tool_call_counts == {}
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert [trajectory.metadata for trajectory in restored] == [task_metadata] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("module,strategy_cls", [
    (chain_rollout, chain_rollout.ChainRollout),
    (step_rollout, step_rollout.StepRollout),
])
async def test_execution_time_includes_setup_reward_and_cleanup_but_not_queueing(
    module, strategy_cls, monkeypatch, isolated_runtime,
):
    clock = [100.0]
    # Replace only this module's clock, not asyncio's process-wide monotonic clock.
    monkeypatch.setattr(module, "time", types.SimpleNamespace(monotonic=lambda: clock[0]), raising=False)

    class TimedContext(FakeContext):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            clock[0] += 2.0

        async def release_resource(self, **kwargs):
            clock[0] += 11.0

        async def end_resource(self, **kwargs):
            clock[0] += 13.0

    async def prepare(context, step):
        clock[0] += 3.0

    async def generate(*args, **kwargs):
        clock[0] += 5.0
        # Let the other episode start waiting for the occupied concurrency slot.
        await asyncio.sleep(0)
        return await fake_generation(*args, **kwargs)

    async def reward(*args, **kwargs):
        clock[0] += 7.0
        return await fake_reward(*args, **kwargs)

    monkeypatch.setattr(module, "Context", TimedContext)
    monkeypatch.setattr(module, "generate_response", generate)
    monkeypatch.setattr(rollout_base, "calculate_reward", reward)
    monkeypatch.setattr(step_rollout, "calculate_reward", reward)
    agent = FakeAgent([[]])
    monkeypatch.setattr(agent, "prepare_first_step", prepare, raising=False)
    strategy = strategy_cls()
    assert not hasattr(strategy, "chain_rollout_seconds")
    result = await strategy.run(
        agent=agent,
        messages=[{"messages": [{"role": "user", "content": "O0"}]}],
        max_turns=1, num_chains=2, max_concurrent_chains=1,
    )
    # Each episode takes 2 + 3 + 5 + 7 + 11 + 13 seconds on the controlled clock.
    assert [trajectory.rollout_time_sec for trajectory in result] == [41.0, 41.0]
    restored = RunResult.model_validate_json(result.model_dump_json())
    assert [trajectory.rollout_time_sec for trajectory in restored] == [41.0, 41.0]
    assert not hasattr(strategy, "chain_rollout_seconds")
    # Both strategies report the same durations through the shared trajectory protocol.
    timing_events = {
        event.name: event.value for event in isolated_runtime
        if event.name.startswith("agent/rollout/time/")
    }
    assert timing_events == {
        "agent/rollout/time/min": 41.0,
        "agent/rollout/time/max": 41.0,
        "agent/rollout/time/avg": 41.0,
    }
    slowest_events = [
        event for event in isolated_runtime if event.name == "agent/rollout/slowest_message"
    ]
    assert len(slowest_events) == 1
    # Tied durations pick the first trajectory; the log uses the complete public schema.
    assert json.loads(slowest_events[0].value) == result[0].model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
@pytest.mark.parametrize("tool_names", [[[]], [["act"]]])
async def test_no_reward_function_returns_none(strategy_cls, tool_names, monkeypatch, isolated_runtime):
    async def unexpected_reward(*args, **kwargs):
        pytest.fail("No reward function is configured; reward evaluation must not run.")

    monkeypatch.setattr(rollout_base, "calculate_reward", unexpected_reward)
    monkeypatch.setattr(step_rollout, "calculate_reward", unexpected_reward)
    agent = FakeAgent(tool_names, terminal_after=1)
    agent._reward_fn = None
    result = await run(strategy_cls(), agent)
    assert result.rewards == [None]
    assert not any(event.name == "agent/rollout/reward" for event in isolated_runtime)
    assert RunResult.model_validate_json(result.model_dump_json()).rewards == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_computed_zero_reward_is_not_missing(strategy_cls, monkeypatch, isolated_runtime):
    async def zero_reward(*args, **kwargs):
        return RewardResult(reward=0.0, metrics={"accuracy": 0.0})

    monkeypatch.setattr(rollout_base, "calculate_reward", zero_reward)
    monkeypatch.setattr(step_rollout, "calculate_reward", zero_reward)
    result = await run(strategy_cls(), FakeAgent([[]]))
    assert result.rewards == [0.0]
    assert result[0].metrics["accuracy"] == 0.0
    assert any(event.name == "agent/rollout/reward" and event.value == 0.0
               for event in isolated_runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls,expected_calls", [
    (chain_rollout.ChainRollout, 2), (step_rollout.StepRollout, 1),
])
async def test_migration_does_not_change_multi_tool_execution(
    strategy_cls, expected_calls, isolated_runtime,
):
    agent = FakeAgent([["first", "second"]])
    result = await run(strategy_cls(), agent)
    assert agent.tool_calls == expected_calls
    assert len(result[0].steps) == expected_calls
    assert result[0].num_segments == 1
    assistant = next(message for message in result[0].segments[0].messages
                     if message["role"] == "assistant")
    assert len(assistant["tool_calls"]) == 2
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == ({"first": 1, "second": 1} if expected_calls == 2 else {"first": 1})
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_turns"] == 1.0
    assert averages["agent/rollout/avg_tool_calls"] == expected_calls


def test_gather_responses_reads_segment_messages():
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]
    trajectories = [
        Trajectory(segments=[Segment(messages=deepcopy(messages))], chain_id="chain_1"),
        Trajectory(segments=[Segment(messages=[])], metadata={"id": "task_2"}),
        Trajectory(segments=[], chain_id="chain_3"),
    ]
    assert gather_responses(trajectories) == [
        {"id": "chain_1", "response": "answer"},
        {"id": "task_2", "response": ""},
        {"id": "chain_3", "response": ""},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
@pytest.mark.parametrize("reward_enabled", [True, False])
async def test_rollouts_merge_the_same_tool_metric_averages(
    strategy_cls, reward_enabled, monkeypatch, isolated_runtime,
):
    agent = FakeAgent([["act"], ["act"], []])
    if not reward_enabled:
        agent._reward_fn = None
    results = [
        ToolResult(name="act", arguments={}, observation="O1", step_reward=0.0,
                   metrics={"valid": True, "cost": 2, "note": "text"}),
        ToolResult(name="act", arguments={}, observation="O2", step_reward=1.0,
                   metrics={"valid": False, "cost": 4}),
    ]
    pending = iter(results)

    async def execute(*args, **kwargs):
        agent.tool_calls += 1
        return next(pending)

    async def reward(*args, **kwargs):
        return RewardResult(reward=0.25, metrics={"score": 7, "tool/act/cost": 999})

    monkeypatch.setattr(agent, "execute_tool_call", execute)
    monkeypatch.setattr(rollout_base, "calculate_reward", reward)
    monkeypatch.setattr(step_rollout, "calculate_reward", reward)
    result = await run(strategy_cls(), agent)
    trajectory = result[0]
    expected = {"tool/act/valid": 0.5, "tool/act/cost": 3.0, "tool/act/step_reward": 0.5}
    if reward_enabled:
        expected["score"] = 7
    assert trajectory.metrics == expected
    assert trajectory.reward == (0.25 if reward_enabled else None)
    assert trajectory.finish_reason == "no_tool_calls"
    assert agent.tool_calls == 2
    # Raw environment signals remain intact; aggregates are a separate logging view.
    assert [s.tool_result for s in trajectory.steps if s.tool_result is not None] == results
    assert RunResult.model_validate_json(result.model_dump_json())[0].metrics == expected
    logged = {event.name: event.value for event in isolated_runtime}
    assert logged["agent/rollout/tool/act/cost"] == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
@pytest.mark.parametrize("project_actions", [False, True])
@pytest.mark.parametrize("status,control,policy,expected", [
    ("success", "end", {}, "tool_control_end"),
    ("invalid", None, {}, "invalid_end"),
    ("error", None, {"on_tool_error": "end"}, "tool_error_end"),
    ("error", "end", {"on_tool_error": "raise"}, "tool_control_end"),
    ("invalid", "end", {"on_invalid_tool_call": "raise"}, "tool_control_end"),
    ("success", None, {}, "max_turns"),
    ("error", None, {}, "max_turns"),
    ("invalid", "continue", {"on_invalid_tool_call": "raise"}, "max_turns"),
    ("error", "continue", {"on_tool_error": "end"}, "max_turns"),
])
async def test_equivalent_tool_stops_have_the_same_finish_reason(
    strategy_cls, project_actions, status, control, policy, expected, monkeypatch, isolated_runtime,
):
    agent = FakeAgent([[], []] if project_actions else [["act"], ["act"]])
    agent.action_tool_name = "act"

    async def execute(*args, **kwargs):
        agent.tool_calls += 1
        return ToolResult(name="act", arguments={}, observation="obs", status=status, control=control)

    monkeypatch.setattr(agent, "execute_tool_call", execute)
    strategy = strategy_cls(project_actions=project_actions, **policy)
    result = await run(strategy, agent)
    trajectory = result[0]
    assert trajectory.finish_reason == expected
    assert FinishReason(trajectory.finish_reason).value == expected
    assert agent.tool_calls == (2 if expected == "max_turns" else 1)
    assert trajectory.num_turns == agent.tool_calls
    assert trajectory.tool_call_counts == {"act": agent.tool_calls}
    assert RunResult.model_validate_json(result.model_dump_json())[0].finish_reason == expected
    if isinstance(strategy, chain_rollout.ChainRollout):
        assert strategy.chains[trajectory.chain_id].info["finish_reason"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
@pytest.mark.parametrize("model_status", ["terminal", "finish"])
async def test_model_stop_remains_terminal_without_executing_tool(
    strategy_cls, model_status, monkeypatch, isolated_runtime,
):
    async def generate(*args, **kwargs):
        message, length = await fake_generation(*args, **kwargs)
        message["status"] = model_status
        return message, length

    monkeypatch.setattr(chain_rollout, "generate_response", generate)
    monkeypatch.setattr(step_rollout, "generate_response", generate)
    agent = FakeAgent([["act"]])
    result = await run(strategy_cls(), agent)
    assert result[0].finish_reason == "terminal"
    assert agent.tool_calls == 0
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
@pytest.mark.parametrize("project_actions", [False, True])
@pytest.mark.parametrize("status,control,policy,error", [
    ("invalid", None, {"on_invalid_tool_call": "raise"}, InvalidToolCallError),
    ("error", None, {"on_tool_error": "raise"}, RuntimeError),
    ("success", "raise", {}, RuntimeError),
])
async def test_tool_raise_still_raises_instead_of_returning_a_finish_reason(
    strategy_cls, project_actions, status, control, policy, error, monkeypatch, isolated_runtime,
):
    async def execute(*args, **kwargs):
        return ToolResult(name="act", arguments={}, observation="failure", status=status, control=control)

    monkeypatch.setattr(FakeAgent, "execute_tool_call", execute)
    agent = FakeAgent([[]] if project_actions else [["act"]])
    agent.action_tool_name = "act"
    with pytest.raises(error):
        await run(strategy_cls(project_actions=project_actions, **policy), agent)


@pytest.mark.asyncio
async def test_chain_context_limit_takes_precedence_over_tool_end(isolated_runtime):
    agent = FakeAgent([["act"]], terminal_after=1)
    # Generation uses 10 tokens, then observation + wrapper exceeds the 11-token guard.
    agent.max_model_len = rollout_base.TOKEN_SAFETY_MARGIN + 11
    result = await run(chain_rollout.ChainRollout(), agent)
    assert agent.tool_calls == 1
    assert result[0].finish_reason == "max_model_len"
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {"act": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_generation_only_turns_count_even_without_tools(strategy_cls, isolated_runtime):
    result = await run(strategy_cls(on_no_tool_call="continue"), FakeAgent([[], [], []]))
    assert result[0].num_turns == 3
    assert result[0].tool_call_counts == {}
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_turns"] == 3.0
    assert averages["agent/rollout/avg_tool_calls"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [
    {"history_length": 0},
    {"history_length": 2},
    {"history_length": 2, "prompt_builder": "alfworld_flat"},
])
async def test_step_counts_do_not_depend_on_prompt_representation(config, isolated_runtime):
    agent = FakeAgent([["act"], ["act"], ["act"]], terminal_after=3)
    result = await run(step_rollout.StepRollout(**config), agent)
    assert result[0].num_turns == 3
    assert result[0].tool_call_counts == {"act": 3}
    assert agent.tool_calls == 3
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_turns"] == 3.0
    assert averages["agent/rollout/avg_tool_calls"] == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_early_tool_stop_counts_only_attempted_calls(strategy_cls, isolated_runtime):
    agent = FakeAgent([["first", "second"]], terminal_after=1)
    result = await run(strategy_cls(), agent)
    assert agent.tool_calls == 1
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {"first": 1}


@pytest.mark.asyncio
async def test_failed_chain_generation_has_no_completed_turn(monkeypatch, isolated_runtime):
    async def fail(*args, **kwargs):
        raise ValueError("maximum context length exceeded")

    monkeypatch.setattr(chain_rollout, "generate_response", fail)
    agent = FakeAgent([[]])
    agent._reward_fn = None
    result = await run(chain_rollout.ChainRollout(), agent)
    assert result[0].finish_reason == "max_model_len"
    assert result[0].num_turns == 0
    assert result[0].tool_call_counts == {}
    averages = {event.name: event.value for event in isolated_runtime}
    assert averages["agent/rollout/avg_turns"] == 0.0
    assert averages["agent/rollout/avg_tool_calls"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_projected_calls_count_from_tool_results(strategy_cls, monkeypatch, isolated_runtime):
    agent = FakeAgent([[]], terminal_after=1)
    monkeypatch.setattr(agent, "action_tool_name", "act", raising=False)
    result = await run(strategy_cls(project_actions=True), agent)
    assert agent.tool_calls == 1
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {"act": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("call_fields", [{}, {"tool_calls": []}, {"tool_calls": None}],
                         ids=["missing-calls", "empty-calls", "null-calls"])
@pytest.mark.parametrize("tool_source", ["action_tool_name", "first_tool"])
@pytest.mark.parametrize("content_format", ["text", "parts"])
async def test_chain_projects_before_recording_and_emitting_messages(
    call_fields, tool_source, content_format, monkeypatch, isolated_runtime,
):
    agent = FakeAgent([[]], terminal_after=1)
    if tool_source == "action_tool_name":
        agent.action_tool_name = "act"
        agent.tools = [types.SimpleNamespace(name="not_selected")]
    else:
        agent.tools = [types.SimpleNamespace(name="act")]
    text = "I Think I Should Just Wander Around The Room For A While"
    content = text if content_format == "text" else [{"type": "text", "text": text}]
    message = {"role": "assistant", "content": content, "token_ids": [11, 12, 2],
               "status": "terminal", **call_fields}

    async def generate(*args, **kwargs):
        return deepcopy(message), 10

    monkeypatch.setattr(chain_rollout, "generate_response", generate)
    executed_calls, execution_messages, emitted_messages = [], [], []
    original_execute = agent.execute_tool_call

    async def execute(context, tool_call, messages, *args):
        executed_calls.append(deepcopy(tool_call))
        execution_messages.append(deepcopy(messages.messages[-1]))
        return await original_execute(context, tool_call, messages, *args)

    monkeypatch.setattr(agent, "execute_tool_call", execute)
    strategy = chain_rollout.ChainRollout(project_actions=True)
    original_run_turn = strategy._run_turn

    async def capture_turn(**kwargs):
        async for event in original_run_turn(**kwargs):
            if isinstance(event, chain_rollout.MessageProduced):
                # Snapshot at yield time, before any later mutation could mask the bug.
                emitted_messages.append(deepcopy(event.message))
            yield event

    monkeypatch.setattr(strategy, "_run_turn", capture_turn)
    result = await run(strategy, agent)

    expected_call = {
        "id": "call_projected", "type": "function",
        "function": {"name": "act", "arguments": json.dumps({"action": text.lower()[-30:]})},
    }
    assert agent.tool_calls == 1
    assert executed_calls == [expected_call]
    assert emitted_messages == [{**message, "tool_calls": [expected_call]}]
    # Message normalization still happens, but response text and sampled IDs are preserved.
    expected_stored = {**message, "content": [{"type": "text", "text": text}],
                       "tool_calls": [expected_call]}
    thought = next(s for s in strategy.chains[result[0].chain_id].steps() if s.type == "Thought")
    assert thought.messages[-1] == expected_stored
    assert execution_messages == [expected_stored]
    assert result[0].segments[0].messages[-2] == expected_stored
    assert result[0].segments[0].messages[-1]["tool_call_id"] == expected_call["id"]
    assert result[0].finish_reason == "tool_control_end"
    assert result[0].tool_call_counts == {"act": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("project_actions", [False, True])
@pytest.mark.parametrize("model_status", ["continue", "terminal", "finish"])
async def test_chain_projection_preserves_existing_calls_and_model_stop_policy(
    project_actions, model_status, monkeypatch, isolated_runtime,
):
    async def generate(*args, **kwargs):
        message, length = await fake_generation(*args, **kwargs)
        message["status"] = model_status
        return message, length

    monkeypatch.setattr(chain_rollout, "generate_response", generate)
    strategy = chain_rollout.ChainRollout(project_actions=project_actions)

    def unexpected_projection(*args):
        pytest.fail("An existing tool call must not be projected")

    monkeypatch.setattr(strategy, "_projected_tool_call", unexpected_projection)
    agent = FakeAgent([["act"]], terminal_after=1)
    result = await run(strategy, agent)
    should_execute = project_actions or model_status == "continue"
    assert agent.tool_calls == int(should_execute)
    assert result[0].finish_reason == ("tool_control_end" if should_execute else "terminal")
    thought = next(s for s in strategy.chains[result[0].chain_id].steps() if s.type == "Thought")
    assert thought.messages[-1]["tool_calls"] == [
        {"id": "call_0", "type": "function", "function": {"name": "act", "arguments": "{}"}},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("project_actions,has_action_tool", [(False, True), (True, False)])
@pytest.mark.parametrize("policy", ["continue", "end", "raise"])
async def test_chain_without_projection_keeps_no_tool_policy(
    project_actions, has_action_tool, policy, isolated_runtime,
):
    agent = FakeAgent([[], []])
    agent.action_tool_name = "act" if has_action_tool else None
    agent.tools = []
    strategy = chain_rollout.ChainRollout(project_actions=project_actions, on_no_tool_call=policy)
    if policy == "raise":
        with pytest.raises(RuntimeError, match="on_no_tool_call='raise'"):
            await run(strategy, agent)
    else:
        result = await run(strategy, agent)
        assert result[0].finish_reason == ("max_turns" if policy == "continue" else "no_tool_calls")
        assert result[0].num_turns == (2 if policy == "continue" else 1)
        assert result[0].tool_call_counts == {}
        assert not any(message.get("tool_calls") for segment in result[0].segments
                       for message in segment.messages)
    assert agent.tool_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("budget,expected_turns,expected_calls", [
    (0, 0, 0),  # Pre-generation guard.
    (10, 1, 0),  # Generation reaches the limit: record the proposed call, do not execute it.
    (11, 1, 1),  # Observation exceeds the limit: it still takes precedence over tool end.
])
async def test_chain_projection_respects_context_limits(
    budget, expected_turns, expected_calls, isolated_runtime,
):
    agent = FakeAgent([[]], terminal_after=1)
    agent.action_tool_name = "act"
    agent.max_model_len = rollout_base.TOKEN_SAFETY_MARGIN + budget
    result = await run(chain_rollout.ChainRollout(project_actions=True), agent)
    assert result[0].finish_reason == "max_model_len"
    assert result[0].num_turns == expected_turns
    assert agent.tool_calls == expected_calls
    assert result[0].tool_call_counts == ({"act": 1} if expected_calls else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [chain_rollout.ChainRollout, step_rollout.StepRollout])
async def test_initial_prompt_history_is_not_execution(strategy_cls, isolated_runtime):
    result = await strategy_cls().run(
        agent=FakeAgent([[]]),
        messages=[{"messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current question"},
        ]}],
        max_turns=1, num_chains=1,
    )
    assert result[0].num_turns == 1
    assert result[0].tool_call_counts == {}
