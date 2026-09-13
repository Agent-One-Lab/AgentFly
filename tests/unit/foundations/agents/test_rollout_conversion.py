"""Characterize the common batch fields and the two legacy signal layouts."""

import sys
import types
from copy import deepcopy

import numpy as np
import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout import conversion
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.rollout.conversion import build_segment_batch
from agentfly.agents.rollout.structures import Step
from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.agents.utils.messages import Messages
from agentfly.tools.types import ToolResult


def make_trajectory(cid, group, count, reward, metrics):
    steps = []
    for i in range(count):
        messages = [
            {"role": "user", "content": f"{cid}/obs{i}"},
            {"role": "assistant", "content": "<think>x</think><action>act</action>",
             "token_ids": [100 + i, 2]},
        ]
        steps.append(Step(
            type="Action Input", messages=Messages.from_turns(messages),
            tool_result=ToolResult(name="act", arguments={}, observation=f"{cid}/obs{i + 1}",
                                   anchor=f"{cid}/state{i + 1}", step_reward=float(i)),
        ))
    return Trajectory(
        segments=[Segment(messages=s.messages.messages) for s in steps],
        steps=steps, chain_id=cid, group_id=group, reward=reward, metrics=metrics,
    )


def make_result():
    return RunResult(trajectories=[
        make_trajectory("A", "G1", 2, 0.75,
                        {"score": 2.0, "explicit_none": None, "text": "ok"}),
        make_trajectory("empty", "G1", 0, None, {"empty_only": 5.0}),
        make_trajectory("B", "G2", 1, -0.25,
                        {"score": 4.0, "explicit_none": 3.0, "other": False}),
    ])


def fake_inputs(n):
    import torch

    ids = torch.arange(n * 5, dtype=torch.long).reshape(n, 5)
    actions = torch.zeros(n, 5, dtype=torch.long)
    rewards = torch.zeros(n, 5, dtype=torch.float32)
    for i in range(n):
        end = 1 + i % 3
        actions[i, 1:end + 1] = 1
        rewards[i, end] = 1.0
    return {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(5).repeat(n, 1), "labels": ids.clone(),
        "action_mask": actions, "reward_mask": rewards,
    }


@pytest.fixture
def fake_verl(monkeypatch):
    class DataProto:
        @staticmethod
        def from_single_dict(inputs, meta_info=None):
            return types.SimpleNamespace(inputs=inputs, meta_info=meta_info)

    package = types.ModuleType("agentfly.verl")
    protocol = types.ModuleType("agentfly.verl.protocol")
    protocol.DataProto = DataProto
    package.protocol = protocol
    monkeypatch.setitem(sys.modules, "agentfly.verl", package)
    monkeypatch.setitem(sys.modules, "agentfly.verl.protocol", protocol)


@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
@pytest.mark.parametrize("multiple", [None, 1, 4])
@pytest.mark.parametrize("entrypoint", ["converter", "agent"])
def test_exporter_batch_contract(strategy_cls, multiple, entrypoint, fake_verl, monkeypatch):
    import torch

    result = make_result()
    result.rollout = "chain" if strategy_cls is ChainRollout else "step"
    before = deepcopy(result.model_dump())
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        assert kwargs["return_reward_mask"] is True
        assert kwargs["concatenate_mm_inputs"] is False
        return fake_inputs(len(messages_list))

    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    agent = types.SimpleNamespace(tokenizer=None, processor=None)
    if entrypoint == "agent":
        batch = BaseAgent.to_verl_dataproto(
            agent, result, pad_to_multiple_of=multiple, log_token_drift=False,
        )
    else:
        convert = conversion.chain_to_dataproto if strategy_cls is ChainRollout else conversion.step_to_dataproto
        batch = convert(agent, result, pad_to_multiple_of=multiple, log_token_drift=False)
    inputs = batch.inputs
    assert seen == [s.messages for t in result for s in t.segments]
    assert result.model_dump() == before
    rows = [0, 1, 2, 2] if multiple == 4 else [0, 1, 2]
    original = fake_inputs(3)
    for key, value in original.items():
        expected = value[rows].clone()
        if key == "action_mask" and multiple == 4:
            expected[-1] = 0
        torch.testing.assert_close(inputs[key], expected, rtol=0, atol=0)
    outcomes = torch.tensor([0.75, 0.75, -0.25])[rows]
    torch.testing.assert_close(inputs["rm_scores"], original["reward_mask"][rows] * outcomes[:, None],
                               rtol=0, atol=0)
    assert inputs["uid"].tolist() == ["G1", "G1"] + ["G2"] * (len(rows) - 2)
    assert inputs["uid"].dtype == object

    is_step = strategy_cls is StepRollout
    missing = 0.0 if is_step else None
    columns = {
        "score": [2.0, 2.0, 4.0], "explicit_none": [None, None, 3.0],
        "text": ["ok", "ok", missing], "other": [missing, missing, False],
        "empty_only": [missing] * 3,
    }
    for key, values in columns.items():
        expected = np.array([values[i] for i in rows], dtype=object if is_step else None)
        np.testing.assert_array_equal(inputs[f"rm_{key}"], expected)
        assert inputs[f"rm_{key}"].dtype == expected.dtype

    common_keys = set(original) | {"uid", "rm_scores"} | {f"rm_{k}" for k in columns}
    if is_step:
        assert set(inputs) == common_keys | {
            "traj_uid", "anchor_obs", "step_env_reward", "is_action_valid", "active_masks",
        }
        assert inputs["traj_uid"].tolist() == ["A", "A"] + ["B"] * (len(rows) - 2)
        assert inputs["anchor_obs"].tolist() == ["__init__", "A/state1"] + ["__init__"] * (len(rows) - 2)
        assert inputs["step_env_reward"].tolist() == [0.0, 1.0] + [0.0] * (len(rows) - 2)
        assert inputs["is_action_valid"].tolist() == [1.0] * len(rows)
        assert inputs["active_masks"].tolist() == [1.0] * 3 + [0.0] * (len(rows) - 3)
        assert inputs["step_env_reward"].dtype == np.float32
        assert inputs["active_masks"].dtype == np.float32
    else:
        assert set(inputs) == common_keys | {
            "batch_idx", "segment_idx", "step_observations", "step_rewards", "step_invalids",
        }
        assert inputs["batch_idx"].tolist() == [0, 0] + [2] * (len(rows) - 2)
        assert inputs["segment_idx"].tolist() == [0, 1] + [0] * (len(rows) - 2)
        assert inputs["batch_idx"].dtype == inputs["segment_idx"].dtype == np.int32
        assert inputs["step_observations"].tolist() == [["__init__", "A/state1"]] * 2 + [["__init__"]] * (len(rows) - 2)
        assert inputs["step_rewards"].tolist() == [[0.0, 1.0]] * 2 + [[0.0]] * (len(rows) - 2)
        assert inputs["step_invalids"].tolist() == [[0.0, 0.0]] * 2 + [[0.0]] * (len(rows) - 2)
        assert inputs["step_observations"].shape == (len(rows),)

    expected_meta = {
        "use_agent": True,
        "repeat_times": [2, 0, len(rows) - 2],
        "layout": "per_step" if is_step else "per_segment",
    }
    # Chain batches also carry the loop-policy counters (capped/tool-less/nudged
    # turns); compare the contract keys only.
    meta = {k: v for k, v in batch.meta_info.items() if k != "rollout_stats"}
    assert meta == expected_meta


@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
def test_exporter_missing_outcome_behavior(strategy_cls, fake_verl, monkeypatch):
    result = RunResult(
        trajectories=[make_trajectory("A", "G", 1, None, {})],
        rollout="chain" if strategy_cls is ChainRollout else "step",
    )
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda agent, messages_list, **kwargs: fake_inputs(len(messages_list)))
    agent = types.SimpleNamespace(tokenizer=None, processor=None)
    if strategy_cls is ChainRollout:
        with pytest.raises(TypeError):
            BaseAgent.to_verl_dataproto(agent, result)
    else:
        batch = BaseAgent.to_verl_dataproto(agent, result, log_token_drift=False)
        assert batch.inputs["rm_scores"].sum().item() == 0.0


@pytest.mark.parametrize("strategy_cls", [ChainRollout, StepRollout])
@pytest.mark.parametrize("multiple", [None, 4])
def test_exporter_multimodal_metadata_stays_row_aligned(strategy_cls, multiple, fake_verl, monkeypatch):
    result = make_result()
    result.rollout = "chain" if strategy_cls is ChainRollout else "step"
    mm = [{"pixel_values": i} for i in range(3)]
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda agent, messages_list, **kwargs: {**fake_inputs(3), "mm_inputs": mm})
    batch = BaseAgent.to_verl_dataproto(
        types.SimpleNamespace(tokenizer=None, processor=None), result,
        pad_to_multiple_of=multiple, log_token_drift=False,
    )
    expected = mm + [mm[-1]] if multiple == 4 else mm
    if strategy_cls is ChainRollout:
        assert batch.inputs["multi_modal_inputs"].tolist() == expected
        assert "mm_inputs" not in batch.inputs
    else:
        assert batch.inputs["mm_inputs"] == expected
        assert "multi_modal_inputs" not in batch.inputs
    assert len(mm) == 3  # Padding must not mutate tokenizer-owned metadata.


def test_shared_conversion_uses_segments_without_runtime_steps_or_cached_state():
    import torch

    result = make_result()
    for trajectory in result:
        trajectory.steps = []
    original = deepcopy(result.model_dump())
    tokenizer_outputs = fake_inputs(3)
    tokenizer_before = {key: value.clone() for key, value in tokenizer_outputs.items()}
    agent = types.SimpleNamespace(tokenizer=object(), processor=object())

    def tokenize(host, messages_list, **kwargs):
        assert host is agent
        assert messages_list == [s.messages for t in result for s in t.segments]
        assert kwargs["tokenizer"] is agent.tokenizer
        assert kwargs["processor"] is agent.processor
        assert kwargs["train_on_last_turn"] is True
        return tokenizer_outputs

    padded, indices, repeats, pad_size = build_segment_batch(
        agent, result, tokenize=tokenize, train_on_last_turn=True, pad_to_multiple_of=4,
    )
    assert indices == [(0, 0), (0, 1), (2, 0), (2, 0)]
    assert repeats == [2, 0, 2]
    assert pad_size == 1
    assert padded["rm_scores"].sum(dim=-1).tolist() == [0.75, 0.75, -0.25, -0.25]
    assert not {"step_rewards", "step_env_reward", "traj_uid", "batch_idx"} & padded.keys()
    for key, value in tokenizer_before.items():
        torch.testing.assert_close(tokenizer_outputs[key], value, rtol=0, atol=0)
    assert "rm_scores" not in tokenizer_outputs

    # Repeated conversion does not retain the previous padding or mutate the result.
    _, indices, repeats, pad_size = build_segment_batch(
        agent, result, tokenize=tokenize, train_on_last_turn=True,
    )
    assert indices == [(0, 0), (0, 1), (2, 0)]
    assert repeats == [2, 0, 1]
    assert pad_size == 0
    assert result.model_dump() == original


def test_shared_conversion_rejects_misaligned_token_rows():
    with pytest.raises(ValueError, match="one row per segment"):
        build_segment_batch(
            types.SimpleNamespace(tokenizer=None, processor=None), make_result(),
            tokenize=lambda *args, **kwargs: fake_inputs(2),
        )


def test_shared_conversion_rejects_misaligned_multimodal_padding():
    with pytest.raises(ValueError, match="one entry per segment"):
        build_segment_batch(
            types.SimpleNamespace(tokenizer=None, processor=None), make_result(),
            tokenize=lambda *args, **kwargs: {**fake_inputs(3), "mm_inputs": [{}]},
            pad_to_multiple_of=4,
        )


def test_chain_discarded_rows_keep_masks_and_scores_zero(fake_verl, monkeypatch):
    class DiscardedTrajectory(Trajectory):
        discarded: bool = True

    result = make_result()
    result.trajectories[-1] = DiscardedTrajectory(**dict(result.trajectories[-1]))
    result.rollout = "chain"
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda *args, **kwargs: fake_inputs(3))
    batch = BaseAgent.to_verl_dataproto(types.SimpleNamespace(tokenizer=None, processor=None), result,
                                pad_to_multiple_of=4)
    for key in ("action_mask", "reward_mask", "rm_scores"):
        assert batch.inputs[key][2:].sum().item() == 0.0
    assert batch.inputs["rm_scores"][:2].sum(dim=-1).tolist() == [0.75, 0.75]
    assert batch.inputs["rm_score"].tolist() == [2.0, 2.0, 4.0, 4.0]


def test_step_diagnostics_count_only_real_rows(fake_verl, monkeypatch):
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda *args, **kwargs: fake_inputs(3))
    result = make_result()
    result.rollout = "step"
    seen = []

    def drift(agent, conversations, inputs, **kwargs):
        seen.extend(conversations)
        return {"token_drift/n_sampled": float(len(conversations))}

    monkeypatch.setattr(conversion, "token_drift_stats", drift)
    batch = BaseAgent.to_verl_dataproto(types.SimpleNamespace(tokenizer=None, processor=None), result,
                                pad_to_multiple_of=4)
    assert seen == [s.messages for t in result for s in t.segments]
    assert batch.meta_info["token_drift"] == {"token_drift/n_sampled": 3.0}


def test_step_diagnostic_failure_does_not_break_conversion(fake_verl, monkeypatch):
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda *args, **kwargs: fake_inputs(3))

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(conversion, "token_drift_stats", fail)
    result = make_result()
    result.rollout = "step"
    batch = BaseAgent.to_verl_dataproto(types.SimpleNamespace(tokenizer=None, processor=None), result,
                                pad_to_multiple_of=4)
    assert "token_drift" not in batch.meta_info
    assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [0.75, 0.75, -0.25, -0.25]


@pytest.mark.parametrize("rollout", [None, "", "unknown"])
def test_agent_conversion_rejects_missing_or_unsupported_rollout(rollout, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid identifiers must fail before conversion/tokenization")

    monkeypatch.setattr(conversion, "tokenize_trajectories", forbidden)
    monkeypatch.setattr(conversion, "chain_to_dataproto", forbidden)
    monkeypatch.setattr(conversion, "step_to_dataproto", forbidden)
    result = RunResult(trajectories=[], rollout=rollout)
    with pytest.raises(ValueError, match="RunResult.rollout"):
        BaseAgent.to_verl_dataproto(object(), result)


def test_agent_conversion_requires_explicit_result():
    with pytest.raises(TypeError):
        BaseAgent.to_verl_dataproto(object())
    with pytest.raises(TypeError, match="Expected a RunResult"):
        BaseAgent.to_verl_dataproto(object(), False)


@pytest.mark.parametrize("marker", ["chain", "step"])
def test_agent_dispatches_only_from_result_marker(marker, monkeypatch):
    result = RunResult(trajectories=[], rollout=marker)
    agent = object()
    seen = []

    def convert(host, supplied, **kwargs):
        assert host is agent and supplied is result
        seen.append(marker)
        return "converted"

    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong converter selected")

    other = "step" if marker == "chain" else "chain"
    monkeypatch.setattr(conversion, f"{marker}_to_dataproto", convert)
    monkeypatch.setattr(conversion, f"{other}_to_dataproto", forbidden)
    assert BaseAgent.to_verl_dataproto(agent, result) == "converted"
    assert seen == [marker]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_kind", ["chain", "step"])
@pytest.mark.parametrize("second_kind", ["chain", "step"])
@pytest.mark.parametrize("dict_postprocess", [False, True])
async def test_convert_earlier_run_without_execution_state(
    first_kind, second_kind, dict_postprocess, fake_verl, monkeypatch,
):
    import agentfly.agents.agent_base as agent_base

    def postprocess(trajectories):
        if dict_postprocess:
            # Legacy hooks may return dicts; retain runtime-only steps explicitly.
            return [dict(t.model_dump(), steps=t.steps) for t in trajectories]
        return trajectories

    agent = types.SimpleNamespace(
        tokenizer=None, processor=None,
        _preprocess_messages=lambda messages: messages,
        _preprocess_backends=lambda: None,
        _postprocess_backends=lambda: None,
        postprocess_trajectories=postprocess,
    )
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        return fake_inputs(len(messages_list))

    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)

    async def run_result(kind, cid, reward):
        strategy = ChainRollout() if kind == "chain" else StepRollout()
        produced = RunResult(
            rollout=kind, trajectories=[make_trajectory(cid, cid, 2, reward, {})],
        )
        produced[0].num_turns = 2
        produced[0].tool_call_counts = {"act": 2}

        async def fake_run(**kwargs):
            return produced

        monkeypatch.setattr(strategy, "run", fake_run)
        result = await BaseAgent.run(agent, messages=[], max_turns=2, rollout=strategy)
        assert result.rollout == kind
        assert isinstance(result[0], Trajectory)
        assert result[0].steps == produced[0].steps
        assert result[0].num_turns == 2
        assert result[0].tool_call_counts == {"act": 2}
        assert not hasattr(agent, "_last_run_result")
        assert not hasattr(agent, "_last_rollout")
        return result

    first = await run_result(first_kind, "first", 0.75)
    first_before = deepcopy(first.model_dump())
    second = await run_result(second_kind, "second", -0.25)
    assert second[0].chain_id == "second"
    assert second.rewards == [-0.25]
    assert first.model_dump() == first_before

    def forbidden(*args, **kwargs):
        raise AssertionError("Conversion must not resolve, construct, or call a rollout")

    monkeypatch.setattr(agent_base, "resolve_rollout", forbidden)
    for cls in (ChainRollout, StepRollout):
        monkeypatch.setattr(cls, "__init__", forbidden)
        monkeypatch.setattr(cls, "to_dataproto", forbidden, raising=False)
    # Convert A, B, then A again without any cached result or rollout on the agent.
    for supplied, kind, cid, reward in (
        (first, first_kind, "first", 0.75),
        (second, second_kind, "second", -0.25),
        (first, first_kind, "first", 0.75),
    ):
        seen.clear()
        batch = BaseAgent.to_verl_dataproto(agent, supplied, log_token_drift=False)
        assert seen == [s.messages for s in supplied[0].segments]
        assert batch.inputs["uid"].tolist() == [cid, cid]
        assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [reward, reward]
        assert batch.meta_info["layout"] == ("per_step" if kind == "step" else "per_segment")
        signal = "step_env_reward" if kind == "step" else "step_rewards"
        assert batch.inputs[signal].tolist() == ([0.0, 1.0] if kind == "step" else [[0.0, 1.0]] * 2)
        assert first.model_dump() == first_before


@pytest.mark.parametrize("enabled", [False, True])
def test_agent_token_drift_uses_explicit_option_without_execution_state(enabled, fake_verl, monkeypatch):
    result = make_result()
    result.rollout = "step"
    agent = types.SimpleNamespace(tokenizer=None, processor=None)
    monkeypatch.setattr(conversion, "tokenize_trajectories",
                        lambda agent, messages_list, **kw: fake_inputs(len(messages_list)))
    seen = []

    def drift(agent, conversations, inputs, **kwargs):
        seen.extend(conversations)
        return {"token_drift/test": 1.0}

    monkeypatch.setattr(conversion, "token_drift_stats", drift)
    batch = BaseAgent.to_verl_dataproto(
        agent, result, log_token_drift=enabled, pad_to_multiple_of=4,
    )
    assert ("token_drift" in batch.meta_info) is enabled
    assert len(seen) == (3 if enabled else 0)


@pytest.mark.parametrize("kind", ["chain", "step"])
@pytest.mark.parametrize("multiple", [None, 3])
def test_conversion_selects_training_rows_without_mutating_result(
    kind, multiple, fake_verl, monkeypatch,
):
    import torch

    trajectory = make_trajectory("A", "G1", 6, 0.75, {"score": 2.0})
    # Keep original indices (1, 3, 4); never compact the canonical segment list.
    trajectory.segments[0] = Segment(messages=[{"role": "user", "content": "context"}])
    trajectory.segments[2] = Segment(messages=[])
    trajectory.segments[5] = Segment(messages=[{"role": "tool", "content": "tail"}])
    masked_trajectory = make_trajectory("B", "G2", 1, None, {})
    result = RunResult(rollout=kind, trajectories=[
        Trajectory(segments=[], chain_id="empty"),
        trajectory,
        masked_trajectory,
        # Matches the zero-generation step rollout's placeholder view as well.
        Trajectory(segments=[Segment(messages=[])], chain_id="context_only"),
    ])
    before = deepcopy(result.model_dump())
    step_lists = [t.steps for t in result]
    step_messages = [[deepcopy(s.messages.messages) for s in t.steps] for t in result]
    tool_results = [[deepcopy(s.tool_result) for s in t.steps] for t in result]
    candidates = [trajectory.segments[i].messages for i in (1, 3, 4)] + [
        masked_trajectory.segments[0].messages,
    ]
    seen = []
    tokenized = fake_inputs(4)
    # Assistant messages can have no targets after masking/truncation.
    tokenized["action_mask"][[1, 3]] = 0
    tokenized["reward_mask"][[1, 3]] = 0
    tokenized_before = {key: value.clone() for key, value in tokenized.items()}
    mm = [{"pixel_values": i} for i in range(4)]
    tokenized["mm_inputs"] = mm

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        return tokenized

    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    diagnostic_rows = []

    def drift(agent, conversations, inputs, **kwargs):
        diagnostic_rows.extend(conversations)
        return {"token_drift/n_sampled": float(len(conversations))}

    monkeypatch.setattr(conversion, "token_drift_stats", drift)
    agent = types.SimpleNamespace(tokenizer=None, processor=None)
    for _ in range(2):  # Repeated conversion must not accumulate filtering/padding.
        seen.clear()
        diagnostic_rows.clear()
        batch = BaseAgent.to_verl_dataproto(agent, result, pad_to_multiple_of=multiple)
        assert seen == candidates
        selected = [0, 2, 2] if multiple else [0, 2]
        n_rows = len(selected)
        for key, value in tokenized_before.items():
            expected = value[selected].clone()
            if key == "action_mask" and multiple:
                expected[-1] = 0
            torch.testing.assert_close(batch.inputs[key], expected, rtol=0, atol=0)
        assert batch.meta_info["repeat_times"] == [0, n_rows, 0, 0]
        assert batch.inputs["uid"].tolist() == ["G1"] * n_rows
        assert batch.inputs["rm_score"].tolist() == [2.0] * n_rows
        assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [0.75] * n_rows
        if kind == "chain":
            assert batch.inputs["batch_idx"].tolist() == [1] * n_rows
            assert batch.inputs["segment_idx"].tolist() == [1, 4] + ([4] if multiple else [])
            assert batch.inputs["step_rewards"].tolist() == [list(map(float, range(6)))] * n_rows
            assert batch.inputs["multi_modal_inputs"].tolist() == [mm[i] for i in selected]
            # The chain path runs the same token-drift diagnostic as the step
            # path, over the real (non-padding) training rows only.
            assert diagnostic_rows == [candidates[0], candidates[2]]
            assert batch.meta_info["token_drift"] == {"token_drift/n_sampled": 2.0}
        else:
            assert batch.inputs["step_env_reward"].tolist() == [1.0, 4.0] + ([4.0] if multiple else [])
            # Anchors come from the ORIGINAL step order, including skipped rows.
            assert batch.inputs["anchor_obs"].tolist() == ["A/state1", "A/state4"] + (
                ["A/state4"] if multiple else []
            )
            assert batch.inputs["active_masks"].tolist() == [1.0, 1.0] + ([0.0] if multiple else [])
            assert batch.inputs["mm_inputs"] == [mm[i] for i in selected]
            assert diagnostic_rows == [trajectory.segments[i].messages for i in (1, 4)]
            assert batch.meta_info["token_drift"]["token_drift/n_sampled"] == 2.0
        assert result.model_dump() == before
        for i, t in enumerate(result):
            assert t.steps is step_lists[i]
            assert [s.messages.messages for s in t.steps] == step_messages[i]
            assert [s.tool_result for s in t.steps] == tool_results[i]
        for key, value in tokenized_before.items():
            torch.testing.assert_close(tokenized[key], value, rtol=0, atol=0)
        assert tokenized["mm_inputs"] is mm and len(mm) == 4


@pytest.mark.parametrize("kind", ["chain", "step"])
@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "context"}]])
def test_conversion_rejects_context_only_batch_before_tokenization(kind, messages, fake_verl, monkeypatch):
    result = RunResult(rollout=kind, trajectories=[
        Trajectory(segments=[Segment(messages=messages)], reward=1.0),
    ])
    before = deepcopy(result.model_dump())

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty/context-only segments must not reach tokenization")

    monkeypatch.setattr(conversion, "tokenize_trajectories", forbidden)
    with pytest.raises(ValueError, match="No trainable segments"):
        BaseAgent.to_verl_dataproto(
            types.SimpleNamespace(tokenizer=None, processor=None), result,
        )
    assert result.model_dump() == before


@pytest.mark.parametrize("kind", ["chain", "step"])
def test_conversion_rejects_batch_with_no_action_tokens(kind, fake_verl, monkeypatch):
    result = RunResult(rollout=kind, trajectories=[make_trajectory("A", "G", 1, None, {})])
    inputs = fake_inputs(1)
    inputs["action_mask"][:] = 0
    monkeypatch.setattr(conversion, "tokenize_trajectories", lambda *a, **kw: inputs)
    # Selection happens before rewards: an unscored, fully masked row is not a sample.
    with pytest.raises(ValueError, match="No trainable segments.*action-mask"):
        BaseAgent.to_verl_dataproto(
            types.SimpleNamespace(tokenizer=None, processor=None), result,
        )


def test_chain_keeps_assistant_segment_ending_in_tool_observation(fake_verl, monkeypatch):
    result = RunResult(rollout="chain", trajectories=[make_trajectory("A", "G", 1, 1.0, {})])
    result[0].segments[0].messages.append({"role": "tool", "content": "final observation"})
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        return fake_inputs(len(messages_list))

    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    batch = BaseAgent.to_verl_dataproto(
        types.SimpleNamespace(tokenizer=None, processor=None), result,
    )
    # First call is the training tokenization; the token-drift diagnostic may
    # re-tokenize a token_ids-stripped copy afterwards.
    assert seen[0] == result[0].segments[0].messages
    assert batch.meta_info["repeat_times"] == [1]


def test_real_tokenization_skips_context_and_truncated_responses(fake_verl):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    template = "Qwen/Qwen2.5-1.5B-Instruct"
    agent = types.SimpleNamespace(
        tokenizer=tok, processor=None, template=template, max_model_len=64,
        prompt_tools=lambda: [],
    )
    segments = [
        Segment(messages=[]),
        Segment(messages=[{"role": "user", "content": "context only"}]),
        Segment(messages=[
            {"role": "user", "content": "long prompt " * 100},
            {"role": "assistant", "content": "truncated response"},
        ]),
        Segment(messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]),
        Segment(messages=[{"role": "user", "content": "unused folded context"}]),
    ]
    result = RunResult(rollout="chain", trajectories=[
        Trajectory(segments=segments, reward=0.5, group_id="G"),
    ])
    before = deepcopy(result.model_dump())
    batch = BaseAgent.to_verl_dataproto(agent, result, pad_to_multiple_of=2)
    assert batch.inputs["segment_idx"].tolist() == [3, 3]
    assert batch.meta_info["repeat_times"] == [2]
    assert batch.inputs["action_mask"][0].sum().item() > 0
    assert batch.inputs["action_mask"][1].sum().item() == 0
    assert batch.inputs["rm_scores"].sum(dim=-1).tolist() == [0.5, 0.5]
    assert result.model_dump() == before
