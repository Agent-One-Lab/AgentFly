"""SWE-bench result plumbing and export tests without model execution."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.utils import swebench


@pytest.fixture
def run_result():
    return RunResult(
        rollout="chain",
        trajectories=[
            Trajectory(
                segments=[
                    Segment(messages=[
                        {"role": "user", "content": "Fix café parsing."},
                        {"role": "assistant", "content": "Fixed.", "token_ids": [3, 7]},
                    ]),
                    # Export preserves unused context views; it is not training conversion.
                    Segment(messages=[{"role": "user", "content": "Folded context."}]),
                    Segment(messages=[]),
                ],
                reward=1.0,
                metrics={"resolved": True, "output": {"patch": "+fixed"}},
                metadata={"instance_id": "org/repo:1", "task_id": 17},
                chain_id="chain-1",
                group_id="group-1",
                chain_idx=0,
                group_idx=0,
                finish_reason="terminal",
                rollout_time_sec=1.5,
                runtime_info={"attempt": 1},
                steps=[object()],
            ),
            Trajectory(
                segments=[Segment(messages=[{"role": "assistant", "content": "Failed."}])],
                reward=0.0,
                metrics={"resolved": False},
                metadata={"uid": "fallback-id"},
            ),
            Trajectory(segments=[], reward=0.5),
        ],
    )


@pytest.mark.parametrize("rollout", [None, "chain", "step"])
def test_save_results_preserves_export_contract(tmp_path, run_result, rollout):
    run_result.rollout = rollout
    before = run_result.model_dump()
    original_step = run_result[0].steps[0]
    result_dir = tmp_path / "results"

    assert swebench._save_per_sample_results(result_dir, run_result) == (0.5, 3)

    filenames = ["org_repo_1.json", "fallback-id.json", "sample_00002.json"]
    instance_ids = ["org/repo:1", "fallback-id", None]
    expected_extras = [
        {"output": {"patch": "+fixed"}, "resolved": True},
        {"output": None, "resolved": False},
        {"output": None, "resolved": None},
    ]
    for index, filename in enumerate(filenames):
        raw = (result_dir / filename).read_text(encoding="utf-8")
        trajectory = run_result[index]
        assert raw.endswith("\n")
        assert json.loads(raw) == {
            "index": index,
            "instance_id": instance_ids[index],
            "reward": trajectory.reward,
            "reward_full": {"reward": trajectory.reward, "extras": trajectory.metrics},
            "reward_extras": expected_extras[index],
            "trajectory": before["trajectories"][index],
        }
    assert "café" in (result_dir / filenames[0]).read_text(encoding="utf-8")
    assert json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8")) == {
        "num_samples": 3,
        "mean_reward": 0.5,
        "accuracy": 0.5,
        "per_sample": [
            {"file": filename, "index": index, "instance_id": instance_ids[index],
             "reward": run_result[index].reward}
            for index, filename in enumerate(filenames)
        ],
    }
    assert {path.name for path in result_dir.iterdir()} == {*filenames, "run_summary.json"}
    assert run_result.model_dump() == before
    assert run_result[0].steps[0] is original_step


def test_save_empty_result(tmp_path):
    result_dir = tmp_path / "empty"

    assert swebench._save_per_sample_results(result_dir, RunResult(trajectories=[])) == (0.0, 0)

    assert [path.name for path in result_dir.iterdir()] == ["run_summary.json"]
    assert json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8")) == {
        "num_samples": 0,
        "mean_reward": 0.0,
        "accuracy": 0.0,
        "per_sample": [],
    }


@pytest.mark.parametrize("agent_kind,tools_mode", [("bash", "bash"), ("qwen3_coder", "file")])
def test_run_helper_returns_exact_result(monkeypatch, run_result, agent_kind, tools_mode):
    # Neither agent has cached results or a _require_last_run accessor.
    agent = SimpleNamespace(run=AsyncMock(return_value=run_result))
    bash_constructor = Mock(return_value=agent)
    coder_constructor = Mock(return_value=agent)
    monkeypatch.setattr(swebench, "BashSWEAgent", bash_constructor)
    monkeypatch.setattr(swebench, "Qwen3CoderSWEAgent", coder_constructor)
    reward_fn = object()
    reward_lookup = Mock(return_value=reward_fn)
    monkeypatch.setattr(swebench, "get_reward_from_name", reward_lookup)
    messages = [{"messages": [{"role": "user", "content": "Fix this."}], "instance_id": "a"}]

    result = asyncio.run(swebench._run_agent_async(
        agent_kind=agent_kind,
        model_name_or_path="test-model",
        template=None,
        max_model_len=4096,
        tools_mode=tools_mode,
        reward_name="test-reward",
        backend="client",
        vllm_base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        tensor_parallel_size=1,
        data_parallel_size=1,
        messages=messages,
        max_turns=5,
        num_chains=2,
        max_concurrent_chains=3,
        temperature=0.25,
        resource_backend="local",
    ))

    assert result is run_result
    reward_lookup.assert_called_once_with("test-reward")
    if agent_kind == "bash":
        constructor, unused = bash_constructor, coder_constructor
        prompt = swebench.InstructionSystemPrompt
        tools = [swebench.run_shell_command]
    else:
        constructor, unused = coder_constructor, bash_constructor
        prompt = swebench.Qwen3CoderToolPrompt
        tools = swebench._file_tool_list()
    constructor.assert_called_once_with(
        system_prompt=prompt,
        model_name_or_path="test-model",
        template=None,
        max_model_len=4096,
        tools=tools,
        backend_config={"backend": "client", "base_url": "http://localhost:8000/v1", "api_key": "EMPTY"},
        reward_fn=reward_fn,
        monitors=[],
    )
    unused.assert_not_called()
    agent.run.assert_awaited_once()
    context_config = agent.run.await_args.kwargs["context_config"]
    assert context_config.resource_backend == "local"
    agent.run.assert_awaited_once_with(
        messages=messages,
        max_turns=5,
        num_chains=2,
        generation_config={"temperature": 0.25},
        max_concurrent_chains=3,
        context_config=context_config,
    )


@pytest.mark.parametrize("sample_count", [0, 3])
def test_cli_exports_returned_result(monkeypatch, tmp_path, run_result, sample_count):
    result = RunResult(trajectories=run_result.trajectories[:sample_count], rollout="chain")
    run = AsyncMock(return_value=result)
    save = Mock(wraps=swebench._save_per_sample_results)
    monkeypatch.setattr(swebench, "_run_agent_async", run)
    monkeypatch.setattr(swebench, "_save_per_sample_results", save)
    # Let monkeypatch restore the environment variable changed by the CLI.
    monkeypatch.setenv("ENROOT_ASYNC", "0")
    data_path = tmp_path / "data.json"
    rows = [{"instance_id": str(i), "problem_statement": "Fix this."} for i in range(sample_count)]
    data_path.write_text(json.dumps(rows), encoding="utf-8")
    result_dir = tmp_path / "results"

    invocation = CliRunner().invoke(swebench.main, [
        "--data-path", str(data_path),
        "--result-dir", str(result_dir),
        "--model-name-or-path", "test-model",
    ])

    assert invocation.exit_code == 0, invocation.output
    run.assert_awaited_once()
    assert run.await_args.kwargs["messages"] == [
        {"instance_id": str(i), "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Fix this."}]},
        ]}
        for i in range(sample_count)
    ]
    save.assert_called_once_with(result_dir, result)
    assert save.call_args.args[1] is result
    assert f"Wrote {sample_count} sample(s) under {result_dir.resolve()} (see run_summary.json)." in invocation.output
    expected_accuracy = (
        "Final accuracy: 0.5000 (50.00%, n=3)" if sample_count else "Final accuracy: n/a (no samples)"
    )
    assert expected_accuracy in invocation.output
    summary = json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == sample_count
    assert summary["accuracy"] == (0.5 if sample_count else 0.0)
