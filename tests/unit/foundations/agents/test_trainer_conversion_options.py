"""Exercise the trainer's actual option routing without importing Ray/GPU workers.

Compile the small configuration helper and the adjacent run/conversion calls
from the trainer source, supplying fake worker plumbing. Both validation and
training must keep the legacy diagnostic setting out of rollout construction.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from agentfly.agents.rollout.registry import resolve_rollout


@pytest.fixture(scope="module")
def trainer_routing():
    path = Path(__file__).resolve().parents[4] / "verl/verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_agent_rollout_options"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    blocks = {}
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for i, stmt in enumerate(value):
                if not (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "_agent_rollout_options"
                ):
                    continue
                block = value[i:i + 3]
                assert len(block) == 3
                assert isinstance(block[1], ast.Assign) and block[1].targets[0].id == "run_result"
                assert isinstance(block[2], ast.Assign)
                assert block[2].value.func.attr == "to_verl_dataproto"
                target = block[2].targets[0].id
                blocks[target] = compile(ast.Module(body=block, type_ignores=[]), str(path), "exec")
    assert set(blocks) == {"test_output_gen_batch_padded", "gen_batch_output"}
    return namespace["_agent_rollout_options"], blocks


@pytest.mark.parametrize("target", ["test_output_gen_batch_padded", "gen_batch_output"])
@pytest.mark.parametrize("raw_options,expected_drift", [
    ("missing", True),
    (None, True),
    ({}, True),
    ({"history_length": 2, "log_token_drift": False}, False),
    ({"history_length": 3, "log_token_drift": True}, True),
    ({"log_token_drift": None}, None),
])
def test_trainer_routes_diagnostic_only_to_conversion(
    target, raw_options, expected_drift, trainer_routing,
):
    split_options, blocks = trainer_routing
    run_config = {
        "rollout": "step", "max_turns": 3, "num_chains": 2,
        "max_concurrent_chains": 4, "generation_config": {"max_tokens": 16},
        "context_config": {},
    }
    if raw_options != "missing":
        run_config["rollout_config"] = raw_options
    config = OmegaConf.create({
        "agent": {"run_config": run_config, "train_on_last_turn": False},
        "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 4}},
    })
    before = OmegaConf.to_container(config, resolve=True)
    OmegaConf.set_readonly(config, True)
    result, batch = object(), object()
    calls = []

    def run(**kwargs):
        assert "log_token_drift" not in kwargs["rollout_config"]
        # Real construction would fail if the removed option leaked through.
        resolve_rollout(kwargs["rollout"], **kwargs["rollout_config"])
        calls.append(("run", kwargs))
        return result

    def convert(supplied, **kwargs):
        assert supplied is result
        assert kwargs["log_token_drift"] is expected_drift
        calls.append(("convert", kwargs))
        return batch

    trainer = SimpleNamespace(
        config=config,
        agent_wrapper=SimpleNamespace(run=run, to_verl_dataproto=convert),
        actor_rollout_wg=SimpleNamespace(world_size=2),
        run_on_bg=lambda value: value,
    )
    trainer._agent_rollout_options = lambda: split_options(trainer)
    input_batch = SimpleNamespace(non_tensor_batch={"messages": ["task"]})
    namespace = {
        "self": trainer,
        "generation_config": {"temperature": 0.0},
        "test_gen_batch_padded": input_batch,
        "gen_batch": input_batch,
    }
    exec(blocks[target], namespace)
    assert namespace[target] is batch
    assert [name for name, _ in calls] == ["run", "convert"]
    expected_options = {} if raw_options in ("missing", None) else {
        key: value for key, value in raw_options.items() if key != "log_token_drift"
    }
    assert calls[0][1]["rollout_config"] == expected_options
    assert OmegaConf.to_container(config, resolve=True) == before
