"""Tests for the rollout-strategy resolver (the composition seam).

``BaseAgent`` no longer inherits the rollout; ``run(rollout=...)`` resolves a
strategy per call via :func:`resolve_rollout` — a ``Rollout`` instance passes
through, a registered name (``"chain"``) or an import path (``"pkg.mod:Cls"``)
is resolved and default-constructed. This mirrors the name-or-path idiom used for
tools/rewards/agents.
"""
from types import SimpleNamespace
from functools import partial

import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.types import RunResult, Segment, Trajectory

from agentfly.agents.rollout.base import Rollout
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.rollout.registry import (
    ROLLOUT_REGISTRY,
    register_rollout,
    resolve_rollout,
)


def test_registry_contains_chain():
    assert ROLLOUT_REGISTRY["chain"] is ChainRollout


def test_resolve_by_registered_name():
    r = resolve_rollout("chain")
    assert isinstance(r, ChainRollout)
    assert isinstance(r, Rollout)


def test_resolve_instance_passes_through():
    inst = ChainRollout()
    assert resolve_rollout(inst) is inst  # a configured strategy is used as-is


def test_resolve_by_import_path():
    r = resolve_rollout("agentfly.agents.rollout.strategies.chain_rollout:ChainRollout")
    assert isinstance(r, ChainRollout)


def test_resolve_unknown_raises():
    with pytest.raises((KeyError, ValueError, ImportError)):
        resolve_rollout("definitely_not_a_registered_rollout")


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [None, "inference_only"])
async def test_register_and_run_custom_rollout_without_training_methods(marker, monkeypatch):
    class MyRollout(Rollout):
        async def run(self, agent, messages, max_turns, **kwargs):
            return RunResult(
                rollout=marker,
                trajectories=[Trajectory(segments=[Segment(messages=messages)])],
            )

    # Restore the registry after the test, including across parameterized cases.
    monkeypatch.setitem(ROLLOUT_REGISTRY, "myrollout_unittest", None)
    register_rollout("myrollout_unittest", MyRollout)
    assert isinstance(resolve_rollout("myrollout_unittest"), MyRollout)
    assert not hasattr(MyRollout, "to_dataproto")

    agent = SimpleNamespace(
        _preprocess_messages=lambda messages: messages,
        _preprocess_backends=lambda: None,
        _postprocess_backends=lambda: None,
        postprocess_trajectories=lambda trajectories: trajectories,
    )
    messages = [{"role": "assistant", "content": "inference result"}]
    result = await BaseAgent.run(
        agent, messages=messages, max_turns=1, rollout="myrollout_unittest",
    )
    assert result.rollout == marker
    assert result[0].segments[0].messages == messages
    # Registering an execution strategy does not register a training converter.
    with pytest.raises(ValueError, match="RunResult.rollout"):
        BaseAgent.to_verl_dataproto(agent, result)


def test_chain_rollout_is_a_rollout_subclass():
    # composition contract: the concrete strategy implements the base interface
    assert issubclass(ChainRollout, Rollout)
    assert Rollout.run.__isabstractmethod__
    assert Rollout.__abstractmethods__ == {"run"}
    with pytest.raises(TypeError, match="abstract"):
        Rollout()


def test_chain_rollout_has_no_unused_timer():
    rollout = ChainRollout()
    assert not hasattr(rollout, "timer")
    assert not hasattr(rollout, "timing_data")


@pytest.mark.parametrize("rollout_cls", [Rollout, ChainRollout, StepRollout])
def test_rollout_interface_has_no_training_wrappers(rollout_cls):
    for name in ("to_dataproto", "tokenize_trajectories", "_token_drift_stats"):
        assert not hasattr(rollout_cls, name)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("extra_options", [{}, {"history_lenght": 99}])
def test_step_rejects_constructor_diagnostic_option(enabled, extra_options):
    with pytest.raises(TypeError, match="agent.to_verl_dataproto"):
        StepRollout(log_token_drift=enabled, **extra_options)
    with pytest.raises(TypeError, match="conversion option"):
        resolve_rollout("step", log_token_drift=enabled, **extra_options)
    assert not hasattr(StepRollout(), "log_token_drift")


@pytest.mark.parametrize("construct", [StepRollout, partial(resolve_rollout, "step")],
                         ids=["direct", "registry"])
@pytest.mark.parametrize("unknown_options", [
    {"history_lenght": 99},
    {"max_promt_length": None},
    {"history_lenght": 99, "extra_config": "private-value"},
])
def test_step_rejects_unknown_constructor_options(construct, unknown_options):
    with pytest.raises(TypeError) as error:
        construct(history_length=3, **unknown_options)

    message = str(error.value)
    assert "StepRollout" in message
    assert "unexpected" in message.lower()
    for name in unknown_options:
        assert repr(name) in message
    assert "private-value" not in message


@pytest.mark.parametrize("construct", [StepRollout, partial(resolve_rollout, "step")],
                         ids=["direct", "registry"])
def test_step_accepts_supported_constructor_options(construct):
    rollout = construct(
        history_length=4,
        prompt_builder="chat_window",
        max_prompt_length=1234,
        on_no_tool_call="continue",
        on_invalid_tool_call="raise",
        on_tool_error="end",
        project_actions=True,
    )

    assert rollout.history_length == 4
    assert rollout.prompt_builder.max_prompt_length == 1234
    assert rollout.on_no_tool_call == "continue"
    assert rollout.on_invalid_tool_call == "raise"
    assert rollout.on_tool_error == "end"
    assert rollout.project_actions is True


def test_inference_only_run_does_not_load_training_dependencies():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import asyncio
        import sys
        from types import SimpleNamespace
        from agentfly.agents.agent_base import BaseAgent
        from agentfly.agents.rollout.base import Rollout
        from agentfly.agents.types import RunResult

        class InferenceOnly(Rollout):
            async def run(self, **kwargs):
                return RunResult(trajectories=[], rollout="inference_only")

        agent = SimpleNamespace(
            _preprocess_messages=lambda messages: messages,
            _preprocess_backends=lambda: None,
            _postprocess_backends=lambda: None,
            postprocess_trajectories=lambda trajectories: trajectories,
        )
        result = asyncio.run(BaseAgent.run(
            agent, messages=[], max_turns=1, rollout=InferenceOnly(),
        ))
        assert result.rollout == "inference_only"
        for module in ("torch", "transformers", "agentfly.verl.protocol"):
            assert module not in sys.modules, module
    """)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
