"""Shared rollout helpers + ChainRollout(project_actions): the chain arm uses the same
always-step projection and the same FORMAT validity as StepRollout, so step-vs-chain
comparisons don't confound rollout type with termination or penalty semantics."""
import json
import types

from agentfly.agents.rollout.base import Rollout
from agentfly.agents.rollout.registry import resolve_rollout
from agentfly.agents.rollout.strategies.chain_rollout import ChainRollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout


def _agent():
    return types.SimpleNamespace(action_tool_name="alfworld_step", tools=[types.SimpleNamespace(name="alfworld_step")])


def test_format_validity_is_shared_and_identical():
    ok = "<think>t</think><action>go to desk 1</action>"
    assert Rollout._action_format_valid(ok) == 1.0
    assert StepRollout._action_format_valid(ok) == 1.0
    assert ChainRollout._action_format_valid(ok) == 1.0
    assert Rollout._action_format_valid("<action>go</action>") == 0.0          # no <think>
    assert Rollout._action_format_valid("<think>t</think> go") == 0.0          # no <action>
    assert Rollout._action_format_valid("<think>t</think><action>去</action>") == 0.0  # Chinese
    assert Rollout._action_format_valid("") == 0.0


def test_projected_tool_call_is_last_30_chars_lowercased():
    msg = {"role": "assistant", "content": [{"type": "text", "text": "I Think I Should Just Wander Around The Room For A While"}]}
    call = Rollout._projected_tool_call(_agent(), msg)
    assert call["function"]["name"] == "alfworld_step"
    assert json.loads(call["function"]["arguments"]) == {"action": "wander around the room for a while"[-30:]}
    assert call["id"] == "call_projected"


def test_projected_tool_call_none_without_action_tool():
    agent = types.SimpleNamespace(action_tool_name=None, tools=[])
    assert Rollout._projected_tool_call(agent, {"role": "assistant", "content": "x"}) is None


def test_chain_rollout_accepts_project_actions_via_registry():
    r = resolve_rollout("chain", project_actions=True)
    assert isinstance(r, ChainRollout) and r.project_actions is True
    assert resolve_rollout("chain").project_actions is False       # default unchanged
    assert resolve_rollout("step", project_actions=True).project_actions is True
