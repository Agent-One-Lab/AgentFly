"""ActionAgent history content: full generated text by default (history == what was
sampled == what training splices), truncation at the first </action> as an opt-in
ablation. The tool call is parsed from the truncated copy either way."""
import json
import types

import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.types import Segment, Trajectory


def _tool():
    return types.SimpleNamespace(name="alfworld_step", schema={"type": "function", "function": {"name": "alfworld_step"}})


@pytest.fixture
def make_agent(monkeypatch):
    import agentfly.agents.agent_base as agent_base
    from agentfly.agents.specialized import action_agent as aa

    # Run the real constructor without loading a model or starting a backend.
    monkeypatch.setattr(agent_base, "create_tokenizer", lambda model: None)
    monkeypatch.setattr(agent_base, "create_processor", lambda model: None)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: None)
    monkeypatch.setattr(BaseAgent, "_init_llm_engine", lambda *args: None)

    def _make(**kwargs):
        return aa.ActionAgent("dummy", tools=[_tool()], backend_config={"backend": "client"}, monitors=[], **kwargs)

    return _make


RAW = "<think>find the lamp</think><action>go to desk 1</action>\nLet me also check the drawer."


def test_default_keeps_full_text_and_parses_first_action(make_agent):
    agent = make_agent()
    assert agent.truncate_history_at_action is False
    msg = agent._parse_single_response(RAW, current_segment=[])
    # history content is the FULL sample (trailing text included)
    assert msg["content"][0]["text"] == RAW
    # the tool call still comes from the first <action> block
    call = msg["tool_calls"][0]["function"]
    assert call["name"] == "alfworld_step"
    assert json.loads(call["arguments"]) == {"action": "go to desk 1"}
    assert msg["status"] == "continue"


def test_truncation_flag_cuts_history_at_first_close_tag(make_agent):
    agent = make_agent(truncate_history_at_action=True)
    msg = agent._parse_single_response(RAW, current_segment=[])
    assert msg["content"][0]["text"] == "<think>find the lamp</think><action>go to desk 1</action>"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"action": "go to desk 1"}


def test_second_action_after_first_is_ignored_but_kept_in_history(make_agent):
    agent = make_agent()
    raw = "<action>look</action> then <action>inventory</action>"
    msg = agent._parse_single_response(raw, current_segment=[])
    assert msg["content"][0]["text"] == raw                       # history keeps everything
    assert len(msg["tool_calls"]) == 1                            # parse: first action only
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"action": "look"}


def test_no_action_keeps_full_text_and_is_terminal(make_agent):
    agent = make_agent()
    raw = "<think>hmm</think> I am not sure what to do"
    msg = agent._parse_single_response(raw, current_segment=[])
    assert msg["content"][0]["text"] == raw
    assert msg["tool_calls"] == []
    assert msg["status"] == "terminal"


@pytest.mark.parametrize("tail", [[], [{"role": "user", "content": "unused context"}]])
def test_postprocessing_preserves_all_rollout_segments(make_agent, tail):
    segments = [
        Segment(messages=[{"role": "assistant", "content": RAW}]),
        Segment(messages=tail),
    ]
    trajectory = Trajectory(
        segments=segments, reward=0.75, metrics={"score": 1.0},
        metadata={"task_id": "task"}, steps=[object()],
    )
    trajectories = [trajectory]
    processed = make_agent().postprocess_trajectories(trajectories)
    assert processed is trajectories
    assert processed[0] is trajectory
    assert trajectory.num_segments == 2
    assert trajectory.segments[1].messages == tail
