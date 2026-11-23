"""ActionAgent ``strip_reasoning`` with the action-agent-qwen35-think template: tool tags are parsed only after the
native reasoning (``</think>``), the stored content leaves the reasoning out, and the rollout drops the sampled ids
of such turns so the next prompt is rendered from the stored text."""
import json
import types

import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout.loop.generation import REASONING_STRIPPED


def _tool():
    return types.SimpleNamespace(name="alfworld_step", schema={"type": "function", "function": {"name": "alfworld_step"}})


@pytest.fixture
def make_agent(monkeypatch):
    import agentfly.agents.agent_base as agent_base
    from agentfly.agents.specialized import action_agent as aa

    monkeypatch.setattr(agent_base, "create_tokenizer", lambda model: None)
    monkeypatch.setattr(agent_base, "create_processor", lambda model: None)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: None)
    monkeypatch.setattr(BaseAgent, "_init_llm_engine", lambda *args: None)

    def _make(**kwargs):
        return aa.ActionAgent("dummy", tools=[_tool()], backend_config={"backend": "client"}, monitors=[], **kwargs)

    return _make


# What the model writes after a generation prompt that ends with "<think>\n".
RAW = "Maybe <action>look</action> first? No, the desk.\n</think>\n\nGo to the desk.\n<action>go to desk 1</action>"


def test_tags_inside_reasoning_are_not_run(make_agent):
    msg = make_agent(strip_reasoning=True)._parse_single_response(RAW, current_segment=[])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"action": "go to desk 1"}
    assert msg["content"][0]["text"] == "Go to the desk.\n<action>go to desk 1</action>"
    assert msg[REASONING_STRIPPED] is True


def test_unclosed_reasoning_holds_no_action(make_agent):
    msg = make_agent(strip_reasoning=True)._parse_single_response("Thinking <action>look</action> still", current_segment=[])
    assert msg["tool_calls"] == [] and msg["content"][0]["text"] == ""


def test_off_by_default(make_agent):
    agent = make_agent()
    msg = agent._parse_single_response(RAW, current_segment=[])
    assert agent.strip_reasoning is False and REASONING_STRIPPED not in msg
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"action": "look"}


def test_with_truncation_the_cut_applies_to_the_answer(make_agent):
    agent = make_agent(strip_reasoning=True, truncate_history_at_action=True)
    msg = agent._parse_single_response(RAW + "\nextra", current_segment=[])
    assert msg["content"][0]["text"] == "Go to the desk.\n<action>go to desk 1</action>"
    assert "content_is_prefix" not in msg


def test_think_template_prompts():
    pytest.importorskip("chat_bricks")
    from chat_bricks import Chat

    import agentfly.templates  # noqa: F401

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<action>look</action>"}, {"role": "user", "content": "<observation>o</observation>"}]
    prompt = Chat("action-agent-qwen35-think", messages).prompt(add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n<action>look</action><|im_end|>" in prompt
