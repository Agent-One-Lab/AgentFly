"""ActionAgent's generic tags: ``<name attr="v">body</name>`` calls the agent's tool ``name``.

The body fills the tool's one required text argument and attributes fill the arguments they
name, cast by the tool's schema. ``spawn`` / ``ask`` / ``return`` are short names for the
subagent tools. ``<action>`` goes to the first tool only when it takes an ``action`` argument
(or declares no parameters).
"""

import json
import types

import pytest

from agentfly.agents.agent_base import BaseAgent


def tool(name, properties=None, required=None):
    parameters = {"type": "object", "properties": properties or {}, "required": required or []}
    return types.SimpleNamespace(
        name=name, schema={"type": "function", "function": {"name": name, "parameters": parameters}}
    )


ENV = tool("scienceworld_explorer", {"action": {"type": "string"}}, ["action"])
SPAWN = tool("spawn_subagent", {"brief": {"type": "string"}}, ["brief"])
ASK = tool("ask_subagent", {"child": {"type": "integer"}, "message": {"type": "string"}}, ["child", "message"])
RETURN = tool("return_to_parent", {"message": {"type": "string"}}, ["message"])
FINISH = tool("finish", {"summary": {"type": "string"}}, ["summary"])


@pytest.fixture
def make_agent(monkeypatch):
    import agentfly.agents.agent_base as agent_base
    from agentfly.agents.specialized import action_agent as aa

    # Run the real constructor without loading a model or starting a backend.
    monkeypatch.setattr(agent_base, "create_tokenizer", lambda model: None)
    monkeypatch.setattr(agent_base, "create_processor", lambda model: None)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: None)
    monkeypatch.setattr(BaseAgent, "_init_llm_engine", lambda *args: None)

    def _make(tools):
        return aa.ActionAgent("dummy", tools=tools, backend_config={"backend": "client"}, monitors=[])

    return _make


def calls(message):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in message["tool_calls"]]


def parse(agent, text):
    return agent._parse_single_response(text, current_segment=[])


def test_coordinator_tags_with_aliases_and_a_cast_attribute(make_agent):
    coordinator = make_agent([SPAWN, ASK, FINISH])
    assert coordinator.action_tool_name is None
    assert calls(parse(coordinator, "<think>plan</think><spawn>find the thermometer</spawn>")) == [
        ("spawn_subagent", {"brief": "find the thermometer"})
    ]
    assert calls(parse(coordinator, '<ask child="2">what did you see?</ask>')) == [
        ("ask_subagent", {"child": 2, "message": "what did you see?"})
    ]
    message = parse(coordinator, "<finish>done</finish>")
    assert calls(message) == [("finish", {"summary": "done"})] and message["status"] == "continue"


def test_full_tool_names_work_too(make_agent):
    coordinator = make_agent([SPAWN, ASK, FINISH])
    assert calls(parse(coordinator, "<spawn_subagent>go</spawn_subagent>")) == [("spawn_subagent", {"brief": "go"})]


def test_subagent_action_and_return(make_agent):
    subagent = make_agent([ENV, RETURN])
    assert subagent.action_tool_name == "scienceworld_explorer"
    assert calls(parse(subagent, "<action>look around</action>")) == [
        ("scienceworld_explorer", {"action": "look around"})
    ]
    assert calls(parse(subagent, "<return>the kitchen has a stove</return>")) == [
        ("return_to_parent", {"message": "the kitchen has a stove"})
    ]


def test_action_without_an_action_tool_is_an_invalid_call(make_agent):
    message = parse(make_agent([SPAWN, ASK, FINISH]), "<action>look around</action>")
    assert message["tool_calls"] == [] and message["status"] == "terminal"


def test_tags_of_tools_the_agent_lacks_are_not_calls(make_agent):
    subagent = make_agent([ENV, RETURN])  # no spawn tool: nesting depth 1
    message = parse(subagent, "<spawn>more help</spawn>")
    assert message["tool_calls"] == [] and message["status"] == "terminal"


def test_a_tag_inside_think_is_found_and_the_first_tag_wins(make_agent):
    subagent = make_agent([ENV, RETURN])
    text = "<think>I should <action>open door</action> first</think><return>later</return>"
    assert calls(parse(subagent, text)) == [("scienceworld_explorer", {"action": "open door"})]


def test_tag_names_ignore_case(make_agent):
    assert calls(parse(make_agent([ENV, RETURN]), "<Return>ok</Return>")) == [("return_to_parent", {"message": "ok"})]


def test_a_body_with_no_argument_to_fill_is_an_invalid_call(make_agent):
    # Both required arguments of ask_subagent are text: the body cannot pick one.
    two_texts = tool("ask_subagent", {"child": {"type": "string"}, "message": {"type": "string"}}, ["child", "message"])
    message = parse(make_agent([SPAWN, two_texts]), "<ask>hello</ask>")
    assert message["tool_calls"] == []


def test_history_keeps_the_full_text_and_parsing_stops_at_the_first_tool_tag(make_agent):
    coordinator = make_agent([SPAWN, ASK, FINISH])
    text = "<spawn>go north</spawn> then <finish>x</finish>"
    message = parse(coordinator, text)
    assert message["content"][0]["text"] == text
    assert calls(message) == [("spawn_subagent", {"brief": "go north"})]
