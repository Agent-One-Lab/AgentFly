"""Derived agents share the engine; declared subagents are built on demand.

``derive`` copies an agent and re-applies its own settings (prompt, tools, length, …) with some
replaced; the copy keeps the engine, tokenizer, processor and template. ``subagents`` declares
settings only, and ``create_subagent`` builds a new agent from them on each call.
"""

import pytest

import agentfly.agents.agent_base as agent_base
from agentfly.agents import ActionAgent, AutoAgent, HFAgent
from agentfly.agents.agent_base import BaseAgent
from agentfly.tools import calculator, tool
from agentfly.tools.src.scienceworld.tools import scienceworld_explorer


@tool(name="derive_test_return", description="Return to the parent.", auto_register=False)
def return_tool(message: str):
    return "Returned."


class Counts:
    def __init__(self):
        self.engines = 0
        self.tokenizers = 0
        self.processors = 0


@pytest.fixture
def stubbed(monkeypatch):
    """Build agents without models or a backend, counting what gets built."""
    counts = Counts()
    tokenizer, engine = object(), object()

    def make_tokenizer(model):
        counts.tokenizers += 1
        return tokenizer

    def make_processor(model):
        counts.processors += 1
        return None

    def make_engine(self, *args):
        counts.engines += 1
        return engine

    monkeypatch.setattr(agent_base, "create_tokenizer", make_tokenizer)
    monkeypatch.setattr(agent_base, "create_processor", make_processor)
    monkeypatch.setattr(agent_base, "get_jinja_template", lambda template: "JINJA")
    monkeypatch.setattr(agent_base.BaseAgent, "_init_llm_engine", make_engine)
    return counts


def make_main(**kwargs):
    kwargs.setdefault("subagents", {
        "explorer": dict(system_prompt="SUB", tools=[scienceworld_explorer, return_tool], max_model_len=16384),
    })
    return ActionAgent(
        "test-model", tools=[calculator], system_prompt="MAIN", max_model_len=4096,
        backend_config={"backend": "client"}, monitors=[], **kwargs,
    )


def test_derived_agent_shares_the_engine_and_builds_nothing(stubbed, monkeypatch):
    main = make_main()
    built = (stubbed.engines, stubbed.tokenizers, stubbed.processors)

    def no_sink(*args, **kwargs):
        pytest.fail("derive must not add a logging sink")

    monkeypatch.setattr(agent_base.Monitor, "add_sink", no_sink)
    child = main.derive(system_prompt="CHILD", tools=[scienceworld_explorer])
    assert (stubbed.engines, stubbed.tokenizers, stubbed.processors) == built
    assert child is not main and type(child) is ActionAgent
    for name in ("llm_engine", "tokenizer", "processor", "template", "jinja_template", "backend", "backend_config"):
        assert getattr(child, name) is getattr(main, name), name


def test_derived_agent_has_its_own_settings_and_the_parent_is_unchanged(stubbed):
    main = make_main()
    child = main.derive(system_prompt="CHILD\\nLINE", tools=[scienceworld_explorer, return_tool], max_model_len=16384)

    assert child.system_prompt == "CHILD\nLINE"
    assert child.tool_names == ["scienceworld_explorer", "derive_test_return"]
    assert child.max_model_len == 16384
    assert child.action_tool_name == "scienceworld_explorer"
    assert child.render_tools_in_prompt is False  # kept from the parent (ActionAgent's default)
    assert [t["function"]["name"] for t in child.prompt_tools() or []] == []

    assert main.system_prompt == "MAIN"
    assert main.tool_names == ["calculator"]
    assert main.max_model_len == 4096
    assert main.action_tool_name == "calculator"


def test_action_agent_settings_can_be_replaced(stubbed, monkeypatch):
    monkeypatch.delenv("CONTEXT_TRIGGER_TURNS", raising=False)
    main = make_main(context_trigger_turns=3, truncate_history_at_action=True)
    child = main.derive(context_trigger_turns=None, truncate_history_at_action=False)
    assert (child.context_trigger_turns, child.truncate_history_at_action) == (None, False)
    assert (main.context_trigger_turns, main.truncate_history_at_action) == (3, True)
    with pytest.raises(ValueError, match="context_trigger_turns must be >= 1"):
        main.derive(context_trigger_turns=0)


class CounterAgent(HFAgent):
    def __init__(self, **kwargs):
        self.count = 0
        super().__init__(tools=[self.counter_increment], **kwargs)

    @tool(name="derive_test_counter", description="Add to this agent's counter.", auto_register=False)
    def counter_increment(self, by: int = 1):
        """
        Args:
            by (int): Amount to add.
        """
        self.count += by
        return str(self.count)


def test_method_tools_of_a_derived_agent_are_bound_to_it(stubbed):
    main = CounterAgent(model_name_or_path="test-model", backend_config={"backend": "client"}, monitors=[])
    child = main.derive()
    assert child.counter_increment is not main.counter_increment
    assert child.counter_increment.instance is child
    assert child.tools[0] is child.counter_increment
    assert main.counter_increment.instance is main


def test_derive_rejects_unknown_settings_and_drops_subagents(stubbed):
    main = make_main()
    with pytest.raises(ValueError, match="Unused arguments"):
        main.derive(model_name_or_path="other")
    assert main.derive().subagents == {}
    assert main.derive(subagents={"helper": {"max_model_len": 8}}).subagents == {"helper": {"max_model_len": 8}}


def test_subagents_are_built_on_demand(stubbed, monkeypatch):
    calls = []
    original = BaseAgent.derive

    def counting_derive(self, **settings):
        calls.append(settings)
        return original(self, **settings)

    monkeypatch.setattr(BaseAgent, "derive", counting_derive)
    main = make_main()
    assert calls == []  # declared, nothing built

    first = main.create_subagent("explorer")
    second = main.create_subagent("explorer")
    assert len(calls) == 2 and first is not second
    assert first.system_prompt == "SUB" and first.max_model_len == 16384
    assert first.tool_names == ["scienceworld_explorer", "derive_test_return"]
    assert first.llm_engine is main.llm_engine
    assert first.subagents == {}

    with pytest.raises(ValueError, match=r"No subagent named 'planner'.*\['explorer'\]"):
        main.create_subagent("planner")


def test_a_bad_declaration_fails_when_the_main_agent_is_built(stubbed):
    with pytest.raises(ValueError, match=r"Subagent 'explorer' has unknown settings \['model'\]"):
        make_main(subagents={"explorer": {"model": "x"}})
    with pytest.raises(TypeError, match="takes tool objects"):
        make_main(subagents={"explorer": {"tools": ["scienceworld_explorer"]}})
    assert stubbed.engines == 0  # checked before anything is built


def test_from_config_resolves_tool_names_in_declarations(stubbed):
    agent = AutoAgent.from_config({
        "agent_type": "action",
        "model_name_or_path": "test-model",
        "backend_config": {"backend": "client"},
        "monitors": [],
        "tools": ["calculator"],
        "subagents": {"explorer": {"tools": ["scienceworld_explorer"], "max_model_len": 16384}},
    })
    declared = agent.subagents["explorer"]
    assert [t.name for t in declared["tools"]] == ["scienceworld_explorer"]  # tool objects, not names
    assert agent.create_subagent("explorer").tool_names == ["scienceworld_explorer"]
