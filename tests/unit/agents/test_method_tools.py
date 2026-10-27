"""``@tool`` on an agent method: each agent gets its own tool instance bound to itself."""

import pytest

from agentfly.agents import HFAgent
from agentfly.tools import calculator, tool
from agentfly.tools.tool_base import BaseTool


class CounterAgent(HFAgent):
    def __init__(self, **kwargs):
        self.count = 0
        # ``self.counter_increment`` is still the shared tool class here; BaseAgent binds it.
        super().__init__(tools=[self.counter_increment, calculator], **kwargs)

    @tool(name="counter_increment", description="Add to this agent's counter.")
    def counter_increment(self, by: int = 1):
        """
        Args:
            by (int): Amount to add.
        """
        self.count += by
        return str(self.count)


def make_agent():
    return CounterAgent(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        backend_config={"backend": "client", "base_url": "http://localhost:1/v1", "api_key": "EMPTY"},
        monitors=[],
    )


def test_method_tool_is_bound_to_its_agent():
    agent = make_agent()
    assert isinstance(agent.counter_increment, BaseTool)  # an instance, not the shared class
    assert agent.counter_increment.instance is agent
    assert agent.tools[0] is agent.counter_increment
    assert agent.counter_increment(by=2).observation == "2"
    assert agent.count == 2


def test_agents_of_one_class_keep_their_own_state():
    first, second = make_agent(), make_agent()
    assert first.counter_increment is not second.counter_increment
    first.counter_increment(by=5)
    second.counter_increment(by=1)
    assert (first.count, second.count) == (5, 1)


def test_plain_tools_are_left_untouched():
    agent = make_agent()
    assert agent.tools[1] is calculator
    assert agent.tool_names == ["counter_increment", "calculator"]


async def run_tool(agent, by):
    """Run the tool the way a rollout does: by name, through the agent."""
    from agentfly.core import Context

    tool_call = {"function": {"name": "counter_increment", "arguments": f'{{"by": {by}}}'}}
    return await agent.execute_tool_call(
        Context(rollout_id="test_method_tools"), tool_call, [], None, "chain", 0, True
    )


@pytest.mark.asyncio
async def test_rollout_tool_calls_reach_the_right_agent():
    # The global tool registry holds one entry per name; each agent must still run its own tool.
    first, second = make_agent(), make_agent()
    result = await run_tool(first, 4)
    await run_tool(second, 1)
    assert result.status == "success" and result.observation == "4"
    assert (first.count, second.count) == (4, 1)


def test_string_tools_and_rewards_get_a_clear_error():
    # Direct construction takes objects; names and import references go through AutoAgent.
    kwargs = dict(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        backend_config={"backend": "client", "base_url": "http://localhost:1/v1", "api_key": "EMPTY"},
        monitors=[],
    )
    with pytest.raises(TypeError, match="takes tool objects.*AutoAgent.from_config"):
        HFAgent(tools=["calculator"], **kwargs)
    with pytest.raises(TypeError, match="takes a reward object.*AutoAgent.from_config"):
        HFAgent(reward_fn="math_equal_reward", **kwargs)
