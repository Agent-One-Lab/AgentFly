import pytest
from agentfly.tools import tool


@pytest.mark.asyncio
async def test_args_validation():
    @tool(name="add", description="Adds two numbers.")
    async def add(a, b):
        return a + b

    result = await add(a=1, b=2, c=3)
    # A tool call now returns a typed ToolResult (not the legacy dict shape).
    assert result.name == "add"
    assert result.arguments == {"a": 1, "b": 2, "c": 3}
    assert result.observation == 'Invalid argument "c" for tool add.'
    assert result.status == "success"
    assert result.metrics == {}
