from agentfly.tools import tool
import pytest


def test_tool_call_error():
    @tool(name="test_tool", description="test tool")
    def test_tool(arg1: str, arg2: int):
        return "test"

    # Don't raise error
    result = test_tool(arg1="test")
    print(result)


@pytest.mark.asyncio
async def test_tool_call_error_async():
    @tool(name="test_tool", description="test tool")
    async def test_tool(arg1: str, arg2: int):
        return "test"

    result = await test_tool(arg1="test")
    print(result)
