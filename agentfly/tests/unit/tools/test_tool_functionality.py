import pytest
from ....tools import tool


@pytest.mark.asyncio
async def test_args_validation():
    @tool(name="add", description="Adds two numbers.")
    async def add(a, b):
        return a + b
    
    result = await add(a=1, b=2, c=3)
    assert result == {"name": "add", "arguments": {"a": 1, "b": 2, "c": 3}, "observation": "Invalid argument \"c\" for tool add.", "status": "success", "info": {}}
    