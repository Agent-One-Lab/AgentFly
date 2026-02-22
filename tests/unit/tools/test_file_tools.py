from agentfly.tools import grep_search, list_files
from agentfly.core import Context
import pytest

@pytest.mark.asyncio
async def test_tool_run():
    context = Context(
        rollout_id="test_0123",
        group_id="group_0123",
        metadata={
            "image_id": "swebench/swesmith.x86_64.andialbrecht_1776_sqlparse.e57923b3",
        }
    )

    result = await list_files(path="/", context=context)

    print(result)
