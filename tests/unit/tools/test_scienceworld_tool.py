import pytest

from agentfly.core import Context
from agentfly.tools import scienceworld_explorer


@pytest.mark.asyncio
async def test_science_world_explorer():
    ctx = Context(rollout_id="test_scienceworld_explorer")
    result = await scienceworld_explorer(
        action="look around", context=ctx
    )
    assert result['observation'].startswith("This room is called")
    print(result['observation'])
    
    await ctx.release_resource(scope="rollout")
