import pytest

from agentfly.core import Context
from agentfly.tools import alfworld_step


@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_step():
    ctx = Context(rollout_id="test_alfworld_step")
    try:
        result = await alfworld_step(action="look", context=ctx)
        assert isinstance(result, dict)
        assert "observation" in result
        assert "Admissible actions:" in result["observation"]
    finally:
        # alfworld_step acquires the env with scope="global", so end that scope.
        await ctx.end_resource(scope="global")
