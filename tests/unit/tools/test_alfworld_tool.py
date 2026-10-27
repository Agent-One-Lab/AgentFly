import pytest

from agentfly.core import Context
from agentfly.tools import alfworld_step
from agentfly.tools.types import ToolResult


@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_step():
    ctx = Context(rollout_id="test_alfworld_step")
    try:
        result = await alfworld_step(action="look", context=ctx)
        # The decorated tool normalizes its dict return into a ToolResult.
        assert isinstance(result, ToolResult)
        assert "Admissible actions:" in result.observation
        # ``look`` does not resolve the task, so the env build's broken ``done`` must NOT
        # end the episode: control stays None (see alfworld_step: end on won/lost, not done).
        assert result.control is None
    finally:
        # alfworld_step acquires the env with scope="global", so end that scope.
        await ctx.end_resource(scope="global")
