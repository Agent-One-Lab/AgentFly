import pytest

from agentfly.core import Context
from agentfly.rewards import webshop_reward


@pytest.mark.asyncio(loop_scope="session")
async def test_webshop_reward():
    ctx = Context(rollout_id="test_webshop_reward")
    try:
        result = await webshop_reward(task_id=0, context=ctx)
        assert "reward" in result
        assert result["reward"] >= 0.0
    finally:
        await ctx.end_resource(scope="global")
