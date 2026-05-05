import pytest

from agentfly.core import Context
from agentfly.rewards import scienceworld_reward


@pytest.mark.asyncio(loop_scope="session")
async def test_scienceworld_reward():
    ctx = Context(rollout_id="test_scienceworld_reward")
    
    prediction = "Task not completed"
    result = await scienceworld_reward(final_response=prediction, context=ctx)
    assert "reward" in result
    assert result["reward"] >= 0.0

    await ctx.end_resource(scope="rollout")
