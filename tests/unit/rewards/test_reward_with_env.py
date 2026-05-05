import pytest

from agentfly.core import Context
from agentfly.rewards import code_reward_test


@pytest.mark.asyncio(loop_scope="session")
async def test_code_reward_test():
    ctx = Context(rollout_id="test_code_reward")
    try:
        code = "print('Hello, World!')"
        result = await code_reward_test(prediction=code, context=ctx)
        assert result["reward"] == 1.0
        assert "Hello, World!" in result["output"]
    finally:
        await ctx.end_resource(scope="rollout")
