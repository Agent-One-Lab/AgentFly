import pytest

from agentfly.core import Context
from agentfly.rewards import reward
from agentfly.tools import tool
from agentfly.envs.webshop_text_env import WebShopSpec


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_reward_env():
    ctx = Context(rollout_id="test_env_id_webshop")
    try:
        @tool(name="test_webshop_tool")
        async def test_tool(prediction: str, context: Context):
            env = await context.acquire_resource(
                spec=WebShopSpec, scope="rollout", backend="local"
            )
            await env.step("search[protein]")
            await env.step("click[B079HGJ5MH]")
            return await env.step("click[Buy Now]")

        @reward(name="test_webshop_reward")
        async def test_reward(prediction, context: Context):
            env = await context.acquire_resource(
                spec=WebShopSpec, scope="rollout", backend="local"
            )
            result = await env.step("get_reward", task_id=0)
            return {"reward": result.get("reward", 0), "result": result}

        result = await test_tool(prediction="random", context=ctx)
        assert result is not None

        result = await test_reward(prediction="random", context=ctx)
        assert "reward" in result
    finally:
        await ctx.release_resource(scope="rollout")
