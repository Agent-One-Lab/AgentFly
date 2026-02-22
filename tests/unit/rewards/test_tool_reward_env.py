import pytest

from agentfly.core import Context
from agentfly.envs.python_env import PythonSandboxSpec
from agentfly.tools import tool
from agentfly.rewards import reward


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_reward_env():
    ctx = Context(rollout_id="test_tool_reward_python")
    try:
        @tool(name="test_python_tool")
        async def test_tool(code: str, context: Context):
            env = await context.acquire_resource(
                spec=PythonSandboxSpec, scope="rollout", backend="local"
            )
            return await env.step(code)

        @reward(name="test_python_reward")
        async def test_reward(prediction, context: Context):
            env = await context.acquire_resource(
                spec=PythonSandboxSpec, scope="rollout", backend="local"
            )
            result = await env.step(prediction)
            return {"reward": 1.0, "result": result}

        result = await test_tool(
            code="import os; os.environ['TEST'] = 'test'", context=ctx
        )
        assert result is not None

        result = await test_reward(
            prediction="import os; print(os.environ.get('TEST', ''))",
            context=ctx,
        )
        assert result["reward"] == 1.0
        assert "test" in result["result"]
    finally:
        await ctx.release_resource(scope="rollout")
