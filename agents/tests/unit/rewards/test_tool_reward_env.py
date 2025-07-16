import pytest

from agents.envs.manager.env_manager import EnvironmentManager
from agents.envs import PythonSandboxEnv
from agents.tools import tool
from agents.rewards import reward


@pytest.mark.asyncio()
async def test_tool_reward_env():
    @tool(env_cls=PythonSandboxEnv, name="test_tool", pool_size=4)
    async def test_tool(code: str, env: PythonSandboxEnv):
        result = await env.step(code)
        return result
    
    @reward(env_cls=PythonSandboxEnv, name="test_reward", pool_size=4)
    async def test_reward(prediction, env: PythonSandboxEnv):
        result = await env.step(prediction)
        return {
            "reward": 1,
            "result": result
        }
    
    result = await test_tool(code="import os; os.environ['TEST'] = 'test'", id="test_0")
    print(result)
    
    result = await test_reward(prediction="import os; print(os.environ['TEST'])", id="test_0")
    print(result)
    