import pytest

from agents.rewards.reward_base import reward
from agents.tools.tool_base import tool
from agents.envs.webshop_text_env import WebAgentTextEnv

@pytest.mark.asyncio()
async def test_tool_reward_env():
    @tool(env_cls=WebAgentTextEnv, name="test_tool", pool_size=4)
    async def test_tool(code: str, env: WebAgentTextEnv):
        result = await env.step('search[protein]')
        result = await env.step('click[B079HGJ5MH]')
        result = await env.step('click[Buy Now]')
        return result
    

    @reward(env_cls=WebAgentTextEnv, name="test_reward", pool_size=4)
    async def test_reward(prediction, env: WebAgentTextEnv):
        result = await env.step('get_reward')

        return {
            "reward": 1,
            "result": qresult
        }
    
    result = await test_tool(code="random", id="test_0")
    print(result)

    result = await test_reward(prediction="random", id="test_0")
    print(result)
