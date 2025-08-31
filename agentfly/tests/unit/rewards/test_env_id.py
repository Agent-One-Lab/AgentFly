import pytest

from ....rewards.reward_base import reward
from ....tools.tool_base import tool
from ....envs.webshop_text_env import WebAgentTextEnv

@pytest.mark.asyncio(loop_scope="session")
async def test_tool_reward_env():
    @tool(env_cls=WebAgentTextEnv, name="test_tool", pool_size=4)
    async def test_tool(prediction: str, env: WebAgentTextEnv):
        result = await env.step('search[protein]')
        result = await env.step('click[B079HGJ5MH]')
        result = await env.step('click[Buy Now]')
        return result
    

    @reward(env_cls=WebAgentTextEnv, name="test_reward", pool_size=4)
    async def test_reward(prediction, env: WebAgentTextEnv):
        result = await env.step('get_reward', task_id=0)

        return {
            "reward": 1,
            "result": result
        }
    
    result = await test_tool(prediction="random", id="test_0")
    print(result)

    result = await test_reward(prediction="random", id="test_0")
    print(result)
