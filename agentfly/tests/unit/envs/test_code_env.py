from ....envs.python_env import PythonSandboxEnv
import asyncio
import pytest

@pytest.mark.asyncio
async def test_env_run():
    env = PythonSandboxEnv()
    await env.start()
    observation = await env.step("print('Hello, World!')")
    assert observation == "Hello, World!\n"
    await env.aclose()

@pytest.mark.asyncio
async def test_env_async_step():
    env = PythonSandboxEnv()
    await env.start()
    tasks = [env.step(f"print('{i}')") for i in range(10)]
    observations = await asyncio.gather(*tasks)
    assert observations == [f"{i}\n" for i in range(10)]
    await env.aclose()

# @pytest.mark.asyncio
# async def test_env_keep_state():
#     env = PythonSandboxEnv()
#     await env.start()
#     code = """
# import os
# os.environ['TEST'] = 'test'
# """
#     observation = await env.step(code)
#     code = """
# import os
# print(os.environ['TEST'])
# """
#     observation = await env.step(code)
#     assert observation == 'test\n', f"Observation: {observation}"
#     await env.aclose()

