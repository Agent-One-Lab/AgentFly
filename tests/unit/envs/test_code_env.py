from agentfly.envs import PythonSandboxSpec
import asyncio
import pytest


@pytest.mark.asyncio
async def test_env_run(local_runner):
    spec = PythonSandboxSpec
    env = await local_runner.start_resource(spec)
    # env = PythonSandboxEnv()
    # await env.start()
    observation = await env.step("print('Hello, World!')")
    assert observation == "Hello, World!\n"
    await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_env_async_step(local_runner):
    spec = PythonSandboxSpec
    env = await local_runner.start_resource(spec)
    tasks = [env.step(f"print('{i}')") for i in range(10)]
    observations = await asyncio.gather(*tasks)
    assert observations == [f"{i}\n" for i in range(10)]
    await local_runner.end_resource(env)
