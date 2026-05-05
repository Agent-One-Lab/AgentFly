import pytest
from agentfly.envs import ScienceWorldSpec


@pytest.mark.asyncio
async def test_env_start_and_close(local_runner):
    env = await local_runner.start_resource(ScienceWorldSpec)
    try:
        assert env._client is not None
        await env.reset()
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_env_reset(local_runner):
    env = await local_runner.start_resource(ScienceWorldSpec)
    try:
        await env.reset()
        assert env.score == 0
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_observation_is_deterministic(local_runner):
    env = await local_runner.start_resource(ScienceWorldSpec)
    try:
        await env.reset()
        obs_orig = await env.step("look around")

        for _ in range(15):
            await env.reset()
            obs = await env.step("look around")
            assert obs == obs_orig
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_multiple_instances(local_runner):
    env1 = await local_runner.start_resource(ScienceWorldSpec)
    env2 = await local_runner.start_resource(ScienceWorldSpec)
    try:
        await env1.reset()
        await env2.reset()

        obs1 = await env1.step("look around")
        obs2 = await env2.step("look around")
        assert obs1 == obs2

        await env1.step("open door to art studio")
        obs1_1 = await env1.step("look around")
        obs2_1 = await env2.step("look around")
        assert obs1_1 != obs2_1

        await env2.reset()
        obs1_2 = await env1.step("look around")
        obs2_2 = await env2.step("look around")
        assert obs1_1 == obs1_2
    finally:
        await local_runner.end_resource(env2)
        await local_runner.end_resource(env1)
