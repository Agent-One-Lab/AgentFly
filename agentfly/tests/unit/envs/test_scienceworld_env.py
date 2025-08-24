import pytest
from ....envs.scienceworld_env import ScienceWorldEnv
from ast import literal_eval

    
@pytest.mark.asyncio
async def test_env_start_and_close():
    env = ScienceWorldEnv()
    await env.start()
    assert env._client is not None
    await env.reset()


@pytest.mark.asyncio
async def test_env_reset():
    env = ScienceWorldEnv()
    await env.start()
    await env.reset()
    assert env.score == 0

@pytest.mark.asyncio
async def test_observation_is_deterministic():
    env = ScienceWorldEnv()
    await env.start()
    await env.reset()
    obs_orig = await env.step('look around')

    for _ in range(15):
        await env.reset()
        obs = await env.step('look around')
        assert obs == obs_orig

@pytest.mark.asyncio
async def test_multiple_instances():
    env1 = ScienceWorldEnv()
    env2 = ScienceWorldEnv()
    await env1.start()
    await env2.start()
    await env1.reset()
    await env2.reset()

    obs1 = await env1.step('look around')
    obs2 = await env2.step('look around')
    assert obs1 == obs2

    obs1 = await env1.step('open door to art studio')
    obs1_1 = await env1.step('look around')
    obs2_1 = await env2.step('look around')
    assert obs1_1 != obs2_1

    await env2.reset()
    obs1_2 = await env1.step('look around')
    obs2_2 = await env2.step('look around')
    assert obs1_1 == obs1_2
