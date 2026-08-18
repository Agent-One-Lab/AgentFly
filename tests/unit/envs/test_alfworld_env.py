"""ALFWorld environment tests, driven through the resource runner.

ALFWorldEnv is a ``ContainerResource``: it is acquired via the runner (which
starts the backing container), reset to a task, then stepped. Each test gets a
unique ``resource_id`` so container names never collide, and ends the resource
in a ``finally`` so nothing is left running.
"""

import asyncio

import pytest

from agentfly.envs.alfworld_env import ALFWorldSpec


async def _start_env(local_runner, resource_id: str):
    """Start an ALFWorld env and reset it to the default task."""
    env = await local_runner.start_resource(ALFWorldSpec, resource_id=resource_id)
    await env.reset()
    return env


@pytest.mark.asyncio
async def test_alfworld_env_lifecycle(local_runner):
    """Start, reset, list admissible commands, and take one step."""
    env = await _start_env(local_runner, "test_alfworld_lifecycle")
    try:
        commands = await env.get_admissible_commands()
        assert isinstance(commands, list)

        obs, reward, done, info = await env.step("look")
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_alfworld_env_multiple_steps(local_runner):
    """Take several steps in one episode."""
    env = await _start_env(local_runner, "test_alfworld_multiple_steps")
    try:
        for action in ("look", "inventory", "go north"):
            obs, reward, done, info = await env.step(action)
            assert isinstance(obs, str)
            if done:
                break
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_alfworld_env_admissible_commands(local_runner):
    """Admissible commands are a (possibly empty) list of strings."""
    env = await _start_env(local_runner, "test_alfworld_admissible")
    try:
        commands = await env.get_admissible_commands()
        assert isinstance(commands, list)
        assert all(isinstance(cmd, str) for cmd in commands)
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_alfworld_env_get_info(local_runner):
    """get_info returns the normalized episode-info shape reward/tools rely on."""
    env = await _start_env(local_runner, "test_alfworld_get_info")
    try:
        info = await env.get_info()
        assert isinstance(info, dict)
        for key in ("task", "goal", "won", "lost", "admissible_commands_count"):
            assert key in info
        assert isinstance(info["task"], str)
        assert isinstance(info["goal"], str)
        assert isinstance(info["won"], bool)
        assert isinstance(info["lost"], bool)
        assert isinstance(info["admissible_commands_count"], int)

        # Info stays consistent after a step (goal carries over).
        await env.step("look")
        info_after = await env.get_info()
        assert info_after["goal"] == info["goal"]
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_alfworld_env_reset_after_steps(local_runner):
    """A second reset starts a fresh episode that can still be stepped."""
    env = await _start_env(local_runner, "test_alfworld_reset_after_steps")
    try:
        await env.step("look")
        await env.step("inventory")

        obs2, info2 = await env.reset()
        assert isinstance(obs2, str)
        assert isinstance(info2, dict)

        obs, reward, done, info = await env.step("look")
        assert isinstance(obs, str)
    finally:
        await local_runner.end_resource(env)


@pytest.mark.asyncio
async def test_alfworld_env_error_handling(local_runner):
    """A nonsense action still returns a well-formed step tuple."""
    env = await _start_env(local_runner, "test_alfworld_error_handling")
    try:
        obs, reward, done, info = await env.step("xyzabc123nonsense")
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    finally:
        await local_runner.end_resource(env)


N_REQUESTS = 3  # kept small for memory safety


@pytest.mark.asyncio
async def test_alfworld_env_concurrent_requests(local_runner):
    """Concurrent steps against a single env instance all complete."""
    env = await _start_env(local_runner, "test_alfworld_concurrent")
    try:
        async def run_action(i: int):
            action = ("look", "inventory", "help")[i % 3]
            obs, reward, done, info = await env.step(action)
            return i, obs, done

        results = await asyncio.gather(*(run_action(i) for i in range(N_REQUESTS)))
        for i, obs, done in results:
            assert isinstance(obs, str)
            assert isinstance(done, bool)
    finally:
        await local_runner.end_resource(env)
