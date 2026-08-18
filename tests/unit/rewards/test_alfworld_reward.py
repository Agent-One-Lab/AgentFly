"""Unit tests for alfworld_episode_reward.

The reward reads the terminal ``won`` flag from the rollout env and coerces it
to a binary {0.0, 1.0} float. We stub the env/context so the coercion logic is
tested without standing up the ALFWorld Docker environment.
"""
import pytest

from agentfly.rewards import alfworld_episode_reward


class _FakeEnv:
    def __init__(self, info):
        self._info = info

    async def get_info(self):
        return self._info


class _FakeContext:
    """Minimal Context stand-in: hands back a fixed env from acquire_resource."""

    def __init__(self, info):
        self._env = _FakeEnv(info)
        self.acquire_calls = []

    async def acquire_resource(self, spec=None, scope=None, backend=None):
        self.acquire_calls.append({"spec": spec, "scope": scope, "backend": backend})
        return self._env


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"won": True}, 1.0),
        ({"won": False}, 0.0),
        ({"won": 1}, 1.0),      # int truthy
        ({"won": 0}, 0.0),      # int falsy
        ({}, 0.0),              # missing key -> 0
        (None, 0.0),            # no info at all -> 0
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_reward_binary(info, expected):
    ctx = _FakeContext(info)
    result = await alfworld_episode_reward(context=ctx)

    assert result == {"reward": expected}
    # reward must be a plain float (not a numpy scalar / list), and exactly 0/1.
    assert isinstance(result["reward"], float)
    assert result["reward"] in (0.0, 1.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_reward_uses_global_local_resource():
    ctx = _FakeContext({"won": True})
    await alfworld_episode_reward(context=ctx)

    # Reward must read the shared rollout env (same one the tools use).
    assert len(ctx.acquire_calls) == 1
    call = ctx.acquire_calls[0]
    assert call["scope"] == "global"
    assert call["backend"] == "local"
