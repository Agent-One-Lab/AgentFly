"""Unit tests for alfworld_reward.

The reward reads the terminal ``won`` flag from the rollout env, coerces it to
{0.0, 1.0}, and returns ``10 * won`` as the reward (verl-agent scale) plus ``won``
as ``trajectory/accuracy``. We stub the env/context so the coercion logic is
tested without standing up the ALFWorld Docker environment.
"""
import pytest

from agentfly.rewards import alfworld_reward


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
    "info, won",
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
async def test_alfworld_reward_binary(info, won):
    ctx = _FakeContext(info)
    result = await alfworld_reward(context=ctx)

    assert result == {"reward": 10.0 * won, "trajectory/accuracy": won}
    # Values must be plain floats (not numpy scalars / lists).
    assert isinstance(result["reward"], float)
    assert isinstance(result["trajectory/accuracy"], float)


@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_reward_uses_global_local_resource():
    ctx = _FakeContext({"won": True})
    await alfworld_reward(context=ctx)

    # Reward must read the shared rollout env (same one the tools use).
    assert len(ctx.acquire_calls) == 1
    call = ctx.acquire_calls[0]
    assert call["scope"] == "global"
    assert call["backend"] == "local"
