"""Resource ids: one rollout can hold different resources, and the same kind is shared."""

import pytest
import pytest_asyncio

from agentfly.core import Context
from agentfly.envs.env_base import BaseEnv
from agentfly.resources import LocalEnvResourceSpec, ResourceEngine


class EnvA(BaseEnv):
    name = "A"

    async def start(self):
        pass

    async def reset(self, **kwargs):
        pass

    async def step(self, action):
        return self.name

    async def aclose(self):
        pass

    @staticmethod
    async def acquire():
        return EnvA()


class EnvB(EnvA):
    name = "B"


SPEC_A = LocalEnvResourceSpec(env_cls_path=f"{__name__}.EnvA")
SPEC_B = LocalEnvResourceSpec(env_cls_path=f"{__name__}.EnvB")


@pytest_asyncio.fixture
async def ctx():
    context = Context(rollout_id="rid-test")
    yield context
    await context.end_resource(scope="rollout")
    await context.end_resource(scope="global")


@pytest.mark.asyncio
async def test_different_resources_in_one_rollout_are_different(ctx):
    a = await ctx.acquire_resource(spec=SPEC_A, scope="rollout")
    b = await ctx.acquire_resource(spec=SPEC_B, scope="rollout")
    assert a is not b
    assert await a.step("x") == "A" and await b.step("x") == "B"
    # The first kind keeps the rollout id (container names are unchanged); others get their own.
    assert a.resource_id == "rid-test"
    assert b.resource_id.startswith("rid-test-") and b.resource_id != a.resource_id


@pytest.mark.asyncio
async def test_same_kind_is_shared_in_one_rollout(ctx):
    first = await ctx.acquire_resource(spec=SPEC_A, scope="global")
    await ctx.acquire_resource(spec=SPEC_B, scope="global")
    again = await ctx.acquire_resource(spec=LocalEnvResourceSpec(env_cls_path=f"{__name__}.EnvA"), scope="global")
    assert again is first  # e.g. a tool and its reward


@pytest.mark.asyncio
async def test_explicit_id_reused_for_another_kind_is_an_error(ctx):
    await ctx.acquire_resource(spec=SPEC_A, id="shared-id", scope="rollout")
    with pytest.raises(ValueError, match="already acquired"):
        await ctx.acquire_resource(spec=SPEC_B, id="shared-id", scope="rollout")


@pytest.mark.asyncio
async def test_rollouts_do_not_share():
    c1, c2 = Context(rollout_id="rid-1"), Context(rollout_id="rid-2")
    try:
        a1 = await c1.acquire_resource(spec=SPEC_A, scope="rollout")
        a2 = await c2.acquire_resource(spec=SPEC_A, scope="rollout")
        assert a1 is not a2
    finally:
        await c1.end_resource(scope="rollout")
        await c2.end_resource(scope="rollout")
        await ResourceEngine.close()
