import pytest

from agentfly.core import Context
from agentfly.tools import (
    alfworld_step,
    alfworld_get_admissible_commands,
    alfworld_get_task_objective,
    alfworld_reset,
)

@pytest.mark.skip(reason="Skipping for now")
@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_reset():
    ctx = Context(rollout_id="test_alfworld_reset")
    try:
        result = await alfworld_reset(context=ctx)
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        await ctx.release_resource(scope="rollout")


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.skip(reason="Skipping for now")
async def test_alfworld_get_objective():
    ctx = Context(rollout_id="test_alfworld_objective")
    try:
        await alfworld_reset(context=ctx)
        result = await alfworld_get_task_objective(context=ctx)
        assert isinstance(result, str)
        assert "Task:" in result
    finally:
        await ctx.release_resource(scope="rollout")


