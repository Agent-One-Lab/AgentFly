from agentfly.core import Context
from agentfly.resources import ResourceSpec
import pytest

@pytest.mark.asyncio
async def test_context_resource():
    context = Context(
        rollout_id="test_0123",
        group_id="group_0123",
        metadata=None
    )

    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await context.acquire_resource(
        id="id_0123",
        spec=spec,
        backend="local",
    )

    result = await container.run_cmd("echo 'hello, world!'")
    print(result)

    await context.release_resource(scope="rollout")
