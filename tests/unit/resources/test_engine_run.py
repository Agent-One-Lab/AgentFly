from agentfly.resources import ResourceEngine, ResourceSpec
import pytest

@pytest.mark.asyncio
async def test_engine_run():
    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await ResourceEngine.acquire(
        id="test_0123",
        spec=spec,
        backend="local"
    )

    result = await container.run_cmd("echo 'hello, world!'")
    print(result)

        await ResourceEngine.release(id="test_0123")
    