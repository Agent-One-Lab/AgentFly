from agentfly.resources import ContainerResourceSpec
import pytest

@pytest.mark.asyncio
async def test_file_env_mount(local_runner):
    spec = ContainerResourceSpec(
        category="container",
        image="ubuntu:22.04",
        mount={"src/agentfly/tools/src/file": "/root:ro,rbind"}
    )
    env = await local_runner.start_resource(spec)
    output = await env.run_cmd("cd /root && ls -l && head -n 10 file_manager.py")
    print(output)
    await local_runner.end_resource(env)