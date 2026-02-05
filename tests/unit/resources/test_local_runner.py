"""
Unit tests for LocalRunner.

Tests start an Ubuntu container via LocalRunner, run a command inside it,
then close the container. Requires enroot and a suitable image (e.g. ubuntu:22.04).
"""

from __future__ import annotations

import pytest

from agentfly.resources.runner import LocalRunner
from agentfly.resources.types import ResourceCategory, ResourceSpec, ResourceStatus


@pytest.mark.asyncio(loop_scope="session")
async def test_local_runner_start_execute_close():
    """Start an Ubuntu container, execute a command in it, then close it."""
    runner = LocalRunner()
    spec = ResourceSpec(
        category=ResourceCategory.CONTAINER,
        image="ubuntu:22.04",
    )

    # Start container
    resource = await runner.start_resource(spec)
    assert resource is not None
    assert resource.resource_id
    assert resource.category == ResourceCategory.CONTAINER

    # Ensure resource is running
    status = await runner.get_status(resource)
    assert status == ResourceStatus.RUNNING

    # Execute a command in the container (ubuntu:22.04 has python3)
    output = await resource.run_cmd("echo 'hello from container'")
    print(output)

    # Close the container
    await runner.end_resource(resource)
    assert resource.resource_id not in runner._containers
