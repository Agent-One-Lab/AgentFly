from __future__ import annotations

import asyncio

import pytest

from agentfly.resources import ResourceEngine, ResourceSpec


@pytest.mark.asyncio
async def test_container_run_cmd_timeout():
    """Ensure that run_cmd enforces timeout and raises asyncio.TimeoutError."""
    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await ResourceEngine.acquire(
        id="test_timeout_0123",
        spec=spec,
        backend="local",
    )

    try:
        # Use a long sleep with a short timeout to trigger the shell `timeout`
        with pytest.raises(asyncio.TimeoutError):
            await container.run_cmd("sleep 10", timeout=1)
    finally:
        await ResourceEngine.release(id="test_timeout_0123")

