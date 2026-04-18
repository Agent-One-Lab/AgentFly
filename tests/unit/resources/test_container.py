from __future__ import annotations

import asyncio
import uuid

import pytest

from agentfly.resources import ContainerResourceSpec, ResourceEngine


@pytest.mark.asyncio
async def test_container_acquire_times_out_on_enroot_create():
    """Acquire should time out with a heavy image and tiny startup timeout."""
    pytest.importorskip("enroot", reason="enroot client required")

    rid = f"test_enroot_start_timeout_{uuid.uuid4().hex[:12]}"
    spec = ContainerResourceSpec(
        category="container",
        image="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
    )
    with pytest.raises(asyncio.TimeoutError):
        await ResourceEngine.acquire(
            id=rid,
            spec=spec,
            backend="local",
            timeout=2.0,
        )


@pytest.mark.asyncio
async def test_container_acquire_times_out_on_enroot_start():
    """Another timeout scenario with a heavy image and tiny startup timeout."""
    pytest.importorskip("enroot", reason="enroot client required")

    rid = f"test_enroot_start_only_{uuid.uuid4().hex[:12]}"
    spec = ContainerResourceSpec(
        category="container",
        image="pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime",
    )

    with pytest.raises(asyncio.TimeoutError):
        await ResourceEngine.acquire(
            id=rid,
            spec=spec,
            backend="local",
            timeout=2.0,
        )


@pytest.mark.asyncio
async def test_container_run_cmd_timeout():
    """Ensure that run_cmd enforces timeout and raises asyncio.TimeoutError."""
    spec = ContainerResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await ResourceEngine.acquire(
        id="test_timeout_0123",
        spec=spec,
        backend="local",
    )

    try:
        # Long sleep with short exec_run timeout should raise asyncio.TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await container.run_cmd("sleep 10", timeout=1)
    finally:
        await ResourceEngine.end(id="test_timeout_0123")



@pytest.mark.asyncio
async def test_container_run_complex_cmd():
    spec = ContainerResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await ResourceEngine.acquire(
        id="test_timeout_4567",
        spec=spec,
        backend="local",
    )

#     cmd = """# Modify GifImagePlugin.py to correctly map comment extension to 'info'
# sed -i "/^# Process comment extensions.*/a\
#             self.info['comment'] = self.info.get('comments', '').join(comment)\
#         if comment:\
#             self.info['comment'] = comment" /path/to/Pillow/src/PIL/GifImagePlugin.py"
# """
    cmd = """sed -i 's/addurls.extract(/addurls.extract(input_type='list', /' test_add_urls.py"""
    try:
        result = await container.run_cmd(cmd, timeout=1)
        print(result)
    finally:
        await ResourceEngine.end(id="test_timeout_4567")
