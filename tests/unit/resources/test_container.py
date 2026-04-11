from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentfly.resources import ResourceEngine, ResourceSpec


@pytest.mark.asyncio
async def test_container_acquire_times_out_on_enroot_create():
    """Acquire should raise asyncio.TimeoutError when ``containers.create`` hits enroot's deadline.

    This patches the first blocking enroot call (container create) to raise
    ``enroot.errors.TimeoutError``, which the runner maps to ``asyncio.TimeoutError``.

    For an integration check without mocks, use a very short ``timeout`` (e.g. a few
    seconds) with a large image when it is not already cached (e.g.
    ``pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime``); cold pulls often exceed the
    budget and trigger the same error path.
    """
    pytest.importorskip("enroot", reason="enroot client required")
    from enroot.errors import TimeoutError as EnrootTimeoutError

    import agentfly.resources.runner as runner_mod

    orig_to_thread = asyncio.to_thread
    n_calls = 0

    async def to_thread_side_effect(fn, *args, **kwargs):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            raise EnrootTimeoutError("simulated create timeout")
        return await orig_to_thread(fn, *args, **kwargs)

    rid = f"test_enroot_start_timeout_{uuid.uuid4().hex[:12]}"
    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    with patch.object(runner_mod.asyncio, "to_thread", side_effect=to_thread_side_effect):
        with pytest.raises(asyncio.TimeoutError):
            await ResourceEngine.acquire(
                id=rid,
                spec=spec,
                backend="local",
                timeout=600.0,
            )


@pytest.mark.asyncio
async def test_container_acquire_times_out_on_enroot_start():
    """Same as create timeout, but the failure happens on ``container.start`` (second blocking call)."""
    pytest.importorskip("enroot", reason="enroot client required")
    from enroot.errors import TimeoutError as EnrootTimeoutError

    import agentfly.resources.runner as runner_mod

    orig_to_thread = asyncio.to_thread
    n_calls = 0
    fake_container = MagicMock()
    fake_container.name = "fake_start_timeout_container"

    async def to_thread_side_effect(fn, *args, **kwargs):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 1:
            return fake_container
        if n_calls == 2:
            raise EnrootTimeoutError("simulated start timeout")
        return await orig_to_thread(fn, *args, **kwargs)

    rid = f"test_enroot_start_only_{uuid.uuid4().hex[:12]}"
    spec = ResourceSpec(category="container", image="ubuntu:22.04")

    with patch.object(runner_mod.asyncio, "to_thread", side_effect=to_thread_side_effect):
        with pytest.raises(asyncio.TimeoutError):
            await ResourceEngine.acquire(
                id=rid,
                spec=spec,
                backend="local",
                timeout=600.0,
            )


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
        # Long sleep with short exec_run timeout should raise asyncio.TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await container.run_cmd("sleep 10", timeout=1)
    finally:
        await ResourceEngine.release(id="test_timeout_0123")



@pytest.mark.asyncio
async def test_container_run_complex_cmd():
    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await ResourceEngine.acquire(
        id="test_timeout_0123",
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
        await ResourceEngine.release(id="test_timeout_0123")
