from __future__ import annotations

import os

import pytest

from agentfly.core import Context
from agentfly.resources import RayContainerResource, ResourceSpec


def _ensure_ray_driver_connected() -> None:
    """Attach this pytest process to the Ray cluster (``RAY_ADDRESS`` or ``auto``)."""
    import ray

    if ray.is_initialized():
        return
    address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=address, ignore_reinit_error=True)
    except Exception as e:
        pytest.skip(
            "Ray driver not connected; set RAY_ADDRESS or call ray.init() first. "
            f"({e})"
        )
    if not ray.is_initialized():
        pytest.skip("Ray.init did not connect; check RAY_ADDRESS / cluster reachability.")


@pytest.mark.asyncio
async def test_context_resource_ray_backend():
    """Same flow as ``test_context_resource`` using ``ResourceEngine`` backend ``ray``."""
    pytest.importorskip("ray", reason="ray required for ray backend")
    _ensure_ray_driver_connected()

    context = Context(
        rollout_id="test_ray_ctx_0123",
        group_id="group_ray_ctx_0123",
        metadata=None,
    )

    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await context.acquire_resource(
        id="id_ray_ctx_0123",
        spec=spec,
        backend="ray",
    )

    assert isinstance(container, RayContainerResource)

    cmd = "echo 'print(\"hello, world!\")' > test_add_urls.py"
    result = await container.run_cmd(cmd)
    print(result)
    cmd = "sed -i 's/addurls\\\\.extract(/addurls.extract(input_type=\\\\'list\\\\', /' test_add_urls.py"
    result = await container.run_cmd(cmd)
    print(result)
    result = await container.run_cmd("cat test_add_urls.py")
    print(result)
    assert "hello, world!" in result

    await context.end_resource(scope="rollout")


@pytest.mark.asyncio
async def test_context_resource_ray_backend_via_context_config():
    """``resource_backend=ray`` on :class:`ContextConfig` when ``acquire_resource`` omits ``backend``."""
    pytest.importorskip("ray", reason="ray required for ray backend")
    _ensure_ray_driver_connected()

    context = Context(
        rollout_id="test_ray_ctx_cfg_0123",
        group_id="group_ray_ctx_cfg_0123",
        metadata=None,
        context_config={"resource_backend": "ray"},
    )

    spec = ResourceSpec(
        category="container",
        image="ubuntu:22.04",
    )

    container = await context.acquire_resource(
        id="id_ray_ctx_cfg_0123",
        spec=spec,
    )

    assert isinstance(container, RayContainerResource)
    result = await container.run_cmd("echo via_context_config", timeout=60)
    print(result)
    assert "via_context_config" in result

    await context.end_resource(scope="rollout")