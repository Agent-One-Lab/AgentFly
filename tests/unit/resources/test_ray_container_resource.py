from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("enroot", reason="enroot client required")
pytest.importorskip("ray", reason="ray required for RayContainerResource")


def _ensure_ray_driver_connected() -> None:
    """
    Connect *this* Python process to the Ray cluster if needed.

    ``ray status`` only shows that a cluster exists; the pytest driver must still
    call ``ray.init``. Uses ``RAY_ADDRESS`` when set, otherwise ``address='auto'``.
    """
    import ray

    if ray.is_initialized():
        return
    address = os.environ.get("RAY_ADDRESS", "auto")
    try:
        ray.init(address=address, ignore_reinit_error=True)
    except Exception as e:
        pytest.skip(
            "This process is not a Ray driver. A running cluster (e.g. `ray status`) does not "
            f"connect pytest. Set RAY_ADDRESS=<head_ip>:6379 or call ray.init(address=...) first. ({e})"
        )
    if not ray.is_initialized():
        pytest.skip(
            "Ray.init did not connect. Set RAY_ADDRESS to the head (e.g. <ip>:6379) or pass "
            "the correct address to ray.init()."
        )


def test_ray_enroot_container_actor_is_remote_class():
    from agentfly.resources.containers.ray_container_resource import RayEnrootContainerActor

    assert RayEnrootContainerActor is not None


@pytest.mark.asyncio
async def test_ray_containers_run_on_distinct_nodes_when_ray_cluster_has_multiple_nodes():
    """
    Pin two :class:`RayEnrootContainerActor` instances with hard node affinity and
    assert each reports the expected Ray node id.

    Connects the driver via :func:`_ensure_ray_driver_connected` if needed
    (``RAY_ADDRESS`` or ``ray.init(address='auto')``).

    Requires:
    - A reachable Ray cluster (often set ``RAY_ADDRESS=<head_ip>:6379``).
    - At least two alive Ray nodes with distinct ``NodeID``s.
    - Enroot and image ``ubuntu:22.04`` available on those workers.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from agentfly.resources import ContainerResourceSpec, create_ray_container_resource

    _ensure_ray_driver_connected()

    alive = [n for n in ray.nodes() if n.get("Alive")]
    node_ids: list[str] = []
    for n in alive:
        nid = n.get("NodeID")
        if nid and nid not in node_ids:
            node_ids.append(nid)
    if len(node_ids) < 2:
        pytest.skip(
            f"Need at least 2 Ray nodes with distinct NodeIDs for this test; found {len(node_ids)}"
        )

    spec = ContainerResourceSpec(category="container", image="ubuntu:22.04")
    r1 = None
    r2 = None
    try:
        r1 = await create_ray_container_resource(
            spec,
            f"ray_node_test_a_{uuid.uuid4().hex[:8]}",
            start_timeout=600.0,
            ray_actor_options={
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=node_ids[0],
                    soft=False,
                ),
            },
        )
        r2 = await create_ray_container_resource(
            spec,
            f"ray_node_test_b_{uuid.uuid4().hex[:8]}",
            start_timeout=600.0,
            ray_actor_options={
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=node_ids[1],
                    soft=False,
                ),
            },
        )

        n1 = await r1.get_ray_node_id()
        n2 = await r2.get_ray_node_id()
        assert n1 == node_ids[0]
        assert n2 == node_ids[1]
        assert n1 != n2

        out1 = (await r1.run_cmd("echo node_one", timeout=60)).strip()
        out2 = (await r2.run_cmd("echo node_two", timeout=60)).strip()
        assert out1 == "node_one"
        assert out2 == "node_two"
    finally:
        if r1 is not None:
            await r1.end()
        if r2 is not None:
            await r2.end()
