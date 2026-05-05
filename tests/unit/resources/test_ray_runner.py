from __future__ import annotations

from agentfly.resources import RayRunner, ResourceEngine


def test_resource_engine_registers_ray_backend():
    assert "ray" in ResourceEngine._runners
    assert ResourceEngine._runners["ray"].name == "ray"


def test_ray_runner_name():
    assert RayRunner().name == "ray"
