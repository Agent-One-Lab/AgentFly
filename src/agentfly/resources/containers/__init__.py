"""
Container-backed resources: local enroot (:class:`ContainerResource`), docker
(:class:`DockerContainer`), Daytona cloud sandboxes
(:class:`DaytonaContainer`) and Ray actors (:class:`RayContainerResource`).

The Daytona, Docker and Ray symbols are imported lazily via ``__getattr__``: the
``daytona`` SDK is an optional dependency, and ``docker`` and ``ray`` are slow to import,
so importing this package (which loading the built-in tools does) must not pull them in.
"""

from typing import Any

from .container_resource import ContainerResource
from .enroot_containers import (
    ENROOT_CONTAINER_PREFIX,
    clear_agentfly_containers,
    enroot_container_name,
)

# name -> submodule, imported on first access.
_LAZY_EXPORTS = {
    "DockerContainer": "docker_container",
    "start_docker_container": "docker_container",
    "DaytonaContainer": "daytona_container",
    "start_daytona_container": "daytona_container",
    "ensure_snapshot": "daytona_container",
    "reap_stale_sandboxes": "daytona_container",
    "RayContainerResource": "ray_container_resource",
    "RayEnrootContainerActor": "ray_container_resource",
    "create_ray_container_resource": "ray_container_resource",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache so __getattr__ isn't hit again for this name
    return value


__all__ = [
    "ContainerResource",
    "DockerContainer",
    "start_docker_container",
    "ENROOT_CONTAINER_PREFIX",
    "clear_agentfly_containers",
    "enroot_container_name",
    "DaytonaContainer",
    "start_daytona_container",
    "ensure_snapshot",
    "reap_stale_sandboxes",
    "RayContainerResource",
    "RayEnrootContainerActor",
    "create_ray_container_resource",
]
