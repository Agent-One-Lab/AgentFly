"""
Container-backed resources: local enroot (:class:`ContainerResource`), docker
(:class:`DockerContainer`), Daytona cloud sandboxes
(:class:`DaytonaContainer`) and Ray actors (:class:`RayContainerResource`).

The Daytona symbols are imported lazily via ``__getattr__`` because the
``daytona`` SDK is an optional dependency — importing this package must not
require it.
"""

from typing import Any

from .container_resource import ContainerResource
from .docker_container import DockerContainer, start_docker_container
from .ray_container_resource import (
    RayContainerResource,
    RayEnrootContainerActor,
    create_ray_container_resource,
)

_DAYTONA_EXPORTS = {
    "DaytonaContainer",
    "start_daytona_container",
    "ensure_snapshot",
    "reap_stale_sandboxes",
}


def __getattr__(name: str) -> Any:
    if name in _DAYTONA_EXPORTS:
        from . import daytona_container
        return getattr(daytona_container, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContainerResource",
    "DockerContainer",
    "start_docker_container",
    "DaytonaContainer",
    "start_daytona_container",
    "ensure_snapshot",
    "reap_stale_sandboxes",
    "RayContainerResource",
    "RayEnrootContainerActor",
    "create_ray_container_resource",
]
