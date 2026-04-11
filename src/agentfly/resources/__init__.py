"""
Resource engine: decoupled design for managing resources (containers, vLLM)
across backends (local, Ray, AWS, K8s).

Tools and rewards use the engine to acquire, use, and release resources.
"""

from .types import (
    BaseResource,
    ResourceSpec,
    ResourceStatus,
)
from .runner import (
    BaseRunner,
    LocalRunner,
    RayRunner,
    CloudRunner,
    K8sRunner,
)
from .containers import (
    ContainerResource,
    RayContainerResource,
    RayEnrootContainerActor,
    create_ray_container_resource,
)
from .engine import ResourceEngine

__all__ = [
    "BaseResource",
    "ResourceSpec",
    "ResourceStatus",
    "BaseRunner",
    "LocalRunner",
    "RayRunner",
    "CloudRunner",
    "K8sRunner",
    "ContainerResource",
    "ResourceEngine",
    "RayContainerResource",
    "RayEnrootContainerActor",
    "create_ray_container_resource",
]
