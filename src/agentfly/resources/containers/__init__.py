"""
Container-backed resources: local enroot (:class:`ContainerResource`) and Ray actors (:class:`RayContainerResource`).
"""

from .container_resource import ContainerResource
from .ray_container_resource import (
    RayContainerResource,
    RayEnrootContainerActor,
    create_ray_container_resource,
)

__all__ = [
    "ContainerResource",
    "RayContainerResource",
    "RayEnrootContainerActor",
    "create_ray_container_resource",
]
