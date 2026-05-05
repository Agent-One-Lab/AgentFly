"""
Resource engine: decoupled design for managing resources (containers, vLLM)
across backends (local, Ray, AWS, K8s).

Tools and rewards use the engine to acquire, use, and release resources.
"""

from .types import (
    APIModelResourceSpec,
    BaseResource,
    BaseResourceSpec,
    ContainerCategory,
    ContainerResourceSpec,
    LocalEnvResourceSpec,
    ResourceStatus,
    VLLMModelResourceSpec,
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
from .local_env_resource import LocalEnvResource
from .models import APIModelResource, VLLMModelResource
from .engine import ResourceEngine

__all__ = [
    "BaseResource",
    "ResourceStatus",
    "BaseResourceSpec",
    "ContainerCategory",
    "ContainerResourceSpec",
    "LocalEnvResourceSpec",
    "VLLMModelResourceSpec",
    "APIModelResourceSpec",
    "BaseRunner",
    "LocalRunner",
    "RayRunner",
    "CloudRunner",
    "K8sRunner",
    "ContainerResource",
    "LocalEnvResource",
    "VLLMModelResource",
    "APIModelResource",
    "ResourceEngine",
    "RayContainerResource",
    "RayEnrootContainerActor",
    "create_ray_container_resource",
]
