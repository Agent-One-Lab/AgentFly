"""
Resource engine: decoupled design for managing resources (containers, vLLM)
across backends (local, Ray, AWS, K8s).

Tools and rewards use the engine to acquire, use, and release resources.

The lightweight specs/base types (``.types``) are imported eagerly — they are
stdlib-only and are on the hot path (``core.context`` imports them). The engine
machinery (runners, containers, models, engine) is imported lazily via
``__getattr__``: those modules pull heavy backends (ray, and the container/model
stacks) — and ``.containers`` even builds a Ray actor at import time — so a bare
``import agentfly.resources`` (and hence ``import agentfly.agents``) must not load
them until they're actually used.
"""

from typing import TYPE_CHECKING

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

# name -> submodule that defines it. No imports happen here; the module is loaded
# on first access via __getattr__ below.
_LAZY_EXPORTS = {
    "BaseRunner": ".runner",
    "LocalRunner": ".runner",
    "RayRunner": ".runner",
    "CloudRunner": ".runner",
    "K8sRunner": ".runner",
    "ContainerResource": ".containers",
    "RayContainerResource": ".containers",
    "RayEnrootContainerActor": ".containers",
    "create_ray_container_resource": ".containers",
    "LocalEnvResource": ".local_env_resource",
    "APIModelResource": ".models",
    "VLLMModelResource": ".models",
    "ResourceEngine": ".engine",
}


def __getattr__(name):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # cache so __getattr__ isn't hit again for this name
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # let type checkers/IDEs still see the lazily-exported symbols
    from .runner import BaseRunner, LocalRunner, RayRunner, CloudRunner, K8sRunner
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
