"""
Runners (backends) for the resource engine.

Each runner knows how to start, monitor, control, and end a resource
on a specific infrastructure (local, Ray, AWS, K8s).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional
import asyncio
import logging
from enroot import from_env, random_name
from enroot.errors import APIError, EnrootError, TimeoutError as EnrootTimeoutError
from .containers import ContainerResource, create_ray_container_resource
from .models import APIModelResource, VLLMModelResource
from .types import (
    APIModelResourceSpec,
    BaseResource,
    ContainerResourceSpec,
    ResourceStatus,
    BaseResourceSpec,
    VLLMModelResourceSpec,
)

logger = logging.getLogger(__name__)


async def _start_enroot_container(
    client: Any,
    spec: ContainerResourceSpec,
    resource_id: Optional[str],
    timeout: Optional[float] = 1800.0,
    *,
    containers_registry: Dict[str, Any],
    runner_label: str = "EnrootRunner",
) -> BaseResource:
    """Create + start an enroot container on the local enroot client (this machine)."""
    name = resource_id or random_name(prefix="res")
    image = spec.image or "ubuntu:22.04"
    timeout_sec = 1800.0 if timeout is None else timeout
    create_kwargs: Dict[str, Any] = {
        "name": name,
        "environment": spec.environment or {},
        "mount": spec.mount or {},
        "ports": spec.ports,
        "timeout": timeout_sec,
    }

    start_kwargs: Dict[str, Any] = {
        "timeout": timeout_sec,
        "environment": spec.environment or {},
        "mount": spec.mount or {},
    }

    logger.debug(f"[{runner_label}]: creating container image={image} kwargs: {create_kwargs}")
    try:
        container = await client.containers.create_async(
            image,
            **create_kwargs,
        )

        logger.debug(f"[{runner_label}]: starting container name={name} kwargs: {start_kwargs}")

        await container.start_async(**start_kwargs)
        
    except (asyncio.TimeoutError, TimeoutError, EnrootTimeoutError) as e:
        raise asyncio.TimeoutError(
            f"Container create/start timed out for image={image!r}, name={name!r}, timeout={timeout!r}."
        ) from e
    except APIError as e:
        raise RuntimeError(
            f"Container create/start failed for image={image!r}, name={name!r}: {e}"
        ) from e
    except EnrootError as e:
        raise RuntimeError(
            f"Container create/start failed for image={image!r}, name={name!r}: {e}"
        ) from e

    containers_registry[container.name] = container

    if spec.category == "python_env":
        from ..envs.python_env import PythonSandboxEnv
        resource = PythonSandboxEnv(container=container, resource_id=container.name, spec=spec)
    elif spec.category == "scienceworld":
        from ..envs.scienceworld_env import ScienceWorldEnv
        resource = ScienceWorldEnv(container=container, resource_id=container.name, spec=spec)
    elif spec.category == "alfworld":
        from ..envs.alfworld_env import ALFWorldEnv
        resource = ALFWorldEnv(container=container, resource_id=container.name, spec=spec)
    elif spec.category == "webshop":
        from ..envs.webshop_text_env import WebShopEnv
        resource = WebShopEnv(container=container, resource_id=container.name, spec=spec)
    elif spec.category == "container":
        resource = ContainerResource(container=container, resource_id=container.name, spec=spec)
    else:
        raise ValueError(f"Unsupported container resource category: {spec.category}")

    await resource.start()
    return resource


class BaseRunner(abc.ABC):
    """
    Backend that creates and manages resources on a given infrastructure.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runner identifier (e.g. 'local', 'ray', 'aws', 'k8s')."""
        raise NotImplementedError

    @abc.abstractmethod
    async def start_resource(
        self,
        spec: BaseResourceSpec,
        resource_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseResource:
        """
        Start a single resource according to spec.
        Called by ResourceEngine when scaling up or on first acquire.

        ``resource_id`` is set on acquire paths when the engine has a stable id
        (e.g. rollout id) for naming the resource.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def end_resource(self, resource: BaseResource) -> None:
        """End / kill the resource and release handles."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        """Return current execution/lifecycle status of the resource."""
        raise NotImplementedError


class LocalRunner(BaseRunner):
    """
    Run resources on the local machine.
    Containers use the enroot Python client (src/enroot).
    """

    def __init__(self):
        self._containers: Dict[str, Any] = {}
        self.client = from_env()
        self.container_resources_categories = [
            "container",
            "python_env",
            "scienceworld",
            "alfworld",
            "webshop",
        ]

    @property
    def name(self) -> str:
        return "local"

    async def start_resource(
        self,
        spec: BaseResourceSpec,
        resource_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseResource:
        if spec.category in self.container_resources_categories:
            return await self._start_container(spec, resource_id, timeout=timeout)
        if spec.category == "vllm":
            return await self._start_vllm(spec, resource_id, timeout=timeout)
        if spec.category == "api_model":
            return await self._start_api_model(spec, resource_id, timeout=timeout)
        raise ValueError(f"LocalRunner does not support resource category: {spec.category}")

    async def _start_container(
        self,
        spec: ContainerResourceSpec,
        resource_id: Optional[str],
        timeout: Optional[float] = None,
    ) -> BaseResource:
        """Start a container using the enroot client (local placement)."""
        return await _start_enroot_container(
            self.client,
            spec,
            resource_id,
            timeout,
            containers_registry=self._containers,
            runner_label="LocalRunner",
        )

    async def _start_vllm(
        self,
        spec: VLLMModelResourceSpec,
        resource_id: Optional[str],
        timeout: Optional[float] = None,
    ) -> BaseResource:
        startup_timeout = timeout if timeout is not None else 300.0
        resource = VLLMModelResource(
            spec=spec,
            resource_id=resource_id,
            startup_timeout=startup_timeout,
        )
        await resource.start()
        return resource

    async def _start_api_model(
        self,
        spec: APIModelResourceSpec,
        resource_id: Optional[str],
        timeout: Optional[float] = None,
    ) -> BaseResource:
        startup_timeout = timeout if timeout is not None else 60.0
        resource = APIModelResource(
            spec=spec,
            resource_id=resource_id,
            startup_timeout=startup_timeout,
        )
        await resource.start()
        return resource

    async def end_resource(self, resource: BaseResource) -> None:
        await resource.end()
        self._containers.pop(resource.resource_id, None)

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        return await resource.get_status()


class RayRunner(BaseRunner):
    """
    Start containers as :class:`~agentfly.resources.containers.ray_container_resource.RayContainerResource`.

    Requires ``ray.init(...)`` in this process (or ``RAY_ADDRESS`` + init). Pass
    ``spec.ray_actor_options`` (e.g. ``scheduling_strategy``) to control
    which Ray worker runs the enroot-backed actor.
    """

    def __init__(self) -> None:
        self._resources: Dict[str, BaseResource] = {}

    @property
    def name(self) -> str:
        return "ray"

    async def start_resource(
        self,
        spec: ContainerResourceSpec,
        resource_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseResource:
        if spec.category != "container":
            raise ValueError(f"RayRunner currently supports only container, got: {spec.category}")
        rid = resource_id or random_name(prefix="ray_res")
        ray_opts = spec.ray_actor_options
        opts: Optional[dict[str, Any]] = dict(ray_opts) if isinstance(ray_opts, dict) else None
        start_timeout = timeout if timeout is not None else 1800.0
        resource = await create_ray_container_resource(
            spec,
            rid,
            start_timeout=start_timeout,
            ray_actor_options=opts,
        )
        self._resources[resource.resource_id] = resource
        return resource

    async def end_resource(self, resource: BaseResource) -> None:
        await resource.end()
        self._resources.pop(resource.resource_id, None)

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        return await resource.get_status()


class CloudRunner(BaseRunner):
    """Run resources on AWS (ECS/EKS/SageMaker). Stub for K8s/AWS."""

    @property
    def name(self) -> str:
        return "cloud"

    async def start_resource(
        self,
        spec: BaseResourceSpec,
        resource_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseResource:
        raise NotImplementedError("CloudRunner is a stub")

    async def end_resource(self, resource: BaseResource) -> None:
        raise NotImplementedError("CloudRunner is a stub")

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        raise NotImplementedError("CloudRunner is a stub")


class K8sRunner(BaseRunner):
    """Run resources on Kubernetes (Pods, Deployments)."""

    @property
    def name(self) -> str:
        return "k8s"

    async def start_resource(
        self,
        spec: BaseResourceSpec,
        resource_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> BaseResource:
        raise NotImplementedError("K8sRunner is a stub")

    async def end_resource(self, resource: BaseResource) -> None:
        raise NotImplementedError("K8sRunner is a stub")

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        raise NotImplementedError("K8sRunner is a stub")
