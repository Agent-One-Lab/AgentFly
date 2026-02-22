"""
Runners (backends) for the resource engine.

Each runner knows how to start, monitor, control, and end a resource
on a specific infrastructure (local, Slurm, AWS, K8s).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
from enroot import from_env, random_name
from .container_resource import ContainerResource
from .types import BaseResource, ResourceSpec, ResourceStatus

if TYPE_CHECKING:
    from .call_interface import CallInterface


class BaseRunner(abc.ABC):
    """
    Backend that creates and manages resources on a given infrastructure.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runner identifier (e.g. 'local', 'slurm', 'aws', 'k8s')."""
        raise NotImplementedError

    @abc.abstractmethod
    async def start_resource(self, spec: ResourceSpec) -> BaseResource:
        """
        Start a single resource according to spec.
        Called by ResourceEngine when scaling up or on first acquire.
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

    def get_call_interface(self, kind: str) -> Optional["CallInterface"]:
        """
        Return the call interface (MCP or input-text) for this runner and resource kind.
        Override in subclasses to return MCPToolCall or InputTextCall implementation.
        """
        return None


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

    async def start_resource(self, spec: ResourceSpec, resource_id: Optional[str] = None) -> BaseResource:
        if spec.category in self.container_resources_categories:
            return await self._start_container(spec, resource_id)
        if spec.category == "vllm":
            return await self._start_vllm(spec, resource_id)
        raise ValueError(f"LocalRunner does not support resource category: {spec.category}")

    async def _start_container(self, spec: ResourceSpec, resource_id: Optional[str]) -> BaseResource:
        """Start a container using the enroot client.
        All container-based resource default to use enroot to start locally for now. So we directly
        call their start method for local running.
        """
        name = resource_id or random_name(prefix="res")
        image = spec.image or "ubuntu:22.04"
        kwargs = {
            "name": name,
            "detach": True,
            "remove": False,
            "timeout": 60,
            "environment": spec.environment or {},
            "mount": spec.mount or {},
        }
        if spec.ports:
            kwargs["ports"] = spec.ports

        print(f"[LocalRunner]: starting kwargs: {kwargs}")
        container = self.client.containers.run(image, **kwargs)
        self._containers[container.name] = container

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
            from ..envs.webshop_text_env import WebAgentTextEnv
            resource = WebAgentTextEnv(container=container, resource_id=container.name, spec=spec)
        elif spec.category == "container":
            resource = ContainerResource(container=container, resource_id=container.name, spec=spec)
        else:
            raise ValueError(f"LocalRunner does not support resource category: {spec.category}")

        await resource.start()
        return resource

    async def _start_vllm(self, spec: ResourceSpec, resource_id: Optional[str]) -> BaseResource:
        # TODO: vLLM local process or subprocess; return VLLMResource
        raise NotImplementedError("LocalRunner VLLM not implemented.")

    async def end_resource(self, resource: BaseResource) -> None:
        await resource.end()
        self._containers.pop(resource.resource_id, None)

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        return await resource.get_status()


class SlurmRunner(BaseRunner):
    """Run resources via Slurm (sbatch/srun). Containers in job; vLLM in job."""

    @property
    def name(self) -> str:
        return "slurm"

    async def start_resource(self, spec: ResourceSpec, resource_id: Optional[str] = None) -> BaseResource:
        raise NotImplementedError("SlurmRunner is a stub")

    async def end_resource(self, resource: BaseResource) -> None:
        raise NotImplementedError("SlurmRunner is a stub")

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        raise NotImplementedError("SlurmRunner is a stub")


class CloudRunner(BaseRunner):
    """Run resources on AWS (ECS/EKS/SageMaker). Stub for K8s/AWS."""

    @property
    def name(self) -> str:
        return "cloud"

    async def start_resource(self, spec: ResourceSpec, resource_id: Optional[str] = None) -> BaseResource:
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

    async def start_resource(self, spec: ResourceSpec, resource_id: Optional[str] = None) -> BaseResource:
        raise NotImplementedError("K8sRunner is a stub")

    async def end_resource(self, resource: BaseResource) -> None:
        raise NotImplementedError("K8sRunner is a stub")

    async def get_status(self, resource: BaseResource) -> ResourceStatus:
        raise NotImplementedError("K8sRunner is a stub")
