"""
Resource types and specs for the resource engine.

Defines resource kinds (container, vllm), resource specs (config for creating
a resource), and the base resource protocol that all managed resources implement.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


@dataclass
class ResourceSpec:
    """
    Specification for creating or scaling a managed resource.

    `ResourceSpec` is the declarative config passed to the resource engine when
    starting/acquiring resources. A subset of fields is used depending on the
    resource category and backend.

    Args:
        category: Resource kind, for example `"container"`, `"vllm"`, or
            `"python_env"`. Backends/runners use this to choose how to create
            and operate the resource.
        image: Container image reference (for container-like resources).
        cpu_count: Requested CPU amount for the resource (backend-dependent).
        mem_limit: Memory limit for the resource. Can be backend-native string
            (for example `"8g"`) or integer bytes depending on runtime.
        gpus: GPU request/count (for example `1`) or backend-specific selector
            string (for example `"0,1"`).
        ports: Port mapping/configuration used when exposing service ports.
            Structure is backend/runtime dependent.
        environment: Environment variables injected into the runtime.
        mount: Host-to-runtime mount mapping used by container-like resources.
        model_name_or_path: Model identifier/path for model-serving resources
            such as vLLM.
        tensor_parallel_size: Tensor-parallel degree for distributed model
            serving backends.
        port: Primary service port for model/runtime endpoints.
        extra: Free-form backend-specific options not covered by standard
            fields (for example Ray actor options).
        max_global_num: Maximum number of resources allowed for this spec
            (counting both free and acquired instances). For rollout scope this
            bounds concurrent instances; for global scope this caps total pool size.
    """
    category: str  # Resource kind: "container", "vllm", or "python_env"
    # Container-specific
    image: Optional[str] = None
    cpu_count: Optional[float] = None
    mem_limit: Optional[str | int] = None
    gpus: Optional[int | str] = None  # e.g. 1 or "0,1"
    ports: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, str]] = None
    mount: Optional[Dict[str, str]] = None
    # VLLM-specific
    model_name_or_path: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    port: Optional[int] = None
    # Common
    extra: Dict[str, Any] = field(default_factory=dict)
    # Max concurrent resources for this spec (free + acquired). When set:
    # - Rollout-scoped: at most this many can exist; only after one is ended can another start.
    # - Global-scoped: at most this many total; no new resources can be added when at cap.
    max_global_num: Optional[int] = None


class ResourceStatus(str, Enum):
    """Execution / lifecycle status of a resource."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


class BaseResource(abc.ABC):
    """
    Minimal contract for a managed resource.
    Implementations are created by runners (e.g. LocalRunner returns a
    ContainerResource that wraps an enroot Container).
    """

    @property
    @abc.abstractmethod
    def resource_id(self) -> str:
        """Unique identifier for this resource instance (e.g. container name, job id)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def category(self) -> str:
        """Resource kind (container, vllm, or python_env)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def start(self) -> None:
        """Start or ensure the resource is running."""
        raise NotImplementedError

    @abc.abstractmethod
    async def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset resource to initial state (e.g. clear container state)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_status(self) -> ResourceStatus:
        """Current execution/lifecycle status."""
        raise NotImplementedError

    @abc.abstractmethod
    async def control(self, **kwargs: Any) -> None:
        """Update resource limits (e.g. memory, cpu_cores, gpus) where supported."""
        raise NotImplementedError

    @abc.abstractmethod
    async def end(self) -> None:
        """End / kill the resource and release underlying handles."""
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Alias for end(); release all resources."""
        raise NotImplementedError
