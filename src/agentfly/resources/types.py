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
    Specification for creating or scaling a resource.
    Used by ResourceEngine.start() and acquire().
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
