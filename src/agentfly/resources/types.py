"""
Resource types and specs for the resource engine.

Defines resource kinds, structured per-kind specs, and :class:`BaseResource`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional


# --- Structured specs ---

ContainerCategory = Literal[
    "container",
    "python_env",
    "scienceworld",
    "alfworld",
    "webshop",
]


@dataclass(kw_only=True)
class BaseResourceSpec:
    """Shared fields for structured resource specs."""

    max_global_num: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerResourceSpec(BaseResourceSpec):
    """Enroot-backed container resources (including env images)."""

    category: ContainerCategory
    image: Optional[str] = None
    cpu_count: Optional[float] = None
    mem_limit: Optional[str | int] = None
    gpus: Optional[int | str] = None
    ports: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, str]] = None
    mount: Optional[Dict[str, str]] = None
    host_ip: Optional[str] = None
    container_port: Optional[int] = None
    start_timeout: Optional[float] = None
    ray_actor_options: Optional[Dict[str, Any]] = None


@dataclass
class VLLMModelResourceSpec(BaseResourceSpec):
    """Locally launched vLLM model service."""

    category: Literal["vllm"] = "vllm"
    model_name_or_path: str = ""
    tensor_parallel_size: Optional[int] = None
    port: Optional[int] = None
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    template: Optional[str] = None
    gpu_memory_utilization: float = 0.8
    tool_call_parser: str = "hermes"


@dataclass
class APIModelResourceSpec(BaseResourceSpec):
    """Existing OpenAI-compatible HTTP API (no local process)."""

    category: Literal["api_model"] = "api_model"
    model_name_or_path: str = ""
    port: Optional[int] = None
    base_url: Optional[str] = None
    host: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout: Optional[float] = None


@dataclass
class LocalEnvResourceSpec(BaseResourceSpec):
    """In-process local Python env (no container, no remote process).

    The runner imports ``env_cls_path`` (a dotted path to a :class:`BaseEnv`
    subclass), instantiates it with ``init_kwargs``, calls ``start``, and
    wraps it in a :class:`LocalEnvResource` so it satisfies the
    :class:`BaseResource` contract. Attribute access on the wrapper
    delegates to the underlying env, so callers can use the env directly
    (``env.step(...)``, ``env.is_solved``, ...).
    """

    category: Literal["local_env"] = "local_env"
    env_cls_path: str = ""
    init_kwargs: Dict[str, Any] = field(default_factory=dict)



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
