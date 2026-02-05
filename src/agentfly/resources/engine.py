"""
Resource engine: unified facade for managing resources across backends.

Tools and rewards use the engine to start, acquire, release, and optionally
monitor/control resources. The engine holds a single heterogeneous store of
free resources (any type/spec/backend) and arranges them directly—no separate
pool layer.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .types import BaseResource, ResourceSpec, ResourceStatus
from .runner import LocalRunner, SlurmRunner, CloudRunner, K8sRunner
if TYPE_CHECKING:
    from .runner import BaseRunner


def _pool_key(spec: ResourceSpec, backend: str) -> str:
    """Stable key for (spec, backend). Use '|' so backend can be parsed."""
    spec_key = f"{spec.category.value}:{spec.image or spec.model_name_or_path or 'default'}"
    return f"{spec_key}|{backend}"


class ResourceEngine:
    """
    Single entry point for resource management.

    Holds one store: free resources keyed by (spec, backend). Any resource
    type (container, vLLM, etc.) lives in this store. The engine creates
    resources via runners and arranges acquire/release/start/end directly.
    """

    # Backend name -> runner
    _runners: Dict[str, "BaseRunner"] = {
        "local": LocalRunner(),
        "slurm": SlurmRunner(),
        "aws": CloudRunner(),
        "k8s": K8sRunner(),
    }
    # pool_key -> list of free resources (any type)
    _free: Dict[str, List[BaseResource]] = {}
    # id -> (resource, pool_key) for same-id reuse and correct return on release
    _acquired: Dict[str, Tuple[BaseResource, str]] = {}
    _lock = asyncio.Lock()


    @classmethod
    async def start(
        cls,
        spec: ResourceSpec,
        size: int = 1,
        backend: str = "local",
        **kwargs: Any,
    ) -> None:
        """
        Ensure at least `size` free resources for (spec, backend).
        Creates resources via the backend runner and adds them to the store.
        """
        print(f"[ResourceEngine]: runners: {cls._runners}")
        runner = cls._runners.get(backend)
        if runner is None:
            raise ValueError(f"Unknown backend: {backend}")
        key = _pool_key(spec, backend)
        async with cls._lock:
            if key not in cls._free:
                cls._free[key] = []
            needed = size - len(cls._free[key])
        if needed <= 0:
            return
        # Create outside lock so we don't block other operations
        for _ in range(needed):
            resource = await runner.start_resource(spec)
            async with cls._lock:
                cls._free.setdefault(key, []).append(resource)

    @classmethod
    async def acquire(
        cls,
        id: str,
        spec: ResourceSpec,
        backend: str = "local",
    ) -> BaseResource:   
        """
        Acquire a resource for the given id. Same id returns the same resource.
        Uses a free resource for (spec, backend) or creates one via the runner.
        """
        key = _pool_key(spec, backend)
        async with cls._lock:
            if id in cls._acquired:
                return cls._acquired[id][0]
        runner = cls._runners.get(backend)
        if runner is None:
            raise ValueError(f"Unknown backend: {backend}")
        resource: Optional[BaseResource] = None
        async with cls._lock:
            if key in cls._free and cls._free[key]:
                resource = cls._free[key].pop()
        if resource is None:
            resource = await runner.start_resource(spec)
        async with cls._lock:
            cls._acquired[id] = (resource, key)
        return resource

    @classmethod
    async def release(
        cls,
        resource: BaseResource,
        id: str,
        finished: bool = True,
    ) -> None:
        """Return resource to the store and unregister from acquired."""
        async with cls._lock:
            entry = cls._acquired.pop(id, None)
        if entry is None:
            return
        _, key = entry
        async with cls._lock:
            cls._free.setdefault(key, []).append(resource)

    @classmethod
    async def reset(
        cls,
        resource: BaseResource,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """Reset resource state."""
        return await resource.reset(*args, **kwargs)

    @classmethod
    async def get_status(cls, resource: BaseResource) -> ResourceStatus:
        """Current execution/lifecycle status."""
        return await resource.get_status()

    @classmethod
    async def control(cls, resource: BaseResource, **kwargs: Any) -> None:
        """Update resource limits where supported."""
        await resource.control(**kwargs)

    @classmethod
    async def end(cls, resource: BaseResource) -> None:
        """End/kill resource. Removes from acquired if present; does not return to store."""
        async with cls._lock:
            for id, (r, _) in list(cls._acquired.items()):
                if r is resource:
                    del cls._acquired[id]
                    break
        await resource.end()

    @classmethod
    async def close(cls) -> None:
        """Shut down: end all free and acquired resources, clear store."""
        async with cls._lock:
            acquired_list = list(cls._acquired.values())
            free_list: List[Tuple[str, BaseResource]] = []
            for key, resources in cls._free.items():
                for r in resources:
                    free_list.append((key, r))
            cls._acquired.clear()
            cls._free.clear()
        for _, resource in acquired_list:
            try:
                await resource.end()
            except Exception:
                pass
        for _, resource in free_list:
            try:
                await resource.end()
            except Exception:
                pass

