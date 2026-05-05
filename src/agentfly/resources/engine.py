"""
Resource engine: unified facade for managing resources across backends.

Tools and rewards use the engine to start, acquire, release, and optionally
monitor/control resources. The engine holds a single heterogeneous store of
free resources (any type/spec/backend) and arranges them directly—no separate
pool layer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import uuid
from .types import BaseResource, ResourceStatus, BaseResourceSpec
from .runner import LocalRunner, RayRunner, CloudRunner, K8sRunner
if TYPE_CHECKING:
    from .runner import BaseRunner


def _pool_key(spec: BaseResourceSpec, backend: str) -> str:
    """Stable key for (spec, backend). Use '|' so backend can be parsed."""
    image = getattr(spec, "image", None)
    model_name_or_path = getattr(spec, "model_name_or_path", None)
    env_cls_path = getattr(spec, "env_cls_path", None)
    spec_key = f"{spec.category}:{image or model_name_or_path or env_cls_path or 'default'}"
    key = f"{spec_key}|{backend}"
    if backend == "ray":
        # Same image with different Ray placement options must not share a pool entry.
        ropts = getattr(spec, "ray_actor_options", None)
        key = f"{key}|{json.dumps(ropts, sort_keys=True, default=str) if ropts is not None else ''}"
    return key


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
        "ray": RayRunner(),
        "aws": CloudRunner(),
        "k8s": K8sRunner(),
    }
    # pool_key -> list of free resources (any type)
    _free: Dict[str, List[BaseResource]] = {}
    # id -> (resource, pool_key) for same-id reuse and correct return on release
    _acquired: Dict[str, Tuple[Optional[BaseResource], str]] = {}
    _lock = asyncio.Lock()
    _condition = None  # asyncio.Condition(_lock), created after _lock exists

    @classmethod
    def _get_condition(cls) -> asyncio.Condition:
        if cls._condition is None:
            cls._condition = asyncio.Condition(cls._lock)
        return cls._condition

    @classmethod
    def _count_for_pool(cls, key: str) -> int:
        """Return total number of resources for this pool (free + acquired)."""
        free_count = len(cls._free.get(key, []))
        acquired_count = sum(1 for _, (_, k) in cls._acquired.items() if k == key)
        return free_count + acquired_count

    @classmethod
    async def monitor(cls) -> Dict[str, Dict[str, int]]:
        """
        Return a statistics dictionary of existing resource counts per pool.

        Returns:
            Dict mapping pool_key -> {"total": N, "free": F, "acquired": A}.
            Only includes pool keys that have at least one resource.
        """
        stats: Dict[str, Dict[str, int]] = {}
        async with cls._lock:
            keys = set(cls._free.keys())
            for _, (_, key) in cls._acquired.items():
                keys.add(key)
            for key in keys:
                free_count = len(cls._free.get(key, []))
                acquired_count = sum(1 for _, (_, k) in cls._acquired.items() if k == key)
                stats[key] = {
                    "total": free_count + acquired_count,
                    "free": free_count,
                    "acquired": acquired_count,
                }
        return stats

    @classmethod
    async def start(
        cls,
        spec: BaseResourceSpec,
        size: int = 1,
        backend: str = "local",
        **kwargs: Any,
    ) -> None:
        """
        Ensure at least `size` free resources for (spec, backend).
        Creates resources via the backend runner and adds them to the store.
        """
        runner = cls._runners.get(backend)
        if runner is None:
            raise ValueError(f"Unknown backend: {backend}")
        key = _pool_key(spec, backend)
        
        pending_ids = []
        cond = cls._get_condition()
        async with cond:
            if key not in cls._free:
                cls._free[key] = []
            
            current_free = len(cls._free[key])
            if current_free >= size:
                return

            total_existing = cls._count_for_pool(key)
            cap = spec.max_global_num
            
            needed = size - current_free
            if cap is not None:
                needed = min(needed, cap - total_existing)
            
            if needed <= 0:
                return
            
            # Reserve slots with placeholders
            for _ in range(needed):
                p_id = f"_pending_{key}_{uuid.uuid4().hex}"
                cls._acquired[p_id] = (None, key)
                pending_ids.append(p_id)

        start_timeout = kwargs.get("timeout")

        # Create outside lock
        for p_id in pending_ids:
            try:
                resource = await runner.start_resource(spec, timeout=start_timeout)
                async with cond:
                    cls._acquired.pop(p_id, None)
                    cls._free.setdefault(key, []).append(resource)
                    cond.notify_all()
            except Exception:
                async with cond:
                    cls._acquired.pop(p_id, None)
                    cond.notify_all()
                raise

    @classmethod
    async def acquire(
        cls,
        id: str,
        spec: BaseResourceSpec,
        backend: str = "local",
        timeout: Optional[float] = 600.0,
    ) -> BaseResource:
        """
        Acquire a resource for the given id. Same id returns the same resource.

        Semantics:
        - Prefer taking from the free queue for (spec, backend).
        - If none free and under max_global_num, create a new resource (slot reserved under lock).
        - If at max_global_num and none free, wait until another thread releases one.

        Args:
            timeout: Max seconds for backend start_resource() only.
                Waiting for an existing/pending resource is unbounded.
        """
        key = _pool_key(spec, backend)
        runner = cls._runners.get(backend)
        if runner is None:
            raise ValueError(f"Unknown backend: {backend}")
        cond = cls._get_condition()

        async with cond:
            while True:
                # If already acquired (or being acquired) for this id, return or wait.
                if id in cls._acquired:
                    resource, _ = cls._acquired[id]
                    if resource is not None:
                        return resource
                    # placeholder present; wait for the first caller to finish starting it
                    await cond.wait()
                    continue

                # Take from the free queue if available.
                if key in cls._free and cls._free[key]:
                    resource = cls._free[key].pop()
                    cls._acquired[id] = (resource, key)
                    return resource

                current = cls._count_for_pool(key)
                if spec.max_global_num is None or current < spec.max_global_num:
                    # Reserve a slot using the actual id as placeholder
                    cls._acquired[id] = (None, key)
                    break
                await cond.wait()

        # Start the resource outside the lock.
        # Important: do NOT wrap runner.start_resource(...) with asyncio.wait_for here.
        # start_resource uses asyncio.to_thread for backend calls; cancelling wait_for only
        # cancels the coroutine, not the underlying thread/system call, which can leak
        # orphaned container starts. Rely on backend/native timeouts instead.
        try:
            resource = await runner.start_resource(spec, resource_id=id, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as e:
            async with cond:
                cls._acquired.pop(id, None)
                cond.notify_all()
            raise asyncio.TimeoutError(
                f"Resource start timed out (id={id!r}, pool={key}); backend start exceeded its timeout."
            ) from e
        except Exception:
            # Clean up the placeholder if starting fails so others can try
            async with cond:
                cls._acquired.pop(id, None)
                cond.notify_all()
            raise

        async with cond:
            # Complete the acquisition
            cls._acquired[id] = (resource, key)
            cond.notify_all()
        return resource

    @classmethod
    async def release(
        cls,
        id: str
    ) -> None:
        """Return resource to the store and unregister from acquired."""
        cond = cls._get_condition()
        async with cond:
            entry = cls._acquired.pop(id, None)
            if entry is None:
                return
            resource, key = entry
            if resource is None:
                return  # placeholder; already removed
            cls._free.setdefault(key, []).append(resource)
            cond.notify_all()

    @classmethod
    async def reset(
        cls,
        id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Reset resource state for the given id.

        Args:
            id: Identifier used when acquiring the resource.
        """
        async with cls._lock:
            entry = cls._acquired.get(id)
        if entry is None:
            raise ValueError(f"Resource with id {id!r} is not acquired and cannot be reset.")
        resource, _ = entry
        if resource is None:
            raise ValueError(f"Resource with id {id!r} is still being created.")
        return await resource.reset(*args, **kwargs)

    @classmethod
    async def get_status(cls, id: str) -> ResourceStatus:
        """
        Current execution/lifecycle status for the resource associated with id.

        Args:
            id: Identifier used when acquiring the resource.
        """
        async with cls._lock:
            entry = cls._acquired.get(id)
        if entry is None:
            raise ValueError(f"Resource with id {id!r} is not acquired and has no status.")
        resource, _ = entry
        if resource is None:
            raise ValueError(f"Resource with id {id!r} is still being created.")
        return await resource.get_status()

    @classmethod
    async def control(cls, id: str, **kwargs: Any) -> None:
        """
        Update resource limits for the resource associated with id where supported.

        Args:
            id: Identifier used when acquiring the resource.
        """
        async with cls._lock:
            entry = cls._acquired.get(id)
        if entry is None:
            raise ValueError(f"Resource with id {id!r} is not acquired and cannot be controlled.")
        resource, _ = entry
        if resource is None:
            raise ValueError(f"Resource with id {id!r} is still being created.")
        await resource.control(**kwargs)

    @classmethod
    async def end(cls, id: str) -> None:
        """
        End/kill the resource associated with id.

        Removes it from the acquired map and does not return it to the free store.

        Args:
            id: Identifier used when acquiring the resource.
        """
        async with cls._lock:
            entry = cls._acquired.pop(id, None)
        if entry is None:
            return
        resource, _ = entry
        if resource is not None:
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
            if resource is None:
                continue
            try:
                await resource.end()
            except Exception:
                pass
        for _, resource in free_list:
            try:
                await resource.end()
            except Exception:
                pass

