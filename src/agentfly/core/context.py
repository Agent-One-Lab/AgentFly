"""
Context for tool execution in agentic RL framework.

The Context provides access to rollout metadata and resource management
for tools and rewards that need to run in containers or access shared resources.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from ..resources import ResourceEngine
from ..resources.types import BaseResource, ResourceSpec


def _spec_key(spec: ResourceSpec) -> str:
    """Stable hashable key for a spec (category + image or model)."""
    return f"{spec.category}:{spec.image or spec.model_name_or_path or 'default'}"


class Context:
    """
    Execution context for tools and rewards in agentic RL rollouts.

    Provides access to:
    - Rollout metadata (rollout_id, group_id, metadata)
    - Resource engine for acquiring containers and other resources

    The Context is automatically injected into tools and rewards that request it via
    type hints (e.g., `context: Context`).
    """

    def __init__(
        self,
        rollout_id: str,
        group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Context instance. All data here is rollout tied context

        Args:
            rollout_id: Unique identifier for this rollout/chain execution
            group_id: Optional group identifier for grouping related rollouts
            metadata: Optional dictionary containing additional metadata (e.g., image_id)
        """
        self.rollout_id = rollout_id
        self.group_id = group_id
        self.final_response: str = None
        self.trajectory: List[Dict[str, Any]] = []
        self.metadata = metadata or {}
        self.resource_engine = ResourceEngine
        self.rollout_resource_ids: List[str] = []
        self.global_resource_ids: Set[str] = set()
        # Spec keys acquired this rollout (set by acquire_resource). Tools can use
        # last_acquire_was_first to decide whether to reset (e.g. first acquisition of that spec).
        self._resource_acquired: Set[str] = set()
        # Set by acquire_resource: True if this spec was not in _resource_acquired before.
        self.last_acquire_was_first: bool = False

    def is_spec_acquired(self, spec: ResourceSpec) -> bool:
        """Return True if a resource for this spec has been acquired this rollout."""
        return _spec_key(spec) in self._resource_acquired

    async def monitor_resources(self) -> Dict[str, Dict[str, int]]:
        """
        Monitor the resources in the context.
        """
        return await self.resource_engine.monitor()

    async def acquire_resource(
        self,
        id: Optional[str] = None,
        spec: Optional["ResourceSpec"] = None,
        scope: Literal["rollout", "global"] = "rollout",
        backend: str = "local",
    ) -> "BaseResource":
        """
        Acquire a resource (e.g., container).

        Args:
            id: Resource id. For rollout and global scope defaults to rollout_id.
            spec: ResourceSpec (constructed from metadata image_id if not provided).
            scope: "rollout" = released when rollout ends; "global" = shared, reused.
            backend: Backend name (default "local").

        Returns:
            BaseResource: The acquired resource instance.
        """
        # Construct spec if not provided
        if spec is None:
            image_id = self.metadata.get("image_id")
            if image_id is None:
                raise ValueError(
                    "Either id+spec or image_id in metadata must be provided"
                )
            spec = ResourceSpec(
                category="container",
                image=image_id,
            )

        resource_id = id or self.rollout_id
        spec_key = _spec_key(spec)
        self.last_acquire_was_first = spec_key not in self._resource_acquired
        if scope == "rollout":
            if resource_id not in self.rollout_resource_ids:
                self.rollout_resource_ids.append(resource_id)
        else:
            self.global_resource_ids.add(resource_id)

        resource = await self.resource_engine.acquire(
            id=resource_id,
            spec=spec,
            backend=backend,
        )
        self._resource_acquired.add(spec_key)
        return resource

    async def reset_resource(
        self,
        scope: Literal["rollout", "global"] = "rollout",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Reset state for all resources in the given scope.

        Args:
            scope: "rollout" = reset resources acquired for this rollout;
                   "global" = reset resources in the shared global set.
            *args: Passed to ResourceEngine.reset(id, *args, **kwargs).
            **kwargs: Passed to ResourceEngine.reset(id, *args, **kwargs).
        """
        if scope == "rollout":
            ids = list(self.rollout_resource_ids)
        elif scope == "global":
            ids = list(self.global_resource_ids)
        else:
            raise ValueError(f"Unsupported scope: {scope!r}")

        for resource_id in ids:
            try:
                await self.resource_engine.reset(resource_id, *args, **kwargs)
            except ValueError:
                pass  # Resource not acquired (e.g. already released)

    async def release_resource(
        self,
        scope: Literal["rollout", "global"] = "rollout",
    ) -> None:
        """
        Release resources back to the pool for the given scope.
        """
        if scope == "rollout":
            ids = list(self.rollout_resource_ids)
            for resource_id in ids:
                await self.resource_engine.release(id=resource_id)
            self.rollout_resource_ids.clear()
        elif scope == "global":
            ids = list(self.global_resource_ids)
            for resource_id in ids:
                await self.resource_engine.release(id=resource_id)
            self.global_resource_ids.clear()
        else:
            raise ValueError(f"Unsupported scope: {scope!r}")

    async def end_resource(
        self,
        scope: Literal["rollout", "global"] = "rollout",
    ) -> None:
        """
        End/kill resources for the given scope (removed from engine, not returned to pool).
        """
        if scope == "rollout":
            ids = list(self.rollout_resource_ids)
            for resource_id in ids:
                await self.resource_engine.end(id=resource_id)
            self.rollout_resource_ids.clear()
        elif scope == "global":
            ids = list(self.global_resource_ids)
            for resource_id in ids:
                await self.resource_engine.end(id=resource_id)
            self.global_resource_ids.clear()
        else:
            raise ValueError(f"Unsupported scope: {scope!r}")

    def __repr__(self) -> str:
        return (
            f"Context(rollout_id={self.rollout_id!r}, "
            f"group_id={self.group_id!r}, "
            f"metadata={self.metadata!r})"
        )
