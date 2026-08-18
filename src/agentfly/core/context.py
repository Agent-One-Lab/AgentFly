"""
Context for tool execution in agentic RL framework.

The Context provides access to rollout metadata and resource management
for tools and rewards that need to run in containers or access shared resources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set
from ..resources.types import BaseResource, ContainerResourceSpec, BaseResourceSpec
from .context_config import ContextConfig, resolve_resource_backend


def _coerce_context_config(raw: Optional[Any]) -> ContextConfig:
    """Normalize ``context_config`` to :class:`ContextConfig` using dict-like ``.get()``."""
    if raw is None:
        return ContextConfig()
    if isinstance(raw, ContextConfig):
        return raw
    resource_backend = raw.get("resource_backend") or "local"
    return ContextConfig(
        resource_backend=str(resource_backend).strip() or "local",
    )


def _spec_key(spec: BaseResourceSpec) -> str:
    """Stable hashable key for a spec (category + image / model / env_cls)."""
    image = getattr(spec, "image", None)
    model_name_or_path = getattr(spec, "model_name_or_path", None)
    env_cls_path = getattr(spec, "env_cls_path", None)
    return f"{spec.category}:{image or model_name_or_path or env_cls_path or 'default'}"


class Context:
    """
    Execution context for tools and rewards in agentic RL rollouts.
    """

    def __init__(
        self,
        rollout_id: str,
        group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        context_config: Optional[ContextConfig | Any] = None,
    ):
        """
        Args:
            rollout_id: Unique identifier for this rollout/chain execution
            group_id: Optional group identifier for grouping related rollouts
            metadata: Optional dictionary containing additional metadata (e.g., image_id).
                Stored as a shallow copy so mutating ``context.metadata`` (replace the
                whole dict) does not affect the caller's original mapping.
            context_config: :class:`ContextConfig`, or a mapping / object with the same fields
                (e.g. Hydra ``agent.run_config.context_config``).
        """
        self.rollout_id = rollout_id
        self.group_id = group_id
        self.final_response: str = None
        self.trajectory: List[Dict[str, Any]] = []
        self.trajectory_format: str = "flat"
        self.metadata = dict(metadata) if metadata else {}
        self.context_config: ContextConfig = _coerce_context_config(context_config)
        self.rollout_resource_ids: List[str] = []
        self.global_resource_ids: Set[str] = set()
        self._resource_acquired: Set[str] = set()
        self.last_acquire_was_first: bool = False

    @property
    def resource_engine(self):
        """Lazily import ResourceEngine so importing Context does not require enroot."""
        from ..resources import ResourceEngine

        return ResourceEngine

    @property
    def resource_backend(self) -> str:
        """Backend name for ``ResourceEngine.acquire`` (``ContextConfig`` then metadata)."""
        return resolve_resource_backend(self.metadata, self.context_config)

    def is_spec_acquired(self, spec: BaseResourceSpec) -> bool:
        """Return True if a resource for this spec has been acquired this rollout."""
        return _spec_key(spec) in self._resource_acquired

    async def monitor_resources(self) -> Dict[str, Dict[str, int]]:
        """Monitor the resources in the context."""
        return await self.resource_engine.monitor()

    async def acquire_resource(
        self,
        id: Optional[str] = None,
        spec: Optional[BaseResourceSpec] = None,
        scope: Literal["rollout", "global"] = "rollout",
        backend: Optional[str] = None,
        timeout: Optional[float] = 600.0,
    ) -> BaseResource:
        if backend is None:
            backend = self.resource_backend

        if spec is None:
            image_id = self.metadata.get("image_id")
            if image_id is None:
                raise ValueError(
                    "Either id+spec or image_id in metadata must be provided"
                )
            # Engine-specific placement/build fields ride along in metadata so
            # a caller can select and configure the backend per rollout without
            # constructing a spec itself. ``docker_host`` pins the docker
            # daemon; ``dockerfile``/``build_context``/``snapshot`` carry the
            # image DEFINITION that cloud engines (daytona) build server-side,
            # having no local daemon or registry to receive a pre-built tag.
            spec = ContainerResourceSpec(
                category="container",
                image=image_id,
                docker_host=self.metadata.get("docker_host"),
                container_engine=self.metadata.get("container_engine"),
                dockerfile=self.metadata.get("dockerfile"),
                build_context=self.metadata.get("build_context"),
                snapshot=self.metadata.get("snapshot"),
                mount=self.metadata.get("mount"),
                environment=self.metadata.get("environment"),
            )

        else:
            # Tools build their OWN container spec (shell/file/skills all do),
            # so they never take the branch above. Enrich any container spec
            # with the rollout-level fields from metadata, because HOW the
            # container is materialized is a property of the rollout, not of
            # whichever tool happens to touch it first:
            #   * engine + image definition — a cloud engine has no local
            #     daemon, so without these the tool's bare `image` is treated
            #     as a registry ref and the pull fails;
            #   * mounts — the FIRST acquire creates the container, so seeding
            #     declared by the caller must be merged in or it is simply lost
            #     (the tool's own mounts still win on key collision).
            if getattr(spec, "category", None) == "container":
                for field in ("container_engine", "dockerfile", "build_context",
                              "snapshot"):
                    if getattr(spec, field, None) is None:
                        value = self.metadata.get(field)
                        if value is not None:
                            setattr(spec, field, value)
                meta_mount = self.metadata.get("mount")
                if meta_mount:
                    spec.mount = {**meta_mount, **(spec.mount or {})}

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
            timeout=timeout,
        )
        self._resource_acquired.add(spec_key)

        return resource

    async def reset_resource(
        self,
        scope: Literal["rollout", "global"] = "rollout",
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
                pass

    async def release_resource(
        self,
        scope: Literal["rollout", "global"] = "rollout",
    ) -> None:
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
